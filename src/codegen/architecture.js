import { topologicalSort } from '../validation/topology.js';
import { resolveLayerParameters, resolveParameterValue } from '../domain/expressions.js';
import { getEntry } from '../domain/library/index.js';
import { indent, kwArgs, toPyIdent } from './python.js';

/**
 * Generate a runnable PyTorch nn.Module class from an architecture graph.
 * SuperNodeInstance references are inlined; RepeatNodes are unrolled N
 * times. Layer-edge gradient operators are translated into wrapper layers
 * (detach / grad-reversal) inserted in the forward pass.
 *
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} graph
 * @param {Object} opts
 * @param {Object<string, import('../domain/architecture/types.js').ArchitectureTemplate>} [opts.templates]
 * @param {string} [opts.className]
 * @returns {string}  Python source
 */
export function generateArchitectureModule(graph, { templates = {}, className = 'Model' } = {}) {
    const ctx = makeContext(templates);
    const plan = expandGraph(graph, {}, ctx);

    const initLines = [];
    const forwardLines = [];

    // Assign short sequential names (t1, t2, ...) to each plan-node id so
    // the generated forward pass is readable.
    const nameMap = {};
    let counter = 0;
    const outputName = (id) => {
        if (nameMap[id]) return nameMap[id];
        counter += 1;
        nameMap[id] = `t${counter}`;
        return nameMap[id];
    };

    // For passthrough nodes, alias to the upstream name rather than emitting
    // `t2 = t1`. This keeps the forward pass tight.
    const aliasMap = {};
    const exprFor = (id) => (id ? aliasMap[id] || outputName(id) : 'x');

    for (const item of plan.items) {
        if (item.kind === 'instantiate') {
            const out = outputName(item.id);
            const fieldName = `self.${item.field}`;
            const args = renderArgs(item.entry, item.parameters);
            initLines.push(`${fieldName} = ${item.entry.pyTorchFn}(${args})`);
            const [src] = item.inputs;
            forwardLines.push(`${out} = ${fieldName}(${exprFor(src)})`);
        } else if (item.kind === 'passthrough') {
            const [src] = item.inputs;
            // alias rather than emit
            aliasMap[item.id] = exprFor(src);
        } else if (item.kind === 'combine') {
            const out = outputName(item.id);
            const inputs = item.inputs.map((id) => exprFor(id));
            forwardLines.push(`${out} = ${item.expr(inputs)}`);
        } else if (item.kind === 'gradop') {
            const out = outputName(item.id);
            const [src] = item.inputs;
            forwardLines.push(`${out} = ${item.expr(exprFor(src))}`);
        }
    }

    const finalExpr = plan.finalId
        ? exprFor(plan.finalId)
        : (plan.items.length ? exprFor(plan.items[plan.items.length - 1].id) : 'x');

    const init = initLines.length
        ? initLines.join('\n')
        : 'pass';
    const fwd = forwardLines.length
        ? forwardLines.join('\n')
        : 'pass';

    const helpers = ctx.usesGRL
        ? `\nclass _GradReverse(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x, lambd):\n        ctx.lambd = float(lambd)\n        return x.view_as(x)\n\n    @staticmethod\n    def backward(ctx, grad_output):\n        return grad_output.neg() * ctx.lambd, None\n\n\ndef _grad_reverse(x, lambd):\n    return _GradReverse.apply(x, lambd)\n`
        : '';

    return `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n${helpers}\n\nclass ${toPyIdent(className, 'Model')}(nn.Module):\n    def __init__(self):\n        super().__init__()\n${indent(init, 2)}\n\n    def forward(self, x):\n${indent(fwd, 2)}\n        return ${finalExpr}\n`;
}

// ------------------- internals -------------------

function makeContext(templates) {
    return {
        templates,
        fieldCounter: {},
        usesGRL: false,
        nextField(name) {
            const base = toPyIdent(name, 'layer');
            const n = (this.fieldCounter[base] = (this.fieldCounter[base] || 0) + 1);
            return n === 1 ? base : `${base}_${n}`;
        },
    };
}

/**
 * Expand a graph into a flat sequence of plan items (instantiations and
 * forward-pass calls). The output `items` are in topological order so the
 * caller can emit them straight to source.
 *
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} graph
 * @param {Object<string, *>} scope
 * @param {ReturnType<typeof makeContext>} ctx
 */
function expandGraph(graph, scope, ctx) {
    const { order, adjacency } = topologicalSort(graph);

    /** @type {Array<{ id: string, kind: string, [key: string]: any }>} */
    const items = [];
    /** Map an internal-node id → the plan-item id whose forward output represents that node's value. */
    const outRef = {};

    for (const id of order) {
        const node = graph.nodes[id];
        if (!node) continue;
        const inIds = adjacency.inn[id] ?? [];
        const inputRefs = inIds.map((src) => outRef[src] ?? src);

        if (node.kind === 'layer') {
            const entry = getEntry(node.libraryKey);
            // Boundary / IO blocks are pure pass-throughs in codegen.
            if (
                !entry?.pyTorchFn ||
                node.libraryKey === 'deepflow.template.Input' ||
                node.libraryKey === 'deepflow.template.Output' ||
                node.libraryKey === 'deepflow.io.Input' ||
                node.libraryKey === 'deepflow.io.Output' ||
                node.libraryKey === 'deepflow.op.Split'
            ) {
                items.push({ id, kind: 'passthrough', inputs: inputRefs });
                outRef[id] = id;
                continue;
            }
            // Combine / element-wise ops with multiple inputs.
            if (node.libraryKey === 'deepflow.op.Add') {
                items.push({ id, kind: 'combine', inputs: inputRefs, expr: (xs) => xs.length === 2 ? `torch.add(${xs[0]}, ${xs[1]})` : `${xs.slice(1).reduce((a, x) => `torch.add(${a}, ${x})`, xs[0])}` });
                outRef[id] = id;
                continue;
            }
            if (node.libraryKey === 'deepflow.op.Multiply') {
                items.push({ id, kind: 'combine', inputs: inputRefs, expr: (xs) => xs.slice(1).reduce((a, x) => `torch.mul(${a}, ${x})`, xs[0]) });
                outRef[id] = id;
                continue;
            }
            if (node.libraryKey === 'deepflow.op.Concat') {
                const dim = resolveParameterValue(node.parameters?.dim, scope) ?? 1;
                items.push({ id, kind: 'combine', inputs: inputRefs, expr: (xs) => `torch.cat([${xs.join(', ')}], dim=${dim})` });
                outRef[id] = id;
                continue;
            }

            // Standard layer: resolve $param refs, instantiate.
            const resolved = resolveLayerParameters(node, scope);
            const field = ctx.nextField(node.label || entry.name);
            items.push({
                id,
                kind: 'instantiate',
                field,
                entry,
                parameters: resolved.parameters,
                inputs: inputRefs,
            });
            outRef[id] = id;
            continue;
        }

        if (node.kind === 'supernode') {
            const tpl = ctx.templates[node.templateId];
            if (!tpl) {
                items.push({ id, kind: 'passthrough', inputs: inputRefs });
                outRef[id] = id;
                continue;
            }
            const subScope = buildSubScope(tpl.freeParameters ?? [], node.parameterValues ?? {}, scope);
            const subPlan = expandGraph(tpl.graph, subScope, ctx);
            // Wire the supernode's external input(s) into the template's
            // boundary input(s) before namespacing.
            patchPlanInputs(subPlan, tpl.graph, inputRefs);
            renameSubPlanIds(subPlan, `i_${id}`);
            for (const item of subPlan.items) items.push(item);
            outRef[id] = subPlan.finalId ?? id;
            continue;
        }

        if (node.kind === 'repeat') {
            const tpl = ctx.templates[node.templateId];
            const count = Math.max(0, Number(node.count) || 0);
            if (!tpl || count === 0) {
                items.push({ id, kind: 'passthrough', inputs: inputRefs });
                outRef[id] = id;
                continue;
            }
            const iterVar = node.iterationVariable || 'i';
            let carry = inputRefs[0] ?? null;
            for (let i = 0; i < count; i++) {
                const scopeI = { ...scope, [iterVar]: i };
                for (const fp of tpl.freeParameters ?? []) {
                    let raw;
                    const binding = node.bindings?.[fp.name];
                    if (Array.isArray(binding)) raw = binding[i] ?? fp.default;
                    else if (binding !== undefined && binding !== null && binding !== '') raw = binding;
                    else raw = fp.default;
                    scopeI[fp.name] = resolveParameterValue(raw, scopeI);
                }
                const subPlan = expandGraph(tpl.graph, scopeI, ctx);
                patchPlanInputs(subPlan, tpl.graph, [carry]);
                renameSubPlanIds(subPlan, `r_${id}_${i}`);
                for (const item of subPlan.items) items.push(item);
                carry = subPlan.finalId ?? carry;
            }
            outRef[id] = carry ?? id;
            continue;
        }

        items.push({ id, kind: 'passthrough', inputs: inputRefs });
        outRef[id] = id;
    }

    // Apply gradient-op edge wrappers: when an outgoing edge from `src` to
    // `tgt` carries detach/stop/reverse, replace tgt's input reference with
    // a wrapper plan item.
    for (const edge of graph.edges) {
        const op = edge.gradientOp ?? 'none';
        if (op === 'none') continue;
        const tgtItem = items.find((it) => it.id === edge.target);
        if (!tgtItem || !tgtItem.inputs) continue;
        const srcRef = outRef[edge.source];
        const wrapperId = `${edge.id}_${op}`;
        const lambd = edge.gradientOpLambda ?? 1;
        if (op === 'reverse') ctx.usesGRL = true;
        const wrapperExpr = (x) =>
            op === 'detach' ? `${x}.detach()` :
            op === 'stop'   ? `${x}.detach()` :
            op === 'reverse' ? `_grad_reverse(${x}, ${lambd})` :
            x;
        items.splice(items.indexOf(tgtItem), 0, {
            id: wrapperId,
            kind: 'gradop',
            inputs: [srcRef],
            expr: wrapperExpr,
        });
        // Replace this specific input ref in the target item.
        tgtItem.inputs = tgtItem.inputs.map((r) => (r === srcRef ? wrapperId : r));
    }

    // Identify the final node — the last topologically-ordered output / passthrough
    // that has no outgoing edges.
    let finalId = null;
    for (const id of order) {
        const node = graph.nodes[id];
        if (!node) continue;
        const outs = adjacency.out[id]?.length ?? 0;
        if (outs === 0) finalId = outRef[id] ?? id;
    }
    return { items, finalId };
}

function buildSubScope(freeParameters, bindings, parentScope) {
    const sub = { ...parentScope };
    for (const fp of freeParameters) {
        const raw =
            bindings && bindings[fp.name] !== undefined ? bindings[fp.name] : fp.default;
        sub[fp.name] = resolveParameterValue(raw, parentScope);
    }
    return sub;
}

/**
 * Rewrite all item ids inside a sub-plan to be unique by appending a
 * suffix. This is what lets the same template be expanded multiple times
 * (Repeat or repeated SuperNode use) without forward-pass variables
 * colliding into a single name.
 */
function renameSubPlanIds(subPlan, suffix) {
    /** @type {Object<string, string>} */
    const map = {};
    for (const it of subPlan.items) map[it.id] = `${it.id}_${suffix}`;
    for (const it of subPlan.items) {
        it.id = map[it.id];
        if (it.inputs) it.inputs = it.inputs.map((r) => map[r] ?? r);
    }
    if (subPlan.finalId && map[subPlan.finalId]) {
        subPlan.finalId = map[subPlan.finalId];
    }
    return map;
}

/**
 * After expanding a sub-plan, replace any references its boundary-input
 * passthrough items had to "external inputs" with the actual ref(s) coming
 * from the parent.
 */
function patchPlanInputs(subPlan, tplGraph, parentRefs) {
    const tplInputIds = Object.values(tplGraph.nodes ?? {})
        .filter((n) => n.kind === 'layer' && n.libraryKey === 'deepflow.template.Input')
        .map((n) => n.id);
    if (!tplInputIds.length) return;
    const ref = parentRefs[0] ?? null;
    for (const item of subPlan.items) {
        if (tplInputIds.includes(item.id) && item.kind === 'passthrough') {
            item.inputs = [ref];
        }
    }
}

/** Render the keyword args for a layer instantiation, restricted to the
 *  parameters declared by its library entry. */
function renderArgs(entry, params) {
    const allowed = entry.parameters.map((p) => p.name);
    return kwArgs(params, allowed);
}
