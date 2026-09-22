import { topologicalSort } from '../validation/topology.js';
import { buildParameterScope, resolveLayerParameters, resolveRepeatCount } from '../domain/expressions.js';
import { getIncomingEdges, mapTemplateInputs, orderOperationEdges } from '../domain/graphInputs.js';
import { validate } from '../validation/validator.js';
import { getEntry } from '../domain/library/index.js';
import { isShapeCollection } from '../domain/shapeCollections.js';
import { effectiveTemplateId } from '../domain/architecture/instanceStructure.js';
import { indent, kwArgs, toPyIdent, toPyLiteral } from './python.js';

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
    const report = validate(graph, { templates });
    if (!report.ok) {
        throw new Error(`Cannot export invalid architecture: ${[...report.globalErrors, ...Object.values(report.errors).flat()].slice(0, 4).join('; ')}`);
    }
    const ctx = makeContext(templates);
    ctx.rootInputs = new Map(Object.values(graph.nodes).filter((node) => node.libraryKey === 'deepflow.io.Input').map((node, index) => [node.id, index ? `x${index + 1}` : 'x']));
    const forwardArgs = [...ctx.rootInputs.values()].join(', ') || 'x';
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
        if (item.kind === 'constant') {
            initLines.push(...item.initLines);
            aliasMap[item.id] = `self.${item.field}`;
        } else if (item.kind === 'input') {
            aliasMap[item.id] = item.name;
        } else if (item.kind === 'instantiate') {
            const out = outputName(item.id);
            const fieldName = `self.${item.field}`;
            const args = renderArgs(item.entry, item.parameters);
            initLines.push(`${fieldName} = ${item.initExpr ?? `${item.entry.pyTorchFn}(${args})`}`);
            const [src] = item.inputs;
            const call = item.forwardExpr ? item.forwardExpr(fieldName, exprFor(src)) : `${fieldName}(${exprFor(src)})`;
            forwardLines.push(`${out} = ${call}`);
        } else if (item.kind === 'passthrough') {
            const [src] = item.inputs;
            // alias rather than emit
            aliasMap[item.id] = exprFor(src);
        } else if (item.kind === 'combine') {
            const out = outputName(item.id);
            const inputs = item.inputs.map((id) => exprFor(id));
            for (const check of item.checks?.(inputs) ?? []) forwardLines.push(check);
            forwardLines.push(`${out} = ${item.expr(inputs)}`);
        } else if (item.kind === 'gradop') {
            const out = outputName(item.id);
            const [src] = item.inputs;
            forwardLines.push(`${out} = ${item.expr(exprFor(src))}`);
        }
    }

    // Multi-output models return a tuple in Output-node order.
    const outputRefs = plan.outputRefs ?? [];
    const finalExpr = outputRefs.length > 1
        ? outputRefs.map((id) => exprFor(id)).join(', ')
        : plan.finalId
            ? exprFor(plan.finalId)
            : (plan.items.length ? exprFor(plan.items[plan.items.length - 1].id) : 'x');

    const init = initLines.length
        ? initLines.join('\n')
        : 'pass';
    const fwd = forwardLines.length
        ? forwardLines.join('\n')
        : 'pass';

    const attnHelper = ctx.usesCausalAttention ? CAUSAL_ATTENTION_SRC : '';
    const helpers = attnHelper + (ctx.usesGRL
        ? `\nclass _GradReverse(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, x, lambd):\n        ctx.lambd = float(lambd)\n        return x.view_as(x)\n\n    @staticmethod\n    def backward(ctx, grad_output):\n        return grad_output.neg() * ctx.lambd, None\n\n\ndef _grad_reverse(x, lambd):\n    return _GradReverse.apply(x, lambd)\n`
        : '');

    return `import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n${helpers}\n\nclass ${toPyIdent(className, 'Model')}(nn.Module):\n    def __init__(self):\n        super().__init__()\n${indent(init, 2)}\n\n    def forward(self, ${forwardArgs}):\n${indent(fwd, 2)}\n        return ${finalExpr}\n`;
}

// ------------------- internals -------------------

const CAUSAL_ATTENTION_SRC = `
class CausalSelfAttention(nn.Module):
    """GPT-style masked multi-head self-attention (fused QKV + output projection)."""

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)
`;

function makeContext(templates) {
    return {
        templates,
        fieldCounter: {},
        usesGRL: false,
        usesCausalAttention: false,
        valueTypes: new Map(),
        expandedNodes: 0,
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
function expandGraph(graph, scope, ctx, { prefix = '', externalInputs = {}, externalShapes = {}, depth = 0 } = {}) {
    if (depth > 8) throw new Error('Template recursion exceeds depth 8');
    const { order, cyclic, adjacency } = topologicalSort(graph);
    if (cyclic.length) throw new Error('Cannot export a graph containing a cycle');
    // Resolve the same tensor context as validation, including derived input
    // dimensions. Runtime values and their static shapes travel together.
    const shapeReport = validate(graph, { templates: ctx.templates, scope, externalInputs: externalShapes, depth });
    const items = [];
    const outRef = {};
    const unique = (id) => `${prefix}${id}`;
    const typeOf = (ref) => ctx.valueTypes.get(ref) ?? null;
    const setType = (ref, type) => ctx.valueTypes.set(ref, type);
    const emit = (id, inputs, expr, type = null, checks) => {
        items.push({ id, kind: 'combine', inputs, expr, checks });
        setType(id, type);
        return id;
    };
    const select = (ref, index, id) => emit(id, [ref], ([x]) => `${x}[${index}]`, typeOf(ref)?.[index] ?? null);
    const flatten = (refs, label) => refs.flatMap((ref, index) => {
        const type = typeOf(ref);
        if (!Array.isArray(type)) return [ref];
        return flatten(type.map((_, itemIndex) => select(ref, itemIndex, `${label}_${index}_${itemIndex}`)), `${label}_${index}`);
    });

    for (const nodeId of order) {
        if (++ctx.expandedNodes > 20000) throw new Error('Expanded graph exceeds 20000 nodes');
        const node = graph.nodes[nodeId];
        if (!node) continue;
        const id = unique(nodeId);
        const edges = getIncomingEdges(graph, nodeId);
        // Apply edge wrappers before expansion so they also reach template boundaries.
        const edgeRefs = new Map();
        for (const edge of edges) {
            let ref = outRef[edge.source] ?? unique(edge.source);
            const op = edge.gradientOp ?? 'none';
            if (op !== 'none') {
                const wrapperId = unique(`${edge.id}_${op}`);
                const lambda = Number(edge.gradientOpLambda ?? 1);
                if (!Number.isFinite(lambda)) throw new Error('Gradient reversal coefficient must be finite');
                if (op === 'reverse') ctx.usesGRL = true;
                const transform = (x) => op === 'reverse' ? `_grad_reverse(${x}, ${lambda})` : `${x}.detach()`;
                const type = typeOf(ref);
                const transformValue = (x, valueType) => Array.isArray(valueType) ? `(${valueType.map((part, index) => transformValue(`${x}[${index}]`, part)).join(', ')},)` : transform(x);
                ref = emit(wrapperId, [ref], ([x]) => transformValue(x, type), type);
            }
            edgeRefs.set(edge, ref);
        }
        const inputRefs = orderOperationEdges(node, edges).map((edge) => edgeRefs.get(edge) ?? null);

        if (node.kind === 'layer') {
            const key = node.libraryKey;
            const entry = getEntry(key);
            const parameters = resolveLayerParameters(node, scope).parameters;
            if (key === 'deepflow.io.Input') {
                items.push({ id, kind: 'input', name: ctx.rootInputs.get(nodeId) ?? 'x' });
                outRef[nodeId] = id;
                continue;
            }
            if (['deepflow.template.Input', 'deepflow.template.Output', 'deepflow.io.Input', 'deepflow.io.Output'].includes(key)) {
                const refs = key === 'deepflow.template.Input' && Object.hasOwn(externalInputs, nodeId) ? [externalInputs[nodeId]] : inputRefs;
                items.push({ id, kind: 'passthrough', inputs: refs });
                setType(id, typeOf(refs[0]));
                outRef[nodeId] = id;
                continue;
            }
            if (key === 'deepflow.op.Parameter' || key === 'deepflow.op.Arange') {
                const field = ctx.nextField(node.label || entry.name);
                const value = `self.${field}`;
                const initLines = [];
                if (key === 'deepflow.op.Arange') {
                    initLines.push(`self.register_buffer(${toPyLiteral(field)}, torch.arange(${parameters.start ?? 0}, ${parameters.end ?? 8}, ${parameters.step ?? 1}, dtype=torch.float32))`);
                } else {
                    const shape = `(${parameters.shape.join(', ')}${parameters.shape.length === 1 ? ',' : ''})`;
                    const enabled = parameters.enabled !== false;
                    const initializer = enabled ? parameters.init ?? 'normal' : parameters.disabled_init ?? 'zeros';
                    if (enabled) initLines.push(`${value} = nn.Parameter(torch.empty(${shape}), requires_grad=${toPyLiteral(parameters.requires_grad ?? true)})`);
                    else initLines.push(`self.register_buffer(${toPyLiteral(field)}, torch.empty(${shape}))`);
                    if (initializer === 'normal') initLines.push(`nn.init.normal_(${value}, mean=0.0, std=${parameters.scale ?? 1})`);
                    else if (initializer === 'uniform') initLines.push(`nn.init.uniform_(${value}, a=-${parameters.scale ?? 1}, b=${parameters.scale ?? 1})`);
                    else if (initializer === 'kaiming_uniform') initLines.push(`nn.init.kaiming_uniform_(${value}, a=5 ** 0.5)`);
                    else initLines.push(`nn.init.${initializer}_(${value})`);
                    if (parameters.padding_idx !== null && parameters.padding_idx !== undefined) initLines.push(`with torch.no_grad():\n    ${value}[${parameters.padding_idx}].zero_()`);
                }
                items.push({ id, kind: 'constant', field, initLines });
                outRef[nodeId] = id;
                continue;
            }
            if (key === 'deepflow.op.PositionIndices') {
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `torch.arange(${x}.size(1), device=${x}.device, dtype=torch.long).unsqueeze(0).expand(${x}.size(0), -1)`);
                continue;
            }
            if (key === 'deepflow.op.Gather') {
                outRef[nodeId] = emit(id, inputRefs, ([indices, weight]) => `F.embedding(${indices}, ${weight}, padding_idx=${toPyLiteral(parameters.padding_idx)})`);
                continue;
            }
            if (key === 'deepflow.op.Subtract') {
                outRef[nodeId] = emit(id, inputRefs, ([a, b]) => `torch.sub(${a}, ${b})`);
                continue;
            }
            if (key === 'deepflow.op.Mean') {
                const dims = parameters.dim ?? [-1];
                const tuple = `(${dims.join(', ')},)`;
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `torch.mean(${x}, dim=${tuple}, keepdim=${toPyLiteral(parameters.keepdim ?? true)})`);
                continue;
            }
            if (key === 'deepflow.op.AddScalar' || key === 'deepflow.op.Pow') {
                const isPower = key === 'deepflow.op.Pow';
                const scalar = parameters[isPower ? 'exponent' : 'value'] ?? (isPower ? 2 : 0);
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `torch.${isPower ? 'pow' : 'add'}(${x}, ${toPyLiteral(scalar)})`);
                continue;
            }
            if (['deepflow.op.Rsqrt', 'deepflow.op.Exp', 'deepflow.op.Sin', 'deepflow.op.Cos'].includes(key)) {
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `torch.${key.split('.').at(-1).toLowerCase()}(${x})`);
                continue;
            }
            if (key === 'deepflow.op.Unsqueeze') {
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `torch.unsqueeze(${x}, dim=${parameters.dim ?? -1})`);
                continue;
            }
            if (key === 'deepflow.op.Slice') {
                const { dim = -1, start = 0, end = null, step = 1 } = parameters;
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `${x}[(slice(None),) * (${dim < 0 ? `${x}.dim() + ${dim}` : dim}) + (slice(${start}, ${toPyLiteral(end)}, ${step}),)]`);
                continue;
            }
            if (key === 'deepflow.op.Split' || key === 'deepflow.op.Broadcast') {
                const isSplit = key === 'deepflow.op.Split';
                const { count, error } = resolveRepeatCount(parameters[isSplit ? 'chunks' : 'count'] ?? 2);
                if (error) throw new Error(error);
                const dim = parameters.dim ?? -1;
                const type = Array.from({ length: count }, () => typeOf(inputRefs[0]));
                outRef[nodeId] = emit(id, inputRefs, ([x]) => isSplit
                    ? `torch.tensor_split(${x}, ${count}, dim=${dim})`
                    : `(${Array(count).fill(x).join(', ')},)`, type,
                isSplit ? ([x]) => [`assert ${x}.size(${dim}) % ${count} == 0, "Split dimension must be divisible by chunks"`] : undefined);
                continue;
            }
            if (['deepflow.op.Add', 'deepflow.op.Multiply', 'deepflow.op.Concat', 'deepflow.op.Stack'].includes(key)) {
                const flattened = flatten(inputRefs, `${id}_item`);
                const dim = parameters.dim ?? (key === 'deepflow.op.Stack' ? 0 : 1);
                const expr = key === 'deepflow.op.Concat' ? (xs) => `torch.cat([${xs.join(', ')}], dim=${dim})`
                    : key === 'deepflow.op.Stack' ? (xs) => `torch.stack([${xs.join(', ')}], dim=${dim})`
                    : (xs) => xs.slice(1).reduce((acc, x) => `torch.${key === 'deepflow.op.Add' ? 'add' : 'mul'}(${acc}, ${x})`, xs[0]);
                outRef[nodeId] = emit(id, flattened, expr);
                continue;
            }
            if (key === 'deepflow.op.Transpose') {
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `${x}.transpose(${parameters.dim0 ?? -2}, ${parameters.dim1 ?? -1})`);
                continue;
            }
            if (key === 'deepflow.op.MatMul') {
                outRef[nodeId] = emit(id, inputRefs, ([a, b]) => `torch.matmul(${a}, ${b})`);
                continue;
            }
            if (key === 'deepflow.op.Scale') {
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `(${x} * ${toPyLiteral(parameters.factor ?? 1)})`);
                continue;
            }
            if (key === 'deepflow.op.CausalMask') {
                outRef[nodeId] = emit(id, inputRefs, ([x]) => `${x}.masked_fill(torch.ones((${x}.size(-2), ${x}.size(-1)), dtype=torch.bool, device=${x}.device).triu(1), float('-inf'))`);
                continue;
            }
            if (!entry?.pyTorchFn) {
                items.push({ id, kind: 'passthrough', inputs: inputRefs });
                setType(id, typeOf(inputRefs[0]));
                outRef[nodeId] = id;
                continue;
            }
            const field = ctx.nextField(node.label || entry.name);
            const item = { id, kind: 'instantiate', field, entry, parameters, inputs: inputRefs };
            if (key === 'deepflow.nn.PositionalEmbedding') {
                item.initExpr = `torch.nn.Embedding(${toPyLiteral(parameters.max_len)}, ${toPyLiteral(parameters.embedding_dim)})`;
                item.forwardExpr = (f, x) => `${f}(torch.arange(${x}.size(1), device=${x}.device))`;
            } else if (key === 'deepflow.nn.CausalSelfAttention') {
                ctx.usesCausalAttention = true;
                item.initExpr = `CausalSelfAttention(${renderArgs(entry, parameters)})`;
            }
            items.push(item);
            outRef[nodeId] = id;
            continue;
        }

        if (node.kind === 'supernode' || node.kind === 'repeat') {
            const template = ctx.templates[node.templateId];
            if (!template) throw new Error(`Missing template: ${node.templateId}`);
            const isRepeat = node.kind === 'repeat';
            const parallel = isRepeat && node.mode === 'parallel';
            const { count, error } = isRepeat ? resolveRepeatCount(node.count, scope, edges.map((edge) => shapeReport.shapes[edge.source] ?? null)) : { count: 1 };
            if (error) throw new Error(error);
            let carry = inputRefs[0] ?? null;
            let carryShape = shapeReport.shapes[edges[0]?.source] ?? null;
            const outputs = [];
            for (let index = 0; index < count; index++) {
                const instanceTemplateId = effectiveTemplateId(node, index);
                const instanceTemplate = ctx.templates[instanceTemplateId];
                if (!instanceTemplate) throw new Error(`Missing instance template: ${instanceTemplateId}`);
                const initialShapes = mapTemplateInputs(instanceTemplate.graph, edges, (edge) => shapeReport.shapes[edge.source] ?? null);
                const initialInputs = mapTemplateInputs(instanceTemplate.graph, edges, (edge) => edgeRefs.get(edge));
                const iterationScope = isRepeat ? { ...scope, [node.iterationVariable || 'i']: index } : scope;
                const childShapes = Object.fromEntries(Object.entries(initialShapes).map(([port, shape]) => [port,
                    parallel ? isShapeCollection(shape) ? shape.items[index] ?? null : shape : isRepeat && index > 0 ? carryShape : shape]));
                const subScope = buildParameterScope(instanceTemplate.freeParameters, isRepeat ? node.bindings : node.parameterValues, iterationScope, isRepeat ? index : undefined, childShapes);
                const subInputs = {};
                for (const [port, ref] of Object.entries(initialInputs)) {
                    const type = typeOf(ref);
                    if (parallel && Array.isArray(type)) {
                        if (type.length !== count) throw new Error(`Parallel repeat needs ${count} items in every input collection`);
                        subInputs[port] = select(ref, index, `${id}_${index}_input_${port}`);
                    } else {
                        subInputs[port] = !parallel && isRepeat && index > 0 ? carry : ref;
                    }
                }
                const subPlan = expandGraph(instanceTemplate.graph, subScope, ctx, {
                    prefix: `${id}_${index}__`, externalInputs: subInputs, externalShapes: childShapes, depth: depth + 1,
                });
                items.push(...subPlan.items);
                carry = subPlan.finalId ?? carry;
                carryShape = subPlan.finalShape;
                outputs.push(carry);
            }
            outRef[nodeId] = parallel ? emit(id, outputs, (xs) => `(${xs.join(', ')},)`, outputs.map(typeOf)) : carry;
            continue;
        }
        items.push({ id, kind: 'passthrough', inputs: inputRefs });
        outRef[nodeId] = id;
    }
    let finalId = null;
    const outputRefs = [];
    for (const nodeId of order) {
        const node = graph.nodes[nodeId];
        if (!node) continue;
        if ((adjacency.out[nodeId]?.length ?? 0) === 0) finalId = outRef[nodeId] ?? unique(nodeId);
        if (node.kind === 'layer' && node.libraryKey === 'deepflow.io.Output') outputRefs.push(outRef[nodeId] ?? unique(nodeId));
        if (node.kind === 'layer' && node.libraryKey === 'deepflow.template.Output') finalId = outRef[nodeId] ?? unique(nodeId);
    }
    const output = Object.values(graph.nodes).filter((node) => node.libraryKey === 'deepflow.template.Output').at(-1);
    return { items, finalId, outputRefs, finalShape: output ? shapeReport.shapes[output.id] ?? null : null };
}

/** Render the keyword args for a layer instantiation, restricted to the
 *  parameters declared by its library entry. */
function renderArgs(entry, params) {
    const allowed = entry.parameters.map((p) => p.name);
    return kwArgs(params, allowed);
}
