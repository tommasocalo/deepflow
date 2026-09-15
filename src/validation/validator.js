import { topologicalSort } from './topology.js';
import { validateLayerParameters } from './parameters.js';
import { inferShape } from './shapes.js';
import {
    resolveLayerParameters,
    resolveParameterValue,
} from '../domain/expressions.js';

/**
 * @typedef {Object} ValidationReport
 * @property {string[]} topo
 * @property {string[]} cyclic
 * @property {Object<string, import('./shapes.js').Shape | null>} shapes
 * @property {Object<string, string[]>} errors
 * @property {Object<string, string[]>} warnings
 * @property {string[]} globalErrors
 * @property {boolean} ok
 */

const SOURCE_ONLY = new Set(['deepflow.io.Input']);
const SINK_ONLY = new Set(['deepflow.io.Output']);
const TEMPLATE_INPUT = 'deepflow.template.Input';
const TEMPLATE_OUTPUT = 'deepflow.template.Output';

const MAX_TEMPLATE_DEPTH = 8;

/**
 * Validate an architecture graph.
 *
 * Options:
 *   - `templates`: a map of template id → template, used when the graph
 *     contains SuperNodeInstance nodes. Without it, supernodes degrade to
 *     opaque passthrough.
 *   - `scope`: free-parameter scope for `$ref` resolution inside this
 *     graph's layer parameters. Inherited from the parent supernode when
 *     descending.
 *   - `externalInputs`: a map of template-boundary-input node id → shape,
 *     used to inject the parent supernode's incoming shape into the
 *     template's `Template In` block during recursive validation.
 *   - `depth`: recursion depth, capped to avoid infinite template loops.
 *
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} graph
 * @param {{
 *   templates?: Object<string, import('../domain/architecture/types.js').ArchitectureTemplate>,
 *   scope?: Object<string, *>,
 *   externalInputs?: Object<string, import('./shapes.js').Shape | null>,
 *   depth?: number,
 * }} [opts]
 * @returns {ValidationReport}
 */
export function validate(graph, opts = {}) {
    const {
        templates = {},
        scope = {},
        externalInputs = {},
        depth = 0,
    } = opts;

    const { order: topo, cyclic, adjacency: adj } = topologicalSort(graph);

    /** @type {Object<string, string[]>} */
    const errors = {};
    /** @type {Object<string, string[]>} */
    const warnings = {};
    /** @type {Object<string, import('./shapes.js').Shape | null>} */
    const shapes = {};
    /** @type {string[]} */
    const globalErrors = [];

    const push = (bucket, id, msg) => {
        if (!bucket[id]) bucket[id] = [];
        bucket[id].push(msg);
    };

    if (cyclic.length) {
        for (const id of cyclic) push(errors, id, 'This node is part of a cycle');
        globalErrors.push(`Graph has a cycle through ${cyclic.length} node(s)`);
    }

    for (const id of Object.keys(graph.nodes)) {
        const node = graph.nodes[id];
        const inDeg = adj.inn[id]?.length ?? 0;
        const outDeg = adj.out[id]?.length ?? 0;

        if (node.kind === 'layer') {
            const isTplInput = node.libraryKey === TEMPLATE_INPUT;
            const isTplOutput = node.libraryKey === TEMPLATE_OUTPUT;
            const isSourceOnly = SOURCE_ONLY.has(node.libraryKey) || isTplInput;
            const isSinkOnly = SINK_ONLY.has(node.libraryKey) || isTplOutput;

            if (isSourceOnly && inDeg > 0) {
                push(errors, id, 'Input nodes cannot have inputs');
            }
            if (!isSourceOnly && inDeg === 0) {
                push(warnings, id, 'No incoming connections');
            }
            if (isSinkOnly && outDeg > 0) {
                push(errors, id, 'Output nodes cannot have outgoing connections');
            }
            if (!isSinkOnly && outDeg === 0 && Object.keys(graph.nodes).length > 1) {
                push(warnings, id, 'No outgoing connections');
            }
            // Resolve $param refs against the active scope before checking
            // types, so a parameter bound to `$out` validates against the
            // bound value rather than the literal string.
            const resolved = resolveLayerParameters(node, scope);
            for (const issue of validateLayerParameters(resolved)) {
                push(errors, id, issue.message);
            }
        } else if (node.kind === 'supernode' || node.kind === 'repeat') {
            if (inDeg === 0) push(warnings, id, 'No incoming connections');
            if (outDeg === 0 && Object.keys(graph.nodes).length > 1) {
                push(warnings, id, 'No outgoing connections');
            }
            const tpl = templates[node.templateId];
            if (!tpl) {
                push(errors, id, 'Template missing — re-create or remove this node');
            }
            if (node.kind === 'repeat') {
                const c = Number(node.count);
                if (!Number.isFinite(c) || c < 1) {
                    push(errors, id, 'Repeat count must be a positive integer');
                }
            }
        }
    }

    // Shape inference along the topological order. For each node:
    //   - Layers: resolve `$param` references against `scope`, then run inference
    //   - Supernodes: build a sub-scope from the instance's bindings and
    //                 the template's free-parameter defaults, recursively
    //                 validate the template's graph with the parent shape
    //                 injected at its Template In block, and harvest the
    //                 Template Out shape as this instance's output
    //   - Template boundaries: pull from externalInputs[id] when injecting
    for (const id of topo) {
        const node = graph.nodes[id];
        const inputShapes = (adj.inn[id] ?? []).map((src) => shapes[src] ?? null);
        if (!node) {
            shapes[id] = null;
            continue;
        }

        if (node.kind === 'layer') {
            if (node.libraryKey === TEMPLATE_INPUT && externalInputs[id]) {
                shapes[id] = externalInputs[id];
                continue;
            }
            const resolved = resolveLayerParameters(node, scope);
            const result = inferShape(node.libraryKey, inputShapes, resolved.parameters);
            shapes[id] = result.output;
            for (const e of result.errors) push(errors, id, e);
        } else if (node.kind === 'supernode') {
            const tpl = templates[node.templateId];
            if (!tpl) {
                shapes[id] = null;
                continue;
            }
            if (depth >= MAX_TEMPLATE_DEPTH) {
                push(errors, id, `Template recursion exceeds depth ${MAX_TEMPLATE_DEPTH}`);
                shapes[id] = null;
                continue;
            }

            // Sub-scope: start from each free-param's default, then override
            // with whatever the instance has bound. Bindings are evaluated
            // against the current scope so a parent's `$x` can flow down.
            const subScope = {};
            for (const fp of tpl.freeParameters ?? []) {
                const raw =
                    node.parameterValues && node.parameterValues[fp.name] !== undefined
                        ? node.parameterValues[fp.name]
                        : fp.default;
                const resolved = resolveParameterValue(raw, scope);
                subScope[fp.name] = resolved;
            }

            const tplInputIds = Object.values(tpl.graph.nodes ?? {})
                .filter((n) => n.kind === 'layer' && n.libraryKey === TEMPLATE_INPUT)
                .map((n) => n.id);
            const tplOutputIds = Object.values(tpl.graph.nodes ?? {})
                .filter((n) => n.kind === 'layer' && n.libraryKey === TEMPLATE_OUTPUT)
                .map((n) => n.id);

            const childExternalInputs = {};
            const incoming = inputShapes[0] ?? null;
            for (const tid of tplInputIds) childExternalInputs[tid] = incoming;

            const subReport = validate(tpl.graph, {
                templates,
                scope: subScope,
                externalInputs: childExternalInputs,
                depth: depth + 1,
            });

            // Output shape = first templateOutput's shape (if any)
            let outShape = null;
            for (const tid of tplOutputIds) {
                outShape = subReport.shapes[tid] ?? outShape;
            }
            shapes[id] = outShape;

            // Surface internal issues at the instance level as a digest.
            const subErrCount = Object.values(subReport.errors).reduce(
                (a, e) => a + e.length,
                0,
            );
            if (subErrCount > 0) {
                push(
                    warnings,
                    id,
                    `Template "${tpl.name}" has ${subErrCount} internal error${subErrCount !== 1 ? 's' : ''}`,
                );
            }
        } else if (node.kind === 'repeat') {
            const tpl = templates[node.templateId];
            const count = Math.max(0, Number(node.count) || 0);
            if (!tpl || count === 0) {
                shapes[id] = inputShapes[0] ?? null;
                continue;
            }
            if (depth >= MAX_TEMPLATE_DEPTH) {
                push(errors, id, `Template recursion exceeds depth ${MAX_TEMPLATE_DEPTH}`);
                shapes[id] = null;
                continue;
            }
            const iterVar = node.iterationVariable || 'i';
            const free = tpl.freeParameters ?? [];
            const tplInputIds = Object.values(tpl.graph.nodes ?? {})
                .filter((n) => n.kind === 'layer' && n.libraryKey === TEMPLATE_INPUT)
                .map((n) => n.id);
            const tplOutputIds = Object.values(tpl.graph.nodes ?? {})
                .filter((n) => n.kind === 'layer' && n.libraryKey === TEMPLATE_OUTPUT)
                .map((n) => n.id);

            let carryShape = inputShapes[0] ?? null;
            let totalSubErrors = 0;
            for (let i = 0; i < count; i++) {
                const subScope = { ...scope, [iterVar]: i };
                for (const fp of free) {
                    let raw;
                    const binding = node.bindings?.[fp.name];
                    if (Array.isArray(binding)) {
                        raw = binding[i] ?? fp.default;
                    } else if (binding !== undefined && binding !== null && binding !== '') {
                        raw = binding;
                    } else {
                        raw = fp.default;
                    }
                    subScope[fp.name] = resolveParameterValue(raw, subScope);
                }
                const childExternalInputs = {};
                for (const tid of tplInputIds) childExternalInputs[tid] = carryShape;
                const subReport = validate(tpl.graph, {
                    templates,
                    scope: subScope,
                    externalInputs: childExternalInputs,
                    depth: depth + 1,
                });
                let outShape = null;
                for (const tid of tplOutputIds) {
                    outShape = subReport.shapes[tid] ?? outShape;
                }
                carryShape = outShape;
                totalSubErrors += Object.values(subReport.errors).reduce((a, e) => a + e.length, 0);
            }
            shapes[id] = carryShape;
            if (totalSubErrors > 0) {
                push(
                    warnings,
                    id,
                    `Repeat unrolling produced ${totalSubErrors} internal error${totalSubErrors !== 1 ? 's' : ''}`,
                );
            }
        } else {
            shapes[id] = null;
        }
    }

    const ok = Object.keys(errors).length === 0 && globalErrors.length === 0;
    return { topo, cyclic, shapes, errors, warnings, globalErrors, ok };
}
