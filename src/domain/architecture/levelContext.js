import { validate } from '../../validation/validator.js';
import { createValidationCache } from '../../validation/cache.js';
import { buildParameterScope, resolveRepeatCount } from '../expressions.js';
import { getIncomingEdges, getTemplateInputIds, mapTemplateInputs } from '../graphInputs.js';
import { isShapeCollection } from '../shapeCollections.js';
import { effectiveTemplateId, inspectedInstanceIndex } from './instanceStructure.js';

export const EMPTY_REPORT = Object.freeze({
    topo: [],
    cyclic: [],
    shapes: {},
    errors: {},
    warnings: {},
    globalErrors: [],
    ok: true,
});

export const cachedValidate = createValidationCache((graph, options) => {
    try {
        return validate(graph, options);
    } catch (err) {
        console.error('[useValidation] validate() threw', err);
        return {
            ...EMPTY_REPORT,
            ok: false,
            globalErrors: ['Validation could not complete. Review the graph and try again.'],
        };
    }
});

/**
 * Walk the path from the root to the open level. Each parent level is
 * validated so that the entered node's incoming activation and its bound
 * parameters flow into the child template — the block you are editing is
 * shown with the shapes it really sees at the selected Repeat index.
 */
export function resolveLevelContext(rootGraph, currentPath, templates, inspectedIterations = {}) {
    const root = { graph: rootGraph, scope: {}, externalInputs: {}, levels: [] };
    const levels = [];
    let graph = rootGraph;
    let scope = {};
    let externalInputs = {};
    let repeat = null;
    const path = [];
    for (const nodeId of currentPath) {
        path.push(nodeId);
        const node = graph?.nodes?.[nodeId];
        if (!node || !['supernode', 'repeat'].includes(node.kind) || !templates[node.templateId]?.graph) return root;

        const report = cachedValidate(graph, templates, scope, externalInputs);
        const incoming = getIncomingEdges(graph, nodeId);
        const isRepeat = node.kind === 'repeat';
        let nextScope;
        let nextExternal;
        let tpl;
        let inspectedIndex = 0;
        let count = null;
        repeat = null;
        if (isRepeat) {
            count = resolveRepeatCount(node.count, scope, incoming.map((edge) => report.shapes[edge.source] ?? null)).count;
            const index = inspectedInstanceIndex(node, path, inspectedIterations, count);
            inspectedIndex = index;
            let carry = null;
            // Previous sequential variants determine this occurrence's actual
            // input. Parallel variants instead consume their selected slice.
            for (let i = node.mode === 'parallel' ? index : 0; i <= index; i++) {
                tpl = templates[effectiveTemplateId(node, i)];
                if (!tpl?.graph) return root;
                const inputs = mapTemplateInputs(tpl.graph, incoming, (edge) => report.shapes[edge.source] ?? null);
                nextExternal = node.mode === 'parallel'
                    ? Object.fromEntries(Object.entries(inputs).map(([id, shape]) => [id, isShapeCollection(shape) ? shape.items[i] ?? null : shape]))
                    : i === 0 ? inputs : Object.fromEntries(getTemplateInputIds(tpl.graph).map((id) => [id, carry]));
                nextScope = buildParameterScope(tpl.freeParameters ?? [], node.bindings ?? {}, { ...scope, [node.iterationVariable || 'i']: i }, i, nextExternal);
                if (i < index) {
                    const previous = cachedValidate(tpl.graph, templates, nextScope, nextExternal);
                    const output = Object.values(tpl.graph.nodes).filter((n) => n.libraryKey === 'deepflow.template.Output').at(-1);
                    carry = output ? previous.shapes[output.id] ?? null : null;
                }
            }
            repeat = { path: [...path], count: count || 0, index, label: node.label, mode: node.mode ?? 'sequential' };
        } else {
            tpl = templates[effectiveTemplateId(node)];
            if (!tpl?.graph) return root;
            const inputs = mapTemplateInputs(tpl.graph, incoming, (edge) => report.shapes[edge.source] ?? null);
            nextExternal = inputs;
            nextScope = buildParameterScope(tpl.freeParameters ?? [], node.parameterValues ?? {}, scope, undefined, inputs);
        }
        levels.push({ instance: node, template: tpl, templateId: effectiveTemplateId(node, inspectedIndex), path: [...path], index: inspectedIndex, count });
        graph = tpl.graph;
        scope = nextScope;
        externalInputs = nextExternal;
    }
    return { graph, scope, externalInputs, repeat, levels };
}
