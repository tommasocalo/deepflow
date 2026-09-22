import { useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { validate } from '../validation/index.js';
import { createValidationCache } from '../validation/cache.js';
import { buildParameterScope, resolveRepeatCount } from '../domain/expressions.js';
import { getIncomingEdges, mapTemplateInputs } from '../domain/graphInputs.js';
import { isShapeCollection } from '../domain/shapeCollections.js';

const EMPTY_REPORT = Object.freeze({
    topo: [],
    cyclic: [],
    shapes: {},
    errors: {},
    warnings: {},
    globalErrors: [],
    ok: true,
});

const cachedValidate = createValidationCache((graph, options) => {
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
function resolveLevelContext(rootGraph, currentPath, templates, inspectedIterations = {}) {
    const root = { graph: rootGraph, scope: {}, externalInputs: {} };
    let graph = rootGraph;
    let scope = {};
    let externalInputs = {};
    let repeat = null;
    const path = [];
    for (const nodeId of currentPath) {
        path.push(nodeId);
        const node = graph?.nodes?.[nodeId];
        const tpl = node && (node.kind === 'supernode' || node.kind === 'repeat') ? templates[node.templateId] : null;
        if (!node || !tpl?.graph) return root;

        const report = cachedValidate(graph, templates, scope, externalInputs);
        const inputs = mapTemplateInputs(tpl.graph, getIncomingEdges(graph, nodeId), (edge) => report.shapes[edge.source] ?? null);
        const isRepeat = node.kind === 'repeat';
        let nextScope;
        let nextExternal = inputs;
        repeat = null;
        if (isRepeat) {
            const { count } = resolveRepeatCount(node.count, scope);
            const index = Math.max(0, Math.min((count || 1) - 1, inspectedIterations[JSON.stringify(path)] ?? 0));
            const iterationScope = (i) => buildParameterScope(tpl.freeParameters ?? [], node.bindings ?? {}, { ...scope, [node.iterationVariable || 'i']: i }, i);
            nextScope = iterationScope(index);
            if (node.mode === 'parallel') {
                nextExternal = Object.fromEntries(Object.entries(inputs).map(([id, shape]) => [id, isShapeCollection(shape) ? shape.items[index] ?? null : shape]));
            } else {
                for (let i = 0; i < index; i++) {
                    const previous = cachedValidate(tpl.graph, templates, iterationScope(i), nextExternal);
                    const output = Object.values(tpl.graph.nodes).filter((n) => n.libraryKey === 'deepflow.template.Output').at(-1);
                    const carry = output ? previous.shapes[output.id] ?? null : null;
                    nextExternal = Object.fromEntries(Object.keys(inputs).map((id) => [id, carry]));
                }
            }
            repeat = { path: [...path], count: count || 0, index, label: node.label, mode: node.mode ?? 'sequential' };
        } else {
            nextScope = buildParameterScope(tpl.freeParameters ?? [], node.parameterValues ?? {}, scope);
        }
        graph = tpl.graph;
        scope = nextScope;
        externalInputs = nextExternal;
    }
    return { graph, scope, externalInputs, repeat };
}

/**
 * Validate the graph at the current view level with the templates map
 * plumbed in, so SuperNodeInstance / Repeat shape inference can recurse.
 *
 * @returns {import('../validation/validator.js').ValidationReport}
 */
export function useValidationContext() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);

    return useMemo(() => {
        if (!rootGraph || typeof rootGraph !== 'object' || !rootGraph.nodes) return null;
        return resolveLevelContext(rootGraph, currentPath, templates, inspectedIterations);
    }, [rootGraph, currentPath, templates, inspectedIterations]);
}

export function useValidation() {
    const templates = useTemplatesStore((s) => s.templates);
    const context = useValidationContext();
    return useMemo(() => {
        if (!context?.graph) return EMPTY_REPORT;
        return cachedValidate(context.graph, templates, context.scope, context.externalInputs);
    }, [context, templates]);
}
