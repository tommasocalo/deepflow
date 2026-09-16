import { useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { validate } from '../validation/index.js';
import { createValidationCache } from '../validation/cache.js';
import { resolveParameterValue } from '../domain/expressions.js';

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

const TEMPLATE_INPUT = 'deepflow.template.Input';

/**
 * Walk the path from the root to the open level. Each parent level is
 * validated so that the entered node's incoming activation and its bound
 * parameters flow into the child template — the block you are editing is
 * shown with the shapes it really sees (iteration 0 for a Repeat).
 */
function resolveLevelContext(rootGraph, currentPath, templates) {
    const root = { graph: rootGraph, scope: {}, externalInputs: {} };
    let graph = rootGraph;
    let scope = {};
    let externalInputs = {};
    for (const nodeId of currentPath) {
        const node = graph?.nodes?.[nodeId];
        const tpl = node && (node.kind === 'supernode' || node.kind === 'repeat') ? templates[node.templateId] : null;
        if (!node || !tpl?.graph) return root;

        const report = cachedValidate(graph, templates, scope, externalInputs);
        const incomingEdge = graph.edges.find((e) => e.target === nodeId);
        const incoming = incomingEdge ? (report.shapes[incomingEdge.source] ?? null) : null;

        const isRepeat = node.kind === 'repeat';
        const nextScope = isRepeat ? { ...scope, [node.iterationVariable || 'i']: 0 } : {};
        for (const fp of tpl.freeParameters ?? []) {
            let raw;
            if (isRepeat) {
                const binding = node.bindings?.[fp.name];
                if (Array.isArray(binding)) raw = binding[0] ?? fp.default;
                else if (binding !== undefined && binding !== null && binding !== '') raw = binding;
                else raw = fp.default;
                nextScope[fp.name] = resolveParameterValue(raw, nextScope);
            } else {
                raw = node.parameterValues?.[fp.name] !== undefined ? node.parameterValues[fp.name] : fp.default;
                nextScope[fp.name] = resolveParameterValue(raw, scope);
            }
        }

        const nextExternal = {};
        for (const n of Object.values(tpl.graph.nodes ?? {})) {
            if (n.kind === 'layer' && n.libraryKey === TEMPLATE_INPUT) nextExternal[n.id] = incoming;
        }
        graph = tpl.graph;
        scope = nextScope;
        externalInputs = nextExternal;
    }
    return { graph, scope, externalInputs };
}

/**
 * Validate the graph at the current view level with the templates map
 * plumbed in, so SuperNodeInstance / Repeat shape inference can recurse.
 *
 * @returns {import('../validation/validator.js').ValidationReport}
 */
export function useValidation() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);

    const context = useMemo(() => {
        if (!rootGraph || typeof rootGraph !== 'object' || !rootGraph.nodes) return null;
        return resolveLevelContext(rootGraph, currentPath, templates);
    }, [rootGraph, currentPath, templates]);

    return useMemo(() => {
        if (!context?.graph) return EMPTY_REPORT;
        return cachedValidate(context.graph, templates, context.scope, context.externalInputs);
    }, [context, templates]);
}
