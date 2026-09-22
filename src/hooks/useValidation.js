import { useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { cachedValidate, EMPTY_REPORT, resolveLevelContext } from '../domain/architecture/levelContext.js';

export { resolveLevelContext } from '../domain/architecture/levelContext.js';

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
