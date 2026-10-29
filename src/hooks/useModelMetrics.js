import { useDeferredValue, useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { cachedModelMetrics } from '../domain/metrics/index.js';

/**
 * Whole-model metrics for the Model table. The document is deferred as one
 * value, so the graph and its templates always match, and typing elsewhere
 * never waits for the walk. Mounted only while a table is open.
 */
export function useModelMetrics() {
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    const documentState = useMemo(() => ({ graph, templates }), [graph, templates]);
    const deferred = useDeferredValue(documentState);
    return useMemo(() => ({
        graph: deferred.graph,
        templates: deferred.templates,
        metrics: cachedModelMetrics(deferred.graph, deferred.templates),
    }), [deferred]);
}
