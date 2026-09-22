import { useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { analyzeGradientFlow, gradientReportAtPath } from '../domain/gradientFlow.js';

export function useGradientFlow(enabled) {
    const graph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const templates = useTemplatesStore((s) => s.templates);
    const report = useMemo(() => enabled ? analyzeGradientFlow(graph, { templates }) : null, [enabled, graph, templates]);
    return useMemo(() => report ? gradientReportAtPath(report, currentPath, inspectedIterations) : null, [report, currentPath, inspectedIterations]);
}
