import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { resolveLevelContext } from '../domain/architecture/levelContext.js';

/** Match the definition opened by this pane's selected occurrence. */
export function useComponentTemplate(node) {
    const path = useArchitectureStore((s) => s.currentPath);
    const graph = useArchitectureStore((s) => s.graph);
    const iterations = useArchitectureStore((s) => s.inspectedIterations);
    const templates = useTemplatesStore((s) => s.templates);
    if (!node || !['supernode', 'repeat'].includes(node.kind)) return undefined;
    return resolveLevelContext(graph, [...path, node.id], templates, iterations).levels.at(-1)?.template;
}
