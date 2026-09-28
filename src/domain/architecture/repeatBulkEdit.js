import { resolveLevelContext } from './levelContext.js';
import { changeStructureScope } from './instanceStructure.js';

/** Apply one settings change to an immutable snapshot of repeat occurrences.
 * Current/Selected create private definitions before patching. All patches
 * every effective definition (shared plus enabled local variants). A missing
 * field in any selected structural variant aborts the entire edit. */
export function applyRepeatSettingsEdit({ graph, templates, path, inspectedIterations = {}, mode, indices = [], nodeId, kind, transform }) {
    const pathKey = JSON.stringify(path);
    const context = resolveLevelContext(graph, path, templates, inspectedIterations);
    const repeat = context.levels.at(-1);
    if (!repeat || repeat.instance.kind !== 'repeat' || !Number.isInteger(repeat.count) || repeat.count < 1) {
        return { error: 'Open a repeat with a valid instance count' };
    }
    const targets = mode === 'current' ? [repeat.index] : mode === 'all'
        ? Array.from({ length: repeat.count }, (_, index) => index) : indices;
    const unique = [...new Set(targets)].filter((index) => Number.isInteger(index) && index >= 0 && index < repeat.count);
    if (!unique.length) return { error: 'No instances selected' };

    // Validate every target first. A local variant may have removed or
    // replaced the field's node; partial application would be surprising.
    for (const index of unique) {
        const at = resolveLevelContext(graph, path, templates, { ...inspectedIterations, [pathKey]: index });
        const node = at.graph.nodes[nodeId];
        if (!node || (kind && node.kind !== kind)) return { error: `Instance ${index + 1} has no matching ${kind ?? 'node'}` };
    }

    let nextGraph = graph;
    let nextTemplates = templates;
    const patchedTemplates = new Set();
    for (const index of unique) {
        const atIndex = { ...inspectedIterations, [pathKey]: index };
        if (mode !== 'all') {
            const isolated = changeStructureScope(nextGraph, nextTemplates, path, atIndex, true);
            if (!isolated) return { error: `Could not isolate instance ${index + 1}` };
            nextGraph = isolated.graph;
            nextTemplates = isolated.templates;
        }
        const at = resolveLevelContext(nextGraph, path, nextTemplates, atIndex);
        const templateId = at.levels.at(-1)?.templateId;
        if (!templateId || patchedTemplates.has(templateId)) continue;
        const template = nextTemplates[templateId];
        const nextBody = transform(template.graph);
        if (nextBody === template.graph) return { error: `Instance ${index + 1} could not be changed` };
        nextTemplates = { ...nextTemplates, [templateId]: { ...template, graph: nextBody } };
        patchedTemplates.add(templateId);
    }
    return { graph: nextGraph, templates: nextTemplates, count: unique.length };
}
