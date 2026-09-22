import { newDomainId } from './factories.js';
import { resolveRepeatCount } from '../expressions.js';
import { resolveLevelContext } from './levelContext.js';

export const structureOverrideKey = (node, index = 0) => node?.kind === 'repeat' ? String(index) : 'single';

/** The original definition stays available while a local version is enabled. */
export function effectiveTemplateId(node, index = 0) {
    const override = node?.structureOverrides?.[structureOverrideKey(node, index)];
    return override?.enabled ? override.templateId : node?.templateId;
}

export function inspectedInstanceIndex(node, path, inspectedIterations = {}, resolvedCount) {
    if (node?.kind !== 'repeat') return 0;
    const selected = inspectedIterations[JSON.stringify(path)];
    const index = Number.isInteger(selected) && selected >= 0 ? selected : 0;
    const count = resolvedCount === undefined ? resolveRepeatCount(node.count).count : resolvedCount;
    return Number.isInteger(count) && count > 0 ? Math.min(index, count - 1) : resolvedCount === null ? 0 : index;
}

/**
 * Copy only the selected ancestor occurrences, then isolate the chosen subtree.
 * IDs inside definitions are retained, preserving ports, open tabs and selection.
 * Disabling a local version keeps it available for a later return to local editing.
 */
export function changeStructureScope(graph, templates, path, inspectedIterations, local) {
    if (!path?.length) return null;
    const context = resolveLevelContext(graph, path, templates, inspectedIterations);
    if (context.levels.length !== path.length) return null;
    const nextTemplates = { ...templates };
    const copyDefinition = (templateId, deep, copied = new Map()) => {
        if (copied.has(templateId)) return copied.get(templateId);
        const original = nextTemplates[templateId];
        if (!original) return null;
        const id = newDomainId();
        copied.set(templateId, id);
        const copy = { ...structuredClone(original), id, sourceTemplateId: templateId, structureVariant: true,
            componentStyleKey: original.componentStyleKey ?? original.standardKey };
        delete copy.standardKey;
        if (deep) {
            for (const node of Object.values(copy.graph.nodes)) {
                if (!['supernode', 'repeat'].includes(node.kind)) continue;
                node.templateId = copyDefinition(node.templateId, true, copied) ?? node.templateId;
                for (const override of Object.values(node.structureOverrides ?? {})) {
                    override.templateId = copyDefinition(override.templateId, true, copied) ?? override.templateId;
                }
            }
        }
        nextTemplates[id] = copy;
        return id;
    };
    const visit = (currentGraph, depth) => {
        const node = currentGraph.nodes[path[depth]];
        if (!node || !['supernode', 'repeat'].includes(node.kind) || !nextTemplates[node.templateId]) return null;
        const index = context.levels[depth].index;
        const key = structureOverrideKey(node, index);
        const existing = node.structureOverrides?.[key];
        const atTarget = depth === path.length - 1;
        let override;
        if (atTarget) {
            if (Boolean(existing?.enabled) === local) return currentGraph;
            if (!local) override = { ...existing, enabled: false };
            else {
                const templateId = nextTemplates[existing?.templateId] ? existing.templateId : copyDefinition(node.templateId, true);
                override = { templateId, enabled: true };
            }
        } else {
            // Restore a saved local ancestor instead of discarding its edits.
            const templateId = nextTemplates[existing?.templateId] ? existing.templateId : copyDefinition(node.templateId, false);
            if (!templateId || !nextTemplates[templateId]) return null;
            const childGraph = visit(nextTemplates[templateId].graph, depth + 1);
            if (!childGraph) return null;
            nextTemplates[templateId] = { ...nextTemplates[templateId], graph: childGraph };
            override = { templateId, enabled: true };
        }
        return { ...currentGraph, nodes: { ...currentGraph.nodes, [node.id]: {
            ...node, structureOverrides: { ...node.structureOverrides, [key]: override },
        } } };
    };
    const nextGraph = visit(graph, 0);
    return nextGraph ? { graph: nextGraph, templates: nextTemplates } : null;
}
