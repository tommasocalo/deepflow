import { getEntry } from '../../domain/library/index.js';

/** Group errors and warnings using node identity, not display names. */
export function groupGraphIssues(validation, graph, templates = {}) {
    const groups = [];
    if (validation.globalErrors?.length) {
        groups.push({ key: 'graph', nodeId: null, label: 'Graph', errors: validation.globalErrors, warnings: [] });
    }
    const errors = validation.errors ?? {};
    const warnings = validation.warnings ?? {};
    for (const nodeId of new Set([...Object.keys(errors), ...Object.keys(warnings)])) {
        const nodeErrors = errors[nodeId] ?? [];
        const nodeWarnings = warnings[nodeId] ?? [];
        if (!nodeErrors.length && !nodeWarnings.length) continue;
        const node = graph?.nodes?.[nodeId];
        const label = node?.label?.trim()
            || (node?.libraryKey && getEntry(node.libraryKey)?.name)
            || templates[node?.templateId]?.name
            || 'Unavailable node';
        groups.push({ key: `node:${nodeId}`, nodeId, label, errors: nodeErrors, warnings: nodeWarnings });
    }
    return groups;
}
