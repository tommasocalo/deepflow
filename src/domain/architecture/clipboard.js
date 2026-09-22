import { newDomainId } from './factories.js';
import { containsTemplate, referencedTemplateIds } from './templateReferences.js';

export const GRAPH_CLIPBOARD_MIME = 'application/x-deepflow-nodes+json';
const FORMAT = 'deepflow-node-selection';

/** The clipboard is a snapshot of instances; component definitions remain shared. */
export function copyGraphSelection(graph, selectedIds) {
    const ids = new Set(selectedIds.filter((id) => graph.nodes[id]));
    if (!ids.size) return null;
    return structuredClone({
        format: FORMAT,
        version: 1,
        nodes: [...ids].map((id) => graph.nodes[id]),
        edges: graph.edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)),
    });
}

export function isGraphClipboard(value) {
    if (value?.format !== FORMAT || value.version !== 1 || !Array.isArray(value.nodes) || !value.nodes.length || !Array.isArray(value.edges)) return false;
    const ids = new Set();
    for (const node of value.nodes) {
        if (!node || typeof node.id !== 'string' || !node.id || ids.has(node.id)
            || !['layer', 'supernode', 'repeat'].includes(node.kind)
            || !Number.isFinite(node.position?.x) || !Number.isFinite(node.position?.y)
            || (node.kind === 'layer' ? typeof node.libraryKey !== 'string' : typeof node.templateId !== 'string')) return false;
        ids.add(node.id);
    }
    return value.edges.every((edge) => edge && ids.has(edge.source) && ids.has(edge.target));
}

export function readGraphClipboard(text) {
    try {
        const value = JSON.parse(text);
        return isGraphClipboard(value) ? value : null;
    } catch { return null; }
}

/** Validate the whole insertion before changing either the graph or history. */
export function pasteGraphSelection(graph, selection, { templates = {}, templateId = null, offset = { x: 36, y: 36 } } = {}) {
    if (!isGraphClipboard(selection)) return { error: 'The clipboard does not contain graph nodes.' };
    for (const node of selection.nodes) {
        for (const reference of referencedTemplateIds(node)) {
            if (!templates[reference]) return { error: 'The copied component definition is no longer in this project.' };
            if (templateId && containsTemplate(reference, templateId, templates)) return { error: 'A component cannot contain itself. Paste at another level.' };
        }
    }
    const ids = new Map(selection.nodes.map((node) => [node.id, newDomainId()]));
    const nodes = { ...graph.nodes };
    for (const original of selection.nodes) {
        const node = structuredClone(original);
        node.id = ids.get(original.id);
        node.position = { x: node.position.x + offset.x, y: node.position.y + offset.y };
        nodes[node.id] = node;
    }
    const edges = selection.edges.map((edge) => ({
        ...structuredClone(edge), id: newDomainId(), source: ids.get(edge.source), target: ids.get(edge.target),
    }));
    return { graph: { ...graph, nodes, edges: [...graph.edges, ...edges] }, nodeIds: [...ids.values()] };
}
