import { effectiveTemplateId } from '../../domain/architecture/instanceStructure.js';
import { getTemplateInputIds, getOperationInputIds } from '../../domain/graphInputs.js';

export function inputPortsFor(node, templates) {
    const operation = getOperationInputIds(node);
    if (operation) return operation.map((id) => ({ id, label: id }));
    const ids = node.kind === 'repeat' ? [node.templateId, ...Object.values(node.structureOverrides ?? {}).filter((override) => override?.enabled).map((override) => override.templateId)] : [effectiveTemplateId(node)];
    const ports = new Map();
    for (const templateId of ids) {
        const graph = templates[templateId]?.graph;
        if (graph) for (const id of getTemplateInputIds(graph)) ports.set(id, { id, label: graph.nodes[id].label });
    }
    return [...ports.values()];
}
