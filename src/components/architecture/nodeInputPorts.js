import { getTemplateInputIds, getOperationInputIds } from '../../domain/graphInputs.js';

export function inputPortsFor(node, templates) {
    const operation = getOperationInputIds(node);
    if (operation) return operation.map((id) => ({ id, label: id }));
    const graph = templates[node.templateId]?.graph;
    return graph ? getTemplateInputIds(graph).map((id) => ({ id, label: graph.nodes[id].label })) : [];
}
