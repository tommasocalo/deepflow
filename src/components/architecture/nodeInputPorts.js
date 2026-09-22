import { getTemplateInputIds } from '../../domain/graphInputs.js';

export function inputPortsFor(node, templates) {
    if (node.libraryKey === 'deepflow.op.MatMul') return [{ id: 'a', label: 'a' }, { id: 'b', label: 'b' }];
    const graph = templates[node.templateId]?.graph;
    return graph ? getTemplateInputIds(graph).map((id) => ({ id, label: graph.nodes[id].label })) : [];
}
