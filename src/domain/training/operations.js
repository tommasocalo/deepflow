/**
 * Immutable operations over a TrainingGraph.
 */

export function addTrainingNode(graph, node) {
    return { ...graph, nodes: { ...graph.nodes, [node.id]: node } };
}

export function removeTrainingNode(graph, id) {
    const { [id]: _gone, ...rest } = graph.nodes;
    const edges = graph.edges.filter((e) => e.source !== id && e.target !== id);
    // Drop dangling references this node had to others (e.g. PhaseNode's
    // optimizer/loss id lists)
    let nodes = rest;
    for (const key of Object.keys(rest)) {
        const n = rest[key];
        if (n.kind === 'phase') {
            const optimizerIds = n.optimizerIds.filter((x) => x !== id);
            const lossIds = n.lossIds.filter((x) => x !== id);
            if (optimizerIds.length !== n.optimizerIds.length || lossIds.length !== n.lossIds.length) {
                nodes = { ...nodes, [key]: { ...n, optimizerIds, lossIds } };
            }
        }
    }
    return { nodes, edges };
}

export function patchTrainingNode(graph, id, patch) {
    const existing = graph.nodes[id];
    if (!existing) return graph;
    return { ...graph, nodes: { ...graph.nodes, [id]: { ...existing, ...patch } } };
}

export function patchTrainingHyperparameters(graph, id, patch) {
    const existing = graph.nodes[id];
    if (!existing) return graph;
    return {
        ...graph,
        nodes: {
            ...graph.nodes,
            [id]: { ...existing, hyperparameters: { ...existing.hyperparameters, ...patch } },
        },
    };
}

export function moveTrainingNode(graph, id, position) {
    return patchTrainingNode(graph, id, { position });
}

export function addTrainingEdge(graph, edge) {
    if (graph.edges.some((e) => e.source === edge.source && e.target === edge.target)) {
        return graph;
    }
    return { ...graph, edges: [...graph.edges, edge] };
}

export function removeTrainingEdge(graph, edgeId) {
    return { ...graph, edges: graph.edges.filter((e) => e.id !== edgeId) };
}
