/**
 * Immutable operations over an ArchitectureGraph.
 * Every function returns a new graph; nothing is mutated in place.
 */

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {import('./types.js').ArchNode} node
 */
export function addNode(graph, node) {
    return { ...graph, nodes: { ...graph.nodes, [node.id]: node } };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} id
 */
export function removeNode(graph, id) {
    const { [id]: _removed, ...rest } = graph.nodes;
    const edges = graph.edges.filter((e) => e.source !== id && e.target !== id);
    let nodes = rest;
    for (const key of Object.keys(rest)) {
        const n = rest[key];
        if (n.kind === 'supernode' && n.children.includes(id)) {
            nodes = { ...nodes, [key]: { ...n, children: n.children.filter((c) => c !== id) } };
        }
    }
    return { nodes, edges };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} id
 * @param {Partial<import('./types.js').ArchNode>} patch
 */
export function patchNode(graph, id, patch) {
    const existing = graph.nodes[id];
    if (!existing) return graph;
    return { ...graph, nodes: { ...graph.nodes, [id]: { ...existing, ...patch } } };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} id
 * @param {Object<string, *>} parameterPatch
 */
export function patchLayerParameters(graph, id, parameterPatch) {
    const existing = graph.nodes[id];
    if (!existing || existing.kind !== 'layer') return graph;
    return {
        ...graph,
        nodes: {
            ...graph.nodes,
            [id]: { ...existing, parameters: { ...existing.parameters, ...parameterPatch } },
        },
    };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} id
 * @param {import('./types.js').Position} position
 */
export function moveNode(graph, id, position) {
    return patchNode(graph, id, { position });
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {import('./types.js').ArchEdge} edge
 */
export function addEdge(graph, edge) {
    if (graph.edges.some((e) => e.source === edge.source && e.target === edge.target)) {
        return graph;
    }
    return { ...graph, edges: [...graph.edges, edge] };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} edgeId
 */
export function removeEdge(graph, edgeId) {
    return { ...graph, edges: graph.edges.filter((e) => e.id !== edgeId) };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} edgeId
 * @param {Partial<import('./types.js').ArchEdge>} patch
 */
export function patchEdge(graph, edgeId, patch) {
    return {
        ...graph,
        edges: graph.edges.map((e) => (e.id === edgeId ? { ...e, ...patch } : e)),
    };
}
