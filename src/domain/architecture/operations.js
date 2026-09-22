import { resolveLevelContext } from './levelContext.js';

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
    const { [id]: _removed, ...nodes } = graph.nodes;
    const edges = graph.edges.filter((e) => e.source !== id && e.target !== id);
    // Instances reference a separate template graph; they do not own children
    // in this graph. Removing a node must leave those definitions untouched.
    return { ...graph, nodes, edges };
}

/**
 * Resolve the shared template being edited, through both instances and repeats.
 * Every editor surface uses the same traversal so edits and feedback refer to
 * the graph displayed on the canvas. An invalid saved path falls back to root.
 */
export function resolveGraphLevel(graph, currentPath = [], templates = {}, inspectedIterations = {}) {
    const root = { graph, isRoot: true, templateId: null, label: 'main', segments: [] };
    const context = resolveLevelContext(graph, currentPath, templates, inspectedIterations);
    if (context.levels.length !== currentPath.length) return root;
    let level = root;
    for (const { instance: node, template, templateId, index } of context.levels) {
        const label = node.label || template.name || (node.kind === 'repeat' ? 'Repeat' : 'Supernode');
        level = {
            graph: template.graph,
            isRoot: false,
            templateId,
            instance: node,
            index: node.kind === 'repeat' ? index : null,
            label,
            segments: [...level.segments, { id: node.id, label, kind: node.kind, depth: level.segments.length + 1 }],
        };
    }
    return level;
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
    return moveNodes(graph, [{ id, position }]);
}

/** Apply a whole drag selection atomically, retaining unrelated node objects. */
export function moveNodes(graph, moves) {
    let nodes = graph.nodes;
    for (const { id, position } of moves) {
        const node = nodes[id];
        if (!node || !Number.isFinite(position?.x) || !Number.isFinite(position?.y)) continue;
        if (node.position.x === position.x && node.position.y === position.y) continue;
        if (nodes === graph.nodes) nodes = { ...nodes };
        nodes[id] = { ...node, position: { x: position.x, y: position.y } };
    }
    return nodes === graph.nodes ? graph : { ...graph, nodes };
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {import('./types.js').ArchEdge} edge
 */
export function isValidConnection(graph, { source, target, sourceHandle, targetHandle }, ignoredEdgeId) {
    return !!source && !!target && source !== target &&
        !!graph.nodes[source] && !!graph.nodes[target] &&
        !graph.edges.some((edge) => edge.id !== ignoredEdgeId && edge.source === source && edge.target === target &&
            (edge.sourceHandle ?? null) === (sourceHandle ?? null) && (edge.targetHandle ?? null) === (targetHandle ?? null));
}

export function addEdge(graph, edge) {
    if (!isValidConnection(graph, edge)) return graph;
    return { ...graph, edges: [...graph.edges, edge] };
}

/** Rewiring preserves the edge id and training metadata. Invalid drops are no-ops. */
export function reconnectEdge(graph, edgeId, connection) {
    const existing = graph.edges.find((edge) => edge.id === edgeId);
    if (!existing || !isValidConnection(graph, connection, edgeId)) return graph;
    const endpoints = {
        source: connection.source,
        target: connection.target,
        sourceHandle: connection.sourceHandle ?? undefined,
        targetHandle: connection.targetHandle ?? undefined,
    };
    if (Object.entries(endpoints).every(([key, value]) => (existing[key] ?? undefined) === value)) return graph;
    return patchEdge(graph, edgeId, endpoints);
}

/**
 * @param {import('./types.js').ArchitectureGraph} graph
 * @param {string} edgeId
 */
export function removeEdge(graph, edgeId) {
    const edges = graph.edges.filter((e) => e.id !== edgeId);
    return edges.length === graph.edges.length ? graph : { ...graph, edges };
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
