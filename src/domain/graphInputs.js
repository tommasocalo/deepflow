/** Stable boundary-port routing shared by validation, previews, and export. */
export function getTemplateInputIds(graph) {
    return Object.values(graph?.nodes ?? {})
        .filter((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.template.Input')
        .map((node) => node.id);
}

export function getIncomingEdges(graph, nodeId) {
    return (graph.edges ?? []).filter((edge) => edge.target === nodeId && graph.nodes[edge.source]);
}

export function mapInputEdgesToPorts(edges, portIds, { broadcastSingle = true } = {}) {
    const result = Object.fromEntries(portIds.map((id) => [id, null]));
    const unhandled = [];
    for (const edge of edges) {
        if (portIds.includes(edge.targetHandle)) {
            if (!result[edge.targetHandle]) result[edge.targetHandle] = edge;
        } else if (edge.targetHandle === null || edge.targetHandle === undefined || edge.targetHandle === '') {
            unhandled.push(edge);
        }
    }
    if (broadcastSingle && edges.length === 1 && unhandled.length === 1) {
        for (const id of portIds) result[id] = edges[0];
    } else {
        for (const id of portIds) {
            if (!result[id]) result[id] = unhandled.shift() ?? null;
        }
    }
    return result;
}

/** Reject ambiguous/stale explicit ports before inference or code expansion. */
export function validateInputPorts(edges, portIds) {
    const errors = [];
    const namedCounts = new Map();
    let unhandledCount = 0;
    for (const edge of edges) {
        const handle = edge.targetHandle;
        if (handle === null || handle === undefined || handle === '') {
            unhandledCount += 1;
        } else if (!portIds.includes(handle)) {
            errors.push(`Unknown input port "${handle}"`);
        } else {
            namedCounts.set(handle, (namedCounts.get(handle) ?? 0) + 1);
        }
    }
    for (const [port, count] of namedCounts) {
        if (count > 1) errors.push(`Input port "${port}" has more than one incoming connection`);
    }
    if (unhandledCount > portIds.length - namedCounts.size) {
        errors.push('Too many unhandled incoming connections for the available input ports');
    }
    return errors;
}

export function mapTemplateInputs(templateGraph, edges, valueForEdge) {
    const ports = mapInputEdgesToPorts(edges, getTemplateInputIds(templateGraph));
    return Object.fromEntries(Object.entries(ports).map(([id, edge]) => [id, edge ? valueForEdge(edge) : null]));
}

export function getOperationInputIds(node) {
    if (node.libraryKey === 'deepflow.op.MatMul' || node.libraryKey === 'deepflow.op.Subtract') return ['a', 'b'];
    if (node.libraryKey === 'deepflow.op.Gather') return ['indices', 'weight'];
    return null;
}

export function orderOperationEdges(node, edges) {
    const ports = getOperationInputIds(node);
    if (!ports) return edges;
    const mapped = mapInputEdgesToPorts(edges, ports, { broadcastSingle: false });
    return ports.map((id) => mapped[id]);
}
