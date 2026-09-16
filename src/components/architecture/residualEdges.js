/**
 * A residual edge bypasses another directed path between the same endpoints.
 * Classification depends only on connections, so moving a node cannot reroute
 * unrelated wires. Parallel direct edges are not a bypassed computation.
 */
export function residualEdgeIds(edges) {
    const outgoing = new Map();
    const successors = new Map();
    const predecessors = new Map();
    for (const edge of edges) {
        if (edge.source === edge.target) continue;
        if (!outgoing.has(edge.source)) outgoing.set(edge.source, []);
        if (!successors.has(edge.source)) successors.set(edge.source, new Set());
        if (!predecessors.has(edge.target)) predecessors.set(edge.target, new Set());
        outgoing.get(edge.source).push(edge);
        successors.get(edge.source).add(edge.target);
        predecessors.get(edge.target).add(edge.source);
    }

    const residuals = new Set();
    for (const edge of edges) {
        // A bypass needs both a branch and a merge, with distinct neighbours.
        if (edge.source === edge.target ||
            (successors.get(edge.source)?.size ?? 0) < 2 ||
            (predecessors.get(edge.target)?.size ?? 0) < 2) continue;

        const visited = new Set([edge.source]);
        const queue = [];
        for (const next of outgoing.get(edge.source)) {
            // Start through an intermediate node, never a parallel direct edge.
            if (next.id === edge.id || next.target === edge.target || visited.has(next.target)) continue;
            visited.add(next.target);
            queue.push(next.target);
        }
        let found = false;
        for (let index = 0; index < queue.length && !found; index += 1) {
            for (const next of outgoing.get(queue[index]) ?? []) {
                if (next.id === edge.id) continue;
                if (next.target === edge.target) {
                    found = true;
                    break;
                }
                if (!visited.has(next.target)) {
                    visited.add(next.target);
                    queue.push(next.target);
                }
            }
        }
        if (found) residuals.add(edge.id);
    }
    return residuals;
}
