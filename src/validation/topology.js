/**
 * Pure topology routines over an ArchitectureGraph.
 * No React, no store, no side effects — easy to test.
 */

/**
 * Build adjacency list keyed by node id.
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} graph
 */
function adjacency(graph) {
    const out = {};
    const inn = {};
    for (const id of Object.keys(graph.nodes)) {
        out[id] = [];
        inn[id] = [];
    }
    for (const edge of graph.edges) {
        if (out[edge.source] && inn[edge.target]) {
            out[edge.source].push(edge.target);
            inn[edge.target].push(edge.source);
        }
    }
    return { out, inn };
}

/**
 * Kahn's algorithm. Returns the topological order plus any nodes that
 * couldn't be ordered (i.e. participate in a cycle).
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} graph
 */
export function topologicalSort(graph) {
    const { out, inn } = adjacency(graph);
    const indeg = {};
    for (const id of Object.keys(graph.nodes)) {
        indeg[id] = inn[id].length;
    }
    const queue = Object.keys(indeg).filter((id) => indeg[id] === 0);
    const order = [];
    while (queue.length) {
        const id = queue.shift();
        order.push(id);
        for (const neighbor of out[id]) {
            indeg[neighbor] -= 1;
            if (indeg[neighbor] === 0) queue.push(neighbor);
        }
    }
    const cyclic = Object.keys(graph.nodes).filter((id) => !order.includes(id));
    return { order, cyclic, adjacency: { out, inn } };
}

/**
 * Trace the chain of cyclic nodes back to a small representative cycle for
 * the error message. Best-effort, returns null if the graph is acyclic.
 */
export function findCycle(graph) {
    const { out } = adjacency(graph);
    const WHITE = 0, GRAY = 1, BLACK = 2;
    const color = {};
    for (const id of Object.keys(graph.nodes)) color[id] = WHITE;
    const stack = [];
    let cycle = null;

    function dfs(id) {
        if (cycle) return;
        color[id] = GRAY;
        stack.push(id);
        for (const next of out[id] ?? []) {
            if (cycle) return;
            if (color[next] === GRAY) {
                const i = stack.indexOf(next);
                cycle = stack.slice(i).concat(next);
                return;
            }
            if (color[next] === WHITE) dfs(next);
        }
        stack.pop();
        color[id] = BLACK;
    }

    for (const id of Object.keys(graph.nodes)) {
        if (color[id] === WHITE) dfs(id);
        if (cycle) break;
    }
    return cycle;
}
