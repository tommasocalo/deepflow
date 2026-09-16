/**
 * Share one validation report across the canvas, status panel and node cards.
 * Graphs and template maps are immutable store snapshots, so a new reference
 * invalidates their reports. Weak keys let obsolete snapshots be collected.
 * Only scope values and boundary shapes are serialized, never the graph.
 */
export function createValidationCache(evaluate) {
    const graphs = new WeakMap();
    return (graph, templates, scope = {}, externalInputs = {}) => {
        let templateMaps = graphs.get(graph);
        if (!templateMaps) {
            templateMaps = new WeakMap();
            graphs.set(graph, templateMaps);
        }
        let scopes = templateMaps.get(templates);
        if (!scopes) {
            scopes = new Map();
            templateMaps.set(templates, scopes);
        }
        const key = JSON.stringify([
            Object.keys(scope).sort().map((name) => [name, scope[name]]),
            Object.keys(externalInputs).sort().map((id) => [id, externalInputs[id]]),
        ]);
        if (scopes.has(key)) return scopes.get(key);
        const report = evaluate(graph, { templates, scope, externalInputs });
        // Keep repeated exploration of scope variants bounded per snapshot.
        if (scopes.size >= 16) scopes.delete(scopes.keys().next().value);
        scopes.set(key, report);
        return report;
    };
}
