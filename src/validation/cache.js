const VISUAL_NODE_FIELDS = new Set(['position', 'visualization', 'tensorRole']);

function sameFields(left, right, ignored) {
    if (left === right) return true;
    const keys = Object.keys(left).filter((key) => !ignored.has(key));
    return keys.length === Object.keys(right).filter((key) => !ignored.has(key)).length
        && keys.every((key) => Object.hasOwn(right, key) && Object.is(left[key], right[key]));
}

function sameGraphModel(left, right) {
    if (left === right) return true;
    // Editing connections always invalidates the report. Moving nodes keeps
    // this array intact, so we only inspect the shallow node fields below.
    if (left.edges !== right.edges) return false;
    const ids = Object.keys(left.nodes);
    const otherIds = Object.keys(right.nodes);
    return ids.length === otherIds.length && ids.every((id, index) =>
        id === otherIds[index] && sameFields(left.nodes[id], right.nodes[id], VISUAL_NODE_FIELDS),
    );
}

// JSON alone collapses undefined and non-finite numbers to null inside arrays.
// Those values can resolve differently in parameter expressions.
function scopeValue(value) {
    if (value === null) return ['null'];
    if (Array.isArray(value)) return ['array', value.map(scopeValue)];
    if (typeof value === 'object') return ['object', Object.keys(value).sort().map((key) => [key, scopeValue(value[key])])];
    return [typeof value, typeof value === 'number' ? String(value) : value];
}

/**
 * Share reports across consumers and across position-only edits. Parameter,
 * topology, binding and template-definition changes still invalidate them.
 * Immutable object references avoid serializing the graph on a drag commit.
 */
export function createValidationCache(evaluate) {
    const graphKeys = new WeakMap();
    const previousGraphs = new WeakMap(); // one candidate per immutable edge array
    const templateKeys = new WeakMap();
    const reports = new WeakMap();
    const templateGraphField = new Set(['graph']);
    let lastTemplates;
    let lastTemplateKey;

    function graphKey(graph) {
        if (graphKeys.has(graph)) return graphKeys.get(graph);
        const previous = previousGraphs.get(graph.edges);
        const key = previous && sameGraphModel(previous.graph, graph) ? previous.key : {};
        graphKeys.set(graph, key);
        previousGraphs.set(graph.edges, { graph, key });
        return key;
    }

    function templatesKey(templates) {
        if (templateKeys.has(templates)) return templateKeys.get(templates);
        const ids = Object.keys(templates);
        const unchanged = lastTemplates && ids.length === Object.keys(lastTemplates).length
            && ids.every((id) => {
                const before = lastTemplates[id];
                const after = templates[id];
                return before && sameFields(before, after, templateGraphField)
                    && graphKey(before.graph) === graphKey(after.graph);
            });
        const key = unchanged ? lastTemplateKey : {};
        templateKeys.set(templates, key);
        lastTemplates = templates;
        lastTemplateKey = key;
        return key;
    }

    return (graph, templates, scope = {}, externalInputs = {}) => {
        const model = graphKey(graph);
        const definitions = templatesKey(templates);
        let templateMaps = reports.get(model);
        if (!templateMaps) {
            templateMaps = new WeakMap();
            reports.set(model, templateMaps);
        }
        let scopes = templateMaps.get(definitions);
        if (!scopes) {
            scopes = new Map();
            templateMaps.set(definitions, scopes);
        }
        const key = JSON.stringify([
            Object.keys(scope).sort().map((name) => [name, scopeValue(scope[name])]),
            Object.keys(externalInputs).sort().map((id) => [id, scopeValue(externalInputs[id])]),
        ]);
        if (scopes.has(key)) return scopes.get(key);
        const report = evaluate(graph, { templates, scope, externalInputs });
        if (scopes.size >= 16) scopes.delete(scopes.keys().next().value);
        scopes.set(key, report);
        return report;
    };
}
