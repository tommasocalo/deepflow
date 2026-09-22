import { buildParameterScope, resolveRepeatCount } from './expressions.js';
import { getIncomingEdges, mapTemplateInputs } from './graphInputs.js';
import { isShapeCollection } from './shapeCollections.js';

/**
 * Resolve the first/separately selected parallel instance for an inspector.
 * Sequential iterations after zero need the preceding outputs; the open-level
 * validation context performs that simulation instead of guessing them here.
 */
export function resolveComponentContext(node, template, graph, shapes = {}, scope = {}, inspectedIndex = 0) {
    if (!node || !template?.graph || !graph?.nodes) return { scope: { ...scope }, inputs: {}, count: null, error: 'Template missing', index: 0 };
    const incoming = getIncomingEdges(graph, node.id);
    const initial = mapTemplateInputs(template.graph, incoming, (edge) => shapes[edge.source] ?? null);
    const repeated = node.kind === 'repeat';
    const { count, error = null } = repeated ? resolveRepeatCount(node.count, scope, Object.values(initial)) : { count: 1 };
    const index = Math.max(0, Math.min((count || 1) - 1, inspectedIndex));
    const inputs = repeated && node.mode === 'parallel'
        ? Object.fromEntries(Object.entries(initial).map(([port, shape]) => [port, isShapeCollection(shape) ? shape.items[index] ?? null : shape]))
        : initial;
    const parent = repeated ? { ...scope, [node.iterationVariable || 'i']: index } : scope;
    return {
        scope: buildParameterScope(template.freeParameters, repeated ? node.bindings : node.parameterValues, parent, repeated ? index : undefined, inputs),
        inputs, count, error, index,
    };
}
