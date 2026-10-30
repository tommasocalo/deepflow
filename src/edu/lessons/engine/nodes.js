import { getEntry } from '../../../domain/library/index.js';
import { buildParameterScope, repeatBindingValue, resolveParameterValue, resolveRepeatCount } from '../../../domain/expressions.js';
import { getIncomingEdges, mapTemplateInputs } from '../../../domain/graphInputs.js';
import { effectiveTemplateId } from '../../../domain/architecture/instanceStructure.js';

/**
 * Node types and matches (docs/edu/SPEC.md §6.4.2). Checks, actions and spotlight targets all name nodes through
 * these helpers, so a standard Linear component (an editable supernode) and a raw torch.nn.Linear layer are the
 * same `Linear` to a lesson.
 */
export const TYPE_GROUPS = Object.freeze({
    Activation: Object.freeze(['ReLU', 'LeakyReLU', 'GELU', 'Tanh', 'Sigmoid']),
});

const LAYER_TYPES = {
    'deepflow.io.Input': 'Input',
    'deepflow.io.Output': 'Output',
    // Boundary blocks of a template graph never stand for the model's Input/Output.
    'deepflow.template.Input': 'TemplateInput',
    'deepflow.template.Output': 'TemplateOutput',
    'deepflow.nn.CausalSelfAttention': 'Attention',
    'deepflow.nn.PositionalEmbedding': 'LearnedPositions',
    'torch.nn.LayerNorm': 'LayerNorm',
};

const STANDARD_TYPES = {
    linear: 'Linear',
    embedding: 'Embedding',
    attention: 'Attention',
    'attention-head': 'AttentionHead',
    layernorm: 'LayerNorm',
    'learned-position': 'LearnedPositions',
    'sinusoidal-position': 'SinusoidalPositions',
};

export const KNOWN_MATCH_KEYS = Object.freeze(['type', 'id', 'label', 'name', 'nth', 'params', 'count']);

export function nodeType(node, templates = {}) {
    if (!node) return null;
    if (node.kind === 'layer') return LAYER_TYPES[node.libraryKey] ?? node.libraryKey?.split('.').at(-1) ?? null;
    if (node.kind === 'supernode' || node.kind === 'repeat') {
        const template = templates?.[node.templateId];
        return STANDARD_TYPES[template?.standardKey ?? template?.componentStyleKey] ?? 'Component';
    }
    return null;
}

/** `wanted` is a type, a group name or a list of either. */
export function typeMatches(type, wanted) {
    if (wanted === undefined || wanted === null) return true;
    const list = Array.isArray(wanted) ? wanted : [wanted];
    return list.some((item) => item === type || (TYPE_GROUPS[item]?.includes(type) ?? false));
}

function sameScalar(left, right) {
    if (left === right) return true;
    const numeric = (value) => typeof value === 'number' || (typeof value === 'string' && value.trim() !== '' && Number.isFinite(Number(value)));
    if (numeric(left) && numeric(right)) return Object.is(Number(left), Number(right));
    if (typeof left === 'boolean' || typeof right === 'boolean') {
        const asBool = (value) => value === true || value === 'true' ? true : value === false || value === 'false' ? false : value;
        return asBool(left) === asBool(right);
    }
    return false;
}

/** Numbers compare after Number() (so '3' equals 3); arrays element-wise; null equals only null. */
export function sameValue(left, right) {
    if (Array.isArray(left) || Array.isArray(right)) {
        return Array.isArray(left) && Array.isArray(right) && left.length === right.length
            && left.every((item, index) => sameValue(item, right[index]));
    }
    if (left === null || right === null || left === undefined || right === undefined) return left === right;
    return sameScalar(left, right);
}

export function sameShape(left, right) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length
        && left.every((axis, index) => (axis === null || right[index] === null) ? axis === right[index] : Number(axis) === Number(right[index]));
}

/** The scope a supernode or repeat (iteration 0) passes to its template, as validation computes it. */
export function componentScope(graph, node, { templates = {}, scope = {}, report = null } = {}) {
    const isRepeat = node?.kind === 'repeat';
    const template = templates[isRepeat ? effectiveTemplateId(node, 0) : effectiveTemplateId(node)] ?? templates[node?.templateId];
    if (!template) return null;
    const incoming = getIncomingEdges(graph, node.id);
    const inputs = template.graph ? mapTemplateInputs(template.graph, incoming, (edge) => report?.shapes?.[edge.source] ?? null) : {};
    const parent = isRepeat ? { ...scope, [node.iterationVariable || 'i']: 0 } : scope;
    return buildParameterScope(template.freeParameters ?? [], isRepeat ? node.bindings ?? {} : node.parameterValues ?? {}, parent, isRepeat ? 0 : undefined, inputs);
}

/**
 * The resolved value of a node parameter. Layers fall back to the library default, components to the template's
 * free-parameter default, and repeats use their binding for iteration 0 (`count` is the resolved repeat count).
 */
export function paramValue(node, name, { templates = {}, scope = {}, graph = null, report = null } = {}) {
    if (!node) return undefined;
    if (node.kind === 'layer') {
        const raw = node.parameters?.[name] ?? getEntry(node.libraryKey)?.parameters?.find((spec) => spec.name === name)?.default ?? null;
        return resolveParameterValue(raw, scope);
    }
    if (node.kind === 'repeat' && name === 'count') return resolveRepeatCount(node.count, scope).count;
    if (node.kind === 'supernode' || node.kind === 'repeat') {
        const template = templates[node.templateId];
        const declared = template?.freeParameters?.find((parameter) => parameter.name === name);
        if (!declared) {
            const raw = node.kind === 'repeat' ? repeatBindingValue(node.bindings?.[name], undefined, 0) : node.parameterValues?.[name];
            return raw === undefined ? undefined : resolveParameterValue(raw, scope);
        }
        const child = componentScope(graph ?? { nodes: { [node.id]: node }, edges: [] }, node, { templates, scope, report });
        return child ? child[name] : undefined;
    }
    return undefined;
}

/**
 * A deterministic topological order: nodes reachable from the Input come first, ties broken by canvas position
 * (top to bottom, then left to right) and insertion order. `nth` counts in this order.
 */
export function orderedNodeIds(graph) {
    const nodes = graph?.nodes ?? {};
    const ids = Object.keys(nodes);
    const index = new Map(ids.map((id, position) => [id, position]));
    const inDegree = new Map(ids.map((id) => [id, 0]));
    const out = new Map(ids.map((id) => [id, []]));
    for (const edge of graph?.edges ?? []) {
        if (!inDegree.has(edge.source) || !inDegree.has(edge.target)) continue;
        out.get(edge.source).push(edge.target);
        inDegree.set(edge.target, inDegree.get(edge.target) + 1);
    }
    const reachable = new Set();
    const stack = ids.filter((id) => nodes[id]?.libraryKey === 'deepflow.io.Input');
    while (stack.length) {
        const id = stack.pop();
        if (reachable.has(id)) continue;
        reachable.add(id);
        stack.push(...out.get(id));
    }
    const key = (id) => [reachable.has(id) ? 0 : 1, nodes[id]?.position?.y ?? 0, nodes[id]?.position?.x ?? 0, index.get(id)];
    const before = (left, right) => {
        const a = key(left), b = key(right);
        for (let position = 0; position < a.length; position++) if (a[position] !== b[position]) return a[position] < b[position];
        return false;
    };
    const ready = ids.filter((id) => inDegree.get(id) === 0);
    const order = [];
    while (ready.length) {
        let best = 0;
        for (let position = 1; position < ready.length; position++) if (before(ready[position], ready[best])) best = position;
        const [id] = ready.splice(best, 1);
        order.push(id);
        for (const next of out.get(id)) {
            inDegree.set(next, inDegree.get(next) - 1);
            if (inDegree.get(next) === 0) ready.push(next);
        }
    }
    // Nodes on a cycle keep their insertion order at the end.
    for (const id of ids) if (!order.includes(id)) order.push(id);
    return order;
}

/** Does one node satisfy a NodeMatch (ignoring nth)? */
export function nodeMatches(node, match = {}, options = {}) {
    if (!node || !match) return false;
    const { templates = {} } = options;
    if (match.id !== undefined && node.id !== match.id) return false;
    if (match.label !== undefined && node.label !== match.label) return false;
    if (match.name !== undefined && templates[node.templateId]?.name !== match.name) return false;
    if (match.type !== undefined && !typeMatches(nodeType(node, templates), match.type)) return false;
    if (match.params) {
        for (const [name, expected] of Object.entries(match.params)) {
            if (!sameValue(paramValue(node, name, options), expected)) return false;
        }
    }
    return true;
}

/** Every matching node of a graph level in topological order; `nth` keeps only that one. */
export function matchNodes(graph, match = {}, options = {}) {
    const order = options.order ?? orderedNodeIds(graph);
    const matches = order.map((id) => graph.nodes[id]).filter((node) => nodeMatches(node, match, { ...options, graph }));
    if (Number.isInteger(match.nth)) return matches[match.nth] ? [matches[match.nth]] : [];
    return matches;
}

/** A display name for a NodeMatch in check details: its label, id or type. */
export function describeMatch(match = {}) {
    if (match.label) return match.label;
    if (Array.isArray(match.type)) return match.type.join(' / ');
    if (match.type) return match.type;
    if (match.name) return match.name;
    if (match.id) return match.id;
    return 'block';
}

/** "(B, 1, 28, 28)" for shapes with a batch axis. */
export function formatShape(shape) {
    if (!Array.isArray(shape)) return '?';
    return `(${shape.map((axis) => axis === null ? 'B' : String(axis)).join(', ')})`;
}

/** Parse "B, 1, 28, 28", "(B, 8, 28, 28)", "B×8×28×28" or "None, 784" into [null, 1, 28, 28]. */
export function parseShapeInput(text) {
    if (typeof text !== 'string') return null;
    const cleaned = text.trim().replace(/^[([]/, '').replace(/[)\]]$/, '');
    if (!cleaned) return null;
    const parts = cleaned.split(/\s*(?:,|×|\*|\bx\b)\s*/i).map((part) => part.trim()).filter((part) => part !== '');
    if (!parts.length) return null;
    const axes = parts.map((part) => {
        if (/^(b|n|none|null|batch|\?)$/i.test(part)) return null;
        const value = Number(part.replace(/_/g, ''));
        return Number.isSafeInteger(value) && value > 0 ? value : NaN;
    });
    return axes.some((axis) => Number.isNaN(axis)) ? null : axes;
}
