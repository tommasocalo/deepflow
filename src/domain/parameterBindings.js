import { asFreeParamRef, resolveParameterValue } from './expressions.js';

/** References in arithmetic and dimension tuples; numeric exponents are not names. */
export function parameterReferences(value) {
    if (Array.isArray(value)) return [...new Set(value.flatMap(parameterReferences))];
    if (typeof value !== 'string') return [];
    const names = [];
    value.replace(/((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|\$?([A-Za-z_][\w]*)/g, (_, number, name) => {
        if (!number && !['true', 'false', 'null', 'auto'].includes(name)) names.push(name);
        return _;
    });
    return [...new Set(names)];
}

export function parameterFrames(graph, path, templates) {
    const frames = [];
    for (const [depth, id] of path.entries()) {
        const instance = graph?.nodes[id];
        const template = templates[instance?.templateId];
        if (!template) break;
        frames.push({ instance, template, path: path.slice(0, depth), bindings: instance.kind === 'repeat' ? instance.bindings ?? {} : instance.parameterValues ?? {} });
        graph = template.graph;
    }
    return frames;
}

export function availableParameters(frames) {
    const visible = new Map();
    for (const frame of frames) {
        for (const parameter of frame.template.freeParameters ?? []) visible.set(parameter.name, { ...parameter, owner: frame.instance.label });
    }
    return [...visible.values()];
}

/** Follow explicit forwarding through enclosing instances instead of calling every $ref global. */
export function parameterOrigin(name, frames, end = frames.length - 1) {
    for (let i = end; i >= 0; i--) {
        const frame = frames[i];
        const parameter = frame.template.freeParameters?.find((p) => p.name === name);
        if (!parameter) continue;
        const forwarded = asFreeParamRef(frame.bindings[name]);
        const upstream = forwarded ? parameterOrigin(forwarded, frames, i - 1) : null;
        if (upstream) return upstream;
        return { name, label: frame.instance.label, path: frame.path, nodeId: frame.instance.id, parameter };
    }
    return null;
}

export function parameterConsumers(template, name) {
    const uses = [];
    for (const node of Object.values(template?.graph?.nodes ?? {})) {
        const fields = node.kind === 'layer' ? node.parameters : node.kind === 'repeat' ? { ...node.bindings, count: node.count } : node.parameterValues;
        for (const [field, value] of Object.entries(fields ?? {})) {
            if (parameterReferences(value).includes(name)) uses.push({ nodeId: node.id, label: node.label, field });
        }
    }
    for (const parameter of template?.freeParameters ?? []) {
        if (!parameter.derived && parameter.name !== name && parameterReferences(parameter.default).includes(name)) uses.push({ label: parameter.name, field: 'default', parameter: true });
    }
    return uses;
}

export function resolvedFieldValue(value, scope, depth = 0) {
    if (depth > 16) return null;
    if (Array.isArray(value)) return value.map((part) => resolvedFieldValue(part, scope, depth + 1));
    const resolved = resolveParameterValue(value, scope);
    if (Array.isArray(resolved)) return resolved.map((part) => resolvedFieldValue(part, scope, depth + 1));
    // An unresolved arithmetic expression must not masquerade as a resolved value.
    if (typeof resolved === 'string' && (resolved.includes('$') || parameterReferences(value).some((name) => !Object.hasOwn(scope, name) || scope[name] == null))) return null;
    return resolved;
}

export function formatParameterValue(value) {
    if (value === null || value === undefined) return 'Unresolved';
    if (Array.isArray(value)) return value.map(formatParameterValue).join(' × ');
    return String(value);
}

/** One immutable edit adds the interface control and binds the original field. */
export function exposeTemplateField(template, { nodeId, field, name, spec, defaultValue }) {
    if (!template || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(name) || (template.freeParameters ?? []).some((p) => p.name === name)) return null;
    const node = template.graph.nodes[nodeId];
    if (!node || spec?.derived) return null;
    const bucket = node.kind === 'layer' ? 'parameters' : node.kind === 'repeat' ? 'bindings' : 'parameterValues';
    const nextNode = field === 'count' && node.kind === 'repeat'
        ? { ...node, count: `$${name}` }
        : { ...node, [bucket]: { ...node[bucket], [field]: `$${name}` } };
    const { derived: _derived, required: _required, ...definition } = spec ?? {};
    return {
        ...template,
        freeParameters: [...(template.freeParameters ?? []), { ...definition, ...(spec?.name === 'shape' && spec.type === 'tuple' ? { editor: 'dimensions' } : {}), name, default: defaultValue ?? null }],
        graph: { ...template.graph, nodes: { ...template.graph.nodes, [nodeId]: nextNode } },
    };
}
