import { resolveLevelContext } from './architecture/levelContext.js';
import { asFreeParamRef, resolveParameterValue } from './expressions.js';
export { formatParameterValue } from './parameterDisplay.js';

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

export function parameterFrames(graph, path, templates, inspectedIterations = {}) {
    return resolveLevelContext(graph, path, templates, inspectedIterations).levels.map(({ instance, template }, depth) => ({
        instance, template, path: path.slice(0, depth), bindings: instance.kind === 'repeat' ? instance.bindings ?? {} : instance.parameterValues ?? {},
    }));
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
        freeParameters: [...(template.freeParameters ?? []), { ...definition, ...(spec?.name === 'shape' && spec.type === 'tuple' ? { editor: 'dimensions' } : {}), name, default: defaultValue ?? null, exposes: { nodeId, field } }],
        graph: { ...template.graph, nodes: { ...template.graph.nodes, [nodeId]: nextNode } },
    };
}

function fieldValue(node, field) {
    if (!node) return undefined;
    if (node.kind === 'repeat' && field === 'count') return node.count;
    const values = node.kind === 'layer' ? node.parameters : node.kind === 'repeat' ? node.bindings : node.parameterValues;
    return values?.[field];
}

/** A direct reference exposes this field through its immediate component.
 * Hidden controls remain references: visibility never changes model values. */
export function exposedFieldParameter(template, nodeId, field) {
    const value = fieldValue(template?.graph?.nodes?.[nodeId], field);
    const name = asFreeParamRef(typeof value === 'string' ? value.trim() : value);
    return name ? template?.freeParameters?.find((parameter) => parameter.name === name && !parameter.derived) ?? null : null;
}

/** `undefined` clears the field so it inherits its default again. */
function setFieldValue(node, field, value) {
    if (node.kind === 'repeat' && field === 'count') return { ...node, count: value };
    const bucket = node.kind === 'layer' ? 'parameters' : node.kind === 'repeat' ? 'bindings' : 'parameterValues';
    const { [field]: _previous, ...rest } = node[bucket] ?? {};
    return { ...node, [bucket]: value === undefined ? rest : { ...rest, [field]: value } };
}

/** Name the eye gives a field: its label, lowercased and underscored. */
export function friendlyParameterName(label, field) {
    const friendly = String(label || (field === 'p' ? 'dropout' : field)).trim()
        .replace(/([a-z0-9])([A-Z])/g, '$1_$2').toLowerCase()
        .replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '');
    return /^[a-z_]/.test(friendly) ? friendly : `parameter${friendly ? `_${friendly}` : ''}`;
}

/**
 * The parameter the eye created for this field, if any. The field then holds
 * `$alias` internally, but the cell keeps showing the alias default, so the
 * value never appears to change. Only aliases marked by the eye qualify: a
 * variable the user typed stays visible as `$name`.
 */
export function fieldAlias(template, nodeId, field) {
    const parameter = exposedFieldParameter(template, nodeId, field);
    return parameter?.exposes?.nodeId === nodeId && parameter.exposes.field === field ? parameter : null;
}

/**
 * The eye shows or hides a field in the parent component's settings and never
 * changes what the cell shows. Showing a literal or formula creates an alias
 * behind the field; hiding it puts the value back and removes the alias, so
 * every instance uses the cell again. Instance overrides stay stored and apply
 * again if the field is shown later. A field that already uses a variable of
 * this component shows or hides that variable.
 */
export function toggleTemplateFieldExposure(template, { nodeId, field, spec, defaultValue, label, reservedNames = [] }) {
    const node = template?.graph?.nodes?.[nodeId];
    if (!node || !field || spec?.derived || (node.kind === 'repeat' && field === 'count' && node.count === 'auto')) return null;
    const alias = fieldAlias(template, nodeId, field);
    if (alias?.hidden) return {
        ...template,
        freeParameters: template.freeParameters.map((parameter) => parameter === alias ? { ...parameter, hidden: false, exposes: { nodeId, field } } : parameter),
    };
    if (alias) {
        // A field that was empty goes back to empty unless its value was edited.
        const restored = alias.exposes.inherited && alias.default === (spec?.default ?? null) ? undefined : alias.default;
        const graph = { ...template.graph, nodes: { ...template.graph.nodes, [nodeId]: setFieldValue(node, field, restored) } };
        const stillUsed = parameterConsumers({ ...template, graph }, alias.name).length > 0;
        return {
            ...template,
            graph,
            freeParameters: stillUsed
                ? template.freeParameters.map((parameter) => { if (parameter !== alias) return parameter; const { exposes: _exposes, ...rest } = parameter; return rest; })
                : template.freeParameters.filter((parameter) => parameter !== alias),
        };
    }
    const exposed = exposedFieldParameter(template, nodeId, field);
    if (exposed) return {
        ...template,
        freeParameters: template.freeParameters.map((parameter) => parameter === exposed ? { ...parameter, hidden: parameter.hidden !== true } : parameter),
    };

    const current = fieldValue(node, field);
    const candidate = current !== undefined ? current : defaultValue !== undefined ? defaultValue : spec?.default;
    const raw = candidate == null && !spec?.nullable && spec?.default != null ? spec.default : candidate ?? null;
    // Include referenced names even when callers supply no scope. An alias
    // named like its own default would shadow the value it must inherit.
    const reserved = new Set([
        ...(template.freeParameters ?? []).map((parameter) => parameter.name),
        ...reservedNames.map((name) => String(name).replace(/^\$/, '')),
        ...parameterReferences(raw),
    ]);
    const base = friendlyParameterName(label, field);
    let name = base;
    for (let suffix = 2; reserved.has(name); suffix++) name = `${base}_${suffix}`;
    const created = exposeTemplateField(template, { nodeId, field, name, spec: { ...spec, hidden: false }, defaultValue: raw });
    if (!created || current !== undefined) return created;
    return { ...created, freeParameters: created.freeParameters.map((parameter) => parameter.name === name ? { ...parameter, exposes: { ...parameter.exposes, inherited: true } } : parameter) };
}
