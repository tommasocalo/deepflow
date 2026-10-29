import { resolveLevelContext } from './architecture/levelContext.js';
import { asFreeParamRef, globalParameters, resolveParameterValue } from './expressions.js';
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

/**
 * Names a field at `path` can reference, with their values in `scope`: globals,
 * then parameters of the enclosing components (which shadow globals), then
 * each enclosing repeat's instance index.
 */
export function parameterSuggestions(graph, path, templates, inspectedIterations = {}, scope = {}, frames = parameterFrames(graph, path, templates, inspectedIterations)) {
    const visible = new Map(globalParameters(graph).map((parameter) => [parameter.name, { ...parameter, owner: 'Global' }]));
    for (const parameter of availableParameters(frames)) visible.set(parameter.name, parameter);
    const parameters = [...visible.values()];
    for (const frame of frames) {
        const name = frame.instance.iterationVariable || 'i';
        if (frame.instance.kind === 'repeat' && !parameters.some((parameter) => parameter.name === name)) parameters.push({ name, type: 'int', owner: 'Instance index' });
    }
    return parameters.map((parameter) => ({ ...parameter, value: resolvedFieldValue(`$${parameter.name}`, scope) }));
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
    const nodes = template?.graph?.nodes ?? {};
    for (const node of Object.values(nodes)) {
        const fields = node.kind === 'layer' ? node.parameters : node.kind === 'repeat' ? { ...node.bindings, count: node.count } : node.parameterValues;
        for (const [field, value] of Object.entries(fields ?? {})) {
            if (parameterReferences(value).includes(name)) uses.push({ nodeId: node.id, label: node.label, field });
        }
    }
    for (const edge of template?.graph?.edges ?? []) {
        if (!parameterReferences(edge.gradientOpLambda).includes(name)) continue;
        uses.push({ edgeId: edge.id, nodeId: edge.target,
            label: `${nodes[edge.source]?.label ?? 'Source'} → ${nodes[edge.target]?.label ?? 'Target'}`, field: 'reverse factor' });
    }
    for (const parameter of template?.freeParameters ?? []) {
        if (!parameter.derived && parameter.name !== name && parameterReferences(parameter.default).includes(name)) uses.push({ label: parameter.name, field: 'default', parameter: true });
    }
    return uses;
}

/** Fields that can read a global: the root graph, and templates that do not
 * declare a parameter of the same name (which would shadow it). */
export function globalParameterUses(rootGraph, templates, name) {
    const uses = parameterConsumers({ graph: rootGraph, freeParameters: rootGraph?.parameters ?? [] }, name).map((use) => ({ ...use, root: true }));
    for (const template of Object.values(templates ?? {})) {
        if ((template.freeParameters ?? []).some((parameter) => parameter.name === name)) continue;
        for (const use of parameterConsumers(template, name)) uses.push({ ...use, template: template.name });
    }
    return uses;
}

const escapeName = (name) => name.replace(/[$]/g, '\\$&');

/**
 * Put a parameter's value where it was referenced: a direct `$name` becomes
 * the value itself, and a formula gets it in parentheses (`4 * d_model` →
 * `4 * (768)`). Longer names that merely start with `name` are left alone.
 */
export function inlineParameter(value, name, replacement) {
    if (Array.isArray(value)) return value.map((part) => inlineParameter(part, name, replacement));
    if (typeof value !== 'string') return value;
    const trimmed = value.trim();
    if (trimmed === `$${name}` || trimmed === name) return replacement;
    if (!parameterReferences(value).includes(name)) return value;
    const literal = Array.isArray(replacement) ? `(${replacement.join(', ')})`
        : typeof replacement === 'number' || typeof replacement === 'boolean' ? String(replacement)
            : `(${replacement ?? 'None'})`;
    return value.replace(new RegExp(`(?<![\\w$])\\$?${escapeName(name)}(?![\\w])`, 'g'), literal);
}

function inlineInNode(node, name, replacement) {
    const inline = (values) => values && Object.fromEntries(Object.entries(values).map(([field, value]) => [field, inlineParameter(value, name, replacement)]));
    if (node.kind === 'layer') return { ...node, parameters: inline(node.parameters) };
    if (node.kind === 'repeat') return { ...node, count: inlineParameter(node.count, name, replacement), bindings: inline(node.bindings) };
    return { ...node, parameterValues: inline(node.parameterValues) };
}

/** Rewrite every field of a graph and the other parameters' defaults. */
function inlineEverywhere(graph, parameters, name, replacement) {
    return {
        graph: { ...graph,
            nodes: Object.fromEntries(Object.entries(graph.nodes ?? {}).map(([id, node]) => [id, inlineInNode(node, name, replacement)])),
            edges: (graph.edges ?? []).map((edge) => edge.gradientOpLambda === undefined ? edge
                : { ...edge, gradientOpLambda: inlineParameter(edge.gradientOpLambda, name, replacement) }),
        },
        parameters: parameters.filter((parameter) => parameter.name !== name)
            .map((parameter) => parameter.derived ? parameter : { ...parameter, default: inlineParameter(parameter.default, name, replacement) }),
    };
}

/** Removing a used component parameter keeps its fields working with its default value. */
export function removeTemplateParameter(template, name) {
    const parameter = template?.freeParameters?.find((candidate) => candidate.name === name);
    if (!parameter) return null;
    const { graph, parameters } = inlineEverywhere(template.graph, template.freeParameters, name, parameter.default ?? null);
    return { ...template, graph, freeParameters: parameters };
}

/** Removing a global writes its value into the root graph and every template that can see it. */
export function removeGlobalParameter(rootGraph, templates, name) {
    const parameter = rootGraph?.parameters?.find((candidate) => candidate.name === name);
    if (!parameter) return null;
    const value = parameter.default ?? null;
    const root = inlineEverywhere(rootGraph, rootGraph.parameters, name, value);
    const nextTemplates = Object.fromEntries(Object.entries(templates ?? {}).map(([id, template]) => {
        if ((template.freeParameters ?? []).some((candidate) => candidate.name === name)) return [id, template];
        if (!globalParameterUses({ nodes: {}, edges: [] }, { [id]: template }, name).length) return [id, template];
        const inlined = inlineEverywhere(template.graph, template.freeParameters ?? [], name, value);
        return [id, { ...template, graph: inlined.graph, freeParameters: inlined.parameters }];
    }));
    return { graph: { ...root.graph, parameters: root.parameters }, templates: nextTemplates };
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
