import { asFreeParamRef, globalParameters, resolveDimensionTuple, resolveRepeatCount } from '../expressions.js';
import { fieldAlias, globalParameterUses, parameterReferences, resolvedFieldValue } from '../parameterBindings.js';
import { fieldRaw } from './configFields.js';

/**
 * Where a Model table edit is written. Each kind names the real owner of the value:
 * - rootNode      { nodeId, source, field }       a node of the root graph
 * - templateLayer { templateId, nodeId, field }   a layer setting inside a shared definition
 * - templateNode  { templateId, nodeId, source, field } a binding or count inside a definition
 * - templateDefault { templateId, name }          a definition's parameter default
 * - global        { name }                        a model-wide parameter on the root graph
 * - readonly      { reason }                      nothing is written; the reason says where to edit
 */

const NUMBER_LITERAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;
const unset = (raw) => raw === undefined || raw === null || raw === '';
const readonly = (reason, hops = []) => ({ kind: 'readonly', reason, hops });

/** A value written as-is: numbers, flags, enum choices and numeric tuples. */
export function isLiteralValue(raw, spec) {
    if (typeof raw === 'number' || typeof raw === 'boolean') return true;
    if (Array.isArray(raw)) return raw.every((part) => typeof part === 'number' || (typeof part === 'string' && NUMBER_LITERAL.test(part.trim())));
    if (typeof raw !== 'string') return false;
    const text = raw.trim();
    if (NUMBER_LITERAL.test(text) || /^(true|false)$/i.test(text)) return true;
    if (['enum', 'string'].includes(spec?.type)) return !text.startsWith('$');
    // A tuple of plain numbers such as (3, 3) or (1024,).
    return /^[([]?\s*[+-]?\d+(?:\.\d+)?(?:\s*,\s*[+-]?\d+(?:\.\d+)?)*\s*,?\s*[)\]]?$/.test(text) && text.includes(',');
}

/** The containers above a row with their live nodes and single definitions. */
function rowFrames(row, rootGraph, templates) {
    const frames = [];
    let graph = rootGraph;
    let owner = null;
    for (let depth = 0; depth < row.levelPath.length; depth++) {
        const container = graph?.nodes?.[row.levelPath[depth]];
        const templateId = row.trail?.[depth]?.templateId ?? container?.templateId;
        const template = templates[templateId];
        if (!container || !template?.graph) return null;
        frames.push({ container, owner, templateId, template });
        graph = template.graph;
        owner = templateId;
    }
    return frames;
}

const bindingSource = (container) => container.kind === 'repeat' ? 'bindings' : 'parameterValues';
const iterationName = (container) => container.kind === 'repeat' ? container.iterationVariable || 'i' : null;

/**
 * Follow a direct `$name` to the value it reads, mirroring buildParameterScope:
 * a container's binding may use parameters declared earlier in the same
 * definition, then the repeat index, then the enclosing level; globals last.
 */
export function followReference(name, frames, rootGraph) {
    const hops = [];
    let depth = frames.length - 1;
    let before = Infinity;
    while (depth >= 0) {
        const { container, owner, templateId, template } = frames[depth];
        const parameters = template.freeParameters ?? [];
        const index = parameters.findIndex((parameter) => parameter.name === name);
        if (index >= 0 && index < before) {
            const parameter = parameters[index];
            if (parameter.derived) {
                return readonly(`${name} is inferred from the input's ${parameter.derived.axis === -1 ? 'last dimension' : `axis ${parameter.derived.axis}`}; it cannot be set.`, hops);
            }
            const source = bindingSource(container);
            let raw = container[source]?.[name];
            const bound = !(raw === undefined || raw === '' || (raw === null && !parameter.nullable));
            if (!bound) raw = parameter.default;
            hops.push({ name, label: container.label, raw, bound, templateName: template.name });
            if (Array.isArray(raw) && parameter.type !== 'tuple') return readonly(`${name} varies per instance of ${container.label}; set the values in Structure → Instances.`, hops);
            const reference = asFreeParamRef(typeof raw === 'string' ? raw.trim() : raw);
            if (reference) {
                const earlier = parameters.findIndex((candidate) => candidate.name === reference);
                if (earlier >= 0 && earlier < index) {
                    name = reference;
                    before = index;
                    continue;
                }
                if (reference === iterationName(container)) return readonly(`${name} is the instance index of ${container.label}.`, hops);
                name = reference;
                before = Infinity;
                depth -= 1;
                continue;
            }
            if (!isLiteralValue(raw, parameter)) return readonly(`Computed by ${raw} (${container.label}). Switch to Symbols to edit the formula.`, hops);
            return bound
                ? { kind: owner ? 'templateNode' : 'rootNode', templateId: owner, nodeId: container.id, source, field: name, hops }
                : { kind: 'templateDefault', templateId, name, hops };
        }
        if (before === Infinity && name === iterationName(container)) return readonly(`${name} is the instance index of ${container.label}.`, hops);
        before = Infinity;
        depth -= 1;
    }
    if (globalParameters(rootGraph).some((parameter) => parameter.name === name)) return { kind: 'global', name, hops };
    return readonly(`$${name} is unresolved.`, hops);
}

/** Write where the field itself is stored. */
function directTarget(row, field, ownerId) {
    if (!row.levelPath.length) return { kind: 'rootNode', nodeId: row.nodeId, source: field.source, field: field.name };
    if (field.source === 'parameters') return { kind: 'templateLayer', templateId: ownerId, nodeId: row.nodeId, field: field.name };
    return { kind: 'templateNode', templateId: ownerId, nodeId: row.nodeId, source: field.source, field: field.name };
}

/** Tooltip text naming the owner before the edit happens. */
export function describeTarget(target, { row, rootGraph, templates, templateUses = {} }) {
    if (target.kind === 'readonly') return target.reason;
    const uses = (templateId) => templateUses[templateId] ?? 0;
    const usesText = (templateId) => `${uses(templateId)} use${uses(templateId) === 1 ? '' : 's'}`;
    if (target.kind === 'global') {
        const count = globalParameterUses(rootGraph, templates, target.name).length;
        return `Sets global ${target.name} · used by ${count} field${count === 1 ? '' : 's'}`;
    }
    if (target.kind === 'templateDefault') return `Sets the default of ${target.name} in the shared ${templates[target.templateId]?.name ?? 'component'} definition · ${usesText(target.templateId)}`;
    if (target.kind === 'templateLayer' || target.kind === 'templateNode') {
        const binding = target.nodeId !== row.nodeId ? ` (${target.field} on ${templates[target.templateId]?.graph.nodes[target.nodeId]?.label ?? 'its container'})` : '';
        return `Edits the shared ${templates[target.templateId]?.name ?? 'component'} definition${binding} · ${usesText(target.templateId)}`;
    }
    const node = rootGraph?.nodes?.[target.nodeId];
    if (target.nodeId !== row.nodeId) {
        const count = row.multiplicity && row.multiplicity > 1 ? ` · all ${row.multiplicity} instances` : '';
        return `Sets ${target.field} on ${node?.label ?? 'its container'}${count}`;
    }
    return `Edits ${node?.label ?? row.label}`;
}

/**
 * Resolve the owner of one config field, from live store state. Symbols mode
 * edits the expression where it is written. Values mode follows a direct
 * `$ref` to its source (a binding, a definition default or a global); formulas,
 * per-instance arrays, derived and automatic values are read-only there.
 */
export function resolveWriteTarget({ row, field, cells = 'values', rootGraph, templates = {} }) {
    if (field.readOnly) return readonly(field.readOnly);
    if (row.ownerIds.length > 1 || row.ancestorConflict || (row.levelPath.length && !row.ownerIds[0])) {
        return readonly('Instances use different definitions (local versions); open the level to edit one.');
    }
    const ownerId = row.levelPath.length ? row.ownerIds[0] : null;
    const ownerGraph = ownerId ? templates[ownerId]?.graph : rootGraph;
    const node = ownerGraph?.nodes?.[row.nodeId];
    if (!node) return readonly('This node is no longer in the model.');
    if (field.differs) return readonly('Instances use different values; open the level to edit one.');
    const alias = ownerId && field.source !== 'count' ? fieldAlias(templates[ownerId], row.nodeId, field.name) : null;
    const raw = fieldRaw(node, field.source, field.name);
    if (field.source === 'count' && raw === 'auto') {
        const split = Object.values(ownerGraph.nodes).find((candidate) => candidate.libraryKey === 'deepflow.op.Split'
            && (ownerGraph.edges ?? []).some((edge) => edge.source === candidate.id && edge.target === node.id));
        const chunks = split?.parameters?.chunks;
        return readonly(`Counted from the incoming Split${chunks !== undefined ? ` (chunks ${chunks})` : ''}; change the number of pieces there.`);
    }
    if (Array.isArray(raw) && field.source === 'bindings' && field.spec?.type !== 'tuple') {
        return readonly('Varies per instance; set the values in Structure → Instances.');
    }
    if (field.spec?.derived) return readonly('Inferred from the input; it cannot be set.');
    const direct = alias ? { kind: 'templateDefault', templateId: ownerId, name: alias.name } : directTarget(row, field, ownerId);
    if (cells === 'symbols') return { ...direct, hops: [], literalOnly: false };
    const shown = alias ? alias.default : raw;
    if (unset(shown) || isLiteralValue(shown, field.spec)) return { ...direct, hops: [], literalOnly: false };
    const reference = asFreeParamRef(typeof raw === 'string' ? raw.trim() : raw);
    if (!reference) return readonly(`Computed by ${typeof raw === 'string' ? raw.trim() : JSON.stringify(raw)}. Switch to Symbols to edit the formula.`);
    const frames = rowFrames(row, rootGraph, templates);
    if (!frames) return readonly('The level of this row changed; try again.');
    const followed = followReference(reference, frames, rootGraph);
    return followed.kind === 'readonly' ? followed : { ...followed, literalOnly: true };
}

/** The patch that sets one field on a node, keeping its other values. */
export function fieldPatch(node, source, name, value) {
    if (source === 'count') return { count: value };
    return { [source]: { ...(node?.[source] ?? {}), [name]: value } };
}

/** The value currently stored at a target, for no-op detection. */
export function targetValue(target, { rootGraph, templates }) {
    if (target.kind === 'global') return globalParameters(rootGraph).find((parameter) => parameter.name === target.name)?.default;
    if (target.kind === 'templateDefault') return templates[target.templateId]?.freeParameters?.find((parameter) => parameter.name === target.name)?.default;
    const graph = target.kind === 'rootNode' ? rootGraph : templates[target.templateId]?.graph;
    const node = graph?.nodes?.[target.nodeId];
    if (target.kind === 'templateLayer') return node?.parameters?.[target.field];
    return fieldRaw(node, target.source, target.field);
}

/**
 * The grammar of ParameterField: a number literal becomes a number, an empty
 * cell inherits the default, true/false are flags, anything else ($ref,
 * formula, tuple) is kept as text.
 */
export function parseCellValue(text, spec) {
    const value = String(text ?? '').trim();
    if (value === '') return null;
    if (NUMBER_LITERAL.test(value) && Number.isFinite(Number(value)) && spec?.type !== 'string' && spec?.type !== 'enum') return Number(value);
    if (spec?.type === 'bool' && /^(true|false)$/i.test(value)) return value.toLowerCase() === 'true';
    return value;
}

function unknownNames(value, scope) {
    return parameterReferences(value).filter((name) => !Object.hasOwn(scope, name) || scope[name] === null || scope[name] === undefined);
}

/**
 * Check a parsed value before writing it. Returns an error message, or null.
 * Nothing is written while this fails; the editor stays open.
 */
export function precheckCellValue(value, { spec, scope = {}, literalOnly = false, source, required: requiredOverride }) {
    const required = requiredOverride ?? (source === 'count' || (spec?.required && spec?.default === undefined));
    if (value === null) return required ? 'Required' : null;
    if (literalOnly && !isLiteralValue(value, spec)) return 'Values mode sets a value here; switch to Symbols to write an expression.';
    if (source === 'count') {
        const missing = unknownNames(value, scope);
        if (missing.length) return `Unknown name $${missing[0]}`;
        return resolveRepeatCount(value, scope).error ?? null;
    }
    const type = spec?.type;
    if (type === 'enum') return spec.options?.includes(value) ? null : `Choose one of ${spec.options?.join(', ')}`;
    if (type === 'string') return null;
    if (type === 'tuple' || spec?.name === 'shape') {
        if (typeof value === 'number') return null;
        if (resolveDimensionTuple(value, scope) !== null) return null;
        const missing = unknownNames(value, scope);
        return missing.length ? `Unknown name $${missing[0]}` : 'Not a list of sizes';
    }
    const resolved = resolvedFieldValue(value, scope);
    if (type === 'bool') {
        if (typeof resolved === 'boolean') return null;
        const missing = unknownNames(value, scope);
        return missing.length ? `Unknown name $${missing[0]}` : 'Must be true or false';
    }
    if (typeof resolved !== 'number' || !Number.isFinite(resolved)) {
        const missing = unknownNames(value, scope);
        return missing.length ? `Unknown name $${missing[0]}` : 'Not a number';
    }
    if (type === 'int' && !Number.isInteger(resolved)) return 'Must be an integer';
    return null;
}
