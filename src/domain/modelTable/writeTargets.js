import { asFreeParamRef, globalParameters, resolveDimensionTuple, resolveRepeatCount } from '../expressions.js';
import { getEntry } from '../library/index.js';
import { fieldAlias, globalParameterUses, parameterReferences, resolvedFieldValue } from '../parameterBindings.js';
import { NUMBER_LITERAL } from '../parameterDisplay.js';
import { COUNT_SPEC, fieldRaw, unset } from './configFields.js';

/**
 * Where a Model table edit is written. Each kind names the real owner of the value:
 * - rootNode      { nodeId, source, field }       a node of the root graph
 * - templateLayer { templateId, nodeId, field }   a layer setting inside a shared definition
 * - templateNode  { templateId, nodeId, source, field } a binding or count inside a definition
 * - templateDefault { templateId, name }          a definition's parameter default
 * - global        { name }                        a model-wide parameter on the root graph
 * - readonly      { reason }                      nothing is written; the reason says where to edit
 */

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
 * A global's value, following references to earlier globals, as
 * withGlobalParameters resolves them in declaration order.
 */
function followGlobal(name, rootGraph, hops) {
    const globals = globalParameters(rootGraph);
    let before = globals.length;
    for (;;) {
        const index = globals.findIndex((parameter) => parameter.name === name);
        if (index < 0 || index >= before) return readonly(`$${name} is unresolved.`, hops);
        const raw = globals[index].default;
        hops.push({ name, label: 'global', raw, bound: true });
        const reference = asFreeParamRef(typeof raw === 'string' ? raw.trim() : raw);
        if (reference) {
            name = reference;
            before = index;
            continue;
        }
        if (Array.isArray(raw) && globals[index].type !== 'tuple') return readonly(`Global ${name} holds several values; edit it in Parameters → Global.`, hops);
        if (!unset(raw) && !isLiteralValue(raw, globals[index])) return readonly(`Computed by ${raw} (global ${name}). Edit the formula in Parameters → Global.`, hops);
        return { kind: 'global', name, hops };
    }
}

/**
 * Follow a direct `$name` to the value it reads, mirroring buildParameterScope.
 * Inside a definition every parameter is visible, then the repeat index, then
 * the enclosing level; globals come last. A binding (or default) sees only
 * the parameters declared before it in its definition: `before` is that
 * declaration index when the walk starts at a container's own binding.
 */
export function followReference(name, frames, rootGraph, { before: startBefore = Infinity } = {}) {
    const hops = [];
    let depth = frames.length - 1;
    let before = startBefore;
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
                if (reference === iterationName(container)) return readonly(`${reference} is the instance index of ${container.label}.`, hops);
                name = reference;
                before = Infinity;
                depth -= 1;
                continue;
            }
            // None (a nullable value set to null) and an empty default are values too.
            if (!unset(raw) && !isLiteralValue(raw, parameter)) return readonly(`Computed by ${raw} (${container.label}). Switch to Symbols to edit the formula.`, hops);
            return bound
                ? { kind: owner ? 'templateNode' : 'rootNode', templateId: owner, nodeId: container.id, source, field: name, hops }
                : { kind: 'templateDefault', templateId, name, hops };
        }
        // Not a parameter visible here: the repeat index, then the enclosing level.
        if (name === iterationName(container)) return readonly(`${name} is the instance index of ${container.label}.`, hops);
        before = Infinity;
        depth -= 1;
    }
    return followGlobal(name, rootGraph, hops);
}

/** Write where the field itself is stored. */
function directTarget(row, field, ownerId) {
    if (!row.levelPath.length) return { kind: 'rootNode', nodeId: row.nodeId, source: field.source, field: field.name };
    if (field.source === 'parameters') return { kind: 'templateLayer', templateId: ownerId, nodeId: row.nodeId, field: field.name };
    return { kind: 'templateNode', templateId: ownerId, nodeId: row.nodeId, source: field.source, field: field.name };
}

const globalUseCounts = new WeakMap();

/** Fields that read a global, counted once per document state. */
function globalUseCount(rootGraph, templates, name) {
    let byTemplates = globalUseCounts.get(rootGraph);
    if (!byTemplates) {
        byTemplates = new WeakMap();
        globalUseCounts.set(rootGraph, byTemplates);
    }
    let counts = byTemplates.get(templates);
    if (!counts) {
        counts = new Map();
        byTemplates.set(templates, counts);
    }
    if (!counts.has(name)) counts.set(name, globalParameterUses(rootGraph, templates, name).length);
    return counts.get(name);
}

/** Tooltip text naming the owner before the edit happens. */
export function describeTarget(target, { row, rootGraph, templates, templateUses = {} }) {
    if (target.kind === 'readonly') return target.reason;
    const uses = (templateId) => templateUses[templateId] ?? 0;
    const usesText = (templateId) => `${uses(templateId)} use${uses(templateId) === 1 ? '' : 's'}`;
    if (target.kind === 'global') {
        // Direct uses undercount the reach: bindings such as D: $d_model carry
        // the global into every layer of a component.
        const count = globalUseCount(rootGraph, templates, target.name);
        return `Sets global ${target.name} for the whole model: every value that reads it changes, including through component bindings (${count} field${count === 1 ? '' : 's'} name it directly)`;
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

const valueText = (value) => value === null || value === undefined || value === '' ? 'default'
    : Array.isArray(value) ? `(${value.join(', ')})` : String(value);
const usesText = (count) => `${count} use${count === 1 ? '' : 's'}`;

/**
 * Where a value goes when it is not the cell's own field: `global d_model`,
 * `default of stride in ResNet block`, `stride on ResNet stages`. Null when
 * the edit writes the field itself.
 */
export function targetTag(target, { row, rootGraph, templates }) {
    if (target.kind === 'global') return `global ${target.name}`;
    if (target.kind === 'templateDefault') return `default of ${target.name} in ${templates[target.templateId]?.name ?? 'its definition'}`;
    if (target.kind === 'readonly' || target.nodeId === row.nodeId) return null;
    const graph = target.kind === 'rootNode' ? rootGraph : templates[target.templateId]?.graph;
    return `${target.field} on ${graph?.nodes?.[target.nodeId]?.label ?? 'its container'}`;
}

/**
 * One edit in words, naming the value that really changes and how far it
 * reaches: `d_model = 1024 (global, 6 fields)`, `ResNet stages stride = 2
 * (all 3 instances)`, `mlp proj out_features = 1024 (shared Transformer
 * block, 12 uses)`. Undo labels and confirmations use it.
 */
export function targetSummary(target, value, { row, rootGraph, templates, templateUses = {} }) {
    const text = valueText(value);
    const uses = (templateId) => templateUses[templateId] ?? 0;
    const shared = (templateId) => `${templates[templateId]?.name ?? 'component'}, ${usesText(uses(templateId))}`;
    if (target.kind === 'global') {
        const count = globalUseCount(rootGraph, templates, target.name);
        return `${target.name} = ${text} (global, ${count} field${count === 1 ? '' : 's'})`;
    }
    if (target.kind === 'templateDefault') return `${target.name} = ${text} (default in ${shared(target.templateId)})`;
    const own = target.nodeId === row.nodeId;
    const graph = target.kind === 'rootNode' ? rootGraph : templates[target.templateId]?.graph;
    const where = own ? row.label : graph?.nodes?.[target.nodeId]?.label ?? 'its container';
    if (target.kind === 'rootNode') return `${where} ${target.field} = ${text}${!own && row.multiplicity > 1 ? ` (all ${row.multiplicity} instances)` : ''}`;
    return `${where} ${target.field} = ${text}${uses(target.templateId) > 1 ? ` (shared ${shared(target.templateId)})` : ''}`;
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
    const alias = ownerId && field.source !== 'count' ? fieldAlias(templates[ownerId], row.nodeId, field.name) : null;
    const raw = fieldRaw(node, field.source, field.name);
    if (field.source === 'count' && raw === 'auto') {
        const source = (ownerGraph.edges ?? []).filter((edge) => edge.target === node.id).map((edge) => ownerGraph.nodes[edge.source])
            .find((candidate) => ['deepflow.op.Split', 'deepflow.op.Broadcast'].includes(candidate?.libraryKey));
        if (!source) return readonly('Counted from the incoming collections.');
        const split = source.libraryKey === 'deepflow.op.Split';
        const value = split ? source.parameters?.chunks : source.parameters?.count;
        return readonly(`Counted from the incoming ${split ? 'Split' : 'Broadcast'} ${source.label ?? ''}${value !== undefined ? ` (${split ? 'chunks' : 'count'} ${value})` : ''}; change the ${split ? 'number of pieces' : 'count'} there.`);
    }
    if (Array.isArray(raw) && field.source === 'bindings' && field.spec?.type !== 'tuple') {
        return readonly('Varies per instance; set the values in Structure → Instances.');
    }
    if (field.spec?.derived) return readonly('Inferred from the input; it cannot be set.');
    const direct = alias ? { kind: 'templateDefault', templateId: ownerId, name: alias.name } : directTarget(row, field, ownerId);
    // Symbols edits the cell as written; an eye alias is written as its default.
    if (cells === 'symbols') return { ...direct, hops: [], literalOnly: false };
    if (!alias && (unset(raw) || isLiteralValue(raw, field.spec))) return { ...direct, hops: [], literalOnly: false };
    const frames = rowFrames(row, rootGraph, templates);
    if (!frames) return readonly('The level of this row changed; try again.');
    if (alias) {
        // The field reads $alias, and an instance may bind it: Values follows
        // the binding, and edits the alias default only when nothing binds it.
        const followed = followReference(alias.name, frames, rootGraph);
        return followed.kind === 'readonly' ? followed : { ...followed, literalOnly: true };
    }
    const reference = asFreeParamRef(typeof raw === 'string' ? raw.trim() : raw);
    if (!reference) return readonly(`Computed by ${typeof raw === 'string' ? raw.trim() : JSON.stringify(raw)}. Switch to Symbols to edit the formula.`);
    let followed;
    if (field.source === 'bindings' || field.source === 'parameterValues') {
        // A container's binding is resolved inside its own definition first:
        // earlier parameters there, then its repeat index, then this level.
        const template = templates[row.definitionIds[0]];
        if (!template) return readonly('The definition of this component is missing.');
        const index = template.freeParameters?.findIndex((parameter) => parameter.name === field.name) ?? -1;
        followed = followReference(reference, [...frames, { container: node, owner: ownerId, templateId: row.definitionIds[0], template }], rootGraph, { before: Math.max(0, index) });
    } else {
        followed = followReference(reference, frames, rootGraph);
    }
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
 * The spec of the value a target writes, for checks before writing: a global,
 * a definition parameter, a container binding or count, or the field itself.
 */
export function targetSpec(target, { rootGraph, templates }, fieldSpec = null) {
    if (target.kind === 'global') return globalParameters(rootGraph).find((parameter) => parameter.name === target.name) ?? fieldSpec;
    if (target.kind === 'templateDefault') return templates[target.templateId]?.freeParameters?.find((parameter) => parameter.name === target.name) ?? fieldSpec;
    if (target.source === 'count') return COUNT_SPEC;
    if (target.source === 'bindings' || target.source === 'parameterValues') {
        const graph = target.kind === 'rootNode' ? rootGraph : templates[target.templateId]?.graph;
        const container = graph?.nodes?.[target.nodeId];
        return templates[container?.templateId]?.freeParameters?.find((parameter) => parameter.name === target.field) ?? fieldSpec;
    }
    const graph = target.kind === 'rootNode' ? rootGraph : templates[target.templateId]?.graph;
    const node = graph?.nodes?.[target.nodeId];
    return getEntry(node?.libraryKey)?.parameters.find((parameter) => parameter.name === target.field) ?? fieldSpec;
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

/** A name nothing defines, spelled as it was typed: `abc` stays `abc`, `$abc` stays `$abc`. */
function unknownName(value, name) {
    const text = Array.isArray(value) ? value.join(', ') : String(value);
    return `No parameter named ${text.includes(`$${name}`) ? `$${name}` : name}`;
}

const BATCH_MARK = /^(none|null|b|n|\?)$/i;
const POSITIVE_INTEGER = /^\+?[1-9]\d*$/;

/**
 * The model Input's shape, in the grammar validation reads: sizes such as
 * (1024,) or (B, 3, 32, 32). Names are not resolved there, so they are refused.
 */
function inputShapeProblem(value) {
    if (typeof value === 'number') return POSITIVE_INTEGER.test(String(value)) ? null : 'Sizes must be positive whole numbers';
    const text = String(value).trim();
    const inner = /^\(.*\)$|^\[.*\]$/.test(text) ? text.slice(1, -1) : text;
    const parts = inner.split(',').map((part) => part.trim());
    if (parts.at(-1) === '' && parts.length > 1) parts.pop();
    if (!parts.length || parts.some((part) => part === '')) return 'Write sizes such as (1024,) or (3, 32, 32)';
    const names = parts.filter((part) => /[A-Za-z_$]/.test(part) && !BATCH_MARK.test(part));
    if (names.length) return `Input sizes take numbers, not names (${names[0]}); write sizes such as (1024,)`;
    if (parts.slice(1).some((part) => BATCH_MARK.test(part))) return 'Only the first size can be the batch (B)';
    if (parts.some((part) => !BATCH_MARK.test(part) && !POSITIVE_INTEGER.test(part))) return 'Sizes must be positive whole numbers';
    if (parts.every((part) => BATCH_MARK.test(part))) return 'Add at least one size, such as (1024,)';
    return null;
}

/**
 * Check a parsed value before writing it. Returns an error message, or null.
 * Nothing is written while this fails; the editor stays open. `libraryKey`
 * names the layer whose field is written, for fields with their own grammar.
 */
export function precheckCellValue(value, { spec, scope = {}, literalOnly = false, source, required: requiredOverride, libraryKey = null }) {
    const required = requiredOverride ?? (source === 'count' || (spec?.required && spec?.default === undefined));
    if (value === null) return required ? 'Required' : null;
    if (literalOnly && !isLiteralValue(value, spec)) return 'Values mode sets a value here; switch to Symbols to write an expression.';
    if (source === 'count') {
        const missing = unknownNames(value, scope);
        if (missing.length) return unknownName(value, missing[0]);
        return resolveRepeatCount(value, scope).error ?? null;
    }
    const type = spec?.type;
    if (type === 'enum') return spec.options?.includes(value) ? null : `Choose one of ${spec.options?.join(', ')}`;
    // Shapes first: the model Input stores its shape as text.
    if (libraryKey === 'deepflow.io.Input' && spec?.name === 'shape') return inputShapeProblem(value);
    if (type === 'tuple' || spec?.name === 'shape') {
        if (typeof value === 'number') return Number.isFinite(value) ? null : 'Not a number';
        const missing = unknownNames(value, scope);
        if (missing.length) return unknownName(value, missing[0]);
        const dimensions = resolveDimensionTuple(value, scope);
        if (dimensions === null || !dimensions.length) return 'Not a list of sizes';
        if (dimensions.some((dimension) => !Number.isInteger(dimension) || dimension < 0)) return 'Sizes must be whole numbers';
        return null;
    }
    if (type === 'string') return null;
    const resolved = resolvedFieldValue(value, scope);
    if (type === 'bool') {
        if (typeof resolved === 'boolean') return null;
        const missing = unknownNames(value, scope);
        return missing.length ? unknownName(value, missing[0]) : 'Must be true or false';
    }
    if (typeof resolved !== 'number' || !Number.isFinite(resolved)) {
        const missing = unknownNames(value, scope);
        return missing.length ? unknownName(value, missing[0]) : 'Not a number';
    }
    if (type === 'int' && !Number.isInteger(resolved)) return 'Must be an integer';
    return null;
}
