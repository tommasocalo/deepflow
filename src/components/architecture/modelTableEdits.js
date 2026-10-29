import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { describeTarget, fieldPatch, resolveWriteTarget, targetValue } from '../../domain/modelTable/writeTargets.js';

/** Table columns, in order; the config column holds the editable settings. */
export const COLUMNS = ['layer', 'count', 'output', 'config', 'params', 'macs', 'share'];
export const CONFIG_COLUMN = COLUMNS.indexOf('config');

const sameValue = (left, right) => left === right || JSON.stringify(left ?? null) === JSON.stringify(right ?? null);

/** Write one value through the store action that owns it. */
function writeTarget(store, target, value) {
    const definitions = useTemplatesStore.getState();
    if (target.kind === 'rootNode') {
        const node = store.getState().graph.nodes[target.nodeId];
        store.getState().patchRootNode(target.nodeId, fieldPatch(node, target.source, target.field, value));
    } else if (target.kind === 'templateLayer') {
        definitions.patchTemplateLayerParameters(target.templateId, target.nodeId, { [target.field]: value });
    } else if (target.kind === 'templateNode') {
        const node = definitions.templates[target.templateId]?.graph.nodes[target.nodeId];
        definitions.patchTemplateNode(target.templateId, target.nodeId, fieldPatch(node, target.source, target.field, value));
    } else if (target.kind === 'templateDefault') {
        definitions.patchFreeParameter(target.templateId, target.name, { default: value });
    } else if (target.kind === 'global') {
        store.getState().patchGlobalParameter(target.name, { default: value });
    }
}

/**
 * Apply the fields edited in one config cell as a single undo step, labelled
 * "Set <row> <fields>". Targets are resolved again from live store state, so
 * a stale render can never write to the wrong owner. Unchanged values and
 * read-only fields write nothing, and an edit that changes nothing records
 * no history entry.
 */
export function applyModelTableEdit(store, row, edits) {
    const label = `Set ${row.label} ${edits.map((edit) => edit.field.name).join(', ')}`;
    return recordArchitectureEdit(label, () => {
        let written = 0;
        for (const { field, value, cells } of edits) {
            const rootGraph = store.getState().graph;
            const templates = useTemplatesStore.getState().templates;
            const target = resolveWriteTarget({ row, field, cells, rootGraph, templates });
            if (target.kind === 'readonly' || sameValue(targetValue(target, { rootGraph, templates }), value)) continue;
            writeTarget(store, target, value);
            written += 1;
        }
        return written;
    });
}

const fieldCache = new WeakMap();

/**
 * Owner, tooltip and editability of each config field of a row, computed once
 * per row object and cells mode. Rows are rebuilt whenever the document
 * changes, so cached targets always describe the rendered state.
 */
export function fieldTargets(row, cells, context) {
    let byMode = fieldCache.get(row);
    if (!byMode) {
        byMode = new Map();
        fieldCache.set(row, byMode);
    }
    if (byMode.has(cells)) return byMode.get(cells);
    const targets = {};
    for (const field of row.config) {
        if (field.kind !== 'field') continue;
        const target = resolveWriteTarget({ row, field, cells, rootGraph: context.graph, templates: context.templates });
        targets[field.name] = {
            field, target, editable: target.kind !== 'readonly',
            hint: describeTarget(target, { row, rootGraph: context.graph, templates: context.templates, templateUses: context.templateUses }),
        };
    }
    byMode.set(cells, targets);
    return targets;
}

/** Scope a field is evaluated in, for checks before writing. */
export function fieldScope(row, field, templates) {
    if (field.source === 'parameters' || field.source === 'count') return row.scope ?? {};
    // A binding sees its level, the repeat index, and parameters declared before it.
    const scope = { ...(row.scope ?? {}) };
    if (row.kind === 'repeat') scope[row.iterationVariable ?? 'i'] = 0;
    for (const parameter of templates[row.definitionIds[0]]?.freeParameters ?? []) {
        if (parameter.name === field.name) break;
        scope[parameter.name] = row.instanceScope?.[parameter.name] ?? null;
    }
    return scope;
}

const asText = (value) => value === undefined || value === null ? ''
    : Array.isArray(value) ? `(${value.join(', ')})` : String(value);

/** What the editor starts with: the expression as written, or the source value it follows. */
export function initialText(info, context) {
    if (!info.target.literalOnly) return asText(info.field.raw);
    return asText(targetValue(info.target, { rootGraph: context.graph, templates: context.templates }));
}

/** Empty is allowed where it means "use the default", never for a count or a source value. */
export function emptyAllowed(info) {
    return !['global', 'templateDefault'].includes(info.target.kind) && info.field.source !== 'count';
}
