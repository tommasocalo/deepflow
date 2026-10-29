import { STANDARD_COMPONENTS } from '../components/index.js';
import { getEntry } from '../library/index.js';
import { isShapeCollection } from '../shapeCollections.js';
import { instanceTotals, METRICS, subtreeMetrics } from '../metrics/modelMetrics.js';
import { aliasFor, configLayout, fieldRaw, fieldSource, fieldSpec, rawText, resolvedBinding, resolvedLayerField, resolvesFalse, summarizeValues } from './configFields.js';

export const DEPTHS = ['root', 'blocks', 'ops'];
/** Definitions that stay one row at Blocks depth: the library's standard modules. */
export const STANDARD_FAMILIES = new Set(['linear', 'embedding', 'layernorm', 'learned-position', 'sinusoidal-position', 'attention', 'attention-head']);
const HIDDEN_KEYS = new Set(['deepflow.template.Input', 'deepflow.template.Output']);
const SUMMARY = /^(?:Repeat|Template ".*") has \d+ internal errors?: /;

/** Rows are keyed by their absolute node-id path, like level paths. */
export const rowKey = (path) => JSON.stringify(path);

/**
 * Dataflow order; when several nodes are ready, read top to bottom and then
 * left to right, as on the canvas. Nodes in cycles come last.
 */
export function readingOrder(graph) {
    const nodes = graph?.nodes ?? {};
    const ids = Object.keys(nodes);
    const insertion = new Map(ids.map((id, index) => [id, index]));
    const indegree = new Map(ids.map((id) => [id, 0]));
    const outgoing = new Map(ids.map((id) => [id, []]));
    for (const edge of graph?.edges ?? []) {
        if (!outgoing.has(edge.source) || !indegree.has(edge.target)) continue;
        outgoing.get(edge.source).push(edge.target);
        indegree.set(edge.target, indegree.get(edge.target) + 1);
    }
    const position = (id) => nodes[id].position ?? {};
    const before = (a, b) => (position(a).y ?? 0) - (position(b).y ?? 0)
        || (position(a).x ?? 0) - (position(b).x ?? 0) || insertion.get(a) - insertion.get(b);
    const ready = ids.filter((id) => indegree.get(id) === 0).sort(before);
    const order = [];
    while (ready.length) {
        const id = ready.shift();
        order.push(id);
        for (const next of outgoing.get(id)) {
            indegree.set(next, indegree.get(next) - 1);
            if (indegree.get(next) === 0) {
                ready.push(next);
                ready.sort(before);
            }
        }
    }
    const seen = new Set(order);
    return [...order, ...ids.filter((id) => !seen.has(id)).sort(before)];
}

function axisText(values, index) {
    if (values.every((value) => value === values[0])) return values[0] === null ? (index === 0 ? 'B' : '?') : String(values[0]);
    const known = values.filter((value) => Number.isFinite(value));
    if (known.length !== values.length) return '?';
    return `${Math.min(...known)}…${Math.max(...known)}`;
}

function singleShapeText(shape) {
    if (isShapeCollection(shape)) {
        const labels = shape.items.map(singleShapeText);
        return labels.every((label) => label === labels[0]) ? `${labels.length} tensors [${labels[0] ?? '?'}]` : `${labels.length} tensors`;
    }
    if (!Array.isArray(shape) || !shape.length) return null;
    return shape.map((axis, index) => axis === null ? (index === 0 ? 'B' : '?') : String(axis)).join(' × ');
}

/**
 * One label for the output of every merged occurrence, B = batch. Shapes
 * that differ in some axes read as a range there: `B × 64…256 × 32 × 32`.
 */
export function shapeSummary(shapes) {
    const distinct = [];
    const seen = new Set();
    for (const shape of shapes) {
        const key = JSON.stringify(shape ?? null);
        if (seen.has(key)) continue;
        seen.add(key);
        distinct.push(shape ?? null);
    }
    const labels = distinct.map((shape) => singleShapeText(shape) ?? '?');
    if (distinct.length <= 1) return { text: distinct.length && singleShapeText(distinct[0]) || null, varies: false, title: labels[0] ?? '' };
    const ranks = distinct.map((shape) => Array.isArray(shape) ? shape.length : -1);
    const text = ranks.every((rank) => rank > 0 && rank === ranks[0])
        ? Array.from({ length: ranks[0] }, (_, axis) => axisText(distinct.map((shape) => shape[axis]), axis)).join(' × ')
        : `${labels[0]} …`;
    return { text, varies: true, title: labels.join(' · ') };
}

const unique = (values) => [...new Set(values)];

/** Occurrence labels for per-instance values: `i=2`, or `3.5` inside nested repeats. */
function occurrenceLabel(trail) {
    const indices = trail.filter((step) => step.repeat).map((step) => step.index);
    if (!indices.length) return '';
    return indices.length === 1 ? `i=${indices[0]}` : indices.join('.');
}

function metricSummary(list, occurrences, metric) {
    let total = 0;
    const unresolved = new Map();
    for (const merged of list) {
        const totals = subtreeMetrics(merged.metrics);
        total += totals[metric];
        for (const [key, label] of totals.unresolved[metric]) unresolved.set(key, label);
    }
    const values = occurrences.map((occurrence) => occurrence[metric]);
    return {
        total,
        unresolved: unresolved.size,
        unresolvedLabels: unique([...unresolved.values()]),
        each: occurrences.length > 1 ? { min: Math.min(...values), max: Math.max(...values) } : null,
        trainable: metric === 'params' ? list.reduce((sum, merged) => sum + subtreeMetrics(merged.metrics).trainable, 0) : undefined,
    };
}

function standardFor(family) {
    return STANDARD_COMPONENTS.find((component) => component.key === family) ?? null;
}

/** Field cells for one row: written values (Symbols) and resolved values (Values). */
function buildConfig(row, list, templates, occurrenceLabels) {
    const firstNode = list[0].metrics.node;
    const ownerId = row.ownerIds.length === 1 ? row.ownerIds[0] : null;
    const parts = [];
    for (const part of configLayout(row, firstNode, templates)) {
        if (part.text || part.note) {
            parts.push(part.text ? { kind: 'text', text: part.text } : { kind: 'note', text: part.note, title: part.title });
            continue;
        }
        const name = part.field;
        const source = fieldSource(firstNode, name);
        const spec = fieldSpec(row, name, templates);
        const alias = source === 'count' ? null : aliasFor(templates, ownerId, row.nodeId, name);
        const raws = list.map((merged) => fieldRaw(merged.metrics.node, source, name));
        const raw = alias ? alias.default : raws[0];
        const written = rawText(alias?.exposes?.inherited && alias.default === (spec?.default ?? null) ? undefined : raw, spec);
        let values;
        let labels = occurrenceLabels;
        if (source === 'count') values = list.map((merged) => merged.metrics.count);
        else if (source === 'parameters') values = list.map((merged) => resolvedLayerField(merged.metrics.node, merged.metrics.scope, name));
        else {
            values = list.flatMap((merged) => resolvedBinding(merged.metrics, name));
            labels = row.kind === 'repeat' ? list.flatMap((merged) => merged.metrics.instances.map((instance) => `i=${instance.index}`)) : occurrenceLabels;
        }
        if (part.hideDefault && values.every((value) => value === null || value === undefined || value === spec?.default)) continue;
        const resolved = summarizeValues(values, spec, labels);
        parts.push({
            kind: 'field', name, source, spec, prefix: part.prefix ?? null, readOnly: part.readOnly ?? null,
            raw, raws, alias: alias ? { name: alias.name } : null,
            symbols: written, values: resolved,
            differs: raws.some((value) => JSON.stringify(value ?? null) !== JSON.stringify(raws[0] ?? null)),
        });
    }
    // A Linear without bias says so in both modes; it is not an editable field here.
    const biasSource = row.kind === 'layer' && row.libraryKey === 'torch.nn.Linear' ? 'parameters'
        : row.family === 'linear' ? fieldSource(firstNode, 'bias') : null;
    if (biasSource) {
        const values = biasSource === 'parameters'
            ? list.map((merged) => resolvedLayerField(merged.metrics.node, merged.metrics.scope, 'bias'))
            : list.flatMap((merged) => resolvedBinding(merged.metrics, 'bias'));
        if (resolvesFalse(values)) parts.push({ kind: 'note', text: 'no bias', title: 'bias = False: no additive bias is learned. Change it in Settings.' });
    }
    return parts;
}

function groupChildren(list) {
    const groups = new Map();
    for (const merged of list) {
        merged.metrics.instances.forEach((instance) => {
            if (!instance.level) return;
            for (const [childId, child] of Object.entries(instance.level.nodes)) {
                if (!groups.has(childId)) groups.set(childId, []);
                groups.get(childId).push({
                    metrics: child,
                    trail: [...merged.trail, { id: merged.metrics.id, index: instance.index, templateId: instance.templateId, repeat: merged.metrics.kind === 'repeat' }],
                });
            }
        });
    }
    return groups;
}

/**
 * Rows of the Model table at a depth.
 * - root: the root graph's nodes (containers collapsed);
 * - blocks: also expands repeats and custom components, recursively, while
 *   standard modules (Linear, LayerNorm, Embedding, positions, attention)
 *   stay single rows;
 * - ops: expands everything down to primitive layers and operations.
 * `overrides` maps row keys to an explicit expanded state. Rows merge one node
 * across every instance of its ancestors, so ×N multiplies nested counts.
 */
export function buildModelTableRows({ metrics, rootGraph, templates = {}, depth = 'blocks', overrides = {} }) {
    if (!metrics?.root) return [];
    const rows = [];
    const modelTotals = metrics.totals;
    const automatic = (row) => depth === 'ops' || (depth === 'blocks' && !STANDARD_FAMILIES.has(row.family));

    const visit = (groups, orderGraphs, levelPath, level, parentKey, ancestorConflict) => {
        const order = unique([...orderGraphs.flatMap((graph) => readingOrder(graph)), ...groups.keys()]).filter((id) => groups.has(id));
        const visible = order.filter((id) => {
            const first = groups.get(id)[0].metrics;
            return !(first.kind === 'layer' && HIDDEN_KEYS.has(first.node.libraryKey));
        });
        visible.forEach((id, position) => {
            const list = groups.get(id);
            const first = list[0].metrics;
            const path = [...levelPath, id];
            const key = rowKey(path);
            const container = first.kind !== 'layer';
            const definitionIds = container ? unique(list.flatMap((merged) => merged.metrics.instances.map((instance) => instance.templateId))) : [];
            const family = container ? first.family ?? null : null;
            const occurrences = first.kind === 'repeat' ? list.flatMap((merged) => instanceTotals(merged.metrics)) : list.map((merged) => subtreeMetrics(merged.metrics));
            const complete = list.every((merged) => merged.metrics.kind === 'layer' || merged.metrics.complete);
            const live = levelPath.length ? null : rootGraph?.nodes?.[id];
            const row = {
                key, path, levelPath, nodeId: id, parentKey, depth: level, posInSet: position + 1, setSize: visible.length,
                kind: first.kind, libraryKey: first.kind === 'layer' ? first.node.libraryKey : null,
                label: (live ?? first.node).label ?? id,
                typeName: container ? templates[definitionIds[0] ?? first.node.templateId]?.name ?? 'Missing template' : getEntry(first.node.libraryKey)?.name ?? first.node.libraryKey,
                mode: first.kind === 'repeat' ? first.node.mode ?? 'sequential' : null,
                family, standard: !!standardFor(family),
                glyph: first.kind === 'layer' ? { libraryKey: first.node.libraryKey }
                    : first.kind === 'repeat' ? { icon: first.node.mode === 'parallel' ? 'parallel' : 'repeat' }
                        : standardFor(family) ? { libraryKey: standardFor(family).libraryKey, standard: true } : { icon: 'boxes' },
                expandable: container && list.some((merged) => merged.metrics.instances.some((instance) => instance.level)),
                multiplicity: first.kind === 'repeat' ? (complete ? occurrences.length : null) : list.length,
                // Repeat ancestors of the first occurrence, for opening its level.
                firstInstance: list[0].trail.map((step, index) => ({ path: levelPath.slice(0, index + 1), index: step.index, repeat: step.repeat })).filter((step) => step.repeat),
                trail: list[0].trail.map((step) => ({ id: step.id, index: step.index, templateId: step.templateId, repeat: step.repeat })),
                definitionIds,
                ownerIds: levelPath.length ? unique(list.map((merged) => merged.trail.at(-1)?.templateId)) : [],
                ancestorConflict,
                shape: shapeSummary(list.map((merged) => merged.metrics.output)),
                fused: first.kind === 'layer' && first.fused,
                countError: first.kind === 'repeat' ? first.countError : null,
            };
            row.expanded = row.expandable && (overrides[key] ?? automatic(row));
            for (const metric of METRICS) row[metric] = metricSummary(list, occurrences, metric);
            row.share = Object.fromEntries(METRICS.map((metric) => [metric, modelTotals[metric] > 0 ? row[metric].total / modelTotals[metric] : null]));
            const errors = unique(list.flatMap((merged) => merged.metrics.errors));
            // An open container's children carry their own errors.
            row.errors = row.expanded ? errors.filter((message) => !SUMMARY.test(message)) : errors;
            row.warnings = unique(list.flatMap((merged) => merged.metrics.warnings));
            row.parameterIssues = {};
            for (const merged of list) {
                for (const instance of merged.metrics.instances ?? []) {
                    for (const issue of instance.parameterIssues ?? []) (row.parameterIssues[issue.name] ??= new Set()).add(issue.message);
                }
            }
            row.parameterIssues = Object.fromEntries(Object.entries(row.parameterIssues).map(([name, messages]) => [name, [...messages]]));
            if (row.countError) row.parameterIssues.count = [row.countError];
            row.config = buildConfig(row, list, templates, list.map((merged) => occurrenceLabel(merged.trail)));
            rows.push(row);
            if (!row.expanded) return;
            const childGraphs = definitionIds.map((templateId) => templates[templateId]?.graph).filter(Boolean);
            visit(groupChildren(list), childGraphs, path, level + 1, key, ancestorConflict || definitionIds.length > 1);
        });
    };

    const rootGroups = new Map(Object.entries(metrics.root.nodes).map(([id, entry]) => [id, [{ metrics: entry, trail: [] }]]));
    visit(rootGroups, [rootGraph ?? metrics.root.graph], [], 0, null, false);
    return rows;
}

/** Keys of every ancestor row of a path, outermost first. */
export function ancestorKeys(path) {
    return path.slice(0, -1).map((_, index) => rowKey(path.slice(0, index + 1)));
}
