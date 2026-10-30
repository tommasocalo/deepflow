import { STANDARD_COMPONENTS } from '../components/index.js';
import { getEntry } from '../library/index.js';
import { isShapeCollection } from '../shapeCollections.js';
import { ANALYSIS_LIMITS, BUDGET_ERROR, instanceTotals, METRICS, subtreeMetrics, SUMMARY_ERROR } from '../metrics/modelMetrics.js';
import { aliasFor, configLayout, fieldRaw, fieldSource, fieldSpec, rawText, resolvedBinding, resolvedLayerValue, resolvesFalse, runsText, summarizeValues, valueRuns } from './configFields.js';

export const DEPTHS = ['root', 'blocks', 'ops'];
/** Definitions that stay one row at Blocks depth: the library's standard modules. */
export const STANDARD_FAMILIES = new Set(['linear', 'embedding', 'layernorm', 'learned-position', 'sinusoidal-position', 'attention', 'attention-head']);
const HIDDEN_KEYS = new Set(['deepflow.template.Input', 'deepflow.template.Output']);
const BUDGET_TEXT = 'Expanded graph exceeds';
const STOPPED_WARNING = `The analysis stopped at ${ANALYSIS_LIMITS.nodes.toLocaleString('en-US')} expanded nodes here; counts and totals are lower bounds`;
/** Layers whose bias can be switched off; the cell says so when it is. */
const BIAS_LAYERS = /^torch\.nn\.(Linear|Conv[123]d|ConvTranspose[123]d)$/;

/** Rows are keyed by their absolute node-id path, like level paths. */
export const rowKey = (path) => JSON.stringify(path);

const orderCache = new WeakMap();

/**
 * Dataflow order; when several nodes are ready, read top to bottom and then
 * left to right, as on the canvas. Nodes in cycles come last. Graphs are
 * immutable, so the order is computed once per graph object.
 */
export function readingOrder(graph) {
    if (!graph) return [];
    if (orderCache.has(graph)) return orderCache.get(graph);
    const order = computeReadingOrder(graph);
    orderCache.set(graph, order);
    return order;
}

function computeReadingOrder(graph) {
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
    // The ready set stays sorted, largest first, so the next node pops off the end.
    const ready = ids.filter((id) => indegree.get(id) === 0).sort((a, b) => before(b, a));
    const insert = (id) => {
        let low = 0;
        let high = ready.length;
        while (low < high) {
            const middle = (low + high) >> 1;
            if (before(ready[middle], id) > 0) low = middle + 1;
            else high = middle;
        }
        ready.splice(low, 0, id);
    };
    const order = [];
    while (ready.length) {
        const id = ready.pop();
        order.push(id);
        for (const next of outgoing.get(id)) {
            indegree.set(next, indegree.get(next) - 1);
            if (indegree.get(next) === 0) insert(next);
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
 * `null` marks an occurrence whose shape is unknown; `unknownReason` says why.
 * `known` of `count` occurrences have a known shape, so a label that covers
 * only some of them can say so.
 */
export function shapeSummary(shapes, { unknownReason = null } = {}) {
    const unknown = shapes.filter((shape) => shape === null || shape === undefined).length;
    const counts = { known: shapes.length - unknown, count: shapes.length };
    const distinct = [];
    const seen = new Set();
    for (const shape of shapes) {
        if (shape === null || shape === undefined) continue;
        const key = JSON.stringify(shape);
        if (seen.has(key)) continue;
        seen.add(key);
        distinct.push(shape);
    }
    const unknownNote = unknown && distinct.length ? `\nUnknown in ${unknown} of ${shapes.length} occurrences${unknownReason ? `: ${unknownReason}` : ''}` : '';
    if (!distinct.length) return { text: null, varies: false, unknown: shapes.length > 0, title: unknownReason ? `Unknown: ${unknownReason}` : '', ...counts };
    const labels = distinct.map((shape) => singleShapeText(shape) ?? '?');
    if (distinct.length === 1) return { text: singleShapeText(distinct[0]), varies: false, unknown: false, title: `${labels[0]}${unknownNote}`, ...counts };
    const ranks = distinct.map((shape) => Array.isArray(shape) ? shape.length : -1);
    const text = ranks.every((rank) => rank > 0 && rank === ranks[0])
        ? Array.from({ length: ranks[0] }, (_, axis) => axisText(distinct.map((shape) => shape[axis]), axis)).join(' × ')
        : `${labels[0]} …`;
    return { text, varies: true, unknown: false, title: `${labels.join(' · ')}${unknownNote}`, ...counts };
}

const unique = (values) => [...new Set(values)];

/** Occurrence labels for per-instance values: `i=2`, or `3.5` inside nested repeats. */
function occurrenceLabel(trail) {
    const indices = trail.filter((step) => step.repeat).map((step) => step.index);
    if (!indices.length) return '';
    return indices.length === 1 ? `i=${indices[0]}` : indices.join('.');
}

/**
 * A metric over every merged occurrence. Each occurrence is exact (a number),
 * a lower bound (`{ atLeast }`: part of it is unknown) or unknown (`null`).
 * `each` is the range of the exact occurrences only and `knownCount` says how
 * many there are, so a half-analysed instance never reads as a small one and
 * "× N = total" is claimed only when every occurrence is known.
 */
function metricSummary(list, occurrences, metric, labels) {
    let total = 0;
    const unresolved = new Map();
    for (const merged of list) {
        const totals = subtreeMetrics(merged.metrics);
        total += totals[metric];
        for (const [key, value] of totals.unresolved[metric]) unresolved.set(key, value);
    }
    const values = occurrences.map((occurrence) => !occurrence.unresolved[metric].size ? occurrence[metric]
        : occurrence[metric] > 0 ? { atLeast: occurrence[metric] } : null);
    const known = values.filter((value) => typeof value === 'number');
    const each = occurrences.length > 1 && known.length ? { min: Math.min(...known), max: Math.max(...known) } : null;
    const items = [];
    const seen = new Set();
    for (const { label, reason } of unresolved.values()) {
        const key = `${label}\u001f${reason ?? ''}`;
        if (seen.has(key)) continue;
        seen.add(key);
        items.push({ label, reason: reason ?? null });
    }
    return {
        total,
        unresolved: unresolved.size,
        unresolvedItems: items,
        each,
        knownCount: known.length,
        count: occurrences.length,
        // Which instances differ, or are not known exactly: one hover lists them.
        perInstance: occurrences.length > 1 && (known.length < values.length || each.min !== each.max) ? valueRuns(values, labels) : null,
        trainable: metric === 'params' ? list.reduce((sum, merged) => sum + subtreeMetrics(merged.metrics).trainable, 0) : undefined,
    };
}

function standardFor(family) {
    return STANDARD_COMPONENTS.find((component) => component.key === family) ?? null;
}

/** Field cells for one row: written values (Symbols) and resolved values (Values). */
function buildConfig(row, list, templates, occurrenceLabels, instanceLabels) {
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
        // One owner definition (or one root node) means one written value for every occurrence.
        const raw = alias ? alias.default : fieldRaw(firstNode, source, name);
        let written = rawText(alias?.exposes?.inherited && alias.default === (spec?.default ?? null) ? undefined : raw, spec);
        // Local versions can write the field differently: Symbols says so instead of showing the first.
        if (row.ownerIds.length > 1 && !alias) {
            const texts = list.map((merged) => rawText(fieldRaw(merged.metrics.node, source, name), spec).text);
            if (texts.some((value) => value !== texts[0])) written = { text: 'varies', tone: 'varies', title: `As written: ${runsText(valueRuns(texts, occurrenceLabels))}` };
        }
        let values;
        let labels = occurrenceLabels;
        // Why a value reads ? where it rests on a width inferred after a shape error.
        let unknownReason = null;
        if (source === 'count') values = list.map((merged) => merged.metrics.count);
        else if (source === 'parameters') {
            values = list.map((merged) => resolvedLayerValue(merged.metrics, name));
            unknownReason = list.find((merged) => merged.metrics.taintedSettings?.includes(name))?.metrics.settingsReason ?? null;
        } else {
            values = list.flatMap((merged) => resolvedBinding(merged.metrics, name));
            labels = row.kind === 'repeat' ? instanceLabels : occurrenceLabels;
            unknownReason = list.flatMap((merged) => merged.metrics.instances.map((instance) => instance.scopeTaint?.[name]?.reason)).find(Boolean) ?? null;
        }
        // A default worth no words (exact GELU, groups 1) is not drawn, but the editor still offers it.
        const concealed = !!part.hideDefault && values.every((value) => value === null || value === undefined || value === spec?.default);
        const resolved = summarizeValues(values, spec, labels);
        parts.push({
            kind: 'field', name, source, spec, prefix: part.prefix ?? null, readOnly: part.readOnly ?? null, concealed,
            raw, alias: alias ? { name: alias.name } : null,
            symbols: written, values: resolved, unknownReason,
        });
    }
    // A layer without bias says so in both modes; it is not an editable field here.
    const biasSource = row.kind === 'layer' && BIAS_LAYERS.test(row.libraryKey ?? '') ? 'parameters'
        : row.family === 'linear' ? fieldSource(firstNode, 'bias') : null;
    if (biasSource) {
        const values = biasSource === 'parameters'
            ? list.map((merged) => resolvedLayerValue(merged.metrics, 'bias'))
            : list.flatMap((merged) => resolvedBinding(merged.metrics, 'bias'));
        if (resolvesFalse(values)) parts.push({ kind: 'note', text: 'no bias', title: 'bias = False: no additive bias is learned. Change it in Settings.' });
    }
    // A switched-off parameter is not created: it holds no weights.
    if (row.libraryKey === 'deepflow.op.Parameter' && resolvesFalse(list.map((merged) => resolvedLayerValue(merged.metrics, 'enabled')))) {
        parts.push({ kind: 'note', text: 'off', title: 'enabled = False: this parameter is not created and holds no weights. Change it in Settings.' });
    }
    return parts;
}

/**
 * Children of a container's rows, merged by node id across every analysed
 * instance; `instances` lists those instances' trails, so a child can say
 * which instances lack it (a local version without that node).
 */
function groupChildren(list) {
    const groups = new Map();
    const instances = [];
    for (const merged of list) {
        merged.metrics.instances.forEach((instance) => {
            if (!instance.level) return;
            const trail = [...merged.trail, { id: merged.metrics.id, index: instance.index, templateId: instance.templateId, repeat: merged.metrics.kind === 'repeat' }];
            instances.push(trail);
            for (const [childId, child] of Object.entries(instance.level.nodes)) {
                if (!groups.has(childId)) groups.set(childId, []);
                groups.get(childId).push({ metrics: child, trail });
            }
        });
    }
    return { groups, instances };
}

/** Runs of occurrences, by label, where `test(trail)` holds: `i=5`, `i=0–4, i=6–11`. */
function occurrenceRuns(trails, test) {
    return valueRuns(trails.map(test), trails.map(occurrenceLabel)).filter((run) => run.value);
}

/** Outputs of every occurrence; unknown shapes are null. Repeats show one instance's output. */
function occurrenceShapes(first, list) {
    if (first.kind !== 'repeat') return list.map((merged) => merged.metrics.shapeUnknown ? null : merged.metrics.output);
    return list.flatMap((merged) => merged.metrics.complete
        ? merged.metrics.instances.map((instance) => instance.outputTaint ? null : instance.output)
        : [null]);
}

function shapeReason(first, list) {
    for (const merged of list) {
        if (merged.metrics.shapeUnknown) return merged.metrics.shapeUnknown;
        if (first.kind === 'repeat') {
            if (!merged.metrics.complete) return merged.metrics.incompleteReason;
            const tainted = merged.metrics.instances.find((instance) => instance.outputTaint);
            if (tainted) return tainted.outputTaint.reason;
        }
    }
    return null;
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
 * Inside a container that could not be analysed completely (an unresolved
 * count, the node budget), ×N and totals are lower bounds (`lowerBound`).
 */
export function buildModelTableRows({ metrics, rootGraph, templates = {}, depth = 'blocks', overrides = {} }) {
    if (!metrics?.root) return [];
    const rows = [];
    const modelTotals = metrics.totals;
    // Rows with a real error somewhere below them, from the errors the metrics recorded.
    const errorsInside = new Set();
    for (const issue of metrics.issues?.errors ?? []) {
        for (let length = 1; length < issue.path.length; length++) errorsInside.add(rowKey(issue.path.slice(0, length)));
    }
    const automatic = (row) => depth === 'ops' || (depth === 'blocks' && !STANDARD_FAMILIES.has(row.family));

    const visit = (groups, orderGraphs, levelPath, level, parentKey, ancestorConflict, ancestorIncomplete, ancestorFactors, parentInstances = []) => {
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
            const occurrenceLabels = list.map((merged) => occurrenceLabel(merged.trail));
            const instanceLabels = first.kind === 'repeat'
                ? list.flatMap((merged) => merged.metrics.instances.map((instance) => occurrenceLabel([...merged.trail, { repeat: true, index: instance.index }])))
                : occurrenceLabels;
            const complete = list.every((merged) => merged.metrics.kind === 'layer' || merged.metrics.complete);
            const counts = first.kind === 'repeat' ? list.map((merged) => merged.metrics.count) : null;
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
                // Occurrences per forward pass: declared repeat counts, or merged entries.
                multiplicity: counts ? (counts.every((count) => Number.isInteger(count)) ? counts.reduce((sum, count) => sum + count, 0) : null) : list.length,
                // Inside an incompletely analysed container, counts and totals are lower bounds:
                // { label, reason } of that container, or null.
                lowerBound: ancestorIncomplete,
                // How ×N multiplies: one { label, count } per repeat on the way down, this row's included.
                factors: counts && counts.every((count) => Number.isInteger(count) && count === counts[0])
                    ? [...ancestorFactors, { label: (live ?? first.node).label ?? id, count: counts[0] }] : ancestorFactors,
                complete,
                incompleteReason: complete ? null : list.find((merged) => merged.metrics.incompleteReason)?.metrics.incompleteReason ?? null,
                // Repeat ancestors of the first occurrence, for opening its level.
                firstInstance: list[0].trail.map((step, index) => ({ path: levelPath.slice(0, index + 1), index: step.index, repeat: step.repeat })).filter((step) => step.repeat),
                trail: list[0].trail.map((step) => ({ id: step.id, index: step.index, templateId: step.templateId, repeat: step.repeat })),
                definitionIds,
                ownerIds: levelPath.length ? unique(list.map((merged) => merged.trail.at(-1)?.templateId)) : [],
                ancestorConflict,
                shape: shapeSummary(occurrenceShapes(first, list), { unknownReason: shapeReason(first, list) }),
                // A parallel repeat collects its instances' outputs; the row shows one instance's.
                collectedShape: first.kind === 'repeat' && first.node.mode === 'parallel' ? shapeSummary(list.map((merged) => merged.metrics.output)).text : null,
                // Ops inside a standard module: the module's output stands for theirs.
                fusedActivation: first.kind === 'layer' && list.every((merged) => merged.metrics.fusedActivation),
                countError: first.kind === 'repeat' ? first.countError : null,
                // Scopes of the first occurrence, for checking an edit before it is written.
                scope: first.scope ?? {},
                instanceScope: container ? first.instances[0]?.scope ?? null : null,
                iterationVariable: first.kind === 'repeat' ? first.node.iterationVariable || 'i' : null,
            };
            // Instances of the container above that have no such node (a local version without it).
            const present = new Set(list.map((merged) => merged.trail));
            const absent = parentInstances.filter((trail) => !present.has(trail));
            row.absent = absent.length ? { count: absent.length, of: parentInstances.length, runs: occurrenceRuns(parentInstances, (trail) => !present.has(trail)),
                local: absent.every((trail) => templates[trail.at(-1)?.templateId]?.structureVariant === true) } : null;
            // Occurrences whose definition is a local version, when the others share the original.
            const localTrails = list.map((merged) => merged.trail).filter((trail) => templates[trail.at(-1)?.templateId]?.structureVariant === true);
            row.variants = levelPath.length && localTrails.length && localTrails.length < list.length
                ? { count: localTrails.length, runs: occurrenceRuns(list.map((merged) => merged.trail), (trail) => templates[trail.at(-1)?.templateId]?.structureVariant === true),
                    name: templates[localTrails[0].at(-1).templateId]?.name ?? 'its definition' } : null;
            // Where a merge (an Add, a Concat, a MatMul) takes its inputs from, in port order.
            row.inputs = first.kind === 'layer' && first.inputs?.length > 1
                ? first.inputs.map((input, port) => input && { label: input.label, shape: shapeSummary(list.map((merged) => merged.metrics.inputs?.[port]?.unknown ? null : merged.metrics.inputShapes[port])).text }).filter(Boolean)
                : null;
            row.expanded = row.expandable && (overrides[key] ?? automatic(row));
            for (const metric of METRICS) {
                row[metric] = metricSummary(list, occurrences, metric, instanceLabels);
                // A layer says why its own value is unknown; containers list what they could not count.
                if (row[metric].unresolved && first.kind === 'layer') row[metric].reason = list.map((merged) => merged.metrics.reasons?.[metric]).find(Boolean) ?? null;
            }
            row.share = Object.fromEntries(METRICS.map((metric) => [metric, modelTotals[metric] > 0 ? row[metric].total / modelTotals[metric] : null]));
            // The node budget is a limit of the analysis, not a mistake in the
            // model: where validation stopped, and a collapsed container whose
            // summary only repeats such stops, get a warning instead of an error.
            const allErrors = unique(list.flatMap((merged) => merged.metrics.errors));
            const stopped = allErrors.some((message) => BUDGET_ERROR.test(message))
                || (container && !row.expanded && !errorsInside.has(key) && allErrors.some((message) => SUMMARY_ERROR.test(message) && message.includes(BUDGET_TEXT)));
            const errors = allErrors.filter((message) => !BUDGET_ERROR.test(message) && !(stopped && !errorsInside.has(key) && SUMMARY_ERROR.test(message)));
            // An open container's children show their own errors, so its summary
            // gives way to the problems of the container itself (ports, missing
            // definitions, parameters that do not resolve).
            row.errors = row.expanded
                ? unique([...errors.filter((message) => !SUMMARY_ERROR.test(message)),
                    ...list.flatMap((merged) => merged.metrics.instances.flatMap((instance) => instance.instanceIssues ?? []))])
                : errors;
            // A shared definition can fail in some of its instances only: name them.
            const failing = list.map((merged) => [...merged.metrics.errors, ...(row.expanded ? merged.metrics.instances.flatMap((instance) => instance.instanceIssues ?? []) : [])]
                .some((message) => row.errors.includes(message)));
            row.failing = row.errors.length && list.length > 1
                ? { count: failing.filter(Boolean).length, of: list.length, runs: valueRuns(failing, occurrenceLabels).filter((run) => run.value) } : null;
            row.warnings = unique([...(stopped ? [STOPPED_WARNING] : []), ...list.flatMap((merged) => merged.metrics.warnings)]);
            row.parameterIssues = {};
            for (const merged of list) {
                for (const instance of merged.metrics.instances ?? []) {
                    for (const issue of instance.parameterIssues ?? []) (row.parameterIssues[issue.name] ??= new Set()).add(issue.message);
                }
            }
            row.parameterIssues = Object.fromEntries(Object.entries(row.parameterIssues).map(([name, messages]) => [name, [...messages]]));
            if (row.countError) row.parameterIssues.count = [row.countError];
            row.config = buildConfig(row, list, templates, occurrenceLabels, instanceLabels);
            rows.push(row);
            if (!row.expanded) return;
            const childGraphs = definitionIds.map((templateId) => templates[templateId]?.graph).filter(Boolean);
            const children = groupChildren(list);
            visit(children.groups, childGraphs, path, level + 1, key, ancestorConflict || definitionIds.length > 1,
                ancestorIncomplete ?? (complete ? null : { label: row.label, reason: row.incompleteReason ?? 'could not be analysed completely' }), row.factors, children.instances);
        });
    };

    const rootGroups = new Map(Object.entries(metrics.root.nodes).map(([id, entry]) => [id, [{ metrics: entry, trail: [] }]]));
    visit(rootGroups, [rootGraph ?? metrics.root.graph], [], 0, null, false, null, []);
    return rows;
}

/** Keys of every ancestor row of a path, outermost first. */
export function ancestorKeys(path) {
    return path.slice(0, -1).map((_, index) => rowKey(path.slice(0, index + 1)));
}
