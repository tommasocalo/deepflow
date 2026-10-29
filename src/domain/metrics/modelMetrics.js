import { validate } from '../../validation/validator.js';
import { createValidationCache } from '../../validation/cache.js';
import { resolveRepeatCount } from '../expressions.js';
import { getIncomingEdges, mapTemplateInputs, orderOperationEdges } from '../graphInputs.js';
import { ACTIVATION_BYTES, CONSTANT_SOURCE_KEYS, layerMacs, layerSize, numelAtBatchOne, producesActivation } from './layerMetrics.js';

const TEMPLATE_INPUT = 'deepflow.template.Input';
const TEMPLATE_OUTPUT = 'deepflow.template.Output';
export const METRICS = ['params', 'macs', 'activation'];

/**
 * Standard modules that keep one output tensor in eager PyTorch (nn.Linear,
 * nn.Embedding, nn.LayerNorm, position tables). Their internal ops add no
 * activations of their own. Attention internals are counted individually.
 */
export const FUSED_FAMILIES = new Set(['linear', 'embedding', 'layernorm', 'learned-position', 'sinusoidal-position']);

/** The same family lookup the canvas uses for standard component styling. */
export const definitionFamily = (template) => template?.standardKey ?? template?.componentStyleKey ?? null;

const templateOutputIds = (template) => Object.values(template?.graph?.nodes ?? {})
    .filter((node) => node.kind === 'layer' && node.libraryKey === TEMPLATE_OUTPUT).map((node) => node.id);

/** The value a container passes on: its last template Output with a shape, as validation carries it. */
function instanceOutput(template, report) {
    return templateOutputIds(template).reduce((output, port) => report?.shapes[port] ?? output, null);
}

/**
 * Per-node, per-instance facts for one level, recursing through the instance
 * reports that validate({ collect: true }) keeps. Nothing is re-inferred.
 */
function levelMetrics(graph, report, templates, { inputWeightOnly = {}, fused = false, templateId = null } = {}) {
    const scope = report.scope ?? {};
    const nodes = {};
    const weightOnly = {};
    for (const id of [...report.topo, ...report.cyclic]) {
        const node = graph.nodes[id];
        if (!node) continue;
        const incoming = getIncomingEdges(graph, id);
        const inputShapes = orderOperationEdges(node, incoming).map((edge) => edge ? report.shapes[edge.source] ?? null : null);
        const output = report.shapes[id] ?? null;
        const errors = report.errors[id] ?? [];
        const warnings = report.warnings[id] ?? [];
        if (node.kind === 'layer') {
            const key = node.libraryKey;
            weightOnly[id] = CONSTANT_SOURCE_KEYS.has(key) ? true
                : key === TEMPLATE_INPUT ? inputWeightOnly[id] === true
                    : incoming.length > 0 && incoming.every((edge) => weightOnly[edge.source] === true);
            const counted = !fused && producesActivation(key) && !weightOnly[id];
            const elements = counted ? numelAtBatchOne(output) : 0;
            nodes[id] = {
                id, node, kind: 'layer', output, inputShapes, scope, fused, weightOnly: weightOnly[id],
                params: layerSize(node, scope),
                macs: layerMacs(node, scope, inputShapes, output),
                activation: elements === null ? null : elements * ACTIVATION_BYTES,
                errors, warnings,
            };
            continue;
        }
        if (node.kind !== 'supernode' && node.kind !== 'repeat') continue;
        const family = definitionFamily(templates[node.templateId]);
        // The outermost standard module owns the activation of its output.
        const ownsActivation = !fused && FUSED_FAMILIES.has(family);
        const countResult = node.kind === 'repeat' ? resolveRepeatCount(node.count, scope, inputShapes) : { count: 1, error: null };
        const instances = [];
        for (const child of report.children?.[id] ?? []) {
            if (!child) continue;
            const template = templates[child.templateId];
            if (!child.report || !template?.graph) {
                instances.push({ index: child.index, templateId: child.templateId, scope: null, inputs: null, output: null, level: null, parameterIssues: [] });
                continue;
            }
            const portFlags = mapTemplateInputs(template.graph, incoming, (edge) => weightOnly[edge.source] === true);
            const level = levelMetrics(template.graph, child.report, templates, {
                inputWeightOnly: portFlags, fused: fused || ownsActivation, templateId: child.templateId,
            });
            instances.push({
                index: child.index, templateId: child.templateId, scope: child.scope, inputs: child.inputs,
                output: instanceOutput(template, child.report), level, parameterIssues: child.parameterIssues ?? [],
            });
        }
        weightOnly[id] = instances.length > 0 && instances.every((instance) => instance.level
            && templateOutputIds(templates[instance.templateId]).every((port) => instance.level.nodes[port]?.weightOnly));
        const instanceActivations = ownsActivation ? instances.map((instance) => {
            if (weightOnly[id]) return 0;
            const elements = numelAtBatchOne(instance.output);
            return elements === null ? null : elements * ACTIVATION_BYTES;
        }) : null;
        nodes[id] = {
            id, node, kind: node.kind, output, inputShapes, scope, family, fused: fused || ownsActivation, ownsActivation,
            count: countResult.count, countError: countResult.error,
            // Budget or depth limits can stop validation before every instance.
            complete: Number.isInteger(countResult.count) && instances.length === countResult.count && instances.every((instance) => instance.level),
            instances, instanceActivations, weightOnly: weightOnly[id], errors, warnings,
        };
    }
    return { graph, templateId, scope, report, nodes, fused };
}

const SEPARATOR = '\u001f';
const prefixed = (prefix, key) => prefix ? `${prefix}${SEPARATOR}${key}` : key;

function emptyTotals() {
    return { params: 0, trainable: 0, macs: 0, activation: 0, unresolved: { params: new Map(), macs: new Map(), activation: new Map() }, largest: null };
}

function addInto(target, source, prefix = '') {
    target.params += source.params;
    target.trainable += source.trainable;
    target.macs += source.macs;
    target.activation += source.activation;
    for (const metric of METRICS) {
        for (const [key, label] of source.unresolved[metric]) target.unresolved[metric].set(prefixed(prefix, key), label);
    }
    if (source.largest && (!target.largest || source.largest.bytes > target.largest.bytes)) {
        target.largest = { ...source.largest, path: prefix ? [prefix, ...source.largest.path] : source.largest.path };
    }
}

function considerLargest(totals, bytes, path, label, shape) {
    if (bytes > 0 && (!totals.largest || bytes > totals.largest.bytes)) totals.largest = { bytes, path, label, shape };
}

const levelCache = new WeakMap();
const entryCache = new WeakMap();
const instanceCache = new WeakMap();

/**
 * Sum over a level and everything inside it. Unknown values are counted, never
 * added: `unresolved[metric]` maps relative node paths to labels, so totals
 * read "≥ X · N rows unresolved" and are never NaN.
 */
export function levelSubtree(level) {
    if (!level) return emptyTotals();
    if (levelCache.has(level)) return levelCache.get(level);
    const totals = emptyTotals();
    for (const entry of Object.values(level.nodes)) addInto(totals, subtreeMetrics(entry));
    levelCache.set(level, totals);
    return totals;
}

/** One instance of a container, keyed relative to the container's level. */
function instanceSubtree(entry, index) {
    const totals = emptyTotals();
    const instance = entry.instances[index];
    if (instance?.level) addInto(totals, levelSubtree(instance.level), entry.id);
    if (entry.ownsActivation) {
        const bytes = entry.instanceActivations[index];
        if (bytes === null) totals.unresolved.activation.set(entry.id, entry.node.label ?? entry.id);
        else {
            totals.activation += bytes;
            considerLargest(totals, bytes, [entry.id], entry.node.label ?? entry.id, instance?.output ?? null);
        }
    }
    return totals;
}

/** Totals of every instance of a container, in instance order. */
export function instanceTotals(entry) {
    if (entry.kind === 'layer') return [subtreeMetrics(entry)];
    if (instanceCache.has(entry)) return instanceCache.get(entry);
    const list = entry.instances.map((_, index) => instanceSubtree(entry, index));
    instanceCache.set(entry, list);
    return list;
}

/** Totals of one node: a layer itself, or a container summed over its instances. */
export function subtreeMetrics(entry) {
    if (entryCache.has(entry)) return entryCache.get(entry);
    const totals = emptyTotals();
    const label = entry.node.label ?? entry.id;
    if (entry.kind === 'layer') {
        if (entry.params) {
            totals.params = entry.params.total;
            totals.trainable = entry.params.trainable;
        } else totals.unresolved.params.set(entry.id, label);
        if (entry.macs === null) totals.unresolved.macs.set(entry.id, label);
        else totals.macs = entry.macs;
        if (entry.activation === null) totals.unresolved.activation.set(entry.id, label);
        else {
            totals.activation = entry.activation;
            considerLargest(totals, entry.activation, [entry.id], label, entry.output);
        }
    } else {
        if (!entry.complete) for (const metric of METRICS) totals.unresolved[metric].set(entry.id, label);
        for (const instance of instanceTotals(entry)) addInto(totals, instance);
    }
    entryCache.set(entry, totals);
    return totals;
}

function countUses(level, uses) {
    for (const entry of Object.values(level.nodes)) {
        if (entry.kind === 'layer') continue;
        for (const instance of entry.instances) {
            uses[instance.templateId] = (uses[instance.templateId] ?? 0) + 1;
            if (instance.level) countUses(instance.level, uses);
        }
    }
}

/**
 * Parameters, multiply-accumulates and activation sizes of the whole model at
 * B = 1, fp32, forward pass, with no weight tying (each layer owns its weights).
 * Sizes come from the root Input nodes through ordinary shape validation.
 */
export function computeModelMetrics(rootGraph, templates = {}) {
    const report = validate(rootGraph, { templates, collect: true });
    const root = levelMetrics(rootGraph, report, templates);
    const templateUses = {};
    countUses(root, templateUses);
    const inputs = Object.values(root.nodes)
        .filter((entry) => entry.kind === 'layer' && entry.node.libraryKey === 'deepflow.io.Input')
        .map((entry) => ({ id: entry.id, label: entry.node.label, shape: entry.output }));
    return { root, totals: levelSubtree(root), templateUses, inputs };
}

/** Never throws: a failure leaves the table empty instead of breaking the editor. */
function safeComputeModelMetrics(graph, templates) {
    try {
        return computeModelMetrics(graph, templates);
    } catch (error) {
        console.error('[modelMetrics] computeModelMetrics() threw', error);
        return { unavailable: true, error: String(error?.message ?? error) };
    }
}

const metricsCache = createValidationCache((graph, { templates }) => safeComputeModelMetrics(graph, templates));

/**
 * Shared across panes; moving nodes reuses the previous result. The cache
 * compares nodes and edges, so the model-wide parameters that live beside
 * them on the root graph join its key.
 */
export function cachedModelMetrics(graph, templates = {}) {
    return metricsCache(graph, templates, { globals: graph?.parameters ?? [] });
}

/** Totals of the whole model, or of a subtree's metrics root. */
export function modelTotals(metrics) {
    return metrics?.totals ?? emptyTotals();
}

/** Relative unresolved keys use this separator; rows compare them as strings. */
export const PATH_SEPARATOR = SEPARATOR;
