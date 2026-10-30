import { validate } from '../../validation/validator.js';
import { createValidationCache } from '../../validation/cache.js';
import { inferShape } from '../../validation/shapes.js';
import { repeatBindingValue, resolveLayerParameters, resolveRepeatCount } from '../expressions.js';
import { parameterReferences } from '../parameterBindings.js';
import { getEntry } from '../library/index.js';
import { getTemplateInputIds, mapTemplateInputs, orderOperationEdges } from '../graphInputs.js';
import {
    ACTIVATION_BYTES, activationKnown, CONSTANT_SOURCE_KEYS, countsMacs, hasPlaceholderShape, layerMacs, layerSize, moduleMacsKind, numelAtBatchOne,
    placeholderReason, producesActivation, unresolvedReason,
} from './layerMetrics.js';

const TEMPLATE_INPUT = 'deepflow.template.Input';
const TEMPLATE_OUTPUT = 'deepflow.template.Output';
const SOURCE_KEYS = new Set(['deepflow.io.Input', TEMPLATE_INPUT, ...CONSTANT_SOURCE_KEYS]);
const SINK_KEYS = new Set(['deepflow.io.Output', TEMPLATE_OUTPUT]);
/** The validator's budget and nesting limits; metrics report when they stop the walk. */
export const ANALYSIS_LIMITS = { nodes: 20000, depth: 8 };
export const METRICS = ['params', 'macs', 'activation'];
/** Container summaries repeat their children's errors; rows and totals count the children instead. */
export const SUMMARY_ERROR = /^(?:Repeat|Template ".*") has \d+ internal errors?: /;
/** Where validation ran out of its node budget: a limit of the analysis, not a mistake in the model. */
export const BUDGET_ERROR = /^Expanded graph exceeds \d+ nodes$/;

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

/** The port a container passes on: its last template Output with a shape, as validation carries it. */
function carriedPort(template, report) {
    return templateOutputIds(template).reduce((port, id) => report?.shapes[id] ? id : port, null);
}

/** An unknown neural module: no library entry, so its weights, compute and shape are unknown. */
const unknownModule = (key) => /^(torch|deepflow)\.nn\./.test(key ?? '') && !getEntry(key);

/**
 * Why a shape is a guess: it rests on a size the model does not determine (an
 * unresolved count, a missing input, a layer without a shape rule, a shape
 * error that left no output, or one that reconciled inputs which disagree,
 * such as an Add of 768 and 1024 features). Validation still fills such
 * shapes in, so the canvas can go on drawing, but nothing downstream is
 * computed from them: neither sizes nor widths inferred from them (a derived
 * parameter such as D). `origin` names the node where the problem starts;
 * downstream nodes read "after <origin>: <problem>". A shape error whose
 * output the node's own settings fix (a MatMul's rows and columns, a norm
 * that keeps its input shape) does not spread: later layers use the declared
 * settings, and the error is reported on its own row and in the footer.
 */
const taintOf = (problem, origin) => ({ problem, origin, reason: problem });
const worse = (current, next) => current ?? next ?? null;
/** The same problem as seen from a node downstream of its origin. */
const downstream = (taint) => taint && { ...taint, reason: `after ${taint.origin}: ${taint.problem}` };
const sameJson = (a, b) => a === b || JSON.stringify(a ?? null) === JSON.stringify(b ?? null);
const sameSize = (a, b) => (a?.total ?? null) === (b?.total ?? null) && (a?.trainable ?? null) === (b?.trainable ?? null);

/** A layer's settings that read a scope name resting on a guessed shape: { taint, names } or null. */
function taintedSettings(node, scopeTaint) {
    let taint = null;
    const names = [];
    for (const [name, raw] of Object.entries(node.parameters ?? {})) {
        const hit = parameterReferences(raw).find((reference) => scopeTaint[reference]);
        if (!hit) continue;
        taint ??= scopeTaint[hit];
        names.push(name);
    }
    return taint ? { taint, names } : null;
}

const NO_TAINT = Object.freeze({});

/** The scope without the names that rest on a guess, to see what a value depends on. */
const withoutTainted = (scope, scopeTaint) => Object.fromEntries(Object.entries(scope).filter(([name]) => !scopeTaint[name]));

/**
 * The names of an instance's scope that rest on a guessed shape: tainted names
 * it inherits, widths inferred from a tainted input port (derived parameters
 * such as D), and bindings or defaults that read either, in declaration
 * order as buildParameterScope resolves them.
 */
function instanceScopeTaint(template, node, index, parentTaint, portTaint) {
    // Nothing upstream rests on a guess: the common case costs nothing.
    if (!Object.keys(parentTaint).length && !Object.values(portTaint).some(Boolean)) return NO_TAINT;
    const result = { ...parentTaint };
    const repeat = node.kind === 'repeat';
    if (repeat) delete result[node.iterationVariable || 'i'];
    const bindings = (repeat ? node.bindings : node.parameterValues) ?? {};
    for (const parameter of template.freeParameters ?? []) {
        let taint = null;
        if (parameter.derived) {
            const upstream = portTaint[parameter.derived.inputId] ?? null;
            taint = upstream && { ...upstream, reason: `${parameter.name} is inferred from an input ${upstream.reason}` };
        } else {
            let raw = repeatBindingValue(bindings[parameter.name], parameter, repeat ? index : undefined);
            if (raw === undefined || (raw === null && !parameter.nullable) || raw === '') raw = parameter.default;
            const hit = parameterReferences(raw).find((name) => result[name]);
            taint = hit ? result[hit] : null;
        }
        if (taint) result[parameter.name] = taint;
        else delete result[parameter.name];
    }
    return result;
}

/**
 * Per-node, per-instance facts for one level, recursing through the instance
 * reports that validate({ collect: true }) keeps. Nothing is re-inferred: a
 * node's MACs and activation size become unknown when its input shapes rest on
 * an unknown size or when the node has a shape error of its own, never
 * computed from a placeholder. `scopeTaint` maps the scope names that rest on
 * a guessed shape to that problem: what settings that read them size (an
 * output, weights, products) is unknown. Shape rules and size formulas run
 * again without those names only to see whether a value depends on them.
 * `shapeError` is the first shape error in the level or anything inside it,
 * so a fused module can say its output is unsure.
 */
function levelMetrics(graph, report, templates, context) {
    const { inputWeightOnly = {}, inputTaint = {}, scopeTaint = {}, fused = false, templateId = null, path = [], issues } = context;
    const scope = report.scope ?? {};
    const tainting = Object.keys(scopeTaint).length > 0;
    const cleanScope = tainting ? withoutTainted(scope, scopeTaint) : scope;
    const nodes = {};
    const weightOnly = {};
    const taint = {};
    let shapeError = null;
    // The edges into each node, in graph order, as getIncomingEdges returns them.
    const incomingEdges = new Map();
    for (const edge of graph.edges ?? []) {
        if (!graph.nodes[edge.source]) continue;
        if (!incomingEdges.has(edge.target)) incomingEdges.set(edge.target, []);
        incomingEdges.get(edge.target).push(edge);
    }
    for (const message of report.globalErrors ?? []) issues.global.add(message);
    for (const id of [...report.topo, ...report.cyclic]) {
        const node = graph.nodes[id];
        if (!node) continue;
        const label = node.label ?? id;
        const where = context.where ? `${context.where} › ${label}` : label;
        // Problems are located by their last two levels: mlp proj › input × weightᵀ.
        const origin = where.split(' › ').slice(-2).join(' › ');
        const incoming = incomingEdges.get(id) ?? [];
        const ordered = orderOperationEdges(node, incoming);
        const inputShapes = ordered.map((edge) => edge ? report.shapes[edge.source] ?? null : null);
        const output = report.shapes[id] ?? null;
        const errors = report.errors[id] ?? [];
        const warnings = report.warnings[id] ?? [];
        const ownErrors = errors.filter((message) => !SUMMARY_ERROR.test(message) && !BUDGET_ERROR.test(message));
        if (ownErrors.length) recordError(issues, path, id, where, ownErrors);
        if (errors.some((message) => message.startsWith('Template recursion exceeds depth'))) issues.limits.depth = true;
        // Why the inputs cannot be trusted, from the upstream nodes.
        let inputProblem = null;
        if (node.kind === 'layer' && node.libraryKey === TEMPLATE_INPUT) inputProblem = inputTaint[id] ?? null;
        else if (!(node.kind === 'layer' && SOURCE_KEYS.has(node.libraryKey))) {
            if (!incoming.length) inputProblem = taintOf('it has no input', origin);
            for (const edge of ordered) inputProblem = worse(inputProblem, edge ? downstream(taint[edge.source] ?? null) : taintOf('an input is not connected', origin));
        }
        if (node.kind === 'layer') {
            const key = node.libraryKey;
            const shapeIssue = report.shapeIssues?.[id]?.[0] ?? null;
            if (shapeIssue) shapeError ??= { message: shapeIssue, origin };
            // Settings that read a width inferred from a guessed shape: whatever they size is unknown too.
            const settings = tainting && key !== TEMPLATE_INPUT ? taintedSettings(node, scopeTaint) : null;
            const outputFollowsTaint = !!settings && !SINK_KEYS.has(key)
                && !sameJson(inferShape(key, inputShapes, resolveLayerParameters(node, cleanScope).parameters).output, output);
            const macsFollowTaint = !!settings && countsMacs(key) && layerMacs(node, cleanScope, inputShapes, output) !== layerMacs(node, scope, inputShapes, output);
            // Unknown inputs come first; otherwise the node's own shape error, then its settings.
            const ownProblem = inputProblem ?? (shapeIssue ? taintOf(shapeIssue, origin) : null);
            const problem = ownProblem ?? (outputFollowsTaint ? settings.taint : null);
            const macsProblem = problem ?? (macsFollowTaint ? settings.taint : null);
            weightOnly[id] = CONSTANT_SOURCE_KEYS.has(key) ? true
                : key === TEMPLATE_INPUT ? inputWeightOnly[id] === true
                    : incoming.length > 0 && incoming.every((edge) => weightOnly[edge.source] === true);
            const placeholder = hasPlaceholderShape(key) || unknownModule(key);
            // A shape validation only carried on (mismatched inputs reconciled) is a guess as well.
            const guess = report.shapeGuesses?.[id] ?? null;
            taint[id] = worse(inputProblem, placeholder ? taintOf(placeholderReason(node), origin)
                : output === null && !SINK_KEYS.has(key) ? taintOf(shapeIssue ?? 'its output shape is unknown', origin)
                    : guess ? taintOf(guess, origin) : outputFollowsTaint ? settings.taint : null);
            const reasons = {};
            const size = unknownModule(key) ? null : layerSize(node, scope);
            const sizeFollowsTaint = !!settings && !unknownModule(key) && !sameSize(size, layerSize(node, cleanScope));
            const params = sizeFollowsTaint ? null : size;
            if (!params) reasons.params = sizeFollowsTaint ? settings.taint.reason : unresolvedReason(node, 'params');
            let macs;
            if (macsProblem && countsMacs(key)) {
                macs = null;
                reasons.macs = macsProblem.reason;
            } else {
                macs = unknownModule(key) ? null : layerMacs(node, scope, inputShapes, output);
                if (macs === null) reasons.macs = unknownModule(key) ? `${getEntry(key)?.name ?? key} is not in the library; its compute is unknown` : unresolvedReason(node, 'macs');
            }
            if (moduleMacsKind(key)) issues.modules.add(moduleMacsKind(key));
            const counted = !fused && producesActivation(key) && !weightOnly[id];
            let activation = 0;
            if (counted) {
                if (problem) reasons.activation = problem.reason;
                else if (!activationKnown(key) || unknownModule(key)) reasons.activation = unresolvedReason(node, 'activation');
                else {
                    const elements = numelAtBatchOne(output);
                    if (elements === null) reasons.activation = 'the output shape is unknown';
                    else activation = elements * ACTIVATION_BYTES;
                }
                if (reasons.activation) activation = null;
            }
            nodes[id] = {
                id, node, kind: 'layer', output, inputShapes, scope, fused, weightOnly: weightOnly[id],
                // Inside a fused standard module, an op whose output the module's own output stands for.
                fusedActivation: fused && producesActivation(key) && !weightOnly[id],
                params, macs, activation, reasons,
                // The shape shown for this node, or why it is a guess.
                shapeUnknown: taint[id]?.reason ?? null,
                // Settings whose resolved value rests on a guessed shape, and why.
                taintedSettings: settings?.names ?? [], settingsReason: settings?.taint.reason ?? null,
                errors, warnings,
            };
            continue;
        }
        if (node.kind !== 'supernode' && node.kind !== 'repeat') continue;
        const family = definitionFamily(templates[node.templateId]);
        // The outermost standard module owns the activation of its output.
        const ownsActivation = !fused && FUSED_FAMILIES.has(family);
        const countResult = node.kind === 'repeat' ? resolveRepeatCount(node.count, scope, inputShapes) : { count: 1, error: null };
        const sequential = node.kind === 'repeat' && node.mode !== 'parallel';
        const instances = [];
        let carryTaint = null;
        let carryWeightOnly = false;
        for (const child of report.children?.[id] ?? []) {
            if (!child) continue;
            const template = templates[child.templateId];
            if (!child.report || !template?.graph) {
                instances.push({ index: child.index, templateId: child.templateId, scope: null, inputs: null, output: null, level: null,
                    outputTaint: taintOf(`instance ${child.index + 1} has no definition`, origin),
                    parameterIssues: [], instanceIssues: child.instanceIssues ?? [] });
                carryTaint = instances.at(-1).outputTaint;
                continue;
            }
            // A sequential repeat feeds each instance the previous output.
            const carried = sequential && child.index > 0;
            const ports = getTemplateInputIds(template.graph);
            const portTaint = carried ? Object.fromEntries(ports.map((port) => [port, carryTaint]))
                : mapTemplateInputs(template.graph, incoming, (edge) => downstream(taint[edge.source] ?? null));
            const portFlags = carried ? Object.fromEntries(ports.map((port) => [port, carryWeightOnly]))
                : mapTemplateInputs(template.graph, incoming, (edge) => weightOnly[edge.source] === true);
            const childScopeTaint = instanceScopeTaint(template, node, child.index, scopeTaint, portTaint);
            const level = levelMetrics(template.graph, child.report, templates, {
                inputWeightOnly: portFlags, inputTaint: portTaint, scopeTaint: childScopeTaint, fused: fused || ownsActivation, templateId: child.templateId,
                path: [...path, id], issues, where,
            });
            shapeError ??= level.shapeError;
            const port = carriedPort(template, child.report);
            const outputTaint = port ? level.taint[port] ?? null : taintOf('its output shape is unknown', origin);
            carryTaint = outputTaint;
            carryWeightOnly = !!port && level.nodes[port]?.weightOnly === true;
            for (const message of child.instanceIssues ?? []) recordError(issues, path, id, where, [message]);
            instances.push({
                index: child.index, templateId: child.templateId, scope: child.scope, scopeTaint: childScopeTaint, inputs: child.inputs,
                output: port ? child.report.shapes[port] ?? null : null, outputTaint, level,
                parameterIssues: child.parameterIssues ?? [], instanceIssues: child.instanceIssues ?? [],
            });
        }
        weightOnly[id] = instances.length > 0 && instances.every((instance) => instance.level
            && templateOutputIds(templates[instance.templateId]).every((port) => instance.level.nodes[port]?.weightOnly));
        // Budget or depth limits can stop validation before every instance.
        const complete = Number.isInteger(countResult.count) && instances.length === countResult.count && instances.every((instance) => instance.level);
        const incompleteReason = countResult.error ? 'its repeat count is unresolved'
            : !templates[node.templateId] ? 'its definition is missing'
                : errors.some((message) => message.startsWith('Template recursion exceeds depth')) ? `components nest deeper than ${ANALYSIS_LIMITS.depth} levels`
                    : instances.length < (countResult.count ?? 0) ? `the analysis stopped at ${ANALYSIS_LIMITS.nodes.toLocaleString('en-US')} expanded nodes`
                        : 'not every instance could be analysed';
        // Unknown inputs come first: what fails inside follows from them.
        taint[id] = worse(inputProblem, !complete ? taintOf(incompleteReason, origin)
            : sequential ? instances.at(-1)?.outputTaint ?? null
                : instances.reduce((current, instance) => worse(current, instance.outputTaint), null));
        const instanceActivations = ownsActivation ? instances.map((instance) => {
            if (weightOnly[id]) return { bytes: 0, reason: null };
            if (instance.outputTaint) return { bytes: null, reason: instance.outputTaint.reason };
            // A module with a shape error inside has no trustworthy output.
            if (instance.level?.shapeError) return { bytes: null, reason: instance.level.shapeError.message };
            const elements = numelAtBatchOne(instance.output);
            return elements === null ? { bytes: null, reason: 'the output shape is unknown' } : { bytes: elements * ACTIVATION_BYTES, reason: null };
        }) : null;
        nodes[id] = {
            id, node, kind: node.kind, output, inputShapes, scope, family, fused: fused || ownsActivation, ownsActivation,
            count: countResult.count, countError: countResult.error,
            complete, incompleteReason: complete ? null : incompleteReason,
            instances, instanceActivations, weightOnly: weightOnly[id],
            shapeUnknown: taint[id]?.reason ?? null,
            errors, warnings,
        };
    }
    return { graph, templateId, scope, scopeTaint, report, nodes, fused, taint, shapeError };
}

/**
 * One entry per table row with errors: instances of a repeat share node ids,
 * so a mistake in a shared definition counts once, wherever it repeats.
 */
function recordError(issues, path, id, where, messages) {
    const key = [...path, id].join(SEPARATOR);
    const entry = issues.errors.get(key);
    if (entry) {
        for (const message of messages) if (!entry.messages.includes(message)) entry.messages.push(message);
    } else issues.errors.set(key, { path: [...path, id], label: where, messages: [...messages] });
}

const SEPARATOR = '\u001f';
const prefixed = (prefix, key) => prefix ? `${prefix}${SEPARATOR}${key}` : key;

function emptyTotals() {
    return { params: 0, trainable: 0, macs: 0, activation: 0, unresolved: { params: new Map(), macs: new Map(), activation: new Map() }, largest: null };
}

/** `step` names the container instance the source sits in, for locating the largest activation. */
function addInto(target, source, prefix = '', step = null) {
    target.params += source.params;
    target.trainable += source.trainable;
    target.macs += source.macs;
    target.activation += source.activation;
    for (const metric of METRICS) {
        for (const [key, value] of source.unresolved[metric]) target.unresolved[metric].set(prefixed(prefix, key), value);
    }
    if (source.largest && (!target.largest || source.largest.bytes > target.largest.bytes)) {
        target.largest = {
            ...source.largest,
            path: prefix ? [prefix, ...source.largest.path] : source.largest.path,
            where: step ? [step, ...source.largest.where] : source.largest.where,
        };
    }
}

function considerLargest(totals, bytes, path, label, shape) {
    if (bytes > 0 && (!totals.largest || bytes > totals.largest.bytes)) totals.largest = { bytes, path, label, shape, where: [] };
}

const levelCache = new WeakMap();
const entryCache = new WeakMap();
const instanceCache = new WeakMap();

/**
 * Sum over a level and everything inside it. Unknown values are counted, never
 * added: `unresolved[metric]` maps relative node paths to { label, reason }, so
 * totals read "≥ X · N rows unresolved" and are never NaN.
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
    const label = entry.node.label ?? entry.id;
    if (instance?.level) {
        addInto(totals, levelSubtree(instance.level), entry.id, { label, index: instance.index, repeat: entry.kind === 'repeat', count: entry.count });
    }
    if (entry.ownsActivation) {
        const { bytes, reason } = entry.instanceActivations[index];
        if (bytes === null) totals.unresolved.activation.set(entry.id, { label, reason });
        else {
            totals.activation += bytes;
            considerLargest(totals, bytes, [entry.id], label, instance?.output ?? null);
            if (totals.largest?.path.length === 1 && entry.kind === 'repeat' && totals.largest.path[0] === entry.id) {
                totals.largest.where = [{ label, index: instance.index, repeat: true, count: entry.count }];
            }
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
        } else totals.unresolved.params.set(entry.id, { label, reason: entry.reasons.params });
        if (entry.macs === null) totals.unresolved.macs.set(entry.id, { label, reason: entry.reasons.macs });
        else totals.macs = entry.macs;
        if (entry.activation === null) totals.unresolved.activation.set(entry.id, { label, reason: entry.reasons.activation });
        else {
            totals.activation = entry.activation;
            considerLargest(totals, entry.activation, [entry.id], label, entry.output);
        }
    } else {
        if (!entry.complete) for (const metric of METRICS) totals.unresolved[metric].set(entry.id, { label, reason: entry.incompleteReason });
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
 * `issues` lists the model's errors (one per definition node) and whether the
 * validator's node budget or nesting limit stopped the analysis.
 */
export function computeModelMetrics(rootGraph, templates = {}) {
    const budget = { remaining: ANALYSIS_LIMITS.nodes };
    const report = validate(rootGraph, { templates, collect: true, budget });
    const issues = { errors: new Map(), global: new Set(), limits: { budget: false, depth: false }, modules: new Set() };
    const root = levelMetrics(rootGraph, report, templates, { issues });
    issues.limits.budget = budget.remaining < 0;
    const templateUses = {};
    countUses(root, templateUses);
    const inputs = Object.values(root.nodes)
        .filter((entry) => entry.kind === 'layer' && entry.node.libraryKey === 'deepflow.io.Input')
        .map((entry) => ({ id: entry.id, label: entry.node.label, shape: entry.output }));
    return {
        root, totals: levelSubtree(root), templateUses, inputs,
        issues: { errors: [...issues.errors.values()], global: [...issues.global], limits: issues.limits },
        // Module kinds whose MACs come from closed forms, for the assumptions line.
        macModules: [...issues.modules].sort(),
    };
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
