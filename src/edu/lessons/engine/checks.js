import { cachedValidate } from '../../../domain/architecture/levelContext.js';
import { resolveGraphLevel } from '../../../domain/architecture/operations.js';
import { resolveRepeatCount, withGlobalParameters } from '../../../domain/expressions.js';
import { countArchitectureParameters } from '../../../domain/optimization/parameterCount.js';
import { effectiveObjective } from '../../objective.js';
import { formatAccuracy } from '../../trainer/charts.js';
import { friendlyIssues } from './feedback.js';
import { formatNumber, hasText, paramLabel, t } from './i18n.js';
import {
    describeMatch, formatShape, matchNodes, nodeMatches, nodeType, orderedNodeIds, paramValue, sameShape, sameValue, typeMatches,
} from './nodes.js';

/**
 * The check API (docs/edu/SPEC.md §6.4). A CheckSpec is an object with exactly one predicate key. Checks are pure
 * functions of a CheckContext; `createCheckContext` derives the expensive parts (validation report, parameter
 * count, effective objective) lazily, so a step only pays for what it asks about.
 *
 * Result: { pass, detail?: { key, params }, misconception?, runId?, predicate }
 */
export const PREDICATES = Object.freeze([
    'all', 'any', 'not', 'visited', 'tab', 'answered', 'answeredCorrectly', 'dataset', 'split', 'inputShape', 'outputShape',
    'valid', 'has', 'param', 'global', 'chain', 'edge', 'shapeAt', 'paramCount', 'training', 'run', 'runs', 'level', 'ui',
    'repeat', 'component', 'custom',
]);
/** Predicates about the graph itself: when they fail on an invalid graph, the panel shows the friendly error. */
export const GRAPH_PREDICATES = Object.freeze(new Set([
    'inputShape', 'outputShape', 'valid', 'has', 'param', 'global', 'chain', 'edge', 'shapeAt', 'paramCount', 'run', 'runs',
    'repeat', 'component', 'custom', 'training',
]));
const RUN_METRICS = ['valAccuracy', 'trainAccuracy', 'valLoss', 'trainLoss'];
const UI_FLAGS = ['showDimensions', 'showGradientFlow', 'codeOpen', 'codeTab'];

function lazy(target, name, compute) {
    let cached;
    let done = false;
    Object.defineProperty(target, name, {
        enumerable: true,
        configurable: true,
        get() {
            if (!done) { cached = compute(); done = true; }
            return cached;
        },
    });
}

/** Build a CheckContext from plain parts; the store-reading wrapper lives in context.js. */
export function createCheckContext({
    graph, templates = {}, training = null, ui = {}, data = null, trainer = null, step = {}, facts = {}, module = null,
    inspectedIterations = {},
} = {}) {
    const ctx = {
        graph: graph ?? { nodes: {}, edges: [] },
        templates: templates ?? {},
        training: training ?? { nodes: {}, edges: [] },
        ui: { tab: 'architecture', showDimensions: true, showGradientFlow: false, codeOpen: false, codeTab: 'model', currentPath: [], expandedNodeId: null, ...ui },
        data,
        trainer,
        step: { answers: {}, enteredAtSeq: 0, visitedTabs: [], tags: {}, ...step },
        facts: facts ?? {},
        module,
        inspectedIterations: inspectedIterations ?? {},
    };
    lazy(ctx, 'scope', () => withGlobalParameters(ctx.graph));
    lazy(ctx, 'report', () => cachedValidate(ctx.graph, ctx.templates, ctx.scope, {}));
    lazy(ctx, 'order', () => orderedNodeIds(ctx.graph));
    lazy(ctx, 'paramCount', () => countArchitectureParameters(ctx.graph, ctx.templates));
    lazy(ctx, 'objective', () => effectiveObjective(ctx.training, ctx.graph));
    lazy(ctx, 'issues', () => friendlyIssues(ctx.report, ctx.graph, ctx.templates));
    return ctx;
}

const pass = (predicate, extra = {}) => ({ pass: true, predicate, ...extra });
const fail = (predicate, key, params = {}, extra = {}) => ({ pass: false, predicate, detail: { key, params }, ...extra });

function matchOptions(ctx) {
    return { templates: ctx.templates, scope: ctx.scope, graph: ctx.graph, report: ctx.report, order: ctx.order };
}

function typeName(type) {
    if (Array.isArray(type)) return type.map(typeName).join(' / ');
    return hasText(`type.${type}`) ? t(`type.${type}`) : String(type ?? 'block');
}

function matchName(match = {}) {
    if (match.label || match.id || match.name) return describeMatch(match);
    return typeName(match.type ?? 'block');
}

/** The article before a type name in check details ("an activation", "a Flatten"); none before a block's label. */
function articleOf(match = {}) {
    if (match.label || match.id || match.name) return '';
    return /^[aeiou]/i.test(matchName(match)) ? 'an ' : 'a ';
}

function valueText(value) {
    if (value === null || value === undefined || value === '') return t('value.none');
    if (typeof value === 'boolean') return t(value ? 'value.true' : 'value.false');
    if (typeof value === 'number') return formatNumber(value);
    if (Array.isArray(value)) return value.join(', ');
    return String(value);
}

/** { equals | oneOf | min | max } → boolean. */
export function satisfies(value, condition = {}) {
    if (Object.hasOwn(condition, 'equals') && !sameValue(value, condition.equals)) return false;
    if (Array.isArray(condition.oneOf) && !condition.oneOf.some((option) => sameValue(value, option))) return false;
    const numeric = Number(value);
    if (condition.min !== undefined && !(value !== null && value !== '' && Number.isFinite(numeric) && numeric >= condition.min)) return false;
    if (condition.max !== undefined && !(value !== null && value !== '' && Number.isFinite(numeric) && numeric <= condition.max)) return false;
    if (condition.gte !== undefined && !(Number.isFinite(numeric) && numeric >= condition.gte)) return false;
    if (condition.lte !== undefined && !(Number.isFinite(numeric) && numeric <= condition.lte)) return false;
    return true;
}

export function describeCondition(condition = {}) {
    if (Object.hasOwn(condition, 'equals')) return valueText(condition.equals);
    if (Array.isArray(condition.oneOf)) return t('value.oneOf', { values: condition.oneOf.map(valueText).join(t('value.or')) });
    const min = condition.min ?? condition.gte;
    const max = condition.max ?? condition.lte;
    if (min !== undefined && max !== undefined) return t('value.between', { min: valueText(min), max: valueText(max) });
    if (min !== undefined) return t('value.atLeast', { min: valueText(min) });
    if (max !== undefined) return t('value.atMost', { max: valueText(max) });
    return '';
}

const condition = (spec) => Object.fromEntries(['equals', 'oneOf', 'min', 'max'].filter((key) => Object.hasOwn(spec, key)).map((key) => [key, spec[key]]));

function firstOfType(ctx, type) {
    return matchNodes(ctx.graph, { type }, matchOptions(ctx))[0] ?? null;
}

// ---- runs -------------------------------------------------------------------------------------------------------

/** Completed and stopped runs (and a finished `current`) of the trainer source, oldest first. */
export function trainerRuns(trainer) {
    const runs = [...(trainer?.runs ?? [])];
    const current = trainer?.current;
    if (current && ['completed', 'stopped', 'failed'].includes(current.status) && !runs.some((run) => run.id === current.id)) runs.push(current);
    return runs.sort((left, right) => (left.seq ?? 0) - (right.seq ?? 0));
}

export const completedEpochs = (run) => Math.max(0, (run?.history?.length ?? 0) - 1);
export const finalMetrics = (run) => run?.history?.at(-1) ?? null;

/** An "evaluate initial loss" record: 0 epochs. Stage B may mark it with kind: 'init'. */
export function isInitRun(run) {
    if (!run) return false;
    if (run.kind === 'init' || run.initOnly === true) return true;
    return (run.history?.length ?? 0) === 1 && run.status === 'completed' && (run.settings?.epochs === 0 || run.kind === undefined);
}

function runLr(run) {
    const value = run?.optimizer?.hyperparameters?.learning_rate;
    return value === undefined || value === null ? null : Number(value);
}

/** Evaluate one run against a RunQuery; returns a failure detail or null. */
function runFailure(run, query, ctx) {
    if (run.status === 'failed') return { key: 'check.run.failed', params: { error: run.error?.message ?? run.error ?? '' } };
    if (query.dataset && run.datasetId !== query.dataset) {
        return { key: 'check.run.dataset', params: { actual: datasetName(run.datasetId), expected: datasetName(query.dataset) } };
    }
    if (query.initOnly) return isInitRun(run) ? null : { key: 'check.run.init', params: {} };
    if (query.initOnly === undefined && isInitRun(run) && (query.minEpochs ?? 1) > 0) {
        return { key: 'check.run.epochs', params: { actual: 0, expected: query.minEpochs ?? 1 } };
    }
    const epochs = completedEpochs(run);
    if (query.minEpochs !== undefined && epochs < query.minEpochs) return { key: 'check.run.epochs', params: { actual: epochs, expected: query.minEpochs } };
    if (query.lr && !satisfies(runLr(run), query.lr)) {
        return { key: 'check.run.lr', params: { actual: valueText(runLr(run)), expected: describeCondition(query.lr) } };
    }
    const final = finalMetrics(run);
    for (const metric of RUN_METRICS) {
        const bound = query.final?.[metric];
        if (!bound) continue;
        const value = final?.[metric];
        if (value === null || value === undefined || !satisfies(value, bound)) {
            return { key: 'check.run.final', params: { metric: t(`metric.${metric}`), actual: formatMetric(metric, value), expected: describeMetricBound(metric, bound) } };
        }
    }
    if (query.overfit) {
        const accuracy = final?.trainAccuracy;
        if (!(Number(accuracy) >= query.overfit.trainAccuracyGte)) {
            return { key: 'check.run.overfitAccuracy', params: { actual: formatMetric('trainAccuracy', accuracy), expected: formatMetric('trainAccuracy', query.overfit.trainAccuracyGte) } };
        }
        const losses = (run.history ?? []).map((entry) => entry.valLoss).filter((value) => Number.isFinite(value));
        const minimum = losses.length ? Math.min(...losses) : NaN;
        const rise = Number.isFinite(minimum) && minimum > 0 ? final.valLoss / minimum : NaN;
        if (!(rise >= query.overfit.valLossRiseGte)) {
            return { key: 'check.run.overfitRise', params: { actual: Number.isFinite(rise) ? rise.toFixed(2) : '?', expected: query.overfit.valLossRiseGte } };
        }
    }
    if (query.model) {
        const sub = createCheckContext({ graph: run.graph, templates: run.templates ?? {}, training: run.training, ui: ctx.ui, data: ctx.data, trainer: ctx.trainer, step: ctx.step, facts: ctx.facts, module: ctx.module });
        const result = evaluateCheck(query.model, sub);
        if (!result.pass) return { key: 'check.run.model', params: { detail: result.detail ? t(result.detail.key, result.detail.params) : '' } };
    }
    return null;
}

function formatMetric(metric, value) {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return t('value.unknown');
    return metric.endsWith('Accuracy') ? formatAccuracy(Number(value)) : Number(value).toFixed(3);
}

function describeMetricBound(metric, bound) {
    if (bound.gte !== undefined) return t('value.atLeast', { min: formatMetric(metric, bound.gte) });
    if (bound.lte !== undefined) return t('value.atMost', { max: formatMetric(metric, bound.lte) });
    return '';
}

function datasetName(id) {
    if (!id) return t('value.none');
    return hasText(`dataset.${id}`) ? t(`dataset.${id}`) : id;
}

// ---- predicates -------------------------------------------------------------------------------------------------

const handlers = {
    all(list, ctx) {
        // The run that satisfied a `run` sub-check travels up, wherever it sits in the list (step.tagRun needs it).
        let runId = null;
        for (const spec of list ?? []) {
            const result = evaluateCheck(spec, ctx);
            if (!result.pass) return result;
            runId = result.runId ?? runId;
        }
        return pass('all', runId ? { runId } : {});
    },
    any(list, ctx) {
        let first = null;
        for (const spec of list ?? []) {
            const result = evaluateCheck(spec, ctx);
            if (result.pass) return result;
            first ??= result;
        }
        return first ?? fail('any', 'check.any');
    },
    not(spec, ctx) {
        const result = evaluateCheck(spec, ctx);
        return result.pass ? fail('not', 'check.not') : pass('not');
    },
    visited(tab, ctx) {
        return ctx.step.visitedTabs?.includes(tab) ? pass('visited') : fail('visited', 'check.visited', { tab: t(`tab.${tab}`) });
    },
    tab(tab, ctx) {
        return ctx.ui.tab === tab ? pass('tab') : fail('tab', 'check.tab', { tab: t(`tab.${tab}`) });
    },
    answered(id, ctx) {
        return answerFor(ctx, id) ? pass('answered') : fail('answered', 'check.answered', { id });
    },
    answeredCorrectly(id, ctx) {
        return answerFor(ctx, id)?.correct ? pass('answeredCorrectly') : fail('answeredCorrectly', 'check.answeredCorrectly', { id });
    },
    dataset(id, ctx) {
        if (ctx.data?.datasetId === id) return pass('dataset');
        return fail('dataset', 'check.dataset', { name: datasetName(id), actual: datasetName(ctx.data?.datasetId) });
    },
    split(expected, ctx) {
        const split = ctx.data?.split ?? {};
        const ok = ['val', 'test'].every((name) => expected[name] === undefined || Math.abs(Number(split[name] ?? 0) - expected[name]) < 1e-9);
        return ok ? pass('split') : fail('split', 'check.split', { val: expected.val ?? split.val ?? 0, test: expected.test ?? split.test ?? 0 });
    },
    inputShape(shape, ctx) {
        const input = firstOfType(ctx, 'Input');
        const actual = input ? ctx.report.shapes?.[input.id] ?? null : null;
        return sameShape(actual, shape) ? pass('inputShape')
            : fail('inputShape', 'check.inputShape', { actual: formatShape(actual), expected: formatShape(shape) });
    },
    outputShape(shape, ctx) {
        const output = firstOfType(ctx, 'Output');
        const actual = output ? ctx.report.shapes?.[output.id] ?? null : null;
        return sameShape(actual, shape) ? pass('outputShape')
            : fail('outputShape', 'check.outputShape', { actual: formatShape(actual), expected: formatShape(shape) });
    },
    valid(expected, ctx) {
        const ok = !!ctx.report.ok;
        if (ok === (expected !== false)) return pass('valid');
        const issue = ctx.issues[0];
        return issue ? fail('valid', issue.key, issue.params, { nodeId: issue.nodeId }) : fail('valid', 'check.valid');
    },
    has(match, ctx, spec) {
        const nodes = countMatches(ctx, match, spec.deep);
        const wanted = typeof spec.count === 'number' ? { min: spec.count, max: spec.count } : spec.count ?? { min: 1 };
        const ok = (wanted.min === undefined || nodes >= wanted.min) && (wanted.max === undefined || nodes <= wanted.max);
        if (ok) return pass('has');
        if (!nodes) return fail('has', 'check.has.none', { type: matchName(match), a: articleOf(match) });
        return fail('has', 'check.has', { type: matchName(match), actual: nodes, expected: describeCondition(wanted) });
    },
    param(spec, ctx) {
        const nodes = matchNodes(ctx.graph, spec.node ?? {}, matchOptions(ctx));
        if (!nodes.length) return fail('param', 'check.nodeMissing', { type: matchName(spec.node), a: articleOf(spec.node) });
        const wanted = condition(spec);
        for (const node of spec.every ? nodes : nodes.slice(0, 1)) {
            const value = paramValue(node, spec.name, matchOptions(ctx));
            if (!satisfies(value, wanted)) {
                return fail('param', spec.every ? 'check.param.every' : 'check.param', {
                    label: node.label ?? nodeType(node, ctx.templates), type: matchName(spec.node), name: paramLabel(spec.name),
                    actual: valueText(value), expected: describeCondition(wanted),
                }, { nodeId: node.id });
            }
        }
        return pass('param');
    },
    global(spec, ctx) {
        const parameter = (ctx.graph.parameters ?? []).find((item) => item.name === spec.name);
        if (!parameter) return fail('global', 'check.global.missing', { name: spec.name });
        const value = Object.hasOwn(ctx.scope, spec.name) ? ctx.scope[spec.name] : parameter.default;
        const wanted = condition(spec);
        return satisfies(value, wanted) ? pass('global')
            : fail('global', 'check.global', { name: spec.name, actual: valueText(value), expected: describeCondition(wanted) });
    },
    chain(list, ctx, spec) {
        return chainCheck(list ?? [], spec.allow ?? [], ctx);
    },
    edge(spec, ctx) {
        const options = matchOptions(ctx);
        const from = new Set(matchNodes(ctx.graph, spec.from ?? {}, options).map((node) => node.id));
        const to = new Set(matchNodes(ctx.graph, spec.to ?? {}, options).map((node) => node.id));
        const ok = (ctx.graph.edges ?? []).some((edge) => from.has(edge.source) && to.has(edge.target));
        return ok ? pass('edge') : fail('edge', 'check.edge', { from: matchName(spec.from), to: matchName(spec.to) });
    },
    shapeAt(spec, ctx) {
        const node = matchNodes(ctx.graph, spec.node ?? {}, matchOptions(ctx))[0];
        if (!node) return fail('shapeAt', 'check.nodeMissing', { type: matchName(spec.node), a: articleOf(spec.node) });
        const actual = ctx.report.shapes?.[node.id] ?? null;
        return sameShape(actual, spec.equals) ? pass('shapeAt')
            : fail('shapeAt', 'check.shapeAt', { label: node.label ?? matchName(spec.node), actual: formatShape(actual), expected: formatShape(spec.equals) }, { nodeId: node.id });
    },
    paramCount(spec, ctx) {
        const count = ctx.paramCount;
        if (!count?.complete) return fail('paramCount', 'check.paramCount.incomplete', { unsupported: (count?.unsupported ?? []).join(', ') });
        const wanted = condition(spec);
        return satisfies(count.total, wanted) ? pass('paramCount')
            : fail('paramCount', 'check.paramCount', { actual: formatNumber(count.total), expected: describeCondition(wanted) });
    },
    training(spec, ctx) {
        const { optimizer, loss, lossBound } = ctx.objective;
        if (spec.loss !== undefined) {
            if (!loss) return fail('training', 'check.training.noLoss');
            if (loss.type !== spec.loss) return fail('training', 'check.training.loss', { actual: lossName(loss.type), expected: lossName(spec.loss) });
        }
        if (spec.lossBound && !lossBound) return fail('training', 'check.training.lossBound');
        if (spec.optimizer !== undefined || spec.lr !== undefined) {
            if (!optimizer) return fail('training', 'check.training.noOptimizer');
            if (spec.optimizer !== undefined && optimizer.type !== spec.optimizer) {
                return fail('training', 'check.training.optimizer', { actual: optimizer.type, expected: spec.optimizer });
            }
            const lr = Number(optimizer.hyperparameters?.learning_rate);
            if (spec.lr && !satisfies(lr, spec.lr)) return fail('training', 'check.training.lr', { actual: valueText(lr), expected: describeCondition(spec.lr) });
        }
        return pass('training');
    },
    run(query, ctx) {
        if (ctx.trainer?.status === 'running' || ctx.trainer?.status === 'stopping') {
            const candidates = runsSinceEntry(ctx).filter((run) => !runFailure(run, query, ctx));
            if (candidates.length) return pass('run', { runId: candidates.at(-1).id });
            return fail('run', 'check.run.running');
        }
        const runs = runsSinceEntry(ctx);
        const passing = runs.filter((run) => !runFailure(run, query, ctx));
        if (passing.length) return pass('run', { runId: passing.at(-1).id });
        if (!runs.length) return fail('run', query.initOnly ? 'check.run.init' : 'check.run.none');
        const detail = runFailure(runs.at(-1), query, ctx);
        return fail('run', detail.key, detail.params);
    },
    runs(spec, ctx) {
        const runs = runsSinceEntry(ctx).filter((run) => run.status === 'completed' && !isInitRun(run)
            && (!spec.dataset || run.datasetId === spec.dataset)
            && (spec.minEpochs === undefined || completedEpochs(run) >= spec.minEpochs));
        const covered = (lr) => runs.some((run) => runLr(run) !== null && Math.abs(runLr(run) - lr) <= Math.abs(lr) * 0.01);
        const missing = (spec.lr ?? []).filter((lr) => !covered(lr));
        if (!missing.length && runs.length >= (spec.count ?? 0)) return pass('runs');
        return fail('runs', 'check.runs', { missing: (missing.length ? missing : spec.lr ?? []).map(valueText).join(', ') });
    },
    level(spec, ctx) {
        const wanted = spec.open ?? {};
        const path = ctx.ui.currentPath ?? [];
        let graph = ctx.graph;
        for (let depth = 0; depth < path.length; depth++) {
            const node = graph?.nodes?.[path[depth]];
            if (!node) break;
            if (nodeMatches(node, wanted, { templates: ctx.templates, scope: ctx.scope, graph })) return pass('level');
            graph = resolveGraphLevel(ctx.graph, path.slice(0, depth + 1), ctx.templates, ctx.inspectedIterations).graph;
        }
        return fail('level', 'check.level', { type: matchName(wanted) });
    },
    ui(spec, ctx) {
        for (const flag of UI_FLAGS) {
            if (spec[flag] === undefined) continue;
            if (flag === 'codeTab') {
                if (ctx.ui.codeTab !== spec.codeTab || !ctx.ui.codeOpen) return fail('ui', 'check.ui.codeTab', { tab: spec.codeTab === 'train' ? 'train.py' : 'model.py' });
                continue;
            }
            if (!!ctx.ui[flag] !== !!spec[flag]) return fail('ui', `check.ui.${flag}.${spec[flag] ? 'on' : 'off'}`);
        }
        return pass('ui');
    },
    repeat(spec, ctx) {
        const repeats = matchNodes(ctx.graph, spec.node ?? {}, matchOptions(ctx)).filter((node) => node.kind === 'repeat');
        if (!repeats.length) return fail('repeat', 'check.repeat.missing');
        let detail = null;
        for (const node of repeats) {
            const mode = node.mode ?? 'sequential';
            const count = resolveRepeatCount(node.count, ctx.scope).count;
            if (spec.mode && mode !== spec.mode) { detail ??= { key: 'check.repeat.mode', params: { actual: mode, expected: spec.mode } }; continue; }
            if (spec.count !== undefined && count !== spec.count) { detail ??= { key: 'check.repeat.count', params: { actual: valueText(count), expected: spec.count } }; continue; }
            return pass('repeat');
        }
        return fail('repeat', detail.key, detail.params);
    },
    component(spec, ctx) {
        const containers = Object.values(ctx.graph.nodes ?? {}).filter((node) => (spec.kind ? node.kind === spec.kind : ['supernode', 'repeat'].includes(node.kind)));
        let best = null;
        for (const node of containers) {
            const graph = ctx.templates[node.templateId]?.graph;
            if (!graph) continue;
            const missing = (spec.contains ?? []).find((wanted) => {
                const count = Object.values(graph.nodes ?? {}).filter((child) => nodeMatches(child, { ...wanted, count: undefined }, { templates: ctx.templates, scope: {} })).length;
                return count < (wanted.count ?? 1);
            });
            if (!missing) return pass('component');
            best ??= missing;
        }
        if (!best) return fail('component', 'check.component');
        return fail('component', 'check.component.contains', { type: matchName(best), count: best.count ?? 1 });
    },
    custom(name, ctx) {
        const check = ctx.module?.checks?.[name];
        if (typeof check !== 'function') return fail('custom', 'check.unknown', { name });
        try {
            const result = check(ctx);
            if (typeof result === 'boolean') return result ? pass('custom') : fail('custom', 'check.custom');
            return result?.pass ? { ...result, predicate: 'custom' } : { detail: { key: 'check.custom', params: {} }, ...result, pass: false, predicate: 'custom' };
        } catch (error) {
            return fail('custom', 'check.custom', { error: error.message });
        }
    },
};

function lossName(type) {
    return hasText(`loss.${type}`) ? t(`loss.${type}`) : String(type);
}

function answerFor(ctx, id) {
    return ctx.step.answers?.[id] ?? null;
}

function runsSinceEntry(ctx) {
    const entered = ctx.step.enteredAtSeq ?? 0;
    return trainerRuns(ctx.trainer).filter((run) => (run.seq ?? 0) > entered);
}

function countMatches(ctx, match, deep) {
    const options = matchOptions(ctx);
    let count = matchNodes(ctx.graph, { ...match, nth: undefined }, options).length;
    if (!deep) return count;
    const seen = new Set();
    const visit = (graph) => {
        for (const node of Object.values(graph.nodes ?? {})) {
            if (!['supernode', 'repeat'].includes(node.kind) || seen.has(node.templateId)) continue;
            seen.add(node.templateId);
            const child = ctx.templates[node.templateId]?.graph;
            if (!child) continue;
            count += Object.values(child.nodes ?? {}).filter((item) => nodeMatches(item, { ...match, nth: undefined, params: undefined }, { templates: ctx.templates })).length;
            visit(child);
        }
    };
    visit(ctx.graph);
    return count;
}

/** A directed root path visiting nodes that match each element in order; `allow` types may sit in between. */
function chainCheck(list, allow, ctx) {
    if (!list.length) return pass('chain');
    const options = matchOptions(ctx);
    const out = new Map();
    for (const edge of ctx.graph.edges ?? []) {
        if (!out.has(edge.source)) out.set(edge.source, []);
        out.get(edge.source).push(edge.target);
    }
    const nodes = ctx.graph.nodes ?? {};
    const matchesAt = (id, index) => nodeMatches(nodes[id], list[index], options);
    const allowed = (id) => allow.length > 0 && typeMatches(nodeType(nodes[id], ctx.templates), allow);
    let best = { depth: -1, nodeId: null };
    const search = (id, index, visited) => {
        if (index > best.depth) best = { depth: index, nodeId: id };
        if (index === list.length - 1) return true;
        for (const next of out.get(id) ?? []) {
            if (visited.has(next)) continue;
            const seen = new Set(visited).add(next);
            if (matchesAt(next, index + 1) && search(next, index + 1, seen)) return true;
            if (allowed(next) && search(next, index, seen)) return true;
        }
        return false;
    };
    const starts = ctx.order.filter((id) => matchesAt(id, 0));
    if (Number.isInteger(list[0].nth)) starts.splice(0, starts.length, ...matchNodes(ctx.graph, list[0], options).map((node) => node.id));
    for (const start of starts) if (search(start, 0, new Set([start]))) return pass('chain');
    if (best.depth < 0) return fail('chain', 'check.chain.start', { type: matchName(list[0]), a: articleOf(list[0]) });
    const after = nodes[best.nodeId];
    return fail('chain', 'check.chain.next', {
        after: after?.label ?? matchName(list[best.depth]),
        next: matchName(list[best.depth + 1]),
        a: articleOf(list[best.depth + 1]),
        expected: list.map(matchName).join(' → '),
    }, { nodeId: best.nodeId });
}

/** Evaluate any CheckSpec. Unknown or malformed specs fail with check.unknown rather than throwing. */
export function evaluateCheck(spec, ctx) {
    if (!spec || typeof spec !== 'object') return fail('unknown', 'check.unknown');
    const keys = Object.keys(spec).filter((key) => PREDICATES.includes(key));
    const name = keys[0];
    const handler = handlers[name];
    if (!handler) return fail('unknown', 'check.unknown', { name: Object.keys(spec)[0] });
    try {
        const result = handler(spec[name], ctx, spec);
        return { ...result, predicate: result.predicate ?? name };
    } catch (error) {
        if (import.meta.env?.DEV) console.warn('[edu checks] check threw', name, error);
        return fail(name, 'check.unknown', { error: error.message });
    }
}

/** The node feeding the root Output directly, if any. */
function nodeBeforeOutput(ctx) {
    const output = firstOfType(ctx, 'Output');
    if (!output) return null;
    const edge = (ctx.graph.edges ?? []).find((item) => item.target === output.id);
    return edge ? ctx.graph.nodes[edge.source] ?? null : null;
}

/** Global diagnostics for failing task steps (§6.4.3). */
export function globalDiagnostics(ctx) {
    const before = nodeBeforeOutput(ctx);
    const type = before ? nodeType(before, ctx.templates) : null;
    const loss = ctx.objective?.loss?.type;
    if (type === 'Softmax' && loss === 'cross_entropy') return 'softmax-before-ce';
    if (type === 'Sigmoid' && loss === 'bce') return 'sigmoid-before-bce';
    return null;
}

/**
 * Evaluate a whole step: its check, its diagnose rules while it fails, the global diagnostics for task steps and
 * the friendly rewrite of the most relevant validation error when a graph predicate fails on an invalid graph.
 */
export function evaluateStep(step, ctx) {
    if (!step) return { pass: false, detail: { key: 'panel.empty', params: {} } };
    const result = evaluateCheck(step.check, ctx);
    if (result.pass) return result;
    let misconception = result.misconception ?? null;
    for (const rule of step.diagnose ?? []) {
        if (misconception) break;
        if (evaluateCheck(rule.when, ctx).pass) misconception = rule.misconception;
    }
    if (!misconception && step.kind === 'task') misconception = globalDiagnostics(ctx);
    let issue = null;
    if (GRAPH_PREDICATES.has(result.predicate) && ctx.report && !ctx.report.ok) issue = ctx.issues[0] ?? null;
    return { ...result, misconception, issue };
}
