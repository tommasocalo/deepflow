import { finalMetrics, isInitRun, trainerRuns } from './checks.js';

/**
 * Live reveal values (docs/edu/SPEC.md §6.3.1 RevealSpec): the learner's own measurements interpolated into the
 * reveal text, such as the untrained loss ({measured}) or a before/after comparison with a tagged run.
 */
const METRICS = {
    gap: (metrics) => (Number.isFinite(metrics?.valLoss) && Number.isFinite(metrics?.trainLoss) ? metrics.valLoss - metrics.trainLoss : null),
    valLoss: (metrics) => (Number.isFinite(metrics?.valLoss) ? metrics.valLoss : null),
    valAccuracy: (metrics) => (Number.isFinite(metrics?.valAccuracy) ? metrics.valAccuracy : null),
};
const LOWER_IS_BETTER = new Set(['gap', 'valLoss']);

const formatMetric = (metric, value) => (metric === 'valAccuracy' ? `${(value * 100).toFixed(1)}%` : value.toFixed(3));

/** @returns {{ params: object, variant: 'better'|'notBetter'|null }} */
export function revealValues(reveal, { trainer, enteredAtSeq = 0, runTags = {} } = {}) {
    const params = {};
    let variant = null;
    if (!reveal) return { params, variant };
    const runs = trainerRuns(trainer).filter((run) => (run.seq ?? 0) > enteredAtSeq && run.status !== 'failed');
    if (reveal.run === 'initLoss') {
        const init = [...runs].reverse().find((run) => isInitRun(run)) ?? runs.at(-1);
        const first = init?.history?.[0];
        const value = first?.valLoss ?? first?.trainLoss;
        if (Number.isFinite(value)) params.measured = value.toFixed(2);
    }
    if (reveal.run === 'finalValAccuracy') {
        const value = finalMetrics([...runs].reverse().find((run) => !isInitRun(run)))?.valAccuracy;
        if (Number.isFinite(value)) params.measured = `${(value * 100).toFixed(1)}%`;
    }
    if (reveal.compare) {
        const { tag, metric } = reveal.compare;
        const measure = METRICS[metric];
        const before = measure?.(runTags[tag]);
        const latest = [...runs].reverse().find((run) => !isInitRun(run) && run.status === 'completed');
        const after = measure?.(finalMetrics(latest));
        if (before !== null && before !== undefined && after !== null && after !== undefined) {
            params.before = formatMetric(metric, before);
            params.after = formatMetric(metric, after);
            variant = (LOWER_IS_BETTER.has(metric) ? after < before : after > before) ? 'better' : 'notBetter';
        }
    }
    return { params, variant };
}
