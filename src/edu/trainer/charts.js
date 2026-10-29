/**
 * Pure helpers for the trainer charts and run labels: number formats, axis domains and ticks, SVG paths, and the
 * "what differed" label of an earlier run (docs/edu/SPEC.md §3.5). No DOM; Node tests import this module.
 */
import { tr } from './text.js';

/** Loss values: 3 significant digits below 10 (0.693, 1.10, 0.0123), whole numbers above. */
export function formatLoss(value) {
    if (!Number.isFinite(value)) return '–';
    const magnitude = Math.abs(value);
    if (magnitude >= 100) return Math.round(value).toLocaleString('en-US');
    if (magnitude >= 10) return value.toFixed(1);
    if (magnitude >= 1) return value.toFixed(2);
    if (magnitude >= 0.01 || magnitude === 0) return value.toFixed(3);
    return value.toPrecision(2);
}

/** Accuracy in [0, 1] as a whole percentage ("97%"); keeps one decimal near 100 % so 99.5 % is not "100%". */
export function formatAccuracy(value) {
    if (!Number.isFinite(value)) return '–';
    const percent = value * 100;
    if (percent > 99 && percent < 100) return `${percent.toFixed(1)}%`;
    return `${Math.round(percent)}%`;
}

/** Learning rates as people write them: 0.0001, 0.03, 1, 3e-5. */
export function formatLr(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value ?? '');
    if (number === 0) return '0';
    if (Math.abs(number) < 1e-4) return Number(number.toPrecision(6)).toExponential();
    return String(Number(number.toPrecision(6)));
}

/** The smallest "nice" number (1, 2, 2.5, 5 × 10^k) that is ≥ value. */
export function niceCeil(value) {
    if (!Number.isFinite(value) || value <= 0) return 1;
    const exponent = Math.floor(Math.log10(value));
    const base = 10 ** exponent;
    for (const step of [1, 2, 2.5, 5, 10]) {
        if (step * base >= value * (1 - 1e-12)) return Number((step * base).toPrecision(12));
    }
    return 10 * base;
}

/**
 * Loss axis maximum over all series. Diverging runs (lr too large) would flatten everything else, so values far
 * above the start are cut: the axis stops at 3 × the reference start (epoch-0 loss or chance level).
 */
export function lossAxisMax(series, { chance = null, reference = null } = {}) {
    let max = Number.isFinite(chance) ? chance : 0;
    for (const values of series) for (const value of values) if (Number.isFinite(value) && value > max) max = value;
    const start = Math.max(Number.isFinite(reference) ? reference : 0, Number.isFinite(chance) ? chance : 0);
    const cap = start > 0 ? 3 * start : Infinity;
    return niceCeil(Math.min(max, cap) || 1);
}

/** Axis tick text without trailing zeros: 0, 0.5, 1, 1.25, 2.5. */
export function formatTick(value) {
    if (!Number.isFinite(value)) return '';
    return String(Number(value.toFixed(Math.abs(value) >= 10 ? 0 : 3)));
}

/** Round tick positions inside [min, max] with a 1/2/2.5/5 × 10^k step, about `count` of them. */
export function niceTicks(min, max, count = 4) {
    if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return [min];
    const step = niceCeil((max - min) / count);
    const ticks = [];
    for (let value = Math.ceil(min / step - 1e-9) * step; value <= max + 1e-9; value += step) ticks.push(Number(value.toPrecision(12)));
    return ticks;
}

/** Integer epoch ticks: 0, the middle and the end. */
export function epochTicks(maxEpoch) {
    if (maxEpoch <= 1) return [0, 1];
    const middle = Math.round(maxEpoch / 2);
    return [...new Set([0, middle, maxEpoch])];
}

/**
 * An SVG path through [x, y] points in data units, scaled by `sx` and `sy`. Points with a non-finite y break the
 * line; y values are clamped into [y0, y1] so diverging runs stay inside the chart.
 */
export function linePath(points, sx, sy, [y0, y1] = [-Infinity, Infinity]) {
    let path = '';
    let pen = false;
    for (const [x, y] of points) {
        if (!Number.isFinite(y)) { pen = false; continue; }
        const clamped = Math.min(y1, Math.max(y0, y));
        path += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(clamped).toFixed(1)}`;
        pen = true;
    }
    return path;
}

export const linearScale = (domain0, domain1, range0, range1) => {
    const span = domain1 - domain0 || 1;
    return (value) => range0 + (value - domain0) / span * (range1 - range0);
};

/** Pick about `count` evenly spaced history rows for the text alternative, always with the first and last. */
export function sampleRows(history, count = 8) {
    if (history.length <= count) return history;
    const step = (history.length - 1) / (count - 1);
    const picked = new Set();
    for (let i = 0; i < count; i++) picked.add(Math.round(i * step));
    return [...picked].sort((a, b) => a - b).map((index) => history[index]);
}

const lrOf = (run) => Number(run?.optimizer?.hyperparameters?.learning_rate);
const sameJson = (a, b) => JSON.stringify(a ?? null) === JSON.stringify(b ?? null);

/**
 * What differs between `run` and `reference` (the newest run), for the dimmed curves: "lr 0.0001", "seed 2",
 * "33 params", at most two parts. Identical settings fall back to "run 4".
 */
export function runDifference(run, reference) {
    if (!run) return '';
    if (!reference || run === reference) return tr('label.run', { seq: run.seq });
    const parts = [];
    if (run.optimizer?.type && run.optimizer.type !== reference.optimizer?.type) parts.push(tr('label.optimizer', { value: run.optimizer.type }));
    if (Number.isFinite(lrOf(run)) && Math.abs(lrOf(run) - lrOf(reference)) > Math.abs(lrOf(reference)) * 1e-9) {
        parts.push(tr('label.lr', { value: formatLr(lrOf(run)) }));
    }
    if (run.loss?.type && run.loss.type !== reference.loss?.type) parts.push(tr('label.loss', { value: run.loss.type }));
    if (Number.isFinite(run.paramCount) && run.paramCount !== reference.paramCount) {
        parts.push(tr('label.params', { value: run.paramCount.toLocaleString('en-US') }));
    } else if (run.graph && reference.graph && run.graph !== reference.graph && !sameJson(stripPositions(run.graph), stripPositions(reference.graph))) {
        parts.push(tr('label.model'));
    }
    if (run.settings?.seed !== reference.settings?.seed) parts.push(tr('label.seed', { value: run.settings?.seed }));
    if (run.settings?.batchSize !== reference.settings?.batchSize) parts.push(tr('label.batch', { value: run.settings?.batchSize }));
    if (run.settings?.epochs !== reference.settings?.epochs) parts.push(tr('label.epochs', { value: run.settings?.epochs }));
    if (!sameJson(run.dataOptions, reference.dataOptions) || !sameJson(run.split, reference.split)) parts.push(tr('label.data'));
    return parts.length ? parts.slice(0, 2).join(' · ') : tr('label.run', { seq: run.seq });
}

/** Node positions do not change the model; ignore them when comparing two captured graphs. */
function stripPositions(graph) {
    return {
        edges: graph.edges,
        nodes: Object.fromEntries(Object.entries(graph.nodes ?? {}).map(([id, node]) => {
            const { position: _position, ...rest } = node;
            return [id, rest];
        })),
        parameters: graph.parameters,
    };
}

/** Earlier finished training runs on the same dataset, newest first, at most `limit` (the current run excluded). */
export function previousRuns(runs, current, limit = 5) {
    if (!current) return [];
    return runs
        .filter((run) => run.id !== current.id && run.kind !== 'init' && run.datasetId === current.datasetId && run.history.length > 1)
        .slice(-limit)
        .reverse();
}

/** Spread label y positions so they are at least `gap` apart, keeping their order (used for end-of-line labels). */
export function spreadLabels(items, gap, [min, max]) {
    const sorted = [...items].sort((a, b) => a.y - b.y);
    for (let i = 1; i < sorted.length; i++) if (sorted[i].y - sorted[i - 1].y < gap) sorted[i].y = sorted[i - 1].y + gap;
    const overflow = sorted.length ? sorted.at(-1).y - max : 0;
    if (overflow > 0) for (const item of sorted) item.y -= overflow;
    for (let i = 0; i < sorted.length; i++) sorted[i].y = Math.max(min + i * gap, sorted[i].y);
    return sorted;
}
