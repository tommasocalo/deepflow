/**
 * prepareDataset (docs/edu/SPEC.md §6.9.1): load, split (§5.3), scale with statistics fitted on the training split
 * only, and cache by (id, options, split, seed). Pure apart from the cache; runs in Node.
 */
import { dataSeedOf, resolveOptions } from './options.js';
import { fitStandardize, gatherRows, splitIndices } from './split.js';

const CACHE_LIMIT = 12;
const values = new Map();
const promises = new Map();
const listeners = new Set();

const normalSplit = (split = {}) => ({ val: Number(split.val ?? 0.2), test: Number(split.test ?? 0) });

export function preparedKey(spec, options, split, dataSeed) {
    const resolved = resolveOptions(spec, options);
    return JSON.stringify([spec?.id ?? null, resolved, normalSplit(split), dataSeed ?? dataSeedOf(resolved)]);
}

function remember(key, value) {
    values.delete(key);
    values.set(key, value);
    while (values.size > CACHE_LIMIT) values.delete(values.keys().next().value);
    // Notify after the current task so a synchronous prepare during a React render never updates other components.
    if (listeners.size) queueMicrotask(() => { for (const listener of listeners) listener(key); });
}

/** Square 2D bounds around all points, with a margin of 10 % of the larger span on each side. */
export function squareBounds(x, n) {
    if (!n) return [[-1, 1], [-1, 1]];
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
        const a = x[i * 2], b = x[i * 2 + 1];
        if (a < minX) minX = a;
        if (a > maxX) maxX = a;
        if (b < minY) minY = b;
        if (b > maxY) maxY = b;
    }
    const span = Math.max(maxX - minX, maxY - minY, 1e-6);
    const half = span / 2 + span * 0.1;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    return [[cx - half, cx + half], [cy - half, cy + half]];
}

/** Split and scale an already loaded RawDataset. Synchronous and deterministic. */
export function prepareRaw(spec, raw, options = {}, split = {}, dataSeed) {
    const resolved = resolveOptions(spec, options);
    const seed = dataSeed ?? dataSeedOf(resolved);
    const fractions = normalSplit(split);
    const featureShape = raw.featureShape;
    const featureSize = featureShape.reduce((product, dim) => product * dim, 1);
    const indices = splitIndices(raw.n, fractions, seed);
    let scaling = null;
    let transform = null;
    if (spec.scaling === 'standardize') {
        scaling = fitStandardize(raw.x, featureSize, indices.train);
        transform = (value, feature) => (value - scaling.mean[feature]) / scaling.std[feature];
    } else if (spec.scaling === 'divide16') {
        transform = (value) => value / 16;
    }
    const part = (rows) => gatherRows(raw.x, raw.y, featureSize, rows, transform);
    const classes = spec.target?.kind === 'classes';
    return {
        spec,
        key: preparedKey(spec, resolved, fractions, seed),
        options: resolved,
        split: fractions,
        dataSeed: seed,
        train: part(indices.train),
        val: part(indices.val),
        test: part(indices.test),
        featureShape: [...featureShape],
        featureSize,
        featureNames: raw.featureNames ?? spec.featureNames ?? null,
        numClasses: classes ? spec.target.size : null,
        classNames: classes ? [...spec.target.classes] : null,
        scaling,
        scalingKind: spec.scaling ?? 'none',
        bounds: featureSize === 2 ? squareBounds(raw.x, raw.n) : null,
        raw,
    };
}

/** The cached PreparedDataset, if it has been prepared already. */
export function peekPrepared(spec, options, split, dataSeed) {
    if (!spec) return null;
    return values.get(preparedKey(spec, options, split, dataSeed)) ?? null;
}

/** Prepare without waiting when the raw data is in memory (toy sets, Iris, digits after its first load). */
export function prepareDatasetSync(spec, options, split, dataSeed) {
    if (!spec) return null;
    const key = preparedKey(spec, options, split, dataSeed);
    if (values.has(key)) return values.get(key);
    const raw = spec.loadSync?.(resolveOptions(spec, options)) ?? null;
    if (!raw) return null;
    const prepared = prepareRaw(spec, raw, options, split, dataSeed);
    remember(key, prepared);
    return prepared;
}

/** Load (possibly a dynamic import), split and scale. Rejects for export-only datasets. */
export function prepareDataset(spec, options = {}, split = {}, dataSeed) {
    if (!spec) return Promise.reject(new Error('No dataset'));
    if (spec.training !== 'browser' || typeof spec.load !== 'function') {
        const error = new Error(`${spec.name} is too large to train in the browser. Open Code → train.py and run it in Colab.`);
        error.code = 'dataset-export-only';
        return Promise.reject(error);
    }
    const key = preparedKey(spec, options, split, dataSeed);
    if (values.has(key)) return Promise.resolve(values.get(key));
    if (!promises.has(key)) {
        const resolved = resolveOptions(spec, options);
        promises.set(key, Promise.resolve(spec.load(resolved)).then((raw) => {
            const prepared = prepareRaw(spec, raw, resolved, split, dataSeed);
            remember(key, prepared);
            return prepared;
        }).finally(() => promises.delete(key)));
    }
    return promises.get(key);
}

/** Notified with the cache key whenever a dataset finishes preparing (the Data tab re-renders). */
export function subscribePrepared(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}
