/**
 * Train / validation / test split and feature scaling (docs/edu/SPEC.md §5.3). data/python/common.js ports
 * splitIndices as split_indices(n, val, test, seed), which returns the same index lists.
 */
import { shuffleInPlace, stream } from './rng.js';

/** Split sizes use ⌊n·f + 0.5⌋ (Python must not use round(), which rounds half to even). */
export function splitSizes(n, { val = 0.2, test = 0 } = {}) {
    const nVal = Math.floor(n * val + 0.5);
    const nTest = Math.floor(n * test + 0.5);
    return { nVal, nTest, nTrain: Math.max(0, n - nVal - nTest) };
}

/**
 * Shuffle 0 … n − 1 with the 'split' stream of the dataset seed. val = the first nVal shuffled indices, test = the
 * next nTest, train = the rest; every split keeps the shuffled order.
 */
export function splitIndices(n, split = {}, seed = 1) {
    const order = new Int32Array(n);
    for (let i = 0; i < n; i++) order[i] = i;
    shuffleInPlace(order, stream(seed, 'split'));
    const { nVal, nTest } = splitSizes(n, split);
    return {
        val: order.slice(0, nVal),
        test: order.slice(nVal, nVal + nTest),
        train: order.slice(nVal + nTest),
    };
}

/** Per-feature mean and population std (ddof 0) over the given rows only; std is floored at 1e-12. */
export function fitStandardize(x, featureSize, rows) {
    const mean = new Array(featureSize).fill(0);
    const std = new Array(featureSize).fill(0);
    if (!rows.length) return { mean, std: std.map(() => 1) };
    for (const row of rows) {
        for (let f = 0; f < featureSize; f++) mean[f] += x[row * featureSize + f];
    }
    for (let f = 0; f < featureSize; f++) mean[f] /= rows.length;
    for (const row of rows) {
        for (let f = 0; f < featureSize; f++) {
            const d = x[row * featureSize + f] - mean[f];
            std[f] += d * d;
        }
    }
    for (let f = 0; f < featureSize; f++) std[f] = Math.max(Math.sqrt(std[f] / rows.length), 1e-12);
    return { mean, std };
}

/** Gather rows of a flat feature matrix (and labels), applying an optional per-feature transform. */
export function gatherRows(x, y, featureSize, indices, transform = null) {
    const n = indices.length;
    const outX = new Float64Array(n * featureSize);
    const outY = y instanceof Float64Array ? new Float64Array(n) : new Int32Array(n);
    for (let i = 0; i < n; i++) {
        const row = indices[i];
        const base = row * featureSize;
        for (let f = 0; f < featureSize; f++) {
            const value = x[base + f];
            outX[i * featureSize + f] = transform ? transform(value, f) : value;
        }
        outY[i] = y[row];
    }
    return { x: outX, y: outY, n, indices: Int32Array.from(indices) };
}
