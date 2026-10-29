/**
 * Seeded randomness shared by the browser and the exported Python (docs/edu/SPEC.md §5.3).
 *
 * Every function here is bit-defined: data/python/common.js ports it line by line, and
 * scripts/edu-data-rng.test.mjs pins its outputs. Changing anything here changes which examples land in which split.
 */

/** mulberry32 with a 32-bit state; returns floats in [0, 1). */
export function mulberry32(seed) {
    let a = seed >>> 0;
    return function next() {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const fnv1a = (s) => {
    let h = 0x811c9dc5;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 0x01000193) >>> 0;
    }
    return h;
};

/**
 * Combine a user seed with a stream label so data, split, init, batch order and dropout draw from independent
 * streams. Labels: 'data', 'split' (dataset seed); 'init', 'batches', 'dropout' (training seed).
 */
export function deriveSeed(seed, label) {
    let x = (Math.imul(seed >>> 0, 0x9E3779B1) ^ fnv1a(label)) >>> 0;
    x ^= x >>> 16; x = Math.imul(x, 0x85EBCA6B) >>> 0; x ^= x >>> 13; x = Math.imul(x, 0xC2B2AE35) >>> 0; x ^= x >>> 16;
    return x >>> 0;
}

/** A fresh generator for one stream. */
export const stream = (seed, label) => mulberry32(deriveSeed(seed, label));

/** Marsaglia polar method: one value per call, the second value is discarded. */
export function normal(next, mean = 0, std = 1) {
    let v1, v2, s;
    do { v1 = 2 * next() - 1; v2 = 2 * next() - 1; s = v1 * v1 + v2 * v2; } while (s >= 1 || s === 0);
    return mean + std * v1 * Math.sqrt(-2 * Math.log(s) / s);
}

export const uniform = (next, a, b) => next() * (b - a) + a;

/** In-place Fisher–Yates: for i from n − 1 down to 1, j = ⌊next() · (i + 1)⌋, swap(i, j). */
export function shuffleInPlace(array, next) {
    for (let i = array.length - 1; i >= 1; i--) {
        const j = Math.floor(next() * (i + 1));
        const tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
    return array;
}
