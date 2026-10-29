/**
 * Seeded 2D toy generators (docs/edu/SPEC.md §5.3). Each draws from mulberry32(deriveSeed(seed, 'data')) and returns
 * { x: Float64Array(n·2), y: Int32Array(n) } (regression: Float64Array) in generation order.
 *
 * Expressions keep the evaluation order of data/python/toy.js, so the exported train.py regenerates the same points.
 * blobs, circle, xor, spiral2 and plane are adapted from TensorFlow Playground's dataset.ts (Apache-2.0);
 * spiral3 is the CS231n "Neural Networks Case Study" spiral (MIT).
 */
import { normal, stream, uniform } from './rng.js';

function result(points, regression = false) {
    const n = points.length;
    const x = new Float64Array(n * 2);
    const y = regression ? new Float64Array(n) : new Int32Array(n);
    points.forEach(([x1, x2, label], index) => {
        x[index * 2] = x1;
        x[index * 2 + 1] = x2;
        y[index] = label;
    });
    return { x, y, n };
}

/** Two Gaussian clouds: ⌊n/2⌋ points with label 1 around (2, 2), then ⌊n/2⌋ with label 0 around (−2, −2). */
export function blobs(n, noise, seed) {
    const next = stream(seed, 'data');
    const half = Math.floor(n / 2);
    const variance = 0.5 + 3.5 * (noise / 0.5);
    const std = Math.sqrt(variance);
    const points = [];
    for (const [center, label] of [[2, 1], [-2, 0]]) {
        for (let i = 0; i < half; i++) {
            const x1 = normal(next, center, std);
            const x2 = normal(next, center, std);
            points.push([x1, x2, label]);
        }
    }
    return result(points);
}

/** A disk (label 1, radius < 2.5) inside a ring (label 0, radius 3.5–5). Noise moves only the labelling point. */
export function circle(n, noise, seed) {
    const next = stream(seed, 'data');
    const half = Math.floor(n / 2);
    const points = [];
    for (const [low, high] of [[0, 2.5], [3.5, 5]]) {
        for (let i = 0; i < half; i++) {
            const r = uniform(next, low, high);
            const angle = uniform(next, 0, 2 * Math.PI);
            const x1 = r * Math.sin(angle);
            const x2 = r * Math.cos(angle);
            const noiseX = uniform(next, -5, 5) * noise;
            const noiseY = uniform(next, -5, 5) * noise;
            const dx = x1 + noiseX;
            const dy = x2 + noiseY;
            points.push([x1, x2, Math.sqrt(dx * dx + dy * dy) < 2.5 ? 1 : 0]);
        }
    }
    return result(points);
}

/** Label 1 where x₁ and x₂ have the same sign. Coordinates are pushed 0.3 away from the axes. */
export function xor(n, noise, seed) {
    const next = stream(seed, 'data');
    const points = [];
    for (let i = 0; i < n; i++) {
        let x1 = uniform(next, -5, 5);
        x1 += x1 > 0 ? 0.3 : -0.3;
        let x2 = uniform(next, -5, 5);
        x2 += x2 > 0 ? 0.3 : -0.3;
        const noiseX = uniform(next, -5, 5) * noise;
        const noiseY = uniform(next, -5, 5) * noise;
        points.push([x1, x2, (x1 + noiseX) * (x2 + noiseY) >= 0 ? 1 : 0]);
    }
    return result(points);
}

/** Two interleaved spiral arms: label 1 (Δ = 0), then label 0 (Δ = π). */
export function spiral2(n, noise, seed) {
    const next = stream(seed, 'data');
    const half = Math.floor(n / 2);
    const points = [];
    for (const [delta, label] of [[0, 1], [Math.PI, 0]]) {
        for (let i = 0; i < half; i++) {
            const r = i / half * 5;
            const t = 1.75 * i / half * 2 * Math.PI + delta;
            const x1 = r * Math.sin(t) + uniform(next, -1, 1) * noise;
            const x2 = r * Math.cos(t) + uniform(next, -1, 1) * noise;
            points.push([x1, x2, label]);
        }
    }
    return result(points);
}

/** CS231n spiral with K = 3 arms; class j occupies indices j·perClass … (j + 1)·perClass − 1. */
export function spiral3(perClass, noise, seed) {
    const next = stream(seed, 'data');
    const points = [];
    for (let j = 0; j < 3; j++) {
        for (let i = 0; i < perClass; i++) {
            const r = i / (perClass - 1);
            const t = 4 * j + 4 * i / (perClass - 1) + normal(next, 0, 1) * noise;
            points.push([r * Math.sin(t), r * Math.cos(t), j]);
        }
    }
    return result(points);
}

/** Regression: target = (x₁ + noise_x + x₂ + noise_y) / 10, Playground's linear scale from [−10, 10] to [−1, 1]. */
export function plane(n, noise, seed) {
    const next = stream(seed, 'data');
    const points = [];
    for (let i = 0; i < n; i++) {
        const x1 = uniform(next, -6, 6);
        const x2 = uniform(next, -6, 6);
        const noiseX = uniform(next, -6, 6) * noise;
        const noiseY = uniform(next, -6, 6) * noise;
        points.push([x1, x2, (x1 + noiseX + x2 + noiseY) / 10]);
    }
    return result(points, true);
}

export const GENERATORS = { blobs, circle, xor, spiral2, spiral3, plane };
