import assert from 'node:assert/strict';
import { test } from 'node:test';
import { deriveSeed, mulberry32, normal, shuffleInPlace, stream, uniform } from '../src/edu/data/rng.js';

// Pinned from the reference Python port of docs/edu/SPEC.md §11.4 (checked with Node 22 and Python 3.13).
const VECTORS = {
    0: [0.26642920868471265, 0.0003297457005828619, 0.2232720274478197, 0.1462021479383111, 0.46732782293111086],
    1: [0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849],
    42: [0.6011037519201636, 0.44829055899754167, 0.8524657934904099, 0.6697340414393693, 0.17481389874592423],
};

test('mulberry32 reproduces the pinned first five outputs for seeds 0, 1 and 42', () => {
    for (const [seed, expected] of Object.entries(VECTORS)) {
        const next = mulberry32(Number(seed));
        assert.deepEqual(Array.from({ length: 5 }, () => next()), expected, `seed ${seed}`);
    }
});

test('mulberry32 stays in [0, 1) and treats seeds as unsigned 32-bit integers', () => {
    const next = mulberry32(123);
    for (let i = 0; i < 10000; i++) {
        const value = next();
        assert.ok(value >= 0 && value < 1);
    }
    assert.equal(mulberry32(-1)(), mulberry32(4294967295)());
    assert.equal(mulberry32(2 ** 32 + 5)(), mulberry32(5)());
});

test('deriveSeed matches the spec vectors and separates streams', () => {
    assert.equal(deriveSeed(1, 'data'), 2212395750);
    assert.equal(deriveSeed(1, 'split'), 3534800360);
    assert.equal(deriveSeed(5, 'init'), 2825648146);
    const labels = ['data', 'split', 'init', 'batches', 'dropout'];
    const seeds = labels.map((label) => deriveSeed(1, label));
    assert.equal(new Set(seeds).size, labels.length, 'every stream label gives its own seed');
    assert.notEqual(deriveSeed(1, 'data'), deriveSeed(2, 'data'));
    assert.equal(stream(1, 'data')(), mulberry32(2212395750)());
});

test('the polar normal has mean 0 and std 1 over 100,000 samples', () => {
    const next = mulberry32(7);
    const n = 100000;
    let sum = 0, sumSq = 0;
    for (let i = 0; i < n; i++) {
        const value = normal(next);
        sum += value;
        sumSq += value * value;
    }
    const mean = sum / n;
    const std = Math.sqrt(sumSq / n - mean * mean);
    assert.ok(Math.abs(mean) < 0.01, `mean ${mean}`);
    assert.ok(Math.abs(std - 1) < 0.01, `std ${std}`);
    const shifted = normal(mulberry32(7), 2, 3);
    assert.ok(Math.abs(shifted - (2 + 3 * normal(mulberry32(7)))) < 1e-12, 'mean and std scale one standard draw');
});

test('uniform and Fisher–Yates are deterministic', () => {
    const a = mulberry32(3), b = mulberry32(3);
    for (let i = 0; i < 100; i++) {
        const value = uniform(a, -5, 5);
        assert.ok(value >= -5 && value < 5);
        assert.equal(value, uniform(b, -5, 5));
    }
    const order = shuffleInPlace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mulberry32(9));
    assert.deepEqual([...order].sort((x, y) => x - y), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    assert.deepEqual(shuffleInPlace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mulberry32(9)), order);
    assert.notDeepEqual(shuffleInPlace([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], mulberry32(10)), order);
});
