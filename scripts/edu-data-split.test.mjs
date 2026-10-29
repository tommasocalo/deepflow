import assert from 'node:assert/strict';
import { test } from 'node:test';
import { fitStandardize, splitIndices, splitSizes } from '../src/edu/data/split.js';
import { getDataset } from '../src/edu/data/registry.js';
import { peekPrepared, prepareDataset, prepareDatasetSync, prepareRaw } from '../src/edu/data/prepare.js';

test('split sizes use floor(n·f + 0.5), not round-half-to-even', () => {
    assert.deepEqual(splitSizes(150, { val: 0.2, test: 0 }), { nVal: 30, nTest: 0, nTrain: 120 });
    assert.deepEqual(splitSizes(1797, { val: 0.2, test: 0 }), { nVal: 359, nTest: 0, nTrain: 1438 });
    assert.deepEqual(splitSizes(300, { val: 0.2, test: 0.1 }), { nVal: 60, nTest: 30, nTrain: 210 });
    // 25 · 0.1 = 2.5: Python's round() would give 2.
    assert.equal(splitSizes(25, { val: 0.1 }).nVal, 3);
    assert.equal(splitSizes(15, { val: 0.5 }).nVal, 8);
});

test('splits are disjoint, cover every index, keep shuffled order and are deterministic', () => {
    for (const [n, split] of [[200, { val: 0.2, test: 0 }], [301, { val: 0.3, test: 0.2 }], [1797, { val: 0.1, test: 0.1 }]]) {
        const parts = splitIndices(n, split, 4);
        const all = [...parts.train, ...parts.val, ...parts.test];
        assert.equal(all.length, n);
        assert.equal(new Set(all).size, n);
        assert.deepEqual([...all].sort((a, b) => a - b), Array.from({ length: n }, (_, i) => i));
        assert.equal(parts.val.length, Math.floor(n * split.val + 0.5));
        assert.equal(parts.test.length, Math.floor(n * split.test + 0.5));
        assert.deepEqual(splitIndices(n, split, 4), parts, 'same seed, same split');
        assert.notDeepEqual(splitIndices(n, split, 5).val, parts.val, 'another seed shuffles differently');
    }
    const sorted = splitIndices(100, { val: 0.2 }, 1);
    assert.notDeepEqual([...sorted.train], [...sorted.train].sort((a, b) => a - b), 'train keeps the shuffled order');
});

test('standardisation is fitted on the training rows only', () => {
    const x = Float64Array.from([1, 10, 2, 20, 3, 30, 100, 1000]); // 4 rows × 2 features
    const train = [0, 1, 2];
    const fitted = fitStandardize(x, 2, train);
    assert.deepEqual(fitted.mean, [2, 20]);
    assert.ok(Math.abs(fitted.std[0] - Math.sqrt(2 / 3)) < 1e-15, 'population std (ddof 0)');
    const changedValidation = Float64Array.from(x);
    changedValidation[6] = -5000;
    assert.deepEqual(fitStandardize(changedValidation, 2, train), fitted, 'a validation row never changes the statistics');
    const changedTrain = Float64Array.from(x);
    changedTrain[0] = 7;
    assert.notDeepEqual(fitStandardize(changedTrain, 2, train).mean, fitted.mean);
    assert.deepEqual(fitStandardize(Float64Array.from([5, 5, 5]), 1, [0, 1, 2]).std, [1e-12], 'std is floored at 1e-12');
});

test('Iris prepares to 120 / 30 rows with the training mean and std of the Data tab', async () => {
    const iris = getDataset('iris');
    const prepared = await prepareDataset(iris, { seed: 1 }, { val: 0.2, test: 0 });
    assert.equal(prepared.train.n, 120);
    assert.equal(prepared.val.n, 30);
    assert.equal(prepared.test.n, 0);
    assert.deepEqual(prepared.scaling.mean.map((value) => value.toFixed(2)), ['5.83', '3.07', '3.72', '1.18']);
    assert.deepEqual(prepared.scaling.std.map((value) => value.toFixed(2)), ['0.84', '0.44', '1.79', '0.78']);
    // Scaled training features have mean 0 and std 1; validation features do not have to.
    for (let f = 0; f < 4; f++) {
        let sum = 0, sq = 0;
        for (let i = 0; i < 120; i++) { sum += prepared.train.x[i * 4 + f]; sq += prepared.train.x[i * 4 + f] ** 2; }
        assert.ok(Math.abs(sum / 120) < 1e-12);
        assert.ok(Math.abs(Math.sqrt(sq / 120) - 1) < 1e-12);
    }
    assert.equal(prepared.bounds, null, 'only 2D inputs get decision-boundary bounds');
    assert.equal(await prepareDataset(iris, { seed: 1 }, { val: 0.2, test: 0 }), prepared, 'cached by id, options, split and seed');
    assert.equal(peekPrepared(iris, { seed: 1 }, { val: 0.2, test: 0 }), prepared);
    assert.notEqual(await prepareDataset(iris, { seed: 2 }, { val: 0.2, test: 0 }), prepared);
});

test('toy sets prepare synchronously, unscaled, with square bounds around every point', () => {
    const blobs = getDataset('blobs');
    const prepared = prepareDatasetSync(blobs, { n: 200, noise: 0.1, seed: 1 }, { val: 0.2, test: 0 });
    assert.equal(prepared.train.n + prepared.val.n, 200);
    assert.equal(prepared.scaling, null);
    const raw = blobs.loadSync({ n: 200, noise: 0.1, seed: 1 });
    const first = prepared.train.indices[0];
    assert.equal(prepared.train.x[0], raw.x[first * 2], 'toy features keep their coordinates');
    const [[x0, x1], [y0, y1]] = prepared.bounds;
    assert.ok(Math.abs((x1 - x0) - (y1 - y0)) < 1e-9, 'square');
    for (let i = 0; i < raw.n; i++) {
        assert.ok(raw.x[i * 2] > x0 && raw.x[i * 2] < x1 && raw.x[i * 2 + 1] > y0 && raw.x[i * 2 + 1] < y1);
    }
    const direct = prepareRaw(blobs, raw, { n: 200, noise: 0.1, seed: 1 }, { val: 0.2, test: 0 });
    assert.deepEqual(direct.val.indices, prepared.val.indices);
});

test('export-only presets are rejected with dataset-export-only', async () => {
    const fake = { id: 'mnist-like', name: 'MNIST', training: 'export', options: {}, target: { kind: 'classes', size: 10, classes: [] } };
    await assert.rejects(prepareDataset(fake), (error) => error.code === 'dataset-export-only' && /Colab/.test(error.message));
});
