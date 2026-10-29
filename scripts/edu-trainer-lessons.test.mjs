import assert from 'node:assert/strict';
import { after, test } from 'node:test';
import { compileModel } from '../src/edu/trainer/compile.js';
import { trainSync } from '../src/edu/trainer/run.js';
import { getDataset } from '../src/edu/data/registry.js';
import { prepareDataset } from '../src/edu/data/prepare.js';
import { chainModel, objectiveOf, trainingGraph } from './fixtures/edu-trainer-graphs.mjs';

// Every in-browser lesson configuration of docs/edu/SPEC.md §4 and §11.1, with the fixtures' data and training
// seeds, must reach the threshold its lesson step checks. Runs synchronously in Node, as the headless Show me does.
const timings = [];
let total = 0;
const L = (inFeatures, outFeatures) => ['linear', inFeatures, outFeatures];

async function lesson(name, { dataset, options, split = { val: 0.2, test: 0 }, input, blocks, loss, lr, epochs, batchSize = 32, seed, initOnly = false }) {
    const { graph, templates } = chainModel(input, blocks);
    const compiled = compileModel({ graph, templates });
    assert.ok(compiled.ok, `${name}: ${JSON.stringify(compiled.issues)}`);
    const prepared = await prepareDataset(getDataset(dataset), options, split);
    const started = performance.now();
    const run = trainSync({
        program: compiled.program, dataset: prepared,
        objective: objectiveOf(trainingGraph({ loss, lr })), settings: { epochs, batchSize, seed },
    }, { initOnly });
    const ms = performance.now() - started;
    timings.push(`${name}: ${ms.toFixed(0)} ms`);
    total += ms;
    return run;
}
const final = (run) => run.history.at(-1);

after(() => {
    console.log(`# lesson runs: ${timings.join('; ')}; total ${total.toFixed(0)} ms`);
});

test('M1 blobs: one neuron reaches ≥ 0.9 validation accuracy within 20 epochs', async () => {
    const run = await lesson('m1 blobs', { dataset: 'blobs', options: { n: 200, noise: 0.1, seed: 1 }, input: '(2,)', blocks: [L(2, 1)], loss: 'bce', lr: 0.03, epochs: 20, seed: 1 });
    assert.equal(run.history.length, 21, 'epoch 0 plus 20 epochs');
    assert.ok(final(run).valAccuracy >= 0.9, `val acc ${final(run).valAccuracy}`);
    assert.ok(run.boundary && run.boundary.classes.length === 64 * 64, 'a 64 × 64 decision boundary');
    assert.equal(run.confusion.flat().reduce((a, b) => a + b, 0), 40, 'confusion over the 40 validation points');
});

test('M1 Iris: Linear 4 → 3 with cross-entropy reaches ≥ 0.85 within 30 epochs', async () => {
    const run = await lesson('m1 iris', { dataset: 'iris', options: { seed: 1 }, input: '(4,)', blocks: [L(4, 3)], loss: 'cross_entropy', lr: 0.03, epochs: 30, batchSize: 16, seed: 1 });
    assert.ok(final(run).valAccuracy >= 0.85, `val acc ${final(run).valAccuracy}`);
    assert.equal(run.confusion.length, 3);
    assert.equal(run.boundary, null, 'no 2D boundary for 4 features');
    // Setosa is separable: its row of the validation confusion matrix is all on the diagonal.
    assert.equal(run.confusion[0][1] + run.confusion[0][2], 0);
});

test('M2 circle: a linear model stays ≤ 0.75, 2-8-1 with ReLU reaches ≥ 0.9 at 50 epochs', async () => {
    const circle = { dataset: 'circle', options: { n: 300, noise: 0, seed: 2 }, input: '(2,)', loss: 'bce', lr: 0.03, seed: 2 };
    const neuron = await lesson('m2 one neuron', { ...circle, blocks: [L(2, 1)], epochs: 30 });
    assert.ok(final(neuron).valAccuracy <= 0.75, `one neuron ${final(neuron).valAccuracy}`);
    const stacked = await lesson('m2 two linears', { ...circle, blocks: [L(2, 8), L(8, 1)], epochs: 30 });
    assert.ok(final(stacked).valAccuracy <= 0.75, `stacked linears ${final(stacked).valAccuracy}`);
    const relu = await lesson('m2 2-8-1 relu', { ...circle, blocks: [L(2, 8), ['relu'], L(8, 1)], epochs: 50 });
    assert.ok(final(relu).valAccuracy >= 0.9, `2-8-1 ${final(relu).valAccuracy}`);
});

test('M2 XOR: 2-8-1 reaches ≥ 0.9 at 50 epochs, and a seeded 2-4-1 MLP solves it above 95 %', async () => {
    const xor = { dataset: 'xor', options: { n: 300, noise: 0, seed: 3 }, input: '(2,)', loss: 'bce', lr: 0.03, seed: 2 };
    const wide = await lesson('m2 xor 2-8-1', { ...xor, blocks: [L(2, 8), ['relu'], L(8, 1)], epochs: 50 });
    assert.ok(final(wide).valAccuracy >= 0.9, `2-8-1 ${final(wide).valAccuracy}`);
    const narrow = await lesson('xor 2-4-1', { ...xor, blocks: [L(2, 4), ['relu'], L(4, 1)], epochs: 100 });
    assert.ok(final(narrow).valAccuracy > 0.95, `2-4-1 ${final(narrow).valAccuracy}`);
    assert.ok(final(narrow).trainAccuracy > 0.95, `2-4-1 train ${final(narrow).trainAccuracy}`);
});

test('M3 digits: Flatten → 64-128-10 reaches ≥ 0.9 at 10 epochs and starts near ln 10', async () => {
    const run = await lesson('m3 digits', { dataset: 'digits', options: { seed: 1 }, input: '(1, 8, 8)', blocks: [['flatten'], L(64, 128), ['relu'], L(128, 10)], loss: 'cross_entropy', lr: 0.001, epochs: 10, seed: 3 });
    assert.ok(Math.abs(run.history[0].valLoss - Math.log(10)) < 0.1, `initial loss ${run.history[0].valLoss}`);
    assert.ok(final(run).valAccuracy >= 0.9, `val acc ${final(run).valAccuracy}`);
});

test('M5 spiral3: initial loss within 0.15 of ln 3, and lr 0.01 beats 0.0001 and 1.0 after 60 epochs', async () => {
    const spiral = { dataset: 'spiral3', options: { perClass: 100, noise: 0.2, seed: 5 }, input: '(2,)', blocks: [L(2, 64), ['relu'], L(64, 3)], loss: 'cross_entropy', seed: 5 };
    const init = await lesson('m5 init', { ...spiral, lr: 0.01, epochs: 60, initOnly: true });
    assert.equal(init.history.length, 1, 'Check initial loss records epoch 0 only');
    assert.ok(Math.abs(init.history[0].valLoss - Math.log(3)) < 0.15, `init loss ${init.history[0].valLoss}`);
    const finals = {};
    for (const lr of [0.0001, 0.01, 1]) finals[lr] = final(await lesson(`m5 lr ${lr}`, { ...spiral, lr, epochs: 60 })).valLoss;
    assert.ok(finals[0.01] < finals[0.0001] && finals[0.01] < finals[1], JSON.stringify(finals));
});

test('M5 overfitting: 2-128-128-3 on 75 points for 300 epochs memorises the data while validation loss rises', async () => {
    const run = await lesson('m5 overfit', {
        dataset: 'spiral3', options: { perClass: 50, noise: 0.3, seed: 5 }, split: { val: 0.5, test: 0 }, input: '(2,)',
        blocks: [L(2, 128), ['relu'], L(128, 128), ['relu'], L(128, 3)], loss: 'cross_entropy', lr: 0.01, epochs: 300, batchSize: 16, seed: 5,
    });
    assert.equal(run.dataset.train.n, 75);
    const minimum = Math.min(...run.history.map((entry) => entry.valLoss));
    assert.ok(final(run).trainAccuracy >= 0.97, `train acc ${final(run).trainAccuracy}`);
    assert.ok(final(run).valLoss / minimum >= 1.25, `rise ${final(run).valLoss / minimum}`);
    assert.ok(run.best.epoch < 300 && run.best.valLoss === minimum, 'the best-validation epoch is marked');
});

test('identical inputs give identical histories (dropout and batch order included)', async () => {
    const config = { dataset: 'circle', options: { n: 300, noise: 0.1, seed: 4 }, input: '(2,)', blocks: [L(2, 16), ['relu'], ['dropout', 0.3], L(16, 1)], loss: 'bce', lr: 0.03, epochs: 25, seed: 9 };
    const strip = (run) => run.history.map(({ ms: _ms, ...metrics }) => metrics);
    const a = await lesson('determinism a', config);
    const b = await lesson('determinism b', config);
    assert.deepEqual(strip(b), strip(a));
    assert.deepEqual(b.confusion, a.confusion);
    assert.deepEqual([...b.boundary.classes], [...a.boundary.classes]);
    const c = await lesson('determinism other seed', { ...config, seed: 10 });
    assert.notDeepEqual(strip(c), strip(a), 'another training seed gives another run');
    assert.ok(total < 30000, `all lesson runs take ${total.toFixed(0)} ms`);
});
