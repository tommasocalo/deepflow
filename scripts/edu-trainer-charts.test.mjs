import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
    epochTicks, formatAccuracy, formatLoss, formatLr, formatTick, linePath, lossAxisMax, niceCeil, niceTicks, previousRuns,
    runDifference, sampleRows, spreadLabels,
} from '../src/edu/trainer/charts.js';
import { formatDuration, groupIssues, runAnnouncement, runStatusText, shownRecord } from '../src/edu/trainer/describe.js';
import { boundaryRgb, divergingRgb, hexToRgb } from '../src/edu/data/classStyle.js';

const run = (overrides = {}) => ({
    id: 'r1', seq: 1, kind: 'train', status: 'completed', datasetId: 'spiral3', paramCount: 387,
    optimizer: { type: 'Adam', hyperparameters: { learning_rate: 0.01 } }, loss: { type: 'cross_entropy' },
    settings: { epochs: 60, batchSize: 32, seed: 5 }, dataOptions: { perClass: 100, noise: 0.2, seed: 5 }, split: { val: 0.2, test: 0 },
    graph: null, history: [{ epoch: 0, trainLoss: 1.1, valLoss: 1.1, valAccuracy: 0.33 }, { epoch: 1, trainLoss: 0.5, valLoss: 0.6, valAccuracy: 0.8 }],
    ...overrides,
});

test('number formats: loss, accuracy, learning rate, ticks and durations', () => {
    assert.equal(formatLoss(Math.log(2)), '0.693');
    assert.equal(formatLoss(Math.log(3)), '1.10');
    assert.equal(formatLoss(0.00123), '0.0012');
    assert.equal(formatLoss(12.34), '12.3');
    assert.equal(formatLoss(null), '–');
    assert.equal(formatAccuracy(0.966), '97%');
    assert.equal(formatAccuracy(0.995), '99.5%', 'never rounds 99.5 % up to 100 %');
    assert.equal(formatAccuracy(1), '100%');
    assert.equal(formatLr(0.0001), '0.0001');
    assert.equal(formatLr(0.03), '0.03');
    assert.equal(formatLr(1), '1');
    assert.equal(formatLr(3e-5), '3e-5');
    assert.equal(formatTick(0.5), '0.5');
    assert.equal(formatTick(1), '1');
    assert.equal(formatTick(2.5), '2.5');
    assert.equal(formatDuration(42.4), '42 ms');
    assert.equal(formatDuration(719), '719 ms');
    assert.equal(formatDuration(3400), '3.4 s');
    assert.equal(formatDuration(-1), '?');
});

test('axes: nice maxima and ticks; diverging runs are cut at 3 × the starting loss', () => {
    assert.equal(niceCeil(0.693), 1);
    assert.equal(niceCeil(1.1), 2);
    assert.equal(niceCeil(2.2), 2.5);
    assert.equal(niceCeil(4.2), 5);
    assert.deepEqual(niceTicks(-6.5, 6.2, 4), [-5, 0, 5]);
    assert.deepEqual(niceTicks(0.5, 7.4, 4), [2, 4, 6]);
    assert.equal(lossAxisMax([[1.1, 0.4]], { chance: Math.log(3), reference: 1.1 }), 2);
    assert.equal(lossAxisMax([[1.1, 500]], { chance: Math.log(3), reference: 1.1 }), 5, 'lr 1.0 blowing up does not flatten the chart');
    assert.deepEqual(epochTicks(60), [0, 30, 60]);
    assert.deepEqual(epochTicks(1), [0, 1]);
});

test('linePath breaks on missing values and clamps to the axis', () => {
    const identity = (value) => value;
    assert.equal(linePath([[0, 1], [1, 2], [2, null], [3, 9]], identity, identity, [0, 5]), 'M0.0,1.0L1.0,2.0M3.0,5.0');
    assert.equal(linePath([], identity, identity), '');
});

test('sampleRows keeps the first and last epoch', () => {
    const history = Array.from({ length: 301 }, (_, epoch) => ({ epoch }));
    const rows = sampleRows(history, 8);
    assert.equal(rows.length, 8);
    assert.equal(rows[0].epoch, 0);
    assert.equal(rows.at(-1).epoch, 300);
});

test('earlier runs are labelled with what differed from the newest run', () => {
    const current = run({ id: 'r3', seq: 3, optimizer: { type: 'Adam', hyperparameters: { learning_rate: 1 } } });
    assert.equal(runDifference(run({ id: 'r1' }), current), 'lr 0.01');
    assert.equal(runDifference(run({ id: 'r2', optimizer: { type: 'Adam', hyperparameters: { learning_rate: 0.0001 } } }), current), 'lr 0.0001');
    assert.equal(runDifference(run({ id: 'r2', optimizer: { type: 'SGD', hyperparameters: { learning_rate: 1 } } }), current), 'SGD');
    assert.equal(runDifference(run({ id: 'r2', optimizer: current.optimizer, settings: { epochs: 60, batchSize: 32, seed: 6 } }), current), 'seed 6');
    assert.equal(runDifference(run({ id: 'r2', optimizer: current.optimizer, paramCount: 33 }), current), '33 params');
    assert.equal(runDifference(run({ id: 'r2', seq: 2, optimizer: current.optimizer }), current), 'run 2', 'identical settings fall back to the run number');
    assert.equal(runDifference(run({ id: 'r2', optimizer: { type: 'SGD', hyperparameters: { learning_rate: 0.1 } }, settings: { epochs: 60, batchSize: 32, seed: 9 } }), current), 'SGD · lr 0.1', 'at most two parts');
});

test('previousRuns: same dataset, finished training runs only, newest first', () => {
    const runs = [
        run({ id: 'a', seq: 1 }), run({ id: 'init', seq: 2, kind: 'init', history: [{ epoch: 0 }] }), run({ id: 'b', seq: 3, datasetId: 'circle' }),
        run({ id: 'c', seq: 4 }), run({ id: 'd', seq: 5 }),
    ];
    assert.deepEqual(previousRuns(runs, runs.at(-1)).map((item) => item.id), ['c', 'a']);
    assert.deepEqual(previousRuns(runs, null), []);
});

test('end labels are spread at least `gap` apart and kept inside the plot', () => {
    const spread = spreadLabels([{ id: 'a', y: 50 }, { id: 'b', y: 52 }, { id: 'c', y: 99 }], 10, [0, 100]);
    const ys = spread.map((item) => item.y);
    for (let i = 1; i < ys.length; i++) assert.ok(ys[i] - ys[i - 1] >= 10 - 1e-9, JSON.stringify(ys));
    assert.ok(ys.at(-1) <= 100);
});

test('status lines and announcements describe the run, and failures say why', () => {
    assert.equal(runStatusText(run({ status: 'running', epoch: 12 }), 'running'), 'Epoch 12 / 60');
    assert.equal(runStatusText(run({ status: 'completed', epoch: 60, startedAt: 0, finishedAt: 700 }), 'idle'), 'Done · 60 epochs in 700 ms');
    assert.equal(runStatusText(run({ status: 'stopped', epoch: 7 }), 'idle'), 'Stopped at epoch 7');
    assert.equal(runStatusText(run({ initOnly: true, history: [{ epoch: 0, trainLoss: 1.1, valLoss: 1.0986 }] }), 'idle'), 'Initial loss 1.10 (epoch 0, before any update)');
    assert.equal(runStatusText(run({ status: 'failed', error: { message: 'No dataset yet.' } }), 'error'), 'Not trained: No dataset yet.');
    assert.equal(runAnnouncement(run({ status: 'running' })), '');
    assert.equal(runAnnouncement(run()), 'Training finished. Validation accuracy 80%, validation loss 0.600.');
    assert.equal(shownRecord(run({ id: 'x', status: 'failed' }), [run({ id: 'good' })]).id, 'good', 'a failed attempt keeps the last good charts');
    assert.equal(shownRecord(null, []), null);
});

test('many unsupported blocks are summarised in one preflight line', () => {
    const errors = [
        { code: 'unsupported-op', type: 'Gather', message: 'a', nodeId: 'tok' },
        { code: 'unsupported-op', type: 'Split', message: 'b', nodeId: 'blocks' },
        { code: 'unsupported-op', type: 'Gather', message: 'c', nodeId: 'pos' },
        { code: 'input-mismatch', message: 'd' },
    ];
    const grouped = groupIssues(errors);
    assert.equal(grouped.length, 2);
    assert.equal(grouped[0].message, '3 blocks are not supported by the browser trainer (Gather, Split). Use Code → train.py to train this model in PyTorch.');
    assert.equal(grouped[0].nodeId, 'tok');
    assert.equal(grouped[1].code, 'input-mismatch');
    assert.equal(groupIssues(errors.slice(0, 2)).length, 2, 'two unsupported blocks keep their own lines');
});

test('class colours: Okabe-Ito tints by confidence, diverging regression colours', () => {
    assert.deepEqual(hexToRgb('#0072B2'), [0, 114, 178]);
    const unsure = boundaryRgb(0, 0.5, 2);
    const sure = boundaryRgb(0, 1, 2);
    assert.ok(unsure[0] > sure[0], 'a confident cell is a stronger blue');
    assert.ok(unsure.every((channel) => channel > 220), 'chance level is only a light tint');
    assert.deepEqual(divergingRgb(0), [255, 255, 255]);
    assert.deepEqual(divergingRgb(-1), [0, 114, 178]);
    assert.deepEqual(divergingRgb(5), [230, 159, 0], 'values are clamped to [−1, 1]');
});
