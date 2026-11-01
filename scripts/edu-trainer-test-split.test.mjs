import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compileModel } from '../src/edu/trainer/compile.js';
import { createRun, runSteps, trainSync } from '../src/edu/trainer/run.js';
import { evaluateSplit } from '../src/edu/trainer/metrics.js';
import { testLine } from '../src/edu/trainer/describe.js';
import { getDataset } from '../src/edu/data/registry.js';
import { prepareDataset } from '../src/edu/data/prepare.js';
import { chainModel, objectiveOf, trainingGraph } from './fixtures/edu-trainer-graphs.mjs';

// A test split in the browser trainer (content request, ML review r1): the best epoch's weights (lowest validation
// loss, the ▲ epoch) are kept and the test split is evaluated once with them when the run ends; the final weights stay.

const SPIRAL = { options: { perClass: 50, noise: 0.3, seed: 5 }, split: { val: 0.3, test: 0.2 } };

async function setup(split = SPIRAL.split) {
    const { graph, templates } = chainModel('(2,)', [['linear', 2, 64], ['relu'], ['linear', 64, 64], ['relu'], ['linear', 64, 3]]);
    const compiled = compileModel({ graph, templates });
    assert.ok(compiled.ok);
    const dataset = await prepareDataset(getDataset('spiral3'), SPIRAL.options, split);
    return { program: compiled.program, dataset, objective: objectiveOf(trainingGraph({ loss: 'cross_entropy', lr: 0.01 })), settings: { epochs: 150, batchSize: 16, seed: 5 } };
}

const drainCounting = (run) => {
    const steps = runSteps(run);
    let finalWeights = null;
    for (let step = steps.next(); !step.done; step = steps.next()) {
        // The last epoch's weights, just before finishRun evaluates the test split.
        if (step.value === 'epoch' && run.epoch === run.settings.epochs) finalWeights = new Map([...run.model.params].map(([id, tensor]) => [id, Float64Array.from(tensor.data)]));
    }
    return finalWeights;
};

test('with a test split, the test is scored once with the best epoch\'s weights, and the final weights stay', async () => {
    const args = await setup();
    assert.ok(args.dataset.test.n > 0, 'the prepared dataset holds a test split');
    const run = createRun(args);
    const finalWeights = drainCounting(run);
    assert.ok(run.test, 'the run records a test result');
    assert.equal(run.test.epoch, run.best.epoch, 'scored at the ▲ epoch');
    assert.ok(run.best.epoch < run.settings.epochs, `this run overfits, so the best epoch (${run.best.epoch}) is not the last`);
    assert.equal(run.test.count, args.dataset.test.n);
    assert.ok(run.test.accuracy > 0.5 && run.test.accuracy <= 1, `test accuracy ${run.test.accuracy}`);
    assert.ok(run.history.every((entry) => !('testLoss' in entry) && !('testAccuracy' in entry)), 'no epoch looks at the test split');
    for (const [id, tensor] of run.model.params) assert.deepEqual([...tensor.data], [...finalWeights.get(id)], 'the final weights are put back');

    // The recorded score is the best epoch's: re-evaluate with the kept weights.
    for (const [id, tensor] of run.model.params) tensor.data = Float64Array.from(run.bestWeights.get(id));
    const again = (() => { const steps = evaluateSplit(run.model, args.dataset, args.dataset.test, args.objective); let step = steps.next(); while (!step.done) step = steps.next(); return step.value; })();
    assert.equal(again.accuracy, run.test.accuracy);
    assert.equal(again.loss, run.test.loss);
    assert.match(testLine(run), /^Test accuracy \d+(\.\d)?% \(best epoch \d+, used once\)$/);
});

test('without a test split nothing is kept or scored', async () => {
    const args = await setup({ val: 0.3, test: 0 });
    const run = trainSync({ ...args, settings: { ...args.settings, epochs: 5 } });
    assert.equal(run.test, null);
    assert.equal(run.bestWeights, null);
    assert.equal(testLine(run), '');
});
