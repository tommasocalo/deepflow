/**
 * Runs (docs/edu/SPEC.md §6.10.1, §6.10.7). createRun builds the model, optimizer and streams; runSteps is a
 * generator that trains batch by batch and yields between batches and evaluation chunks, so the store can slice it
 * into ≤ 12 ms pieces. The computation never depends on where it pauses: the same inputs give the same history.
 *
 * Epoch 0 evaluates the untrained model. After every epoch all metrics are computed in evaluation mode (dropout off)
 * over the full training and validation splits. Streams: 'init', 'batches' and 'dropout' of the training seed.
 *
 * A test split is held out of every choice: the weights of the best epoch (lowest validation loss, the ▲ marker) are
 * kept, and the test split is evaluated once, with them, when the run ends (`run.test`), as train.py does.
 */
import { stream, shuffleInPlace } from '../data/rng.js';
import { computeBoundary } from './boundary.js';
import { computeLoss } from './losses.js';
import { batchOf, bestEpoch, confusionMatrix, evaluateSplit } from './metrics.js';
import { createModel, forward } from './model.js';
import { createOptimizer } from './optimizers.js';
import { Tape } from './tensor.js';

const now = () => (globalThis.performance?.now ? globalThis.performance.now() : Date.now());

/**
 * @param {Object} args
 * @param {Object} args.program     compileModel(...).program
 * @param {Object} args.dataset     a PreparedDataset (data/prepare.js)
 * @param {{ optimizer: { type, hyperparameters, parameterGroup? }, loss: { type, hyperparameters } }} args.objective
 * @param {{ epochs: number, batchSize: number, seed: number }} args.settings
 */
export function createRun({ program, dataset, objective, settings }) {
    const epochs = Math.max(0, Math.floor(Number(settings.epochs) || 0));
    const batchSize = Math.max(1, Math.floor(Number(settings.batchSize) || 32));
    const seed = Math.floor(Number(settings.seed) || 0);
    const model = createModel(program, seed);
    const group = new Set(objective.optimizer.parameterGroup ?? []);
    const trainable = [];
    for (const param of program.params) {
        const tensor = model.params.get(param.id);
        if (!tensor.requiresGrad) continue;
        // A non-empty parameter group trains only the parameters of those root nodes; the others stay frozen.
        if (group.size && !group.has(param.origin.rootNodeId)) {
            tensor.requiresGrad = false;
            continue;
        }
        trainable.push(tensor);
    }
    return {
        program,
        model,
        dataset,
        objective,
        settings: { epochs, batchSize, seed },
        optimizer: createOptimizer(objective.optimizer.type, objective.optimizer.hyperparameters, trainable),
        trainable,
        batches: stream(seed, 'batches'),
        dropout: stream(seed, 'dropout'),
        history: [],
        best: null,
        bestWeights: null,
        test: null,
        confusion: null,
        boundary: null,
        epoch: 0,
        done: false,
    };
}

const hasTestSplit = (run) => (run.dataset.test?.n ?? 0) > 0;

/** Copy the weights when this epoch is the best so far (only runs with a test split need them). */
function keepBestWeights(run, epoch) {
    if (!hasTestSplit(run) || run.best?.epoch !== epoch) return;
    run.bestWeights = new Map([...run.model.params].map(([id, tensor]) => [id, Float64Array.from(tensor.data)]));
}

/** The test split, evaluated once with the best epoch's weights; the final weights are put back afterwards. */
function* evaluateTest(run) {
    if (!hasTestSplit(run) || !run.bestWeights || !run.best) return;
    const final = new Map([...run.model.params].map(([id, tensor]) => [id, tensor.data]));
    for (const [id, tensor] of run.model.params) tensor.data = Float64Array.from(run.bestWeights.get(id) ?? tensor.data);
    try {
        const result = yield* evaluateSplit(run.model, run.dataset, run.dataset.test, run.objective);
        run.test = { epoch: run.best.epoch, loss: result.loss, accuracy: result.accuracy, count: result.count };
    } finally {
        for (const [id, tensor] of run.model.params) tensor.data = final.get(id);
    }
}

/** One optimizer step on the given training rows. Returns the training-mode batch loss. */
export function trainBatch(run, indices) {
    const { model, dataset, objective, optimizer } = run;
    const { x, y } = batchOf(dataset.train, dataset.featureShape, 0, 0, indices);
    const tape = new Tape();
    const out = forward(model, x, { training: true, tape, dropoutNext: run.dropout });
    const { loss, grad } = computeLoss(objective.loss.type, out, y, objective.loss.hyperparameters);
    optimizer.zeroGrad();
    tape.backward(out, grad);
    optimizer.step();
    return loss;
}

function* evaluateEpoch(run, epoch) {
    const started = now();
    const train = yield* evaluateSplit(run.model, run.dataset, run.dataset.train, run.objective);
    const val = yield* evaluateSplit(run.model, run.dataset, run.dataset.val, run.objective);
    return {
        epoch,
        trainLoss: train.loss,
        valLoss: val.loss,
        trainAccuracy: train.accuracy,
        valAccuracy: val.accuracy,
        ms: now() - started,
    };
}

/**
 * Confusion matrix (validation split, or training split when there is none), the final boundary, and the test split
 * once with the best epoch's weights.
 */
export function* finishRun(run) {
    const { dataset } = run;
    yield* evaluateTest(run);
    if (dataset.numClasses !== null) {
        const split = dataset.val.n ? dataset.val : dataset.train;
        const result = yield* evaluateSplit(run.model, dataset, split, run.objective, { keepPredictions: true });
        if (result.predictions) run.confusion = confusionMatrix(result.predictions, split.y, dataset.numClasses);
    }
    run.boundary = computeBoundary(run.model, dataset, run.objective.loss.type);
    run.done = true;
}

/**
 * Generator over the whole run: epoch 0, then `epochs` epochs. Yields 'batch', 'eval' or 'epoch' markers; the
 * store checks the clock between them. `initOnly` evaluates epoch 0 and finishes.
 */
export function* runSteps(run, { initOnly = false } = {}) {
    run.history.push({ ...(yield* evaluateEpoch(run, 0)) });
    run.best = bestEpoch(run.history);
    keepBestWeights(run, 0);
    yield 'epoch';
    const n = run.dataset.train.n;
    const epochs = initOnly ? 0 : run.settings.epochs;
    for (let epoch = 1; epoch <= epochs; epoch++) {
        const started = now();
        const order = new Int32Array(n);
        for (let i = 0; i < n; i++) order[i] = i;
        shuffleInPlace(order, run.batches);
        for (let start = 0; start < n; start += run.settings.batchSize) {
            trainBatch(run, order.subarray(start, Math.min(n, start + run.settings.batchSize)));
            yield 'batch';
        }
        const trainMs = now() - started;
        const metrics = yield* evaluateEpoch(run, epoch);
        run.history.push({ ...metrics, ms: trainMs + metrics.ms });
        run.epoch = epoch;
        run.best = bestEpoch(run.history);
        keepBestWeights(run, epoch);
        yield 'epoch';
    }
    yield* finishRun(run);
}

/** Run a generator to its end without yielding to the event loop (tests, headless Show me). */
export function drain(iterator) {
    let step = iterator.next();
    while (!step.done) step = iterator.next();
    return step.value;
}

/** Convenience for Node: create and run synchronously. */
export function trainSync(args, options = {}) {
    const run = createRun(args);
    drain(runSteps(run, options));
    return run;
}
