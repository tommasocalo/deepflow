import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

// Install in-memory storage before the stores hydrate (as startup-regressions.test.mjs does).
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { useTrainerStore, normalizeSettings, TRAINER_KEY, MAX_RUNS, DEFAULT_SETTINGS } = await import('../src/edu/trainer/store.js');
const { applyDataset } = await import('../src/edu/data/useDataset.js');
const { useEduDataStore } = await import('../src/edu/data/store.js');
const { exportDatasets } = await import('../src/edu/data/registry.js');
const { readSource, callSource, workspaceSlices } = await import('../src/edu/bridge.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { restoreWorkspace } = await import('../src/edu/lessons/engine/workspace.js');
const { createCheckContext, evaluateCheck } = await import('../src/edu/lessons/engine/checks.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore } = await import('../src/stores/trainingStore.js');

const trainer = () => useTrainerStore.getState();
const slice = (name) => workspaceSlices().find(([key]) => key === name)?.[1];
const strip = (history) => history.map(({ ms: _ms, ...metrics }) => metrics);
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

/** Resolve once `predicate(state)` holds for the trainer store, checked on every update (no timing assumptions). */
function until(predicate, ms = 30000) {
    return new Promise((resolve, reject) => {
        if (predicate(trainer())) { resolve(); return; }
        let unsubscribe = () => {};
        const timer = setTimeout(() => { unsubscribe(); reject(new Error(`the trainer did not reach the state within ${ms} ms`)); }, ms);
        unsubscribe = useTrainerStore.subscribe((state) => {
            if (!predicate(state)) return;
            clearTimeout(timer);
            unsubscribe();
            resolve();
        });
    });
}

/** M1's start fixture: in (2,) → neuron (Linear 2→1) → out, Adam lr 0.03 and BCE bound to out, blobs bound. */
function loadM1({ hidden = null, loss = 'bce', lr = 0.03, dataset = 'blobs' } = {}) {
    const h = createFixtureHelpers();
    const input = h.input('in', [2], { label: 'features' });
    const blocks = hidden
        ? [h.linear('lin1', 2, hidden), h.layer('relu', 'torch.nn.ReLU'), h.linear('lin2', hidden, 1)]
        : [h.linear('neuron', 2, 1, { label: 'neuron' })];
    const output = h.output('out', { label: 'logit' });
    const nodes = [input, ...blocks, output];
    restoreWorkspace(h.snapshot({
        nodes, edges: h.chain(...nodes),
        training: [h.optimizer('Adam', { learning_rate: lr }), h.loss(loss, output)],
        data: dataset ? h.data(dataset, { n: 200, noise: 0.1, seed: 1 }) : undefined,
        trainer: h.trainer({ epochs: 20, batchSize: 32, seed: 1 }),
    }));
}

beforeEach(() => {
    memory.clear();
    loadM1();
});

test('a synchronous run records the epoch-0 evaluation plus every epoch, with the captured inputs', async () => {
    const graphAtStart = useArchitectureStore.getState().graph;
    const seq = trainer().seq;
    const record = await trainer().start({ sync: true });
    assert.equal(record.status, 'completed', JSON.stringify(record.error));
    assert.equal(record.seq, seq + 1);
    assert.equal(trainer().seq, seq + 1);
    assert.equal(record.history.length, 21, 'epoch 0 + 20 epochs');
    assert.deepEqual(record.history.map((entry) => entry.epoch), Array.from({ length: 21 }, (_, i) => i));
    assert.ok(record.history.at(-1).valAccuracy >= 0.9, `val acc ${record.history.at(-1).valAccuracy}`);
    assert.equal(record.datasetId, 'blobs');
    assert.deepEqual(record.dataOptions, { n: 200, noise: 0.1, seed: 1 });
    assert.deepEqual(record.split, { val: 0.2, test: 0 });
    assert.deepEqual(record.settings, { epochs: 20, batchSize: 32, seed: 1 });
    assert.equal(record.optimizer.type, 'Adam');
    assert.equal(record.optimizer.hyperparameters.learning_rate, 0.03);
    assert.equal(record.loss.type, 'bce');
    assert.equal(record.graph, graphAtStart, 'the graph reference is captured, not copied');
    assert.equal(record.training, useTrainingStore.getState().graph);
    assert.equal(record.templates, useTemplatesStore.getState().templates);
    assert.equal(record.paramCount, 3);
    assert.ok(Math.abs(record.chance - Math.log(2)) < 1e-12, 'chance line ln 2 for BCE');
    assert.deepEqual(record.classNames, ['blue', 'orange']);
    assert.equal(record.confusion.flat().reduce((a, b) => a + b, 0), 40);
    assert.equal(record.boundary.classes.length, 64 * 64);
    assert.ok(record.best && record.best.epoch >= 0);
    assert.equal(trainer().status, 'idle');
    assert.equal(trainer().runs.at(-1), record);
    assert.equal(trainer().current, record);
});

test('Check initial loss records an epoch-0 run that the run checks recognise', async () => {
    const record = await trainer().evaluateInit({ sync: true });
    assert.equal(record.status, 'completed');
    assert.equal(record.kind, 'init');
    assert.equal(record.initOnly, true);
    assert.equal(record.history.length, 1);
    assert.equal(record.history[0].epoch, 0);
    const ctx = createCheckContext({ graph: useArchitectureStore.getState().graph, templates: useTemplatesStore.getState().templates,
        training: useTrainingStore.getState().graph, data: readSource('data'), trainer: readSource('trainer'), step: { enteredAtSeq: record.seq - 1 } });
    assert.equal(evaluateCheck({ run: { dataset: 'blobs', initOnly: true } }, ctx).pass, true);
    assert.equal(evaluateCheck({ run: { dataset: 'blobs', minEpochs: 1 } }, ctx).pass, false, 'an init check is not a training run');
});

test('a lesson run check passes on a run made through the bridge (as the headless Show me does)', async () => {
    const entered = trainer().seq;
    const record = await callSource('trainer', 'start', { sync: true });
    const ctx = createCheckContext({ graph: useArchitectureStore.getState().graph, templates: useTemplatesStore.getState().templates,
        training: useTrainingStore.getState().graph, data: readSource('data'), trainer: readSource('trainer'), step: { enteredAtSeq: entered } });
    const check = { run: { dataset: 'blobs', minEpochs: 20, final: { valAccuracy: { gte: 0.9 } }, model: { has: { type: 'Linear' }, count: 1 } } };
    assert.equal(evaluateCheck(check, ctx).pass, true);
    const later = createCheckContext({ ...ctx, step: { enteredAtSeq: record.seq } });
    assert.equal(evaluateCheck(check, later).pass, false, 'runs from before the step entry do not count');
});

test('preflight errors fail the run without adding it to the run list', async () => {
    slice('data').restore(undefined);
    const before = trainer().runs.length;
    const record = await trainer().start({ sync: true });
    assert.equal(record.status, 'failed');
    assert.equal(record.error.code, 'no-dataset');
    assert.equal(trainer().status, 'error');
    assert.equal(trainer().current, record);
    assert.equal(trainer().runs.length, before);
    applyDataset('iris', { setOutput: false });
    const invalid = await trainer().start({ sync: true });
    assert.equal(invalid.error.code, 'invalid-graph', 'Input (4,) into Linear 2 → 1');
    const neuron = useArchitectureStore.getState().graph.nodes.neuron;
    useArchitectureStore.getState().patchNode('neuron', { parameterValues: { ...neuron.parameterValues, in_features: 4 } });
    const mismatch = await trainer().start({ sync: true });
    assert.equal(mismatch.status, 'failed');
    assert.equal(mismatch.error.code, 'loss-target-mismatch');
    assert.equal(mismatch.error.message, 'BCE is for two classes, but Iris has 3: choose cross_entropy in the Training tab.');
    useTrainingStore.getState().patchNode('loss', { type: 'cross_entropy' });
    const outputs = await trainer().start({ sync: true });
    assert.equal(outputs.error.message, 'Iris has 3 classes: set the last Linear\'s Output to 3 or use the Data tab\'s Use this dataset.');
    assert.equal(trainer().runs.length, before, 'failed attempts never enter the run list');
});

test('export-only datasets are refused with the Colab message', { skip: exportDatasets().length ? false : 'needs Stage C: src/edu/data/presets/images.js' }, async () => {
    applyDataset(exportDatasets()[0].id);
    const record = await trainer().start({ sync: true });
    assert.equal(record.status, 'failed');
    assert.equal(record.error.code, 'dataset-export-only');
    assert.match(record.error.message, /Colab/);
});

test('a sliced run gives the same history as a synchronous one, and publishes progress while it runs', async () => {
    const reference = await trainer().start({ sync: true });
    let updates = 0;
    const unsubscribe = useTrainerStore.subscribe((state, previous) => { if (state.current !== previous.current) updates += 1; });
    const promise = trainer().start();
    assert.equal(trainer().status, 'running');
    assert.equal(trainer().current.status, 'running');
    const record = await promise;
    unsubscribe();
    assert.equal(record.status, 'completed');
    assert.deepEqual(strip(record.history), strip(reference.history));
    assert.deepEqual(record.confusion, reference.confusion);
    assert.deepEqual([...record.boundary.classes], [...reference.boundary.classes]);
    assert.ok(updates >= 2, 'the store published while running');
});

test('Stop ends a sliced run early; the stopped run keeps its epochs, confusion matrix and boundary', async () => {
    trainer().setSettings({ epochs: 400 });
    loadM1({ hidden: 32 });
    applyDataset('circle', { setOutput: false });
    trainer().setSettings({ epochs: 400 });
    const promise = trainer().start();
    // Stop once the run has recorded an epoch, however slowly it gets there (review r2: two ticks were not always
    // enough under parallel load, and the stopped run had no history).
    await until((state) => (state.current?.history?.length ?? 0) >= 1);
    assert.equal(trainer().stop(), true);
    assert.equal(trainer().status, 'stopping');
    const record = await promise;
    assert.equal(record.status, 'stopped');
    assert.ok(record.history.length >= 1 && record.history.length < 401, `history ${record.history.length}`);
    assert.ok(record.confusion, 'confusion matrix of the stopped model');
    assert.ok(record.boundary, 'boundary of the stopped model');
    assert.equal(trainer().status, 'idle');
    assert.equal(trainer().runs.at(-1).id, record.id);
});

test('starting while a run is active stops it first', async () => {
    trainer().setSettings({ epochs: 400 });
    const first = trainer().start();
    await tick();
    const second = await trainer().start({ sync: true });
    const stopped = await first;
    assert.equal(stopped.status, 'stopped');
    assert.equal(second.status, 'completed');
    assert.ok(second.seq > stopped.seq);
});

test('at most MAX_RUNS runs are kept, and runs can be tagged and cleared', async () => {
    trainer().setSettings({ epochs: 1 });
    for (let i = 0; i < MAX_RUNS + 2; i++) await trainer().start({ sync: true });
    assert.equal(trainer().runs.length, MAX_RUNS);
    const last = trainer().runs.at(-1);
    callSource('trainer', 'tagRun', last.id, 'm5.overfit');
    assert.equal(trainer().runs.at(-1).tag, 'm5.overfit');
    assert.equal(trainer().current.tag, 'm5.overfit');
    trainer().clearRuns();
    assert.deepEqual(trainer().runs, []);
    assert.equal(trainer().current, null);
});

test('settings are clamped whole numbers, persisted, and restored by the trainer slice (which clears runs)', async () => {
    assert.deepEqual(normalizeSettings({ epochs: 0, batchSize: 4096.4, seed: -3 }), { epochs: 1, batchSize: 1024, seed: 0 });
    assert.deepEqual(normalizeSettings({ epochs: 'abc' }, { epochs: 30, batchSize: 16, seed: 2 }), { epochs: 30, batchSize: 16, seed: 2 });
    callSource('trainer', 'setSettings', { epochs: 60, batchSize: 16, seed: 5 });
    assert.deepEqual(JSON.parse(memory.get(TRAINER_KEY)), { epochs: 60, batchSize: 16, seed: 5 });
    await trainer().start({ sync: true });
    assert.ok(trainer().runs.length > 0);
    assert.deepEqual(slice('trainer').capture(), { settings: { epochs: 60, batchSize: 16, seed: 5 } });
    slice('trainer').restore(undefined);
    assert.deepEqual(trainer().settings, DEFAULT_SETTINGS);
    assert.deepEqual(trainer().runs, []);
    assert.equal(trainer().current, null);
    slice('trainer').restore({ settings: { epochs: 300, batchSize: 16 } });
    assert.deepEqual(trainer().settings, { epochs: 300, batchSize: 16, seed: 1 });
});

test('restoring a workspace aborts an active run without recording it', async () => {
    trainer().setSettings({ epochs: 400 });
    const promise = trainer().start();
    await tick();
    loadM1();
    const record = await promise;
    assert.equal(record.status, 'stopped');
    assert.deepEqual(trainer().runs, []);
    assert.equal(trainer().status, 'idle');
});

test('the trainer source exposes status, seq, settings, current, runs and the preflight', async () => {
    const read = readSource('trainer');
    assert.deepEqual(Object.keys(read).sort(), ['current', 'preflight', 'runs', 'seq', 'settings', 'status']);
    const preflight = trainer().refreshPreflight();
    assert.equal(preflight.ok, true, JSON.stringify(preflight.errors));
    assert.equal(preflight.datasetId, 'blobs');
    assert.equal(preflight.paramCount, 3);
    assert.equal(preflight.objective.optimizer.type, 'Adam');
    assert.equal('program' in preflight, false, 'the compiled program stays out of the store');
    useEduDataStore.getState().setDataset(null);
    assert.equal(trainer().refreshPreflight().errors[0].code, 'no-dataset');
});
