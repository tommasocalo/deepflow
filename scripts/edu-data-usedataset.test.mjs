import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

// Install in-memory storage before the stores hydrate (as startup-regressions.test.mjs does).
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { applyDataset, findOutputLinear, useDataset } = await import('../src/edu/data/useDataset.js');
const { useEduDataStore, DATA_KEY, normalizeBinding } = await import('../src/edu/data/store.js');
const { readSource, callSource, workspaceSlices } = await import('../src/edu/bridge.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { restoreWorkspace } = await import('../src/edu/lessons/engine/workspace.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useArchitectureHistory } = await import('../src/stores/architectureHistory.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');

const graph = () => useArchitectureStore.getState().graph;
const history = () => useArchitectureHistory.getState();
const slice = (name) => workspaceSlices().find(([key]) => key === name)?.[1];

/**
 * in (2,) → neuron (Linear component 2→1) → [extras…] → out, with Adam and `loss` bound to out.
 * `extras` are library keys inserted between the Linear and the Output.
 */
function load({ loss = 'bce', extras = [], rawLinear = false, outFeatures = 1 } = {}) {
    const h = createFixtureHelpers();
    const input = h.input('in', [2], { label: 'features' });
    const linear = rawLinear
        ? h.layer('neuron', 'torch.nn.Linear', { in_features: 2, out_features: outFeatures, bias: true }, 'neuron')
        : h.linear('neuron', 2, outFeatures, { label: 'neuron' });
    const middle = extras.map((key, index) => h.layer(`x${index}`, key));
    const output = h.output('out', { label: 'logit' });
    const nodes = [input, linear, ...middle, output];
    restoreWorkspace(h.snapshot({ nodes, edges: h.chain(...nodes), training: loss ? [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss(loss, output)] : undefined }));
}

beforeEach(() => {
    memory.clear();
    slice('data').restore(undefined);
    load();
});

test('Use this dataset sets the root Input shape and dtype, and the last Linear Output for 3 classes', () => {
    const result = applyDataset('iris');
    assert.equal(useEduDataStore.getState().datasetId, 'iris');
    assert.equal(graph().nodes.in.parameters.shape, '(4,)');
    assert.equal(graph().nodes.in.parameters.dtype, 'float32');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 3);
    assert.deepEqual(result.changed, ['Input → (4,)', 'last Linear (neuron) Output → 3']);
    assert.deepEqual(result.warnings, []);
    assert.equal(useEduDataStore.getState().lastUse.datasetId, 'iris', 'the Data tab shows the result line');
});

test('setOutput: false leaves the output size to the learner (lesson steps)', () => {
    const result = useDataset('iris', { setOutput: false });
    assert.equal(graph().nodes.in.parameters.shape, '(4,)');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 1);
    assert.deepEqual(result.changed, ['Input → (4,)']);
});

test('it edits the ROOT graph even while the main pane shows a component level', () => {
    useArchitectureStore.getState().openLevel(['neuron']);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['neuron']);
    applyDataset('digits');
    assert.equal(graph().nodes.in.parameters.shape, '(1, 8, 8)');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 10);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['neuron'], 'navigation is not changed');
});

test('the output walk passes activations, Dropout, Softmax and LogSoftmax', () => {
    load({ loss: 'cross_entropy', extras: ['torch.nn.ReLU', 'torch.nn.Dropout', 'torch.nn.LogSoftmax'] });
    const found = findOutputLinear(graph(), useTemplatesStore.getState().templates);
    assert.equal(found?.node.id, 'neuron');
    applyDataset('spiral3');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 3);
});

test('a raw torch.nn.Linear layer is set through its parameters', () => {
    load({ loss: 'cross_entropy', rawLinear: true });
    applyDataset('iris');
    assert.equal(graph().nodes.neuron.parameters.out_features, 3);
});

test('it reports, and changes nothing, when the block before Output is not a Linear', () => {
    load({ extras: ['torch.nn.Flatten'] });
    const before = graph().nodes.neuron.parameterValues.out_features;
    const result = applyDataset('iris');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, before);
    assert.deepEqual(result.warnings, ['Could not set the output size: the last block is not a Linear.']);
    assert.equal(graph().nodes.in.parameters.shape, '(4,)', 'the Input is still set');
});

test('two classes: BCE gives 1 output, cross-entropy gives 2', () => {
    load({ loss: 'bce', outFeatures: 3 });
    applyDataset('circle');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 1);
    load({ loss: 'cross_entropy', outFeatures: 1 });
    applyDataset('circle');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 2);
    load({ loss: null, outFeatures: 1 });
    applyDataset('xor');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 1, 'no loss: an existing single output stays (BCE is the default)');
});

test('one undo entry restores the Input and the Linear', () => {
    assert.equal(history().undoCount, 0);
    applyDataset('iris');
    assert.equal(history().undoCount, 1);
    assert.equal(history().undoLabel, 'Use dataset Iris');
    history().undo();
    assert.equal(graph().nodes.in.parameters.shape, '(2,)');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 1);
    assert.equal(history().undoCount, 0);
    applyDataset('blobs');
    assert.equal(history().undoCount, 0, 'nothing changed for the matching blobs dataset, so no history entry');
});

test('the data source exposes the binding and its actions; options are clamped per dataset', () => {
    callSource('data', 'useDataset', 'blobs', { options: { n: 5000, noise: 0.123, seed: 7 }, split: { val: 0.9, test: 0.18 } });
    const read = readSource('data');
    assert.equal(read.datasetId, 'blobs');
    assert.deepEqual(read.spec, { id: 'blobs', name: 'Two blobs', kind: 'toy2d', training: 'browser', input: { shape: '(2,)', dtype: 'float32' }, target: { kind: 'classes', classes: ['blue', 'orange'], size: 2 } });
    assert.deepEqual(read.options, { n: 1000, noise: 0.1, seed: 7 });
    assert.deepEqual(read.split, { val: 0.5, test: 0.2 });
    assert.equal(read.dataSeed, 7);
    callSource('data', 'setSplit', { val: 0.2, test: 0 });
    callSource('data', 'setOptions', { noise: 0.3 });
    assert.deepEqual(readSource('data').options, { n: 1000, noise: 0.3, seed: 7 });
    assert.deepEqual(readSource('data').split, { val: 0.2, test: 0 });
    callSource('data', 'setDataset', null);
    assert.equal(readSource('data').datasetId, null);
    assert.equal(readSource('data').spec, null);
});

test('the data slice captures and restores the binding, and the live binding is persisted', () => {
    applyDataset('spiral3', { options: { perClass: 50, noise: 0.3, seed: 5 }, split: { val: 0.5 } });
    const captured = structuredClone(slice('data').capture());
    assert.deepEqual(captured, { datasetId: 'spiral3', options: { spiral3: { perClass: 50, noise: 0.3, seed: 5 } }, split: { val: 0.5, test: 0 } });
    assert.deepEqual(JSON.parse(memory.get(DATA_KEY)), captured);
    slice('data').restore(undefined);
    assert.equal(useEduDataStore.getState().datasetId, null);
    slice('data').restore(captured);
    assert.equal(useEduDataStore.getState().datasetId, 'spiral3');
    assert.deepEqual(useEduDataStore.getState().current().options, { perClass: 50, noise: 0.3, seed: 5 });
    // Lesson fixtures write the slice with Stage A's helper: { datasetId, options: { [id]: … }, split }.
    const h = createFixtureHelpers();
    slice('data').restore(h.data('circle', { n: 300, noise: 0, seed: 2 }, { val: 0.2 }));
    assert.deepEqual(useEduDataStore.getState().current(), { spec: useEduDataStore.getState().current().spec, options: { n: 300, noise: 0, seed: 2 }, split: { val: 0.2, test: 0 }, dataSeed: 2 });
    // Unknown ids and malformed values are dropped.
    assert.deepEqual(normalizeBinding({ datasetId: 'nope', options: { nope: { n: 1 } }, split: { val: 'x' } }), { datasetId: null, options: {}, split: { val: 0.2, test: 0 } });
});
