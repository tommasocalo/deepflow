import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const W = await import('../src/edu/lessons/engine/workspace.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { runActions } = await import('../src/edu/lessons/engine/actions.js');
const { registerSlice } = await import('../src/edu/bridge.js');
const { setModuleRegistry } = await import('../src/edu/lessons/engine/registry.js');
const { useLessonSession } = await import('../src/edu/lessons/engine/session.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useArchitectureHistory } = await import('../src/stores/architectureHistory.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore } = await import('../src/stores/trainingStore.js');
const { gpt2Storyboard } = await import('../src/storyboards/gpt2.js');
const { startArchitectureSession } = await import('../src/stores/architectureHistory.js');
const { TEST_MODULE, TEST_TEXT } = await import('./fixtures/edu-test-module.js');

// Stand-ins for Stage B's 'data' and 'trainer' slices.
let dataSlice = { datasetId: null, options: {}, split: { val: 0.2, test: 0 } };
let trainerSlice = { settings: { epochs: 50, batchSize: 32, seed: 1 } };
registerSlice('data', { capture: () => structuredClone(dataSlice), restore: (value) => { dataSlice = value ?? { datasetId: null, options: {}, split: { val: 0.2, test: 0 } }; } });
registerSlice('trainer', { capture: () => structuredClone(trainerSlice), restore: (value) => { trainerSlice = value ?? { settings: { epochs: 50, batchSize: 32, seed: 1 } }; } });

const arch = () => useArchitectureStore.getState();
const session = () => useLessonSession.getState();
/** Storage is JSON: undefined handles disappear, which is harmless. Compare the JSON form. */
const plain = (value) => JSON.parse(JSON.stringify(value));

function seedSandbox() {
    startArchitectureSession(() => gpt2Storyboard({ architectureStore: useArchitectureStore, templatesStore: useTemplatesStore, trainingStore: useTrainingStore }));
}

beforeEach(() => {
    memory.clear();
    setModuleRegistry({ modules: [TEST_MODULE], text: TEST_TEXT });
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model' });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null });
    session().setHeadless(true);
    session().hydrate();
    useTemplatesStore.setState({ templates: {} });
    seedSandbox();
    dataSlice = { datasetId: null, options: {}, split: { val: 0.2, test: 0 } };
});

test('capture and restore round-trip graph, templates, training, slices and view flags', () => {
    dataSlice = { datasetId: 'iris', options: { iris: { seed: 3 } }, split: { val: 0.3, test: 0 } };
    trainerSlice = { settings: { epochs: 12, batchSize: 16, seed: 4 } };
    arch().setShowGradientFlow(true);
    const snapshot = W.captureWorkspace(123);
    assert.equal(snapshot.v, 1);
    assert.equal(snapshot.savedAt, 123);
    assert.deepEqual(snapshot.slices, { data: dataSlice, trainer: trainerSlice });
    assert.deepEqual(snapshot.ui, { showDimensions: true, showGradientFlow: true });
    const json = JSON.parse(JSON.stringify(snapshot));

    W.restoreWorkspace(W.buildFixtureSnapshot(TEST_MODULE, 'start'));
    assert.deepEqual(dataSlice.datasetId, 'blobs', 'the fixture data slice is restored');
    assert.equal(arch().showGradientFlow, false);
    assert.ok(arch().graph.nodes.lin1);

    W.restoreWorkspace(json);
    assert.deepEqual(plain(arch().graph), plain(snapshot.graph));
    assert.deepEqual(plain(useTemplatesStore.getState().templates), plain(snapshot.templates));
    assert.deepEqual(plain(useTrainingStore.getState().graph), plain(snapshot.training));
    assert.deepEqual(dataSlice, { datasetId: 'iris', options: { iris: { seed: 3 } }, split: { val: 0.3, test: 0 } });
    assert.deepEqual(trainerSlice.settings, { epochs: 12, batchSize: 16, seed: 4 });
    assert.equal(arch().showGradientFlow, true);
    assert.equal(useArchitectureHistory.getState().undoCount, 0, 'restoring starts a fresh history');
    assert.deepEqual(arch().currentPath, []);
});

test('slots are stored per workspace under deepflow-edu:slot:<id>:v1', () => {
    W.saveSlot('m3', W.captureWorkspace());
    assert.ok(memory.has('deepflow-edu:slot:m3:v1'));
    assert.deepEqual(W.loadSlot('m3').graph, plain(arch().graph));
    W.deleteSlot('m3');
    assert.equal(W.loadSlot('m3'), null);
    assert.equal(W.loadSlot('nope'), null);
});

test('sandbox ↔ lesson switching keeps both workspaces', async () => {
    const sandboxGraph = arch().graph;
    const sandboxTemplates = useTemplatesStore.getState().templates;
    assert.ok(Object.values(sandboxGraph.nodes).some((node) => node.label === 'Transformer blocks'));

    await session().startModule('demo');
    assert.equal(session().mode, 'lesson');
    assert.equal(session().phase, 'warmup', 'a module with warm-up items opens on the warm-up');
    assert.equal(useEduUiStore.getState().activeTab, 'architecture', 'starting never leaves the learner on Learn');
    assert.ok(arch().graph.nodes.lin1, 'the start fixture is loaded');
    assert.ok(memory.has('deepflow-edu:slot:sandbox:v1'), 'the sandbox was saved before switching');

    await session().skipWarmup();
    await runActions([{ do: 'setParam', node: { id: 'lin2' }, name: 'out_features', value: 3 }]);
    session().exitLesson();
    assert.equal(session().mode, 'sandbox');
    assert.equal(useEduUiStore.getState().activeTab, 'learn');
    assert.deepEqual(plain(arch().graph), plain(sandboxGraph));
    assert.deepEqual(plain(useTemplatesStore.getState().templates), plain(sandboxTemplates));

    await session().startModule('demo');
    assert.equal(session().phase, 'steps', 'warm-up is not repeated after it was skipped');
    assert.equal(arch().graph.nodes.lin2.parameterValues.out_features, 3, 'the lesson slot keeps the learner’s edit');
});

test('entering a step runs its setup once; Reset step restores the entry snapshot and reopens the prediction', async () => {
    await session().startModule('demo');
    await session().skipWarmup();
    await session().goToStep(3);
    assert.equal(session().currentStep().id, 's4');
    assert.equal(arch().showDimensions, false, 'setup turned Dimensions off');
    await runActions([{ do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 5 }]);
    session().submitPrediction('demo.shape', [8]);
    assert.ok(session().progress.modules.demo.answers['demo.shape']);
    assert.ok(session().resetStep());
    assert.equal(arch().graph.nodes.lin1.parameterValues.out_features, 8);
    assert.equal(session().progress.modules.demo.answers['demo.shape'], undefined, 'the prediction can be retried');

    arch().setShowDimensions(true);
    await session().back();
    await session().goToStep(3);
    assert.equal(arch().showDimensions, true, 'setup is not run again on a later entry');
});

test('a storyboard fixture runs on fresh templates, applies its after actions and clears history', async () => {
    const module = {
        ...TEST_MODULE, id: 'story',
        fixtures: { start: { kind: 'storyboard', storyboard: 'gpt2', configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4, nLayers: 2, mlpWidth: 256 }, after: [{ do: 'setOptimizer', type: 'AdamW', hyperparameters: { learning_rate: 0.001, weight_decay: 0.1 } }, { do: 'setUi', showDimensions: true }] } },
    };
    useTemplatesStore.setState({ templates: { stale: { id: 'stale', name: 'Old template', graph: { nodes: {}, edges: [] }, freeParameters: [] } } });
    await W.loadFixtureIntoWorkspace(module, 'start', { runAfter: async (after) => { const result = await runActions(after, { module }); assert.ok(result.ok, result.error); } });
    const globals = Object.fromEntries(arch().graph.parameters.map((parameter) => [parameter.name, parameter.default]));
    assert.deepEqual([globals.vocab_size, globals.context, globals.d_model, globals.n_heads, globals.n_layers], [65, 64, 64, 4, 2]);
    assert.ok(!useTemplatesStore.getState().templates.stale, 'the lesson starts from its own templates');
    const optimizer = Object.values(useTrainingStore.getState().graph.nodes).find((node) => node.kind === 'optimizer');
    assert.equal(optimizer.hyperparameters.learning_rate, 0.001);
    assert.equal(useArchitectureHistory.getState().undoCount, 0);
});

test('exiting without a saved sandbox slot falls back to the GPT-2 example', async () => {
    await session().startModule('demo');
    memory.delete('deepflow-edu:slot:sandbox:v1');
    session().exitLesson();
    assert.ok(Object.values(arch().graph.nodes).some((node) => node.label === 'Transformer blocks'));
});

test('Reset progress clears lesson slots, progress and the step entry but keeps the sandbox slot', async () => {
    await session().startModule('demo');
    await session().skipWarmup();
    session().exitLesson();
    assert.ok(memory.has('deepflow-edu:slot:demo:v1'));
    session().resetProgress();
    assert.ok(!memory.has('deepflow-edu:slot:demo:v1'));
    assert.ok(memory.has('deepflow-edu:slot:sandbox:v1'));
    assert.ok(!memory.has(W.STEP_ENTRY_KEY));
    assert.deepEqual(session().progress.modules, {});
});
