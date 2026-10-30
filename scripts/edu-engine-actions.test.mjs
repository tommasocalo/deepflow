import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

// Install in-memory storage before the stores hydrate (as startup-regressions.test.mjs does).
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { runActions, resolveRootNode } = await import('../src/edu/lessons/engine/actions.js');
const { restoreWorkspace, buildFixtureSnapshot } = await import('../src/edu/lessons/engine/workspace.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useArchitectureHistory, recordArchitectureEdit } = await import('../src/stores/architectureHistory.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore } = await import('../src/stores/trainingStore.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { registerSource } = await import('../src/edu/bridge.js');
const { effectiveObjective } = await import('../src/edu/objective.js');
const { nodeType } = await import('../src/edu/lessons/engine/nodes.js');
const { TEST_MODULE } = await import('./fixtures/edu-test-module.js');

const arch = () => useArchitectureStore.getState();
const graph = () => arch().graph;
const history = () => useArchitectureHistory.getState();
const edgesOf = () => graph().edges.map((edge) => `${edge.source}>${edge.target}`).sort();
const typeOf = (id) => nodeType(graph().nodes[id], useTemplatesStore.getState().templates);

/** in → lin1 (2→8) → lin2 (8→1) → out, plus Adam and BCE. */
function loadMlp({ relu = false } = {}) {
    const h = createFixtureHelpers();
    const input = h.input('in', [2]);
    const lin1 = h.linear('lin1', 2, 8, { label: 'hidden' });
    const middle = relu ? [h.layer('relu', 'torch.nn.ReLU', {}, 'ReLU')] : [];
    const lin2 = h.linear('lin2', 8, 1, { label: 'score' });
    const output = h.output('out');
    const nodes = [input, lin1, ...middle, lin2, output];
    restoreWorkspace(h.snapshot({ nodes, edges: h.chain(...nodes), training: [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss('bce', output)], parameters: [{ name: 'n_heads', default: 4 }] }));
}

beforeEach(() => {
    memory.clear();
    loadMlp();
    useEduUiStore.setState({ activeTab: 'architecture', codeOpen: false, codeTab: 'model' });
});

test('insertBetween removes the edge and wires from → new → to, pushing lower nodes down', async () => {
    const before = { lin2: graph().nodes.lin2.position.y, out: graph().nodes.out.position.y };
    const result = await runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }]);
    assert.deepEqual(result, { ok: true });
    const relu = Object.values(graph().nodes).find((node) => node.libraryKey === 'torch.nn.ReLU');
    assert.ok(relu);
    assert.deepEqual(edgesOf(), ['in>lin1', `lin1>${relu.id}`, 'lin2>out', `${relu.id}>lin2`].sort());
    assert.equal(relu.position.y, graph().nodes.lin1.position.y + 130);
    assert.equal(graph().nodes.lin2.position.y, before.lin2 + 130);
    assert.equal(graph().nodes.out.position.y, before.out + 130);
});

test('lower nodes move only as far as the new rows need, so a distant Output keeps its place', async () => {
    const h = createFixtureHelpers();
    restoreWorkspace(h.snapshot({ nodes: [h.input('in', [2]), h.output('out', { row: 5 })], edges: [] }));
    const inY = graph().nodes.in.position.y;
    const outY = graph().nodes.out.position.y;
    assert.equal(outY - inY, 5 * 130);
    await runActions([{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [{ block: 'torch.nn.ReLU' }, { block: 'torch.nn.Tanh' }] }]);
    await runActions([{ do: 'insertAfter', block: 'torch.nn.Sigmoid', after: { type: 'Tanh' } }]);
    await runActions([{ do: 'insertAfter', block: 'torch.nn.GELU', after: { type: 'Sigmoid' } }]);
    assert.equal(graph().nodes.out.position.y, outY, 'four new rows fit between the Input and the Output');
    await runActions([{ do: 'insertAfter', block: 'torch.nn.Dropout', after: { type: 'GELU' } }]);
    assert.equal(graph().nodes.out.position.y, outY + 130, 'the fifth row moves the Output down by one row');
    const ys = ['torch.nn.ReLU', 'torch.nn.Tanh', 'torch.nn.Sigmoid', 'torch.nn.GELU', 'torch.nn.Dropout']
        .map((key) => Object.values(graph().nodes).find((node) => node.libraryKey === key).position.y - inY);
    assert.deepEqual(ys, [130, 260, 390, 520, 650]);
});

test('insertAfter moves the outgoing edges of `after` to the new block', async () => {
    await runActions([{ do: 'insertAfter', block: 'torch.nn.Dropout', after: { id: 'lin1' }, params: { p: 0.5 } }]);
    const dropout = Object.values(graph().nodes).find((node) => node.libraryKey === 'torch.nn.Dropout');
    assert.equal(dropout.parameters.p, 0.5);
    assert.deepEqual(edgesOf(), ['in>lin1', `lin1>${dropout.id}`, 'lin2>out', `${dropout.id}>lin2`].sort());
});

test('buildChain connects from → blocks → to with component parameters', async () => {
    const h = createFixtureHelpers();
    restoreWorkspace(h.snapshot({ nodes: [h.input('in', [2]), h.output('out')], edges: [] }));
    const result = await runActions([{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [
        { block: 'component:linear', params: { in_features: 2, out_features: 8 } }, { block: 'torch.nn.ReLU' }, { block: 'component:linear', params: { in_features: 8, out_features: 1 } },
    ] }]);
    assert.equal(result.ok, true, result.error);
    const order = [];
    let current = 'in';
    while (current) {
        order.push(typeOf(current));
        current = graph().edges.find((edge) => edge.source === current)?.target;
    }
    assert.deepEqual(order, ['Input', 'Linear', 'ReLU', 'Linear', 'Output']);
    const linears = Object.values(graph().nodes).filter((node) => node.kind === 'supernode');
    assert.deepEqual(linears.map((node) => node.parameterValues.out_features).sort(), [1, 8]);
});

test('setParam patches layers and merges component parameter values', async () => {
    loadMlp({ relu: true });
    await runActions([
        { do: 'setParam', node: { id: 'lin2' }, name: 'out_features', value: 3 },
        { do: 'setParam', node: { type: 'Linear', nth: 0 }, name: 'out_features', value: 4 },
        { do: 'setParam', node: { id: 'relu' }, name: 'inplace', value: true },
    ]);
    assert.deepEqual(graph().nodes.lin2.parameterValues, { in_features: 8, out_features: 3 }, 'other component values are kept');
    assert.equal(graph().nodes.lin1.parameterValues.out_features, 4);
    assert.equal(graph().nodes.relu.parameters.inplace, true);
});

test('a whole Show me is one undo entry, and Undo restores the previous graph', async () => {
    const original = graph();
    const undoBefore = history().undoCount;
    await runActions([
        { do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 3 },
        { do: 'setParam', node: { id: 'lin2' }, name: 'in_features', value: 3 },
        { do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } },
    ], { label: 'show me' });
    assert.equal(history().undoCount, undoBefore + 1);
    assert.equal(history().undoLabel, 'Lesson: show me');
    history().undo();
    assert.deepEqual(graph(), original);
});

test('a failing action stops the list, reports the error and leaves the document unchanged', async () => {
    const original = graph();
    const result = await runActions([
        { do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 3 },
        { do: 'connect', from: { id: 'missing' }, to: { id: 'out' } },
        { do: 'setParam', node: { id: 'lin2' }, name: 'in_features', value: 3 },
    ]);
    assert.equal(result.ok, false);
    assert.match(result.error, /No block matches missing/);
    assert.deepEqual(graph(), original);
    assert.equal((await runActions([{ do: 'fly' }])).ok, false);
});

test('group, repeat and openLevel at the root', async () => {
    loadMlp({ relu: true });
    const result = await runActions([{ do: 'group', nodes: [{ id: 'lin1' }, { id: 'relu' }], name: 'Block' }]);
    assert.equal(result.ok, true, result.error);
    const instance = Object.values(graph().nodes).find((node) => node.kind === 'supernode' && node.label === 'Block');
    assert.ok(instance, 'the grouped component replaces the two blocks');
    assert.equal(useTemplatesStore.getState().templates[instance.templateId].name, 'Block');
    assert.ok(!graph().nodes.lin1 && !graph().nodes.relu);

    await runActions([{ do: 'repeat', node: { label: 'Block' }, count: 3, mode: 'sequential' }]);
    const repeat = graph().nodes[instance.id];
    assert.equal(repeat.kind, 'repeat');
    assert.equal(repeat.count, 3);
    await runActions([{ do: 'repeat', node: { label: 'Block' }, count: 2, mode: 'sequential' }]);
    assert.equal(graph().nodes[instance.id].count, 2, 'an existing repeat is patched');

    await runActions([{ do: 'openLevel', path: [{ label: 'Block' }, { type: 'Linear' }] }]);
    assert.equal(arch().currentPath.length, 2);
    assert.equal(arch().currentPath[0], instance.id);
    await runActions([{ do: 'setParam', node: { id: 'lin2' }, name: 'out_features', value: 2 }]);
    assert.equal(arch().currentPath.length, 0, 'graph actions return to the root level first');
});

test('setGlobal, setOptimizer and setLoss patch the model-wide and training settings', async () => {
    await runActions([{ do: 'setGlobal', name: 'n_heads', value: 8 }]);
    assert.equal(graph().parameters.find((parameter) => parameter.name === 'n_heads').default, 8);

    await runActions([{ do: 'setOptimizer', hyperparameters: { learning_rate: 1 } }]);
    let objective = effectiveObjective(useTrainingStore.getState().graph, graph());
    assert.equal(objective.optimizer.hyperparameters.learning_rate, 1);
    await runActions([{ do: 'setOptimizer', type: 'AdamW', hyperparameters: { learning_rate: 0.001, weight_decay: 0.1 } }]);
    objective = effectiveObjective(useTrainingStore.getState().graph, graph());
    assert.equal(objective.optimizer.type, 'AdamW');
    assert.equal(objective.optimizer.label, 'AdamW');
    assert.equal(objective.optimizer.hyperparameters.weight_decay, 0.1);

    await runActions([{ do: 'setLoss', type: 'cross_entropy' }]);
    objective = effectiveObjective(useTrainingStore.getState().graph, graph());
    assert.equal(objective.loss.type, 'cross_entropy');
    assert.equal(objective.loss.label, 'Cross-Entropy loss', 'a default label follows the new type');
    assert.equal(objective.loss.hyperparameters.label_smoothing, 0);
    assert.equal(objective.lossBound, true);
});

test('setUi drives the tab, the view flags and the code panel', async () => {
    await runActions([{ do: 'setUi', tab: 'training', showDimensions: false, showGradientFlow: true, codeOpen: true, codeTab: 'train' }]);
    assert.equal(useEduUiStore.getState().activeTab, 'training');
    assert.equal(arch().showDimensions, false);
    assert.equal(arch().showGradientFlow, true);
    assert.equal(useEduUiStore.getState().codeOpen, true);
    assert.equal(useEduUiStore.getState().codeTab, 'train');
    await runActions([{ do: 'setUi', codeOpen: false, showDimensions: true }]);
    assert.equal(useEduUiStore.getState().codeOpen, false);
});

test('answer goes through the context; train and evaluateInit await the trainer source', async () => {
    const answers = [];
    await runActions([{ do: 'answer', id: 'demo.params', value: 24 }], { answer: (id, value) => answers.push([id, value]) });
    assert.deepEqual(answers, [['demo.params', 24]]);

    const calls = [];
    const unregister = registerSource('trainer', {
        read: () => ({ status: 'idle', seq: calls.length, runs: [] }),
        actions: {
            start: async (options) => { calls.push(['start', options]); await new Promise((resolve) => setTimeout(resolve, 5)); return { status: 'completed' }; },
            evaluateInit: async (options) => { calls.push(['init', options]); return { status: 'completed' }; },
            setSettings: (patch) => calls.push(['settings', patch]),
        },
    });
    try {
        const result = await runActions([{ do: 'setTrainer', epochs: 50 }, { do: 'train' }, { do: 'evaluateInit' }], { headless: true });
        assert.equal(result.ok, true, result.error);
        assert.deepEqual(calls, [['settings', { epochs: 50 }], ['start', { sync: true }], ['init', { sync: true }]]);
    } finally {
        unregister();
    }
    const missing = await runActions([{ do: 'train' }]);
    assert.equal(missing.ok, false, 'training without a trainer source reports an error');
});

test('useDataset calls the data source inside the same undo entry', async () => {
    const unregister = registerSource('data', {
        read: () => ({ datasetId: null }),
        actions: {
            useDataset: (id, options) => recordArchitectureEdit(`Use dataset ${id}`, () => {
                arch().patchLayerParameters('in', { shape: '(4,)' });
                return { changed: ['Input → (4,)'], warnings: [], options };
            }),
        },
    });
    try {
        const before = history().undoCount;
        const result = await runActions([
            { do: 'useDataset', id: 'iris', setOutput: false },
            { do: 'setParam', node: { id: 'lin1' }, name: 'in_features', value: 4 },
        ]);
        assert.equal(result.ok, true, result.error);
        assert.equal(graph().nodes.in.parameters.shape, '(4,)');
        assert.equal(history().undoCount, before + 1, 'the nested dataset edit merges into one entry');
    } finally {
        unregister();
    }
});

test('loadFixture replaces the workspace and clears history', async () => {
    await runActions([{ do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 5 }]);
    assert.ok(history().undoCount > 0);
    const result = await runActions([{ do: 'loadFixture', id: 'relu' }], { module: TEST_MODULE });
    assert.equal(result.ok, true, result.error);
    assert.equal(history().undoCount, 0);
    assert.ok(graph().nodes.relu, 'the fixture graph is live');
    assert.deepEqual(Object.keys(graph().nodes), Object.keys(buildFixtureSnapshot(TEST_MODULE, 'relu').graph.nodes));
    assert.equal(resolveRootNode({ type: 'Activation' }).id, 'relu');
});

// ---- Show me completes what the learner added in this step (docs/edu/SPEC.md §6.5) -------------------------------

const entryIds = () => Object.keys(graph().nodes);
const ofKey = (key) => Object.values(graph().nodes).filter((node) => node.libraryKey === key);
const chip = (key) => arch().addLayerByLibraryKey(key, { x: 400, y: 261 });

test('insertAfter wires an unconnected block added since the step began instead of adding a duplicate', async () => {
    const h = createFixtureHelpers();
    restoreWorkspace(h.snapshot({ nodes: [h.input('in', [1, 28, 28]), h.output('out', { row: 5 })], edges: [] }));
    const entryNodeIds = entryIds();
    const flatten = chip('torch.nn.Flatten');
    const result = await runActions([{ do: 'insertAfter', block: 'torch.nn.Flatten', after: { type: 'Input' } }], { entryNodeIds });
    assert.equal(result.ok, true, result.error);
    assert.deepEqual(ofKey('torch.nn.Flatten').map((node) => node.id), [flatten], 'the chip block is reused');
    assert.deepEqual(edgesOf(), [`in>${flatten}`]);
    assert.equal(graph().nodes[flatten].position.y, graph().nodes.in.position.y + 130, 'it moves into the column');
});

test('a block already wired to the anchors is completed, and the result is the same as from scratch', async () => {
    const entryNodeIds = entryIds();
    const relu = chip('torch.nn.ReLU');
    arch().connect({ source: 'lin1', target: relu });
    await runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }], { entryNodeIds });
    assert.equal(ofKey('torch.nn.ReLU').length, 1);
    assert.deepEqual(edgesOf(), ['in>lin1', `lin1>${relu}`, 'lin2>out', `${relu}>lin2`].sort());
});

test('buildChain adopts several chips in canvas order and applies the chain parameters', async () => {
    const h = createFixtureHelpers();
    restoreWorkspace(h.snapshot({ nodes: [h.input('in', [2]), h.output('out', { row: 5 })], edges: [] }));
    const entryNodeIds = entryIds();
    const first = arch().addStandardComponent('linear', { x: 700, y: 100 });
    const second = arch().addStandardComponent('linear', { x: 700, y: 400 });
    const relu = chip('torch.nn.ReLU');
    arch().connect({ source: first, target: relu });
    const result = await runActions([{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [
        { block: 'component:linear', params: { in_features: 2, out_features: 8 } }, { block: 'torch.nn.ReLU' }, { block: 'component:linear', params: { in_features: 8, out_features: 1 } },
    ] }], { entryNodeIds });
    assert.equal(result.ok, true, result.error);
    assert.equal(Object.keys(graph().nodes).length, 5, 'no duplicates');
    assert.deepEqual(edgesOf(), ['in>' + first, `${first}>${relu}`, `${relu}>${second}`, `${second}>out`].sort());
    assert.equal(graph().nodes[first].parameterValues.out_features, 8);
    assert.equal(graph().nodes[second].parameterValues.in_features, 8);
});

test('blocks from the step entry, blocks wired elsewhere and calls without entry ids are never adopted', async () => {
    // A ReLU that is part of the entry graph stays where it is.
    loadMlp({ relu: true });
    await runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'in' }, to: { id: 'lin1' } }], { entryNodeIds: entryIds() });
    assert.equal(ofKey('torch.nn.ReLU').length, 2);
    // A learner block wired to an unrelated node keeps its wire; a new block is inserted.
    loadMlp();
    const entryNodeIds = entryIds();
    const elsewhere = chip('torch.nn.ReLU');
    arch().connect({ source: elsewhere, target: 'out' });
    await runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }], { entryNodeIds });
    assert.equal(ofKey('torch.nn.ReLU').length, 2);
    assert.ok(edgesOf().includes(`${elsewhere}>out`));
    // Setup and fixture actions pass no entry ids: a loose block is left alone.
    loadMlp();
    chip('torch.nn.ReLU');
    await runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }]);
    assert.equal(ofKey('torch.nn.ReLU').length, 2);
});
