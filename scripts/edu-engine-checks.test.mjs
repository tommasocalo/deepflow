import assert from 'node:assert/strict';
import { test } from 'node:test';

// Checks are pure: they read a CheckContext built from plain graphs, so no store is needed here.
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { createCheckContext, evaluateCheck, evaluateStep, globalDiagnostics, isInitRun } = await import('../src/edu/lessons/engine/checks.js');
const { nodeType, typeMatches, orderedNodeIds, paramValue, matchNodes, parseShapeInput, sameValue } = await import('../src/edu/lessons/engine/nodes.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate, createEdge } = await import('../src/domain/architecture/index.js');
const { getEntry } = await import('../src/domain/library/index.js');
const { ensureStandardTemplate } = await import('../src/domain/components/index.js');
const { t } = await import('../src/edu/lessons/engine/i18n.js');

/** in (2,) → lin1 Linear 2→8 → [relu] → lin2 Linear 8→1 → out, with Adam + BCE bound to out. */
function mlp({ relu = true, dropout = false, outFeatures = 1, rawLinear = false } = {}) {
    const h = createFixtureHelpers();
    const input = h.input('in', [2]);
    const lin1 = rawLinear ? h.layer('lin1', 'torch.nn.Linear', { in_features: 2, out_features: 8 }, 'hidden') : h.linear('lin1', 2, 8, { label: 'hidden' });
    const middle = [];
    if (relu) middle.push(h.layer('relu', 'torch.nn.ReLU', {}, 'ReLU'));
    if (dropout) middle.push(h.layer('drop', 'torch.nn.Dropout', { p: 0.5 }, 'Dropout'));
    const lin2 = h.linear('lin2', 8, outFeatures, { label: 'score' });
    const output = h.output('out', { label: 'logit' });
    const nodes = [input, lin1, ...middle, lin2, output];
    return h.snapshot({ nodes, edges: h.chain(...nodes), training: [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss('bce', output)] });
}

const contextOf = (snapshot, extra = {}) => createCheckContext({ graph: snapshot.graph, templates: snapshot.templates, training: snapshot.training, ...extra });
const passes = (spec, ctx) => evaluateCheck(spec, ctx).pass;
const detail = (result) => (result.detail ? t(result.detail.key, result.detail.params) : '');

test('nodeType normalises layers, standard components and other templates', () => {
    const templates = {};
    const store = { getState: () => ({ templates, createTemplate: ({ name, description, graph, freeParameters, ...rest }) => { const tpl = { ...rest, ...createTemplate({ name, description, graph, freeParameters }) }; templates[tpl.id] = tpl; return tpl.id; } }) };
    const linearId = ensureStandardTemplate('linear', store);
    const attentionId = ensureStandardTemplate('attention', store);
    const headId = Object.values(templates).find((template) => template.standardKey === 'attention-head').id;
    const custom = createTemplate({ name: 'ResBlock' });
    templates[custom.id] = custom;
    const layer = (key) => createLayerNode({ libraryKey: key, entry: getEntry(key), position: { x: 0, y: 0 } });
    assert.equal(nodeType(layer('torch.nn.Linear'), templates), 'Linear');
    assert.equal(nodeType(createSuperNodeInstance({ templateId: linearId, position: { x: 0, y: 0 } }), templates), 'Linear');
    assert.equal(nodeType(layer('torch.nn.Conv2d'), templates), 'Conv2d');
    assert.equal(nodeType(layer('deepflow.op.Add'), templates), 'Add');
    assert.equal(nodeType(layer('deepflow.io.Input'), templates), 'Input');
    assert.equal(nodeType(layer('deepflow.io.Output'), templates), 'Output');
    assert.equal(nodeType(layer('deepflow.template.Input'), templates), 'TemplateInput');
    assert.equal(nodeType(layer('deepflow.nn.CausalSelfAttention'), templates), 'Attention');
    assert.equal(nodeType(layer('deepflow.nn.PositionalEmbedding'), templates), 'LearnedPositions');
    assert.equal(nodeType(layer('torch.nn.LayerNorm'), templates), 'LayerNorm');
    assert.equal(nodeType(createSuperNodeInstance({ templateId: attentionId, position: { x: 0, y: 0 } }), templates), 'Attention');
    assert.equal(nodeType(createRepeatNode({ templateId: headId, position: { x: 0, y: 0 }, mode: 'parallel' }), templates), 'AttentionHead');
    assert.equal(nodeType(createRepeatNode({ templateId: custom.id, position: { x: 0, y: 0 } }), templates), 'Component');
    assert.ok(typeMatches('ReLU', 'Activation'));
    assert.ok(typeMatches('GELU', ['Dropout', 'Activation']));
    assert.ok(!typeMatches('Softmax', 'Activation'), 'Softmax is not in the Activation group');
});

test('NodeMatch: type, id, label, nth in topological order, params after resolution', () => {
    const ctx = contextOf(mlp());
    const linears = matchNodes(ctx.graph, { type: 'Linear' }, { templates: ctx.templates, scope: ctx.scope });
    assert.deepEqual(linears.map((node) => node.id), ['lin1', 'lin2']);
    assert.equal(matchNodes(ctx.graph, { type: 'Linear', nth: 1 }, { templates: ctx.templates })[0].id, 'lin2');
    assert.equal(matchNodes(ctx.graph, { label: 'score' }, { templates: ctx.templates })[0].id, 'lin2');
    assert.equal(matchNodes(ctx.graph, { type: 'Linear', params: { out_features: '8' } }, { templates: ctx.templates, scope: ctx.scope })[0].id, 'lin1');
    assert.deepEqual(orderedNodeIds(ctx.graph), ['in', 'lin1', 'relu', 'lin2', 'out']);
    assert.equal(paramValue(ctx.graph.nodes.lin1, 'bias', { templates: ctx.templates }), true, 'component defaults come from the template');
    assert.ok(sameValue('3', 3) && sameValue([null, 4], [null, 4]) && !sameValue(null, 0));
    assert.deepEqual(parseShapeInput('B, 1, 28, 28'), [null, 1, 28, 28]);
    assert.deepEqual(parseShapeInput('(B, 8, 28, 28)'), [null, 8, 28, 28]);
    assert.deepEqual(parseShapeInput('B×784'), [null, 784]);
    assert.equal(parseShapeInput('B, foo, 2'), null);
    assert.deepEqual(parseShapeInput('B x 8'), [null, 8]);
});

test('the standard Linear component and torch.nn.Linear both match { type: Linear } for param checks', () => {
    for (const rawLinear of [false, true]) {
        const ctx = contextOf(mlp({ rawLinear }));
        assert.ok(passes({ param: { node: { type: 'Linear' }, name: 'in_features', equals: 2 } }, ctx));
        assert.ok(passes({ param: { node: { type: 'Linear', nth: 0 }, name: 'out_features', max: 8 } }, ctx));
        const failure = evaluateCheck({ param: { node: { type: 'Linear' }, name: 'out_features', equals: 3 } }, ctx);
        assert.equal(failure.pass, false);
        assert.equal(detail(failure), 'hidden: Output is 8; this step needs 3.');
    }
    const ctx = contextOf(mlp());
    assert.ok(passes({ param: { node: { type: 'Linear' }, name: 'in_features', min: 2, every: true } }, ctx));
    assert.ok(!passes({ param: { node: { type: 'Linear' }, name: 'in_features', equals: 2, every: true } }, ctx), 'lin2 has 8 inputs');
    assert.equal(detail(evaluateCheck({ param: { node: { type: 'Conv2d' }, name: 'padding', equals: 1 } }, ctx)), 'Add a Conv2d block first.');
});

test('visited, tab, answered and answeredCorrectly read the step and UI state', () => {
    const ctx = contextOf(mlp(), {
        ui: { tab: 'data' },
        step: { visitedTabs: ['architecture', 'data'], answers: { 'm1.params': { value: 3, correct: true }, 'm1.boundary': { value: 'circle', correct: false } } },
    });
    assert.ok(passes({ visited: 'data' }, ctx));
    assert.ok(!passes({ visited: 'training' }, ctx));
    assert.equal(detail(evaluateCheck({ visited: 'training' }, ctx)), 'Open the **Training** tab in the top bar.');
    assert.ok(passes({ tab: 'data' }, ctx));
    assert.ok(passes({ answered: 'm1.params' }, ctx) && passes({ answered: 'm1.boundary' }, ctx));
    assert.ok(!passes({ answered: 'm1.leak' }, ctx));
    assert.ok(passes({ answeredCorrectly: 'm1.params' }, ctx) && !passes({ answeredCorrectly: 'm1.boundary' }, ctx));
});

test('dataset and split read the data source', () => {
    const ctx = contextOf(mlp(), { data: { datasetId: 'iris', split: { val: 0.2, test: 0 } } });
    assert.ok(passes({ dataset: 'iris' }, ctx));
    const failure = evaluateCheck({ dataset: 'mnist' }, ctx);
    assert.equal(detail(failure), 'Use the **MNIST** dataset (Data tab).');
    assert.ok(passes({ split: { val: 0.2 } }, ctx) && !passes({ split: { test: 0.1 } }, ctx));
    assert.ok(!passes({ dataset: 'iris' }, contextOf(mlp(), { data: null })), 'no data source → not bound');
});

test('inputShape, outputShape and shapeAt use the validation report', () => {
    const ctx = contextOf(mlp({ outFeatures: 3 }));
    assert.ok(passes({ inputShape: [null, 2] }, ctx));
    assert.ok(passes({ outputShape: [null, 3] }, ctx));
    assert.ok(passes({ shapeAt: { node: { id: 'lin1' }, equals: [null, 8] } }, ctx));
    assert.equal(detail(evaluateCheck({ outputShape: [null, 10] }, ctx)), 'The output is (B, 3); this step needs (B, 10).');
    assert.equal(detail(evaluateCheck({ shapeAt: { node: { type: 'Linear' }, equals: [null, 4] } }, ctx)), 'hidden outputs (B, 8); this step needs (B, 4).');
    assert.ok(!passes({ inputShape: [4] }, ctx), 'the batch axis is part of the shape');
});

test('valid passes on a sound graph and fails with the friendly rewrite', () => {
    assert.ok(passes({ valid: true }, contextOf(mlp())));
    const broken = mlp();
    broken.graph.nodes.lin2 = { ...broken.graph.nodes.lin2, parameterValues: { in_features: 5, out_features: 1 } };
    const result = evaluateCheck({ valid: true }, contextOf(broken));
    assert.equal(result.pass, false);
    assert.equal(result.detail.key, 'fb.linear-in');
    assert.equal(detail(result), '**score** expects 5 input features but receives 8.');
});

test('has counts matches with exact, min/max and deep counts', () => {
    const ctx = contextOf(mlp());
    assert.ok(passes({ has: { type: 'Linear' }, count: 2 }, ctx));
    assert.ok(!passes({ has: { type: 'Linear' }, count: 3 }, ctx));
    assert.ok(passes({ has: { type: 'Activation' } }, ctx));
    assert.ok(!passes({ has: { type: 'Dropout' } }, ctx));
    assert.ok(passes({ has: { type: 'Linear' }, count: { min: 1, max: 2 } }, ctx));
    assert.ok(passes({ not: { has: { type: 'Activation' } } }, contextOf(mlp({ relu: false }))));
    assert.ok(passes({ has: { type: 'MatMul' }, deep: true }, ctx), 'deep looks inside the Linear component');
    assert.ok(!passes({ has: { type: 'MatMul' } }, ctx), 'root only by default');
});

test('global reads the root global parameter default', () => {
    const snapshot = mlp();
    snapshot.graph.parameters = [{ name: 'n_heads', type: 'int', default: 8 }, { name: 'd_model', type: 'int', default: 64 }];
    const ctx = contextOf(snapshot);
    assert.ok(passes({ global: { name: 'n_heads', equals: 8 } }, ctx));
    assert.ok(passes({ global: { name: 'd_model', oneOf: [64, 768] } }, ctx));
    assert.equal(detail(evaluateCheck({ global: { name: 'n_heads', equals: 3 } }, ctx)), 'The global parameter n_heads is 8; this step needs 3.');
    assert.ok(!passes({ global: { name: 'n_layers', equals: 12 } }, ctx));
});

test('chain follows directed paths, with and without allow', () => {
    const chain = [{ type: 'Input' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }];
    assert.ok(passes({ chain }, contextOf(mlp())));
    const missing = evaluateCheck({ chain }, contextOf(mlp({ relu: false })));
    assert.equal(missing.pass, false);
    assert.equal(detail(missing), 'After **hidden**, connect **activation (such as ReLU)**.');
    const withDropout = contextOf(mlp({ dropout: true }));
    assert.ok(!passes({ chain }, withDropout), 'Dropout breaks adjacency');
    assert.ok(passes({ chain, allow: ['Dropout'] }, withDropout));
    assert.ok(passes({ chain: [{ type: 'Activation' }, { type: 'Dropout' }] }, withDropout));
    assert.ok(!passes({ chain: [{ type: 'Linear' }, { type: 'Input' }] }, withDropout), 'direction matters');
    assert.equal(detail(evaluateCheck({ chain: [{ type: 'Flatten' }, { type: 'Linear' }] }, withDropout)), 'The model needs a **Flatten** block.');
});

test('edge checks a direct root connection', () => {
    const ctx = contextOf(mlp());
    assert.ok(passes({ edge: { from: { id: 'lin1' }, to: { type: 'Activation' } } }, ctx));
    assert.ok(!passes({ edge: { from: { id: 'lin1' }, to: { id: 'lin2' } } }, ctx));
});

test('paramCount counts the expanded graph', () => {
    const ctx = contextOf(mlp());
    assert.ok(passes({ paramCount: { equals: 2 * 8 + 8 + 8 * 1 + 1 } }, ctx));
    assert.ok(passes({ paramCount: { min: 30, max: 40 } }, ctx));
    assert.equal(detail(evaluateCheck({ paramCount: { equals: 101770 } }, ctx)), 'The model has 33 parameters; this step needs 101,770.');
});

test('training uses the effective optimizer and loss', () => {
    const ctx = contextOf(mlp());
    assert.ok(passes({ training: { loss: 'bce', lossBound: true } }, ctx));
    assert.ok(passes({ training: { optimizer: 'Adam', lr: { min: 0.01, max: 0.05 } } }, ctx));
    assert.equal(detail(evaluateCheck({ training: { loss: 'cross_entropy' } }, ctx)), 'The loss is BCE; this step needs **Cross-Entropy**.');
    const unbound = mlp();
    unbound.training.nodes.loss = { ...unbound.training.nodes.loss, architectureOutputId: null };
    assert.equal(detail(evaluateCheck({ training: { loss: 'bce', lossBound: true } }, contextOf(unbound))), 'Bind the loss to the model output in the **Training** tab.');
});

/** A fake trainer source in the shape of Stage B's read(). */
function trainer(runs, { seq = runs.length, status = 'idle' } = {}) {
    return { status, seq, settings: { epochs: 30, batchSize: 32, seed: 1 }, current: null, runs, preflight: { ok: true, errors: [], warnings: [] } };
}
function run(seq, { dataset = 'blobs', epochs = 30, valAccuracy = 0.95, trainAccuracy = 0.96, valLosses, lr = 0.03, graph, templates, training, status = 'completed', kind } = {}) {
    const history = Array.from({ length: epochs + 1 }, (_, epoch) => ({
        epoch, trainLoss: 0.7 / (epoch + 1), valLoss: valLosses?.[epoch] ?? 0.7 / (epoch + 1),
        trainAccuracy: epoch === epochs ? trainAccuracy : 0.5, valAccuracy: epoch === epochs ? valAccuracy : 0.5, ms: 1,
    }));
    return { id: `run${seq}`, seq, status, datasetId: dataset, settings: { epochs, batchSize: 32, seed: 1 }, optimizer: { type: 'Adam', hyperparameters: { learning_rate: lr } }, loss: { type: 'bce', hyperparameters: {} }, graph, templates, training, history, best: null, confusion: null, ...(kind ? { kind } : {}) };
}

test('run queries filter by entry sequence, dataset, epochs and final metrics', () => {
    const base = mlp();
    const ctx = (runs, enteredAtSeq = 0) => contextOf(base, { trainer: trainer(runs), step: { enteredAtSeq } });
    const query = { dataset: 'blobs', minEpochs: 20, final: { valAccuracy: { gte: 0.9 } } };
    assert.ok(passes({ run: query }, ctx([run(1)])));
    assert.equal(evaluateCheck({ run: query }, ctx([run(1)])).runId, 'run1', 'the satisfying run is reported for tagRun');
    assert.ok(!passes({ run: query }, ctx([run(1)], 1)), 'runs from before the step started do not count');
    assert.equal(detail(evaluateCheck({ run: query }, ctx([], 0))), 'Press **Train** to train the model.');
    assert.equal(detail(evaluateCheck({ run: query }, ctx([run(1, { epochs: 10 })]))), 'The last run trained 10 epochs; train at least 20.');
    assert.equal(detail(evaluateCheck({ run: query }, ctx([run(1, { dataset: 'circle' })]))), 'The last run used Circle; train on Two blobs.');
    assert.equal(detail(evaluateCheck({ run: query }, ctx([run(1, { valAccuracy: 0.5 })]))), 'Validation accuracy reached 50%; this step needs at least 90%.');
    assert.ok(passes({ run: query }, ctx([run(1, { valAccuracy: 0.5 }), run(2)])), 'any satisfying run since entry passes');
    assert.ok(!passes({ run: query }, ctx([run(1, { status: 'failed' })])));
    assert.ok(passes({ run: { lr: { min: 0.02 } } }, ctx([run(1)])));
});

test('run queries: initOnly, overfit and a model sub-check on the captured graph', () => {
    const base = mlp();
    const init = run(1, { epochs: 0, kind: 'init' });
    assert.ok(isInitRun(init));
    const ctx = (runs) => contextOf(base, { trainer: trainer(runs) });
    assert.ok(passes({ run: { dataset: 'blobs', initOnly: true } }, ctx([init])));
    assert.ok(!passes({ run: { dataset: 'blobs', minEpochs: 1 } }, ctx([init])), 'an init record is not a training run');
    assert.ok(!passes({ run: { initOnly: true } }, ctx([run(1)])));

    const rising = Array.from({ length: 301 }, (_, epoch) => (epoch < 50 ? 1 - epoch / 100 : 0.5 + (epoch - 50) / 250));
    const overfit = { trainAccuracyGte: 0.97, valLossRiseGte: 1.25 };
    assert.ok(passes({ run: { minEpochs: 300, overfit } }, ctx([run(1, { epochs: 300, trainAccuracy: 0.99, valLosses: rising })])));
    assert.ok(!passes({ run: { overfit } }, ctx([run(1, { epochs: 300, trainAccuracy: 0.9, valLosses: rising })])));
    assert.ok(!passes({ run: { overfit } }, ctx([run(1, { epochs: 300, trainAccuracy: 0.99 })])), 'monotone falling validation loss has no rise');

    const linear = mlp({ relu: false });
    const model = { all: [{ has: { type: 'Linear' }, count: 2 }, { not: { has: { type: 'Activation' } } }] };
    const captured = run(1, { graph: linear.graph, templates: linear.templates, training: linear.training });
    assert.ok(passes({ run: { model } }, ctx([captured])), 'evaluated on the run’s graph, not the live one');
    const withRelu = run(2, { graph: base.graph, templates: base.templates, training: base.training });
    assert.ok(!passes({ run: { model } }, ctx([withRelu])));
});

test('runs covers every listed learning rate within 1 %', () => {
    const base = mlp();
    const ctx = (runs) => contextOf(base, { trainer: trainer(runs) });
    const spec = { count: 3, lr: [0.0001, 0.01, 1], dataset: 'blobs', minEpochs: 20 };
    assert.ok(passes({ runs: spec }, ctx([run(1, { lr: 0.0001 }), run(2, { lr: 0.01 }), run(3, { lr: 1.005 })])));
    const partial = evaluateCheck({ runs: spec }, ctx([run(1, { lr: 0.0001 }), run(2, { lr: 0.01 })]));
    assert.equal(partial.pass, false);
    assert.equal(detail(partial), 'Still to try: learning rate 1.');
    assert.ok(!passes({ runs: spec }, ctx([run(1, { lr: 0.0001 }), run(2, { lr: 0.01 }), run(3, { lr: 1, epochs: 5 })])), 'minEpochs applies');
});

test('level, ui, repeat and component predicates', () => {
    const snapshot = mlp();
    assert.ok(passes({ level: { open: { type: 'Linear' } } }, contextOf(snapshot, { ui: { currentPath: ['lin1'] } })));
    assert.ok(!passes({ level: { open: { type: 'Linear' } } }, contextOf(snapshot, { ui: { currentPath: [] } })));

    const ui = contextOf(snapshot, { ui: { showDimensions: false, showGradientFlow: true, codeOpen: true, codeTab: 'train' } });
    assert.ok(passes({ ui: { showGradientFlow: true, codeOpen: true, codeTab: 'train' } }, ui));
    assert.equal(detail(evaluateCheck({ ui: { showDimensions: true } }, ui)), 'Turn on **Dimensions** in the ribbon (View).');
    assert.ok(!passes({ ui: { codeTab: 'model' } }, ui));

    const block = createTemplate({ name: 'ResBlock', graph: {
        nodes: Object.fromEntries(['torch.nn.Conv2d', 'torch.nn.Conv2d', 'torch.nn.BatchNorm2d', 'deepflow.op.Add'].map((key, index) => {
            const node = createLayerNode({ libraryKey: key, entry: getEntry(key), position: { x: 0, y: index * 130 }, id: `n${index}` });
            return [node.id, node];
        })), edges: [],
    } });
    const repeated = createRepeatNode({ id: 'blocks', templateId: block.id, position: { x: 400, y: 260 }, count: '$n', mode: 'sequential' });
    const graph = { nodes: { ...snapshot.graph.nodes, blocks: repeated }, edges: snapshot.graph.edges, parameters: [{ name: 'n', type: 'int', default: 3 }] };
    const ctx = createCheckContext({ graph, templates: { ...snapshot.templates, [block.id]: block } });
    assert.ok(passes({ repeat: { node: { type: 'Component' }, mode: 'sequential', count: 3 } }, ctx));
    assert.equal(detail(evaluateCheck({ repeat: { node: { type: 'Component' }, count: 4 } }, ctx)), 'Set the repeat count to 4 (now 3).');
    assert.ok(!passes({ repeat: { node: { type: 'Component' }, mode: 'parallel' } }, ctx));
    assert.ok(passes({ component: { contains: [{ type: 'Conv2d', count: 2 }, { type: 'BatchNorm2d', count: 1 }, { type: 'Add' }] } }, ctx));
    assert.ok(!passes({ component: { contains: [{ type: 'BatchNorm2d', count: 2 }] } }, ctx));
    assert.ok(!passes({ component: { contains: [{ type: 'Conv2d' }], kind: 'supernode' } }, ctx), 'kind narrows to supernodes');
});

test('custom checks, combinators and unknown specs', () => {
    const module = { checks: { always: () => true, never: () => ({ pass: false, detail: { key: 'check.custom', params: {} } }), boom: () => { throw new Error('x'); } } };
    const ctx = contextOf(mlp(), { module });
    assert.ok(passes({ custom: 'always' }, ctx));
    assert.ok(!passes({ custom: 'never' }, ctx));
    assert.ok(!passes({ custom: 'boom' }, ctx), 'a throwing custom check fails instead of crashing');
    assert.ok(!passes({ custom: 'missing' }, ctx));
    assert.ok(passes({ any: [{ custom: 'never' }, { custom: 'always' }] }, ctx));
    assert.ok(!passes({ all: [{ custom: 'always' }, { custom: 'never' }] }, ctx));
    assert.equal(evaluateCheck({ nonsense: 1 }, ctx).detail.key, 'check.unknown');
});

test('global diagnostics: Softmax before cross-entropy and Sigmoid before BCE', () => {
    const h = createFixtureHelpers();
    const input = h.input('in', [4]);
    const linear = h.linear('lin', 4, 3);
    const softmax = h.layer('soft', 'torch.nn.Softmax', {}, 'Softmax');
    const output = h.output('out');
    const snapshot = h.snapshot({ nodes: [input, linear, softmax, output], edges: h.chain(input, linear, softmax, output), training: [h.optimizer(), h.loss('cross_entropy', output)] });
    const ctx = contextOf(snapshot);
    assert.equal(globalDiagnostics(ctx), 'softmax-before-ce');
    const step = { kind: 'task', check: { outputShape: [null, 4] } };
    assert.equal(evaluateStep(step, ctx).misconception, 'softmax-before-ce');
    assert.equal(evaluateStep({ ...step, kind: 'predict' }, ctx).misconception, null, 'only task steps get global diagnostics');

    const h2 = createFixtureHelpers();
    const in2 = h2.input('in', [2]);
    const lin2 = h2.linear('lin', 2, 1);
    const sigmoid = h2.layer('sig', 'torch.nn.Sigmoid', {}, 'Sigmoid');
    const out2 = h2.output('out');
    const binary = h2.snapshot({ nodes: [in2, lin2, sigmoid, out2], edges: h2.chain(in2, lin2, sigmoid, out2), training: [h2.optimizer(), h2.loss('bce', out2)] });
    assert.equal(globalDiagnostics(contextOf(binary)), 'sigmoid-before-bce');
    assert.equal(globalDiagnostics(contextOf(mlp())), null);
});

test('evaluateStep: diagnose rules while failing and the friendly issue for graph predicates', () => {
    const snapshot = mlp();
    const step = {
        kind: 'task',
        check: { all: [{ param: { node: { id: 'lin2' }, name: 'out_features', equals: 3 } }, { valid: true }] },
        diagnose: [{ when: { param: { node: { id: 'lin2' }, name: 'out_features', equals: 1 } }, misconception: 'one-output-multiclass' }],
    };
    const failing = evaluateStep(step, contextOf(snapshot));
    assert.equal(failing.pass, false);
    assert.equal(failing.misconception, 'one-output-multiclass');
    assert.equal(failing.issue, null, 'a valid graph has no friendly issue');

    const broken = mlp();
    broken.graph.nodes.lin2 = { ...broken.graph.nodes.lin2, parameterValues: { in_features: 5, out_features: 3 } };
    const invalid = evaluateStep(step, contextOf(broken));
    assert.equal(invalid.issue.key, 'fb.linear-in');
    assert.equal(invalid.issue.nodeId, 'lin2');

    const fixed = mlp({ outFeatures: 3 });
    assert.equal(evaluateStep(step, contextOf(fixed)).pass, true);
    assert.ok(!evaluateStep({ kind: 'predict', check: { answered: 'x' } }, contextOf(broken)).issue, 'answer-type failures keep their own detail');
});

test('derived parts of a context are computed lazily and only once', () => {
    const snapshot = mlp();
    const ctx = createCheckContext({ graph: snapshot.graph, templates: snapshot.templates, training: snapshot.training });
    assert.equal(typeof Object.getOwnPropertyDescriptor(ctx, 'paramCount').get, 'function');
    const first = ctx.paramCount;
    assert.equal(ctx.paramCount, first, 'the cached value is reused');
    assert.ok(ctx.report.ok);
    assert.equal(createEdge({ source: 'a', target: 'b' }).gradientOp, 'none');
});
