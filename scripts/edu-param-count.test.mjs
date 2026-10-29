import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

// docs/edu/SPEC.md §6.11.4 and §11.3: the Conv scalar-kernel fix and per-node parameter counts.
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key) };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { countArchitectureParameters, countParametersByNode } = await import('../src/domain/optimization/parameterCount.js');
const { createEdge, createLayerNode } = await import('../src/domain/architecture/index.js');
const { getEntry } = await import('../src/domain/library/index.js');
const { validate } = await import('../src/validation/validator.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore } = await import('../src/stores/trainingStore.js');
const { gpt2Storyboard } = await import('../src/storyboards/gpt2.js');

const stores = { architectureStore: useArchitectureStore, templatesStore: useTemplatesStore, trainingStore: useTrainingStore };

beforeEach(() => {
    for (const store of Object.values(stores)) store.setState(store.getInitialState(), true);
    memory.clear();
});

/** Input → layers… → Output as plain layer nodes (no stores). */
function chain(layers, inputShape) {
    const nodes = {};
    const edges = [];
    let previous = null;
    const add = (key, parameters = {}, id = `n${Object.keys(nodes).length}`) => {
        const node = createLayerNode({ id, libraryKey: key, entry: getEntry(key), position: { x: 400, y: Object.keys(nodes).length * 130 } });
        node.parameters = { ...node.parameters, ...parameters };
        nodes[node.id] = node;
        if (previous) edges.push(createEdge({ source: previous.id, target: node.id }));
        previous = node;
        return node;
    };
    add('deepflow.io.Input', { shape: inputShape }, 'in');
    for (const [key, parameters, id] of layers) add(key, parameters, id);
    add('deepflow.io.Output', {}, 'out');
    return { nodes, edges, parameters: [] };
}

const total = (graph, templates = {}) => countArchitectureParameters(graph, templates);
const sumByNode = (byNode) => Object.values(byNode).reduce((sum, entry) => sum + entry.total, 0);

/** The M4 CNN of SPEC §4.4 (8 and 16 filters, 3 × 3, padding 1, two 2 × 2 pools, Linear 784 → 10). */
function m4Cnn(kernel) {
    return chain([
        ['torch.nn.Conv2d', { in_channels: 1, out_channels: 8, kernel_size: kernel, padding: 1 }, 'conv1'],
        ['torch.nn.ReLU', {}, 'relu1'],
        ['torch.nn.MaxPool2d', { kernel_size: 2 }, 'pool1'],
        ['torch.nn.Conv2d', { in_channels: 8, out_channels: 16, kernel_size: kernel, padding: 1 }, 'conv2'],
        ['torch.nn.ReLU', {}, 'relu2'],
        ['torch.nn.MaxPool2d', { kernel_size: 2 }, 'pool2'],
        ['torch.nn.Flatten', {}, 'flat'],
        ['torch.nn.Linear', { in_features: 784, out_features: 10 }, 'fc'],
    ], '(1, 28, 28)');
}

/** The M6 model of SPEC §4.6 built with the editor's own actions: stem, one residual block grouped, then repeated. */
function m6Model({ repeat = 3 } = {}) {
    const arch = useArchitectureStore.getState();
    arch.resetCurrentGraph();
    const graph0 = useArchitectureStore.getState().graph;
    const input = Object.values(graph0.nodes).find((node) => node.libraryKey === 'deepflow.io.Input').id;
    const output = Object.values(graph0.nodes).find((node) => node.libraryKey === 'deepflow.io.Output').id;
    arch.patchLayerParameters(input, { shape: '(3, 32, 32)' });
    let y = 0;
    const add = (key, parameters) => {
        y += 130;
        const id = arch.addLayerByLibraryKey(key, { x: 400, y });
        arch.patchLayerParameters(id, parameters);
        return id;
    };
    // Scalar kernel_size 3, as the palette gives learners (SPEC §4.0).
    const stem = add('torch.nn.Conv2d', { in_channels: 3, out_channels: 32, kernel_size: 3, padding: 1 });
    const bn0 = add('torch.nn.BatchNorm2d', { num_features: 32 });
    const relu0 = add('torch.nn.ReLU', {});
    const c1 = add('torch.nn.Conv2d', { in_channels: 32, out_channels: 32, kernel_size: 3, padding: 1, bias: false });
    const b1 = add('torch.nn.BatchNorm2d', { num_features: 32 });
    const r1 = add('torch.nn.ReLU', {});
    const c2 = add('torch.nn.Conv2d', { in_channels: 32, out_channels: 32, kernel_size: 3, padding: 1, bias: false });
    const b2 = add('torch.nn.BatchNorm2d', { num_features: 32 });
    const sum = add('deepflow.op.Add', {});
    const r2 = add('torch.nn.ReLU', {});
    const pool = add('torch.nn.AdaptiveAvgPool2d', { output_size: 1 });
    const flat = add('torch.nn.Flatten', {});
    const fc = add('torch.nn.Linear', { in_features: 32, out_features: 10 });
    for (const [source, target] of [[input, stem], [stem, bn0], [bn0, relu0], [relu0, c1], [c1, b1], [b1, r1], [r1, c2], [c2, b2],
        [b2, sum], [relu0, sum], [sum, r2], [r2, pool], [pool, flat], [flat, fc], [fc, output]]) arch.connect({ source, target });
    const ids = { input, output, stem, bn0, relu0, fc };
    if (repeat) {
        arch.groupSelection([c1, b1, r1, c2, b2, sum, r2], 'ResBlock');
        const instance = Object.values(useArchitectureStore.getState().graph.nodes).find((node) => node.kind === 'supernode');
        ids.block = instance.id;
        if (repeat > 1) ids.block = useArchitectureStore.getState().convertInstanceToRepeat(instance.id, repeat, 'sequential');
    }
    return { graph: useArchitectureStore.getState().graph, templates: useTemplatesStore.getState().templates, ids };
}

test('Conv2d(1 → 8, kernel 3) has 80 parameters with a scalar or a tuple kernel', () => {
    for (const kernel of [3, '3', '(3, 3)', '3, 3']) {
        const graph = chain([['torch.nn.Conv2d', { in_channels: 1, out_channels: 8, kernel_size: kernel }]], '(1, 28, 28)');
        const count = total(graph);
        assert.equal(count.complete, true);
        assert.equal(count.total, 80, `kernel_size ${JSON.stringify(kernel)}: 8 × 1 × 3 × 3 + 8`);
    }
    const rectangular = chain([['torch.nn.Conv2d', { in_channels: 1, out_channels: 8, kernel_size: '(3, 5)', bias: false }]], '(1, 28, 28)');
    assert.equal(total(rectangular).total, 120, 'a tuple is used as given: 8 × 3 × 5');
    const grouped = chain([['torch.nn.Conv2d', { in_channels: 4, out_channels: 8, kernel_size: 3, groups: 2 }]], '(4, 28, 28)');
    assert.equal(total(grouped).total, 8 * 2 * 9 + 8, 'groups divide the input channels; the scalar kernel still covers 3 × 3');
});

test('Conv1d keeps a scalar kernel on one axis and Conv3d expands it to three', () => {
    const conv1d = chain([['torch.nn.Conv1d', { in_channels: 4, out_channels: 2, kernel_size: 3 }]], '(4, 16)');
    assert.equal(total(conv1d).total, 2 * 4 * 3 + 2);
    const conv3d = chain([['torch.nn.Conv3d', { in_channels: 1, out_channels: 2, kernel_size: 3, bias: true }]], '(1, 8, 8, 8)');
    assert.equal(total(conv3d).total, 2 * 27 + 2);
});

test('the M4 CNN has 9,098 parameters with the palette\'s scalar kernel (8,282 before the fix)', () => {
    for (const kernel of [3, '(3, 3)']) {
        const graph = m4Cnn(kernel);
        const report = validate(graph, { templates: {} });
        assert.equal(report.ok, true);
        assert.deepEqual(report.shapes.conv1, [null, 8, 28, 28]);
        assert.deepEqual(report.shapes.pool2, [null, 16, 7, 7]);
        assert.deepEqual(total(graph), { total: 9098, trainable: 9098, complete: true, unsupported: [] });
        const byNode = countParametersByNode(graph, {});
        assert.equal(byNode.conv1.total, 80);
        assert.equal(byNode.conv2.total, 1168);
        assert.equal(byNode.fc.total, 7850);
        assert.equal(byNode.relu1.total, 0);
        assert.equal(sumByNode(byNode), 9098);
    }
});

test('the M6 model: one ResBlock has 18,560 parameters and three repeated blocks give 56,970', () => {
    const one = m6Model({ repeat: 1 });
    assert.equal(validate(one.graph, { templates: one.templates }).ok, true);
    assert.equal(total(one.graph, one.templates).total, 19850, 'stem 896 + BatchNorm 64 + block 18,560 + Linear 330');
    assert.equal(countParametersByNode(one.graph, one.templates)[one.ids.block].total, 18560);

    const three = m6Model({ repeat: 3 });
    const report = validate(three.graph, { templates: three.templates });
    assert.equal(report.ok, true);
    assert.deepEqual(report.shapes[three.ids.output], [null, 10]);
    const count = total(three.graph, three.templates);
    assert.equal(count.complete, true);
    assert.equal(count.total, 56970);
    const byNode = countParametersByNode(three.graph, three.templates);
    assert.equal(byNode[three.ids.stem].total, 896);
    assert.equal(byNode[three.ids.bn0].total, 64);
    assert.equal(byNode[three.ids.block].total, 3 * 18560, 'each copy has its own weights');
    assert.equal(byNode[three.ids.fc].total, 330);
    assert.equal(sumByNode(byNode), 56970);
    assert.ok(Object.values(byNode).every((entry) => entry.complete && entry.trainable === entry.total));
});

test('GPT-2 is unchanged (163,037,184) and its per-node counts add up', () => {
    gpt2Storyboard(stores);
    const graph = useArchitectureStore.getState().graph;
    const templates = useTemplatesStore.getState().templates;
    const count = total(graph, templates);
    assert.equal(count.complete, true);
    assert.equal(count.total, 163037184);
    const byNode = countParametersByNode(graph, templates);
    assert.deepEqual(Object.keys(byNode).sort(), Object.keys(graph.nodes).sort(), 'one entry per root node');
    assert.equal(sumByNode(byNode), 163037184);
    const blocks = Object.values(graph.nodes).find((node) => node.kind === 'repeat');
    assert.equal(byNode[blocks.id].total, 12 * 7087872, '12 Transformer blocks of 7,087,872 parameters');
});

test('the M7 tiny GPT has 112,512 parameters', () => {
    gpt2Storyboard({ ...stores, configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4, nLayers: 2, mlpWidth: 256 } });
    const graph = useArchitectureStore.getState().graph;
    const templates = useTemplatesStore.getState().templates;
    assert.equal(total(graph, templates).total, 112512);
    assert.equal(sumByNode(countParametersByNode(graph, templates)), 112512);
});

test('countParametersByNode marks only the node whose size is unknown as incomplete', () => {
    const graph = chain([
        ['torch.nn.Linear', { in_features: 4, out_features: 3 }, 'lin'],
        ['torch.nn.LSTM', { input_size: 3, hidden_size: 3 }, 'lstm'],
    ], '(4,)');
    const count = total(graph);
    assert.equal(count.complete, false);
    const byNode = countParametersByNode(graph, {});
    assert.deepEqual(byNode.lin, { total: 15, trainable: 15, complete: true });
    assert.equal(byNode.lstm.complete, false);
    assert.equal(byNode.in.complete, true);
    assert.equal(byNode.in.total, 0);
});

test('frozen parameters count towards the total but not the trainable count', () => {
    const graph = chain([['deepflow.op.Parameter', { shape: '(3, 4)', requires_grad: false }, 'weight']], '(4,)');
    graph.edges = graph.edges.filter((edge) => edge.target !== 'weight');
    const byNode = countParametersByNode(graph, {});
    assert.deepEqual(byNode.weight, { total: 12, trainable: 0, complete: true });
});
