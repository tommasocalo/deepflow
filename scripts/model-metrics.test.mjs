import assert from 'node:assert/strict';
import { test } from 'node:test';
import { validate } from '../src/validation/validator.js';
import { countArchitectureParameters, layerSize } from '../src/domain/optimization/parameterCount.js';
import { cachedModelMetrics, computeModelMetrics, formatBytes, formatCount, formatMacs, formatShare, instanceTotals, layerMacs, subtreeMetrics } from '../src/domain/metrics/index.js';
import { createEdge, createLayerNode, createRepeatNode, createTemplate, moveNode, patchLayerParameters } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { gpt2Storyboard } from '../src/storyboards/gpt2.js';
import { resnetStoryboard } from '../src/storyboards/resnet.js';
import { ganStoryboard } from '../src/storyboards/gan.js';

const storage = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
} };
const { useArchitectureStore: architectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: trainingStore } = await import('../src/stores/trainingStore.js');

function load(storyboard) {
    for (const store of [architectureStore, templatesStore, trainingStore]) store.setState(store.getInitialState(), true);
    storage.clear();
    storyboard({ architectureStore, templatesStore, trainingStore });
    return { graph: architectureStore.getState().graph, templates: templatesStore.getState().templates };
}
const byLabel = (level, label) => {
    const entry = Object.values(level.nodes).find((candidate) => candidate.node.label === label);
    assert.ok(entry, `Expected a node labelled ${label}`);
    return entry;
};
const total = (entry, metric) => subtreeMetrics(entry)[metric];
const layer = (libraryKey, parameters = {}) => {
    const entry = getEntry(libraryKey);
    const node = createLayerNode({ libraryKey, entry, position: { x: 0, y: 0 } });
    return { ...node, parameters: { ...node.parameters, ...parameters } };
};
const chain = (...nodes) => ({
    nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
    edges: nodes.slice(1).map((node, index) => createEdge({ source: nodes[index].id, target: node.id })),
});

test('GPT-2 small (untied lm_head) reproduces reference parameters, MACs and activation sizes', () => {
    const { graph, templates } = load(gpt2Storyboard);
    const metrics = computeModelMetrics(graph, templates);
    const { totals } = metrics;
    assert.equal(totals.params, 163037184);
    assert.equal(totals.trainable, 163037184);
    assert.equal(countArchitectureParameters(graph, templates).total, 163037184, 'The existing counter agrees');
    assert.equal(totals.macs, 145824153600, 'B = 1, T = 1024: MatMul, Linear and Conv multiply-accumulates');
    assert.equal(totals.activation, 4034203648);
    assert.deepEqual([totals.largest.bytes, totals.largest.label, totals.largest.shape], [205852672, 'lm_head', [null, 1024, 50257]]);
    assert.equal(totals.params * 4, 652148736, 'fp32 weights');
    assert.deepEqual([totals.unresolved.params.size, totals.unresolved.macs.size, totals.unresolved.activation.size], [0, 0, 0]);

    const root = metrics.root;
    for (const [label, params] of [['token embedding', 38597376], ['positional embedding', 786432], ['lm_head', 38597376], ['ln_f', 1536]]) {
        assert.equal(total(byLabel(root, label), 'params'), params, label);
    }
    assert.equal(total(byLabel(root, 'token embedding'), 'macs'), 0, 'Embedding lookups are not multiply-accumulates');
    assert.equal(total(byLabel(root, 'positional embedding'), 'macs'), 0);
    assert.equal(total(byLabel(root, 'lm_head'), 'macs'), 39523713024);

    const blocks = byLabel(root, 'Transformer blocks');
    assert.equal(blocks.instances.length, 12);
    assert.equal(total(blocks, 'params'), 85054464);
    assert.equal(total(blocks, 'macs'), 106300440576);
    const perBlock = instanceTotals(blocks);
    assert.ok(perBlock.every((block) => block.params === 7087872 && block.macs === 8858370048), 'Every block has the same size');
    assert.equal(perBlock[0].activation, 317718528);

    const block = blocks.instances[0].level;
    for (const [label, params, macs] of [
        ['ln_1', 1536, 0], ['causal self-attention', 2362368, 1811939328 + 805306368 + 805306368 + 603979776],
        ['mlp fc', 2362368, 2415919104], ['mlp proj', 2360064, 2415919104],
    ]) {
        assert.equal(total(byLabel(block, label), 'params'), params, `${label} params`);
        assert.equal(total(byLabel(block, label), 'macs'), macs, `${label} MACs`);
    }
    const attention = byLabel(block, 'causal self-attention').instances[0].level;
    const qkv = ['Q projection', 'K projection', 'V projection'].reduce((sum, label) => sum + total(byLabel(attention, label), 'macs'), 0);
    assert.equal(qkv, 1811939328);
    assert.equal(total(byLabel(attention, 'output projection'), 'macs'), 603979776);
    const heads = byLabel(attention, 'Attention heads');
    assert.equal(heads.instances.length, 12);
    const headSum = (label) => heads.instances.reduce((sum, instance) => sum + byLabel(instance.level, label).macs, 0);
    assert.equal(headSum('Q × Kᵀ'), 805306368, 'Causal attention counts the full T × T score product');
    assert.equal(headSum('weights × V'), 805306368);
    assert.equal(total(heads, 'activation'), 254803968, 'Attention internals count individually');
    assert.equal(metrics.templateUses[blocks.node.templateId], 12);
    assert.equal(metrics.templateUses[heads.node.templateId], 144);
});

test('ResNet stages count 3 × 3 kernels after the scalar-kernel fix', () => {
    const { graph, templates } = load(resnetStoryboard);
    const metrics = computeModelMetrics(graph, templates);
    assert.equal(metrics.totals.params, 371264);
    assert.equal(countArchitectureParameters(graph, templates).total, 371264, 'Conv2d kernel_size 3 is a 3 × 3 kernel');
    assert.equal(metrics.totals.macs, 379256832);
    assert.equal(metrics.totals.activation, 5505024);
    assert.equal(metrics.totals.largest.bytes, 1048576);
    const stages = instanceTotals(byLabel(metrics.root, 'ResNet stages'));
    assert.deepEqual(stages.map((stage) => stage.params), [1856, 73984, 295424]);
    assert.deepEqual(stages.map((stage) => stage.macs), [1769472, 75497472, 301989888]);
});

test('layer sizes cover convolution kernels, transposed convolutions, frozen and disabled parameters', () => {
    assert.deepEqual(layerSize(layer('torch.nn.Conv2d', { in_channels: 3, out_channels: 64, kernel_size: 3 }), {}), { total: 1792, trainable: 1792 });
    assert.deepEqual(layerSize(layer('torch.nn.Conv2d', { in_channels: 3, out_channels: 64, kernel_size: '(3,3)' }), {}), { total: 1792, trainable: 1792 });
    assert.equal(layerSize(layer('torch.nn.Conv2d', { in_channels: 3, out_channels: 64, kernel_size: '$k' }), { k: 5 }).total, 3 * 64 * 25 + 64);
    assert.equal(layerSize(layer('torch.nn.Conv2d', { in_channels: 3, out_channels: 64, kernel_size: '$missing' }), {}), null, 'An unresolved kernel is not the default kernel');
    assert.deepEqual(layerSize(layer('torch.nn.Conv1d', { in_channels: 3, out_channels: 64, kernel_size: 3 }), {}), { total: 640, trainable: 640 });
    assert.deepEqual(layerSize(layer('torch.nn.ConvTranspose2d', { in_channels: 64, out_channels: 32, kernel_size: 4 }), {}), { total: 32800, trainable: 32800 });
    assert.deepEqual(layerSize(layer('deepflow.op.Parameter', { shape: [768, 3072], requires_grad: false }), {}), { total: 2359296, trainable: 0 });
    assert.deepEqual(layerSize(layer('deepflow.op.Parameter', { shape: [768], enabled: false }), {}), { total: 0, trainable: 0 });
    assert.deepEqual(layerSize(layer('torch.nn.LeakyReLU'), {}), { total: 0, trainable: 0 });
    const conv1d = layer('torch.nn.Conv1d', { in_channels: 3, out_channels: 8, kernel_size: 3 });
    assert.equal(layerMacs(conv1d, {}, [[null, 3, 32]], [null, 3, 32]), null, 'Conv1d has no shape rule, so its MACs stay unknown');

    const { graph, templates } = load(ganStoryboard);
    const count = countArchitectureParameters(graph, templates);
    assert.equal(count.complete, true, 'LeakyReLU has no weights');
    assert.equal(count.total, 210065);
    assert.equal(computeModelMetrics(graph, templates).totals.params, 210065);
});

test('unknown modules and counts give partial totals, never NaN', () => {
    const input = layer('deepflow.io.Input', { shape: '(16, 32)' });
    const lstm = layer('torch.nn.LSTM', { input_size: 32, hidden_size: 64 });
    const linear = layer('torch.nn.Linear', { in_features: 32, out_features: 10 });
    const output = layer('deepflow.io.Output');
    const partial = computeModelMetrics(chain(input, lstm, linear, output), {});
    assert.equal(partial.totals.params, 330, 'Only the Linear weights are known');
    assert.equal(partial.totals.unresolved.params.size, 1);
    assert.deepEqual([...partial.totals.unresolved.params.values()], ['LSTM']);
    assert.equal(partial.totals.unresolved.macs.size, 1);
    assert.deepEqual([...partial.totals.unresolved.activation.values()], ['LSTM'], 'An LSTM output has no shape rule, so its size is unknown');

    const body = createTemplate({ name: 'Body', graph: chain(layer('deepflow.template.Input'), layer('torch.nn.Linear', { in_features: 32, out_features: 32 }), layer('deepflow.template.Output')) });
    const repeat = createRepeatNode({ templateId: body.id, position: { x: 0, y: 0 }, count: '$missing', label: 'Stack' });
    const unresolvedRepeat = computeModelMetrics(chain(layer('deepflow.io.Input', { shape: '(32,)' }), repeat, layer('deepflow.io.Output')), { [body.id]: body });
    const stack = Object.values(unresolvedRepeat.root.nodes).find((entry) => entry.node.label === 'Stack');
    assert.equal(stack.instances.length, 0);
    assert.equal(stack.complete, false);
    assert.deepEqual([...unresolvedRepeat.totals.unresolved.params.values()], ['Stack']);

    const everyNumber = (level) => {
        for (const entry of Object.values(level.nodes)) {
            const totals = subtreeMetrics(entry);
            for (const metric of ['params', 'trainable', 'macs', 'activation']) assert.ok(Number.isFinite(totals[metric]), `${entry.node.label} ${metric}`);
            if (entry.kind === 'layer') {
                for (const value of [entry.macs, entry.activation, entry.params?.total ?? null]) assert.ok(value === null || Number.isFinite(value));
            }
            for (const instance of entry.instances ?? []) if (instance.level) everyNumber(instance.level);
        }
    };
    everyNumber(partial.root);
    everyNumber(unresolvedRepeat.root);
    everyNumber(computeModelMetrics(load(gpt2Storyboard).graph, templatesStore.getState().templates).root);
    for (const text of [formatCount(null), formatMacs(Number.NaN), formatBytes(undefined), formatShare(Number.NaN)]) assert.ok(!String(text).includes('NaN'));
});

test('model metrics are cached across position-only edits and recomputed for parameter edits', () => {
    const { graph, templates } = load(gpt2Storyboard);
    const first = cachedModelMetrics(graph, templates);
    const dropout = Object.values(graph.nodes).find((node) => node.libraryKey === 'torch.nn.Dropout');
    assert.equal(cachedModelMetrics(moveNode(graph, dropout.id, { x: 10, y: 20 }), templates), first, 'Moving a node reuses the metrics');
    const edited = cachedModelMetrics(patchLayerParameters(graph, dropout.id, { p: 0.2 }), templates);
    assert.notEqual(edited, first, 'A parameter edit recomputes the metrics');
    assert.equal(edited.totals.params, first.totals.params);
    const wider = { ...graph, parameters: graph.parameters.map((parameter) => parameter.name === 'd_model' ? { ...parameter, default: 1024 } : parameter) };
    const resized = cachedModelMetrics(wider, templates);
    assert.notEqual(resized, first, 'A global parameter edit recomputes the metrics');
    assert.ok(resized.totals.params > first.totals.params);
});

test('validate collects instance reports only on request', () => {
    const strip = (report) => {
        const { scope: _scope, children: _children, ...rest } = report;
        return rest;
    };
    for (const storyboard of [gpt2Storyboard, resnetStoryboard]) {
        const { graph, templates } = load(storyboard);
        const plain = validate(graph, { templates });
        assert.equal('children' in plain || 'scope' in plain, false, 'Default reports keep their shape');
        assert.deepEqual(strip(validate(graph, { templates, collect: true })), plain);
    }
    const { graph, templates } = load(gpt2Storyboard);
    const metrics = computeModelMetrics(graph, templates);
    const attention = byLabel(byLabel(metrics.root, 'Transformer blocks').instances[0].level, 'causal self-attention');
    assert.equal(byLabel(attention.instances[0].level, 'Attention heads').instances.length, 12);

    const resnet = load(resnetStoryboard);
    const stages = byLabel(computeModelMetrics(resnet.graph, resnet.templates).root, 'ResNet stages');
    assert.deepEqual(stages.instances.map((instance) => instance.scope.in), [3, 64, 128]);
    assert.deepEqual(stages.instances.map((instance) => instance.scope.i), [0, 1, 2]);
});

test('metric formatting stays compact and labelled', () => {
    assert.deepEqual([163037184, 7087872, 786432, 1536, 768, 999999].map(formatCount), ['163.0M', '7.09M', '786.4K', '1.54K', '768', '1.00M']);
    assert.deepEqual([145824153600, 2415919104].map(formatMacs), ['145.8G', '2.42G']);
    assert.deepEqual([205852672, 4034203648, 652].map(formatBytes), ['205.9 MB', '4.03 GB', '652 B']);
    assert.deepEqual([0.5217, 0.0000941, 0, null, 1].map(formatShare), ['52.2%', '<0.1%', '', '', '100%']);
});
