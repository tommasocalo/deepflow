import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeGradientFlow } from '../src/domain/gradientFlow.js';

const layer = (id, libraryKey, parameters = {}) => ({ id, kind: 'layer', libraryKey, parameters });
const edge = (source, target, rest = {}) => ({ id: `${source}_${target}`, source, target, ...rest });
const graph = (nodes, edges) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges });

test('integer architecture inputs are constant index tensors', () => {
    for (const dtype of ['int32', 'int64']) {
        const report = analyzeGradientFlow(graph([
            layer('input', 'deepflow.io.Input', { dtype }), layer('output', 'deepflow.io.Output'),
        ], [edge('input', 'output')]));
        assert.equal(report.nodes.input.dtype, 'index');
        assert.equal(report.nodes.input.requires, 'no');
        assert.equal(report.edges.input_output.status, 'constant');
    }
});

test('opaque LayerNorm without affine terms preserves input gradient requirements', () => {
    const report = analyzeGradientFlow(graph([
        layer('input', 'deepflow.io.Input', { requires_grad: false }),
        layer('norm', 'torch.nn.LayerNorm', { normalized_shape: 4, elementwise_affine: false }),
        layer('output', 'deepflow.io.Output'),
    ], [edge('input', 'norm'), edge('norm', 'output')]));
    assert.equal(report.nodes.norm.requires, 'no');
    assert.equal(report.edges.norm_output.status, 'constant');
});

test('default Split and Broadcast produce two differentiable collection members', () => {
    for (const key of ['deepflow.op.Split', 'deepflow.op.Broadcast']) {
        const report = analyzeGradientFlow(graph([
            layer('parameter', 'deepflow.op.Parameter', { shape: [4] }),
            layer('collection', key), layer('output', 'deepflow.io.Output'),
        ], [edge('parameter', 'collection'), edge('collection', 'output')]));
        assert.equal(report.nodes.collection.items?.length, 2);
        assert.ok(report.nodes.collection.items.every((item) => item.requires === 'yes'));
        assert.equal(report.edges.collection_output.status, 'flow');
    }
});

test('Gather distinguishes table derivatives from generated position indices and frozen sources', () => {
    const source = graph([
        layer('tokens', 'deepflow.io.Input', { requires_grad: true }), layer('positions', 'deepflow.op.PositionIndices'),
        layer('weight', 'deepflow.op.Parameter', { shape: [8, 4] }), layer('gather', 'deepflow.op.Gather'), layer('output', 'deepflow.io.Output'),
    ], [edge('tokens', 'positions'), edge('positions', 'gather', { targetHandle: 'indices' }), edge('weight', 'gather', { targetHandle: 'weight' }), edge('gather', 'output')]);
    const report = analyzeGradientFlow(source);
    assert.equal(report.edges.tokens_positions.status, 'constant');
    assert.equal(report.edges.positions_gather.status, 'constant');
    assert.equal(report.edges.weight_gather.status, 'flow');
    assert.equal(report.edges.gather_output.status, 'flow');
    for (const patch of [{ requires_grad: false }, { enabled: false }]) {
        const frozen = { ...source, nodes: { ...source.nodes, weight: { ...source.nodes.weight, parameters: { shape: [8, 4], ...patch } } } };
        assert.equal(analyzeGradientFlow(frozen).edges.gather_output.status, 'constant');
    }
    for (const [gradientOp, gradientOpLambda, expected] of [['detach', 1, 'stopped'], ['stop', 1, 'stopped'], ['reverse', 0, 'stopped'], ['reverse', 0.5, 'reverse']]) {
        const changed = { ...source, edges: source.edges.map((connection) => connection.source === 'weight' ? { ...connection, gradientOp, gradientOpLambda } : connection) };
        const checked = analyzeGradientFlow(changed);
        assert.equal(checked.edges.weight_gather.status, expected);
        assert.equal(checked.nodes.gather.requires, expected === 'stopped' ? 'no' : 'yes');
    }
});

test('sequential nested templates preserve gradient dependencies and respect an internal stop', () => {
    const leaf = { id: 'leaf', graph: graph([
        layer('input', 'deepflow.template.Input'), layer('scale', 'deepflow.op.Scale', { factor: 2 }), layer('output', 'deepflow.template.Output'),
    ], [edge('input', 'scale'), edge('scale', 'output')]) };
    const parent = { id: 'parent', graph: graph([
        layer('input', 'deepflow.template.Input'), { id: 'repeat', kind: 'repeat', templateId: 'leaf', count: 3 }, layer('output', 'deepflow.template.Output'),
    ], [edge('input', 'repeat'), edge('repeat', 'output')]) };
    const root = graph([
        layer('parameter', 'deepflow.op.Parameter', { shape: [4] }), { id: 'block', kind: 'supernode', templateId: 'parent' }, layer('output', 'deepflow.io.Output'),
    ], [edge('parameter', 'block'), edge('block', 'output')]);
    const templates = { parent, leaf };
    const report = analyzeGradientFlow(root, { templates });
    assert.equal(report.edges.parameter_block.status, 'flow');
    assert.equal(report.edges.block_output.status, 'flow');
    assert.equal(report.children.block[0].children.repeat.length, 3);
    leaf.graph.edges[0].gradientOp = 'stop';
    const stopped = analyzeGradientFlow(root, { templates });
    assert.equal(stopped.edges.parameter_block.status, 'constant');
    assert.equal(stopped.edges.block_output.status, 'constant');
    assert.ok(stopped.children.block[0].children.repeat.every((child) => child.output.requires === 'no'));
});
