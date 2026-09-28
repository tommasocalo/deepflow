import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeGradientFlow } from '../src/domain/gradientFlow.js';
import { resolveGradientFactor } from '../src/domain/gradientFactor.js';
import { validate } from '../src/validation/validator.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';

const layer = (id, libraryKey, parameters = {}) => ({ id, kind: 'layer', libraryKey, parameters });
const edge = (source, target, rest = {}) => ({ id: `${source}_${target}`, source, target, ...rest });
const graph = (nodes, edges) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges });

test('reverse factor resolves numeric parameters in analysis, validation and Python export', () => {
    const model = {
        ...graph([
            layer('input', 'deepflow.io.Input', { shape: '(4,)', requires_grad: true }),
            layer('output', 'deepflow.io.Output'),
        ], [edge('input', 'output', { gradientOp: 'reverse', gradientOpLambda: '$lambda * 0.5' })]),
        parameters: [{ name: 'lambda', type: 'float', default: 0.6 }],
    };
    assert.deepEqual(resolveGradientFactor('$lambda * 0.5', { lambda: 0.6 }), { value: 0.3, error: null });
    assert.equal(validate(model).ok, true);
    assert.equal(analyzeGradientFlow(model).edges.input_output.status, 'reverse');
    assert.match(generateArchitectureModule(model), /_grad_reverse\(x, 0\.3\)/);

    model.parameters[0].default = 0;
    assert.equal(analyzeGradientFlow(model).edges.input_output.status, 'stopped');
    model.edges[0].gradientOpLambda = '$missing';
    assert.equal(validate(model).ok, false, 'An unresolved factor is flagged on its target node');
    assert.match(validate(model).errors.output.join(' '), /Reverse factor/);
    assert.equal(analyzeGradientFlow(model).edges.input_output.status, 'unknown');
    assert.throws(() => generateArchitectureModule(model), /Reverse factor/);
    model.edges[0].gradientOpLambda = -0.2;
    assert.equal(validate(model).ok, false);
});

test('reverse factor can use a component-local value and the repeat index', () => {
    const inner = {
        id: 'body', name: 'Body', freeParameters: [{ name: 'factor', type: 'float', default: 1 }],
        graph: graph([
            layer('in', 'deepflow.template.Input'), layer('out', 'deepflow.template.Output'),
        ], [edge('in', 'out', { gradientOp: 'reverse', gradientOpLambda: '$factor' })]),
    };
    const model = graph([
        layer('input', 'deepflow.io.Input', { shape: '(4,)', requires_grad: true }),
        { id: 'repeat', kind: 'repeat', label: 'Repeat', templateId: 'body', mode: 'parallel', count: 2,
            iterationVariable: 'i', bindings: { factor: 'i * 0.5' } },
        layer('output', 'deepflow.io.Output'),
    ], [edge('input', 'repeat'), edge('repeat', 'output')]);
    const templates = { body: inner };
    assert.equal(validate(model, { templates }).ok, true);
    const report = analyzeGradientFlow(model, { templates });
    assert.equal(report.children.repeat[0].edges.in_out.status, 'stopped');
    assert.equal(report.children.repeat[1].edges.in_out.status, 'reverse');
    const python = generateArchitectureModule(model, { templates });
    assert.match(python, /_grad_reverse\([^,]+, 0\)/);
    assert.match(python, /_grad_reverse\([^,]+, 0\.5\)/);
});

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
