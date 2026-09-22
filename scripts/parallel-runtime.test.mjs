import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { validate } from '../src/validation/validator.js';
import { inferShape, formatShape } from '../src/validation/shapes.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';
import { buildParameterScope, evaluateExpression, resolveRepeatCount } from '../src/domain/expressions.js';
import { buildAttentionHeadGraph, buildMultiHeadAttentionGraph } from '../src/storyboards/attention.js';

const layer = (id, libraryKey, parameters = {}, label = id) => ({ id, kind: 'layer', libraryKey, parameters, label });
const edge = (source, target, targetHandle) => ({ id: `${source}_${target}_${targetHandle ?? ''}`, source, target, targetHandle });
const graph = (nodes, edges) => ({ nodes: Object.fromEntries(nodes.map((n) => [n.id, n])), edges });
const collection = (items) => ({ kind: 'collection', items });
const identity = { id: 'identity', name: 'Identity', graph: graph([
    layer('in', 'deepflow.template.Input'), layer('out', 'deepflow.template.Output'),
], [edge('in', 'out')]), freeParameters: [] };

function repeatFixture(count = 2, mode = 'parallel') {
    return graph([
        layer('input', 'deepflow.io.Input', { shape: '(4, 8)' }),
        layer('split', 'deepflow.op.Split', { chunks: count, dim: -1 }),
        { id: 'repeat', kind: 'repeat', templateId: 'identity', mode, count, bindings: {} },
        layer('concat', 'deepflow.op.Concat', { dim: -1 }),
    ], [edge('input', 'split'), edge('split', 'repeat'), edge('repeat', 'concat')]);
}

function attentionFixture(D = 8, H = 2) {
    const templates = {
        head: { id: 'head', name: 'Attention head', graph: buildAttentionHeadGraph(), freeParameters: [
            { name: 'd_head', type: 'int', derived: { kind: 'input-dimension', inputId: 'head_q', axis: -1 } }, { name: 'dropout', type: 'float', default: 0 },
        ] },
        mha: { id: 'mha', name: 'Multi-head attention', graph: buildMultiHeadAttentionGraph({ headTemplateId: 'head' }), freeParameters: [
            { name: 'D', type: 'int', default: 8 }, { name: 'H', type: 'int', default: 2 },
            { name: 'd_head', type: 'int', default: 'D / H' }, { name: 'dropout', type: 'float', default: 0 },
        ] },
    };
    const root = graph([
        layer('input', 'deepflow.io.Input', { shape: `(4, ${D})` }),
        { id: 'mha', kind: 'supernode', templateId: 'mha', parameterValues: { D, H, dropout: 0 } },
        layer('output', 'deepflow.io.Output'),
    ], [edge('input', 'mha'), edge('mha', 'output')]);
    return { root, templates };
}

test('ordered derived defaults and bare/$ arithmetic resolve identically', () => {
    assert.equal(evaluateExpression('$D / H', { D: 12, H: 3 }), 4);
    assert.equal(evaluateExpression('1 / (d_head ** 0.5)', { d_head: 4 }), 0.5);
    assert.deepEqual(buildParameterScope([{ name: 'D', default: 8 }, { name: 'H', default: 2 }, { name: 'd_head', default: 'D / H' }], { D: 12, H: 3 }), { D: 12, H: 3, d_head: 4 });
    assert.equal(resolveRepeatCount('H', { H: 3 }).count, 3);
    assert.equal(resolveRepeatCount('$H', { H: 3 }).count, 3);
});

test('parallel repeat yields a collection; explicit Concat restores width when count changes', () => {
    for (const count of [1, 2, 4, 8]) {
        const root = repeatFixture(count);
        const report = validate(root, { templates: { identity } });
        assert.equal(report.ok, true, JSON.stringify(report.errors));
        assert.deepEqual(report.shapes.repeat, collection(Array.from({ length: count }, () => [null, 4, 8 / count])));
        assert.deepEqual(report.shapes.concat, [null, 4, 8]);
        const source = generateArchitectureModule(root, { templates: { identity } });
        assert.match(source, /torch\.tensor_split/);
        assert.match(source, /torch\.cat/);
    }
    assert.equal(formatShape(collection([[null, 4], [null, 4]])), '2 × (?, 4)');
});

test('Split, Broadcast, Stack and reductions have explicit collection semantics', () => {
    const parts = inferShape('deepflow.op.Split', [[null, 4, 12]], { chunks: 3, dim: -1 }).output;
    assert.deepEqual(parts, collection(Array.from({ length: 3 }, () => [null, 4, 4])));
    assert.deepEqual(inferShape('deepflow.op.Stack', [parts], { dim: 1 }).output, [null, 3, 4, 4]);
    assert.deepEqual(inferShape('deepflow.op.Add', [parts], {}).output, [null, 4, 4]);
    assert.deepEqual(inferShape('deepflow.op.Broadcast', [[null, 4]], { count: 2 }).output, collection([[null, 4], [null, 4]]));
    assert.ok(inferShape('torch.nn.Linear', [parts], { in_features: 4, out_features: 4 }).errors.length);
    assert.ok(inferShape('deepflow.op.Split', [[null, 7]], { chunks: 2, dim: -1 }).errors.length);
    assert.ok(inferShape('deepflow.op.Concat', [parts], { dim: 5 }).errors.length);
    assert.ok(inferShape('deepflow.op.Stack', [parts], { dim: -9 }).errors.length);
});

test('unhandled sequential repeats remain a chain with per-iteration expressions', () => {
    const dense = { id: 'dense', name: 'Dense', freeParameters: [{ name: 'in', type: 'int', default: 128 }, { name: 'out', type: 'int', default: 64 }], graph: graph([
        layer('in', 'deepflow.template.Input'), layer('linear', 'torch.nn.Linear', { in_features: '$in', out_features: '$out', bias: true }), layer('out', 'deepflow.template.Output'),
    ], [edge('in', 'linear'), edge('linear', 'out')]) };
    const root = graph([
        layer('input', 'deepflow.io.Input', { shape: '(128,)' }),
        { id: 'repeat', kind: 'repeat', templateId: 'dense', count: 3, bindings: { in: '128 / 2 ** i', out: '64 / 2 ** i' } },
    ], [edge('input', 'repeat')]);
    const report = validate(root, { templates: { dense } });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.repeat, [null, 16]);
    assert.equal((generateArchitectureModule(root, { templates: { dense } }).match(/= torch.nn.Linear/g) ?? []).length, 3);
});

test('explicit boundary and MatMul ports win over edge insertion order', () => {
    const multiply = { id: 'multiply', name: 'Matrix product', graph: graph([
        layer('a', 'deepflow.template.Input'), layer('b', 'deepflow.template.Input'),
        layer('mm', 'deepflow.op.MatMul'), layer('out', 'deepflow.template.Output'),
    ], [edge('b', 'mm', 'b'), edge('a', 'mm', 'a'), edge('mm', 'out')]) };
    const root = graph([
        layer('left', 'deepflow.io.Input', { shape: '(3, 5)' }), layer('right', 'deepflow.io.Input', { shape: '(5, 7)' }),
        { id: 'block', kind: 'supernode', templateId: 'multiply', parameterValues: {} },
    ], [edge('right', 'block', 'b'), edge('left', 'block', 'a')]);
    const report = validate(root, { templates: { multiply } });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.block, [null, 3, 7]);
    const fallback = { ...root, edges: [edge('left', 'block'), edge('right', 'block')] };
    assert.deepEqual(validate(fallback, { templates: { multiply } }).shapes.block, [null, 3, 7]);
});

test('invalid, unresolved and excessive repeat counts fail before expansion/export', () => {
    for (const count of [0, -1, 1.5, Infinity, NaN, '$missing', 'missing', '1 / 0', 257, 1e9]) {
        const root = repeatFixture(2);
        root.nodes.repeat.count = count;
        assert.equal(validate(root, { templates: { identity } }).ok, false, String(count));
        assert.throws(() => generateArchitectureModule(root, { templates: { identity } }), /count/);
    }
    const mismatch = repeatFixture(2);
    mismatch.nodes.repeat.count = 3;
    assert.ok(validate(mismatch, { templates: { identity } }).errors.repeat.some((message) => message.includes('3 items')));
});

test('editable MHA validates complete multi-port math and rejects indivisible D/H', () => {
    for (const [D, H] of [[8, 2], [8, 4], [12, 3]]) {
        const { root, templates } = attentionFixture(D, H);
        const report = validate(root, { templates });
        assert.equal(report.ok, true, JSON.stringify(report.errors));
        assert.deepEqual(report.shapes.output, [null, 4, D]);
        const source = generateArchitectureModule(root, { templates });
        assert.equal((source.match(/torch\.matmul/g) ?? []).length, H * 2);
        assert.equal((source.match(/\.masked_fill/g) ?? []).length, H);
        assert.match(source, /torch\.cat/);
        assert.doesNotMatch(source, /CausalSelfAttention|scaled_dot_product_attention/);
    }
    const { root, templates } = attentionFixture(8, 3);
    assert.equal(validate(root, { templates }).ok, false);
    assert.throws(() => generateArchitectureModule(root, { templates }), /integer|divisible/);
});

const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
test('exported explicit MHA matches PyTorch reference outputs and gradients on CPU', { skip: !torchAvailable }, () => {
    const { root, templates } = attentionFixture();
    const source = generateArchitectureModule(root, { templates });
    const python = `${source}
import copy

torch.manual_seed(21)
model = Model().double().eval()
reference = copy.deepcopy(model)
x = torch.randn(2, 4, 8, dtype=torch.double, requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)
actual = model(x)
q_layer, k_layer, v_layer, out_layer = [layer for layer in reference.children() if isinstance(layer, nn.Linear)]
q = q_layer(x_ref).reshape(2, 4, 2, 4).transpose(1, 2)
k = k_layer(x_ref).reshape(2, 4, 2, 4).transpose(1, 2)
v = v_layer(x_ref).reshape(2, 4, 2, 4).transpose(1, 2)
heads = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
expected = out_layer(heads.transpose(1, 2).reshape(2, 4, 8))
torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
actual.square().sum().backward()
expected.square().sum().backward()
torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-10, atol=1e-10)
for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
    torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-10, atol=1e-10)
print('CPU output and gradient equivalence passed')
`;
    const result = spawnSync('python3', ['-c', python], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /equivalence passed/);
});

test('parallel module copies own independent parameters and explicit Broadcast/Stack execute', { skip: !torchAvailable }, () => {
    const dense = { id: 'dense', name: 'Dense', graph: graph([
        layer('in', 'deepflow.template.Input'), layer('linear', 'torch.nn.Linear', { in_features: 4, out_features: 3, bias: true }), layer('out', 'deepflow.template.Output'),
    ], [edge('in', 'linear'), edge('linear', 'out')]) };
    const root = graph([
        layer('input', 'deepflow.io.Input', { shape: '(4,)' }), layer('broadcast', 'deepflow.op.Broadcast', { count: 3 }),
        { id: 'repeat', kind: 'repeat', templateId: 'dense', mode: 'parallel', count: 3 }, layer('stack', 'deepflow.op.Stack', { dim: 1 }),
    ], [edge('input', 'broadcast'), edge('broadcast', 'repeat'), edge('repeat', 'stack')]);
    const source = generateArchitectureModule(root, { templates: { dense } });
    const result = spawnSync('python3', ['-c', `${source}
model = Model()
layers = [layer for layer in model.children() if isinstance(layer, nn.Linear)]
assert len(layers) == 3
assert len({layer.weight.data_ptr() for layer in layers}) == 3
x = torch.randn(2, 4)
assert model(x).shape == (2, 3, 3)
torch.testing.assert_close(model(x), torch.stack([layer(x) for layer in layers], dim=1))
`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});

test('parallel multiple collections route by named ports in exported executable code', { skip: !torchAvailable }, () => {
    const multiply = { id: 'multiply', name: 'Matrix product', graph: graph([
        layer('a', 'deepflow.template.Input'), layer('b', 'deepflow.template.Input'),
        layer('mm', 'deepflow.op.MatMul'), layer('out', 'deepflow.template.Output'),
    ], [edge('b', 'mm', 'b'), edge('a', 'mm', 'a'), edge('mm', 'out')]) };
    const root = graph([
        layer('left', 'deepflow.io.Input', { shape: '(3, 8)' }), layer('right', 'deepflow.io.Input', { shape: '(8, 5)' }),
        layer('leftParts', 'deepflow.op.Split', { chunks: 2, dim: -1 }), layer('rightParts', 'deepflow.op.Split', { chunks: 2, dim: -2 }),
        { id: 'repeat', kind: 'repeat', templateId: 'multiply', mode: 'parallel', count: 2 }, layer('concat', 'deepflow.op.Concat', { dim: -1 }),
    ], [edge('left', 'leftParts'), edge('right', 'rightParts'), edge('rightParts', 'repeat', 'b'), edge('leftParts', 'repeat', 'a'), edge('repeat', 'concat')]);
    const report = validate(root, { templates: { multiply } });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.concat, [null, 3, 10]);
    const source = generateArchitectureModule(root, { templates: { multiply } });
    assert.match(source, /def forward\(self, x, x2\)/);
    const result = spawnSync('python3', ['-c', `${source}
a, b = torch.randn(2, 3, 8), torch.randn(2, 8, 5)
expected = torch.cat([x @ y for x, y in zip(a.chunk(2, -1), b.chunk(2, -2))], -1)
torch.testing.assert_close(Model()(a, b), expected)
`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});

test('nested legal counts respect the total expansion budget', () => {
    const inner = { id: 'inner', name: 'Nested repeat', graph: graph([
        layer('in', 'deepflow.template.Input'), { id: 'repeat', kind: 'repeat', templateId: 'identity', count: 256 }, layer('out', 'deepflow.template.Output'),
    ], [edge('in', 'repeat'), edge('repeat', 'out')]) };
    const root = graph([
        layer('input', 'deepflow.io.Input', { shape: '(4,)' }), { id: 'repeat', kind: 'repeat', templateId: 'inner', count: 256 },
    ], [edge('input', 'repeat')]);
    const templates = { identity, inner };
    assert.equal(validate(root, { templates }).ok, false);
    assert.throws(() => generateArchitectureModule(root, { templates }), /exceeds 20000/);
});

test('duplicate and unknown named template/repeat inputs fail validation and export', () => {
    const threeInputs = { id: 'three', name: 'Three inputs', graph: graph([
        layer('q', 'deepflow.template.Input'), layer('k', 'deepflow.template.Input'), layer('v', 'deepflow.template.Input'),
        layer('sum', 'deepflow.op.Add'), layer('out', 'deepflow.template.Output'),
    ], [edge('q', 'sum'), edge('k', 'sum'), edge('v', 'sum'), edge('sum', 'out')]) };
    const templates = { three: threeInputs };
    for (const kind of ['supernode', 'repeat']) {
        const valid = graph([
            ...['q1', 'q2', 'k1', 'v1'].map((id) => layer(id, 'deepflow.io.Input', { shape: '(4,)' })),
            { id: 'block', kind, templateId: 'three', mode: 'parallel', count: 2 },
        ], [edge('q1', 'block', 'q'), edge('k1', 'block', 'k'), edge('v1', 'block', 'v')]);
        assert.equal(validate(valid, { templates }).ok, true);
        const duplicate = { ...valid, edges: [...valid.edges, edge('q2', 'block', 'q')] };
        assert.ok(validate(duplicate, { templates }).errors.block.some((message) => /port "q" has more than one/.test(message)));
        assert.throws(() => generateArchitectureModule(duplicate, { templates }), /port "q" has more than one/);
        const unknown = { ...valid, edges: valid.edges.map((connection, index) => index ? connection : { ...connection, targetHandle: 'stale_q' }) };
        assert.ok(validate(unknown, { templates }).errors.block.some((message) => message.includes('Unknown input port "stale_q"')));
        assert.throws(() => generateArchitectureModule(unknown, { templates }), /Unknown input port/);
        const excessive = { ...valid, edges: [edge('q1', 'block'), edge('q2', 'block'), edge('k1', 'block'), edge('v1', 'block')] };
        assert.ok(validate(excessive, { templates }).errors.block.some((message) => message.includes('Too many unhandled')));
        const legacy = { ...valid, edges: [edge('q1', 'block'), edge('k1', 'block', null), edge('v1', 'block', '')] };
        assert.equal(validate(legacy, { templates }).ok, true);
        const singleLegacy = { ...valid, edges: [edge('q1', 'block')] };
        assert.equal(validate(singleLegacy, { templates }).ok, true);
        assert.doesNotThrow(() => generateArchitectureModule(singleLegacy, { templates }));
    }
});

test('MatMul rejects duplicate and unknown named inputs while preserving unhandled order', () => {
    const root = graph([
        layer('left', 'deepflow.io.Input', { shape: '(4, 4)' }), layer('right', 'deepflow.io.Input', { shape: '(4, 4)' }),
        layer('mm', 'deepflow.op.MatMul'),
    ], [edge('left', 'mm', 'a'), edge('right', 'mm', 'b')]);
    for (const [handle, expected] of [['a', /more than one incoming/], ['stale_b', /Unknown input port/]]) {
        const malformed = { ...root, edges: [root.edges[0], { ...root.edges[1], targetHandle: handle }] };
        assert.ok(validate(malformed).errors.mm.some((message) => expected.test(message)));
        assert.throws(() => generateArchitectureModule(malformed), expected);
    }
    const legacy = { ...root, edges: [edge('left', 'mm'), edge('right', 'mm', null)] };
    assert.equal(validate(legacy).ok, true);
    assert.doesNotThrow(() => generateArchitectureModule(legacy));
});
