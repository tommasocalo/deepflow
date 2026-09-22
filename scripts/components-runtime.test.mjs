import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { ensureStandardTemplate } from '../src/domain/components/index.js';
import { validate } from '../src/validation/validator.js';
import { inferShape } from '../src/validation/shapes.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';
import { evaluateExpression, resolveDimensionTuple } from '../src/domain/expressions.js';

function fixture(key, parameterValues, shape) {
    const templates = {};
    const state = { templates, createTemplate(template) { const id = `template_${Object.keys(templates).length}`; templates[id] = { ...template, id }; return id; } };
    const templateId = ensureStandardTemplate(key, { getState: () => state });
    const root = {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.io.Input', parameters: { shape } },
            component: { id: 'component', kind: 'supernode', templateId, parameterValues },
            output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.io.Output', parameters: {} },
        },
        edges: [{ id: 'in', source: 'input', target: 'component' }, { id: 'out', source: 'component', target: 'output' }],
    };
    return { root, templates, template: templates[templateId] };
}
function sourceFor(key, parameters, shape, className) {
    const { root, templates } = fixture(key, parameters, shape);
    const report = validate(root, { templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    return generateArchitectureModule(root, { templates, className });
}
const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
function python(source) {
    const result = spawnSync('python3', ['-c', source], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
}

test('component shapes resolve parameter tuples and use real broadcasting', () => {
    assert.deepEqual(resolveDimensionTuple('(out_features, in_features)', { out_features: 5, in_features: 4 }), [5, 4]);
    assert.deepEqual(resolveDimensionTuple(['$D', '2 * H'], { D: 8, H: 3 }), [8, 6]);
    assert.equal(evaluateExpression('eps * 2', { eps: 1e-5 }), 2e-5);
    assert.equal(evaluateExpression('1e-5 * 2'), 2e-5);
    for (const key of ['deepflow.op.Add', 'deepflow.op.Multiply', 'deepflow.op.Subtract']) {
        assert.deepEqual(inferShape(key, [[null, 3, 5], [5]], {}).output, [null, 3, 5]);
        assert.deepEqual(inferShape(key, [[null, 3, 5], [null, 3, 1]], {}).output, [null, 3, 5]);
        assert.ok(inferShape(key, [[null, 3, 5], [4]], {}).errors.length);
    }
    assert.ok(inferShape('deepflow.op.Stack', [[null, 3, 5], [5]], { dim: -1 }).errors.length);
    assert.deepEqual(inferShape('deepflow.op.Mean', [[null, 3, 5]], { dim: [-2, -1], keepdim: false }).output, [null]);
    assert.deepEqual(inferShape('torch.nn.Flatten', [[null, 3, 4, 2]], { start_dim: -2, end_dim: -1 }).output, [null, 3, 8]);
    for (const [key, parameters, shape, expected] of [
        ['linear', { in_features: 4, out_features: 5 }, '(3, 4)', [null, 3, 5]],
        ['embedding', { num_embeddings: 11, embedding_dim: 5 }, '(3,)', [null, 3, 5]],
        ['layernorm', { normalized_shape: 4 }, '(3, 4)', [null, 3, 4]],
        ['learned-position', { max_len: 11, embedding_dim: 5 }, '(3,)', [null, 3, 5]],
        ['sinusoidal-position', { D: 5 }, '(3,)', [null, 3, 5]],
    ]) {
        const { root, templates } = fixture(key, parameters, shape);
        const report = validate(root, { templates });
        assert.equal(report.ok, true, `${key}: ${JSON.stringify(report.errors)}`);
        assert.deepEqual(report.shapes.output, expected);
    }
});

test('explicit Linear and LayerNorm match native modules and all gradients', { skip: !torchAvailable }, () => {
    const linear = sourceFor('linear', { in_features: 4, out_features: 5 }, '(3, 4)', 'LinearModel');
    const norm = sourceFor('layernorm', { normalized_shape: 4, eps: 1e-5 }, '(3, 4)', 'NormModel');
    assert.doesNotMatch(linear, /torch\.nn\.Linear|nn\.Linear\(/);
    assert.doesNotMatch(norm, /torch\.nn\.LayerNorm|nn\.LayerNorm\(/);
    python(`${linear}\n${norm}
for model, reference in [(LinearModel().double(), nn.Linear(4, 5).double()), (NormModel().double(), nn.LayerNorm(4, eps=1e-5).double())]:
    params = list(model.parameters())
    refs = list(reference.parameters())
    assert len(params) == len(refs) == 2
    with torch.no_grad():
        for parameter, ref in zip(params, refs):
            ref.copy_(parameter)
    x = torch.randn(2, 3, 4, dtype=torch.double, requires_grad=True)
    other = x.detach().clone().requires_grad_(True)
    actual, expected = model(x), reference(other)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, other.grad, rtol=1e-9, atol=1e-9)
    for parameter, ref in zip(params, refs):
        torch.testing.assert_close(parameter.grad, ref.grad, rtol=1e-9, atol=1e-9)
`);
});

test('disabled affine terms are buffers and frozen parameters retain requires_grad false', { skip: !torchAvailable }, () => {
    const linear = sourceFor('linear', { in_features: 4, out_features: 5, bias: false }, '(3, 4)', 'NoBias');
    const norm = sourceFor('layernorm', { normalized_shape: 4, elementwise_affine: false }, '(3, 4)', 'NoAffine');
    const frozen = fixture('embedding', { num_embeddings: 11, embedding_dim: 5 }, '(3,)');
    Object.values(frozen.template.graph.nodes).find((node) => node.libraryKey === 'deepflow.op.Parameter').parameters.requires_grad = false;
    const frozenSource = generateArchitectureModule(frozen.root, { templates: frozen.templates, className: 'Frozen' });
    python(`${linear}\n${norm}\n${frozenSource}
x = torch.randn(2, 3, 4)
linear, norm, frozen = NoBias(), NoAffine(), Frozen()
assert len(list(linear.parameters())) == 1
assert len(list(norm.parameters())) == 0
assert len(list(norm.buffers())) == 2
torch.testing.assert_close(linear(x), F.linear(x, linear.weight))
torch.testing.assert_close(norm(x), F.layer_norm(x, (4,), eps=1e-5))
assert len(list(frozen.parameters())) == 1
assert isinstance(frozen.embedding_table, nn.Parameter)
assert not frozen.embedding_table.requires_grad
assert not frozen(torch.tensor([[1, 2, 3]])).requires_grad
`);
});

test('explicit Embedding preserves padding initialization and indexed weight gradients', { skip: !torchAvailable }, () => {
    const source = sourceFor('embedding', { num_embeddings: 11, embedding_dim: 5, padding_idx: 0 }, '(3,)', 'EmbeddingModel');
    python(`${source}
model = EmbeddingModel().double()
reference = nn.Embedding(11, 5, padding_idx=0).double()
assert torch.count_nonzero(model.embedding_table[0]) == 0
with torch.no_grad():
    reference.weight.copy_(model.embedding_table)
indices = torch.tensor([[0, 1, 2], [3, 2, 1]], dtype=torch.long)
actual, expected = model(indices), reference(indices)
torch.testing.assert_close(actual, expected)
actual.square().sum().backward()
expected.square().sum().backward()
torch.testing.assert_close(model.embedding_table.grad, reference.weight.grad)
assert torch.count_nonzero(model.embedding_table.grad[0]) == 0
assert indices.grad is None
`);
});

test('learned positions ignore token values and only train their table', { skip: !torchAvailable }, () => {
    const source = sourceFor('learned-position', { max_len: 11, embedding_dim: 5 }, '(3,)', 'Positions');
    python(`${source}
model = Positions().double()
x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
actual = model(x)
indices = torch.arange(3, device=x.device).expand(2, -1)
expected = F.embedding(indices, model.embedding_table)
torch.testing.assert_close(actual, expected)
torch.testing.assert_close(actual, model(x * 17 + 3))
actual.sum().backward()
assert x.grad is None
assert torch.all(model.embedding_table.grad[:3] == 2)
assert torch.count_nonzero(model.embedding_table.grad[3:]) == 0
assert actual.device == x.device
`);
});

test('explicit sinusoidal even/odd positions match the mathematical formula without gradients', { skip: !torchAvailable }, () => {
    for (const D of [4, 5]) {
        const source = sourceFor('sinusoidal-position', { D }, '(3,)', 'Positions');
        assert.match(source, /torch\.sin/);
        assert.match(source, /torch\.cos/);
        assert.match(source, /torch\.exp/);
        python(`${source}
import math
model = Positions().double()
x = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
actual = model(x)
positions = torch.arange(3, dtype=torch.double).unsqueeze(-1)
frequency = torch.exp(torch.arange(0, ${D}, 2, dtype=torch.double) * (-math.log(10000) / ${D}))
angles = positions * frequency
expected = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)[..., :${D}].expand(2, -1, -1)
torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
assert actual.shape == (2, 3, ${D})
assert not actual.requires_grad
assert len(list(model.parameters())) == 0
assert len(list(model.buffers())) == 1
assert next(model.buffers()).dtype == torch.double
`);
    }
});

test('detach edges stop parameter gradients on explicit tensor paths', { skip: !torchAvailable }, () => {
    const { root, templates, template } = fixture('linear', { in_features: 4, out_features: 5 }, '(3, 4)');
    const weight = Object.values(template.graph.nodes).find((node) => node.label === 'weight');
    template.graph.edges.find((edge) => edge.source === weight.id).gradientOp = 'detach';
    const source = generateArchitectureModule(root, { templates });
    python(`${source}
model = Model()
x = torch.randn(2, 3, 4, requires_grad=True)
model(x).sum().backward()
assert x.grad is not None
assert model.weight.grad is None
assert model.bias.grad is not None
`);
});

test('invalid source and reduction parameters prevent export', () => {
    const { root, templates, template } = fixture('linear', { in_features: 4, out_features: 5 }, '(3, 4)');
    const weight = Object.values(template.graph.nodes).find((node) => node.label === 'weight');
    for (const shape of [[4, 1.5], [-1, 4], ['$missing', 4]]) {
        weight.parameters.shape = shape;
        assert.equal(validate(root, { templates }).ok, false);
        assert.throws(() => generateArchitectureModule(root, { templates }), /shape/);
    }
    assert.ok(inferShape('deepflow.op.Arange', [], { start: 0, end: 5, step: 0 }).errors.length);
    assert.ok(inferShape('deepflow.op.Mean', [[null, 3, 4]], { dim: [1, 1] }).errors.length);
    assert.ok(inferShape('deepflow.op.PositionIndices', [[null, 3, 4]], {}).errors.length);
    assert.ok(inferShape('deepflow.op.Slice', [[null, 3, 4]], { dim: -1, step: 0 }).errors.length);
});

test('stop and reversal apply to every tensor in nested parallel collections', { skip: !torchAvailable }, () => {
    const { templates, template } = fixture('linear', { in_features: 4, out_features: 4 }, '(3, 4)');
    const inner = { id: 'inner', name: 'Inner parallel', freeParameters: [], graph: {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.template.Input', parameters: {} },
            repeat: { id: 'repeat', kind: 'repeat', templateId: template.id, mode: 'parallel', count: 2, bindings: { in_features: 4, out_features: 4, bias: true } },
            output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.template.Output', parameters: {} },
        }, edges: [{ id: 'i', source: 'input', target: 'repeat' }, { id: 'o', source: 'repeat', target: 'output' }],
    } };
    templates.inner = inner;
    const root = { nodes: {
        input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.io.Input', parameters: { shape: '(3, 4)' } },
        repeat: { id: 'repeat', kind: 'repeat', templateId: 'inner', mode: 'parallel', count: 2 },
        sum: { id: 'sum', kind: 'layer', libraryKey: 'deepflow.op.Add', parameters: {} },
    }, edges: [{ id: 'i', source: 'input', target: 'repeat' }, { id: 'c', source: 'repeat', target: 'sum' }, { id: 'skip', source: 'input', target: 'sum' }] };
    const reference = generateArchitectureModule(root, { templates, className: 'Reference' });
    root.edges[1].gradientOp = 'stop';
    const stopped = generateArchitectureModule(root, { templates, className: 'Stopped' });
    root.edges[1].gradientOp = 'reverse';
    root.edges[1].gradientOpLambda = 0.25;
    const reversed = generateArchitectureModule(root, { templates, className: 'Reversed' });
    python(`${reference}\n${stopped}\n${reversed}
reference, stopped, reversed_model = Reference().double(), Stopped().double(), Reversed().double()
stopped.load_state_dict(reference.state_dict())
reversed_model.load_state_dict(reference.state_dict())
x = torch.randn(2, 3, 4, dtype=torch.double)
inputs = [x.clone().requires_grad_(True) for _ in range(3)]
for model, value in zip([reference, stopped, reversed_model], inputs):
    torch.testing.assert_close(model(value), reference(x))
    model(value).sum().backward()
assert all(parameter.grad is None for parameter in stopped.parameters())
torch.testing.assert_close(inputs[1].grad, torch.ones_like(x))
for actual, expected in zip(reversed_model.parameters(), reference.parameters()):
    torch.testing.assert_close(actual.grad, expected.grad * -0.25)
torch.testing.assert_close(inputs[2].grad, 1 - 0.25 * (inputs[0].grad - 1))
`);
});

test('standard attention uses explicit Parameter projections and matches reference gradients', { skip: !torchAvailable }, () => {
    const source = sourceFor('attention', { D: 8, H: 2, dropout: 0 }, '(3, 8)', 'Attention');
    assert.doesNotMatch(source, /nn\.Linear\(|CausalSelfAttention|scaled_dot_product_attention/);
    python(`${source}
import copy
model = Attention().double()
reference = copy.deepcopy(model)
weights = [parameter for name, parameter in reference.named_parameters() if name.startswith('weight')]
biases = [parameter for name, parameter in reference.named_parameters() if name.startswith('bias')]
assert len(weights) == len(biases) == 4
x = torch.randn(2, 3, 8, dtype=torch.double, requires_grad=True)
y = x.detach().clone().requires_grad_(True)
q, k, v = [F.linear(y, weight, bias).reshape(2, 3, 2, 4).transpose(1, 2) for weight, bias in zip(weights[:3], biases[:3])]
attention = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
expected = F.linear(attention.transpose(1, 2).reshape(2, 3, 8), weights[3], biases[3])
actual = model(x)
torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
actual.square().sum().backward()
expected.square().sum().backward()
torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-9)
for parameter, expected_parameter in zip(model.parameters(), reference.parameters()):
    torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=1e-9, atol=1e-9)
`);
});
