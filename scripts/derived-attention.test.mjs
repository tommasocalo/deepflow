import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { buildParameterScope, resolveRepeatCount } from '../src/domain/expressions.js';
import { resolveComponentContext } from '../src/domain/componentContext.js';
import { attentionHeadParameters, attentionParameters, buildAttentionHeadGraph, buildMultiHeadAttentionGraph } from '../src/storyboards/attention.js';
import { ensureStandardTemplate } from '../src/domain/components/index.js';
import { validate } from '../src/validation/validator.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';
import { analyzeGradientFlow } from '../src/domain/gradientFlow.js';

const layer = (id, libraryKey, parameters = {}) => ({ id, label: id, kind: 'layer', libraryKey, parameters });
const edge = (source, target, targetHandle) => ({ id: `${source}_${target}_${targetHandle ?? ''}`, source, target, targetHandle });
const graph = (nodes, edges) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges });
const collection = (items) => ({ kind: 'collection', items });
const head = () => ({ id: 'head', name: 'Head', graph: buildAttentionHeadGraph(), freeParameters: attentionHeadParameters() });

function headFixture() {
    const component = { id: 'head-instance', kind: 'supernode', templateId: 'head', parameterValues: { d_head: 999, dropout: 0 } };
    return {
        templates: { head: head() },
        root: graph([
            layer('query', 'deepflow.io.Input', { shape: '(5, 8)' }),
            layer('key', 'deepflow.io.Input', { shape: '(7, 8)' }),
            layer('value', 'deepflow.io.Input', { shape: '(7, 6)' }),
            component, layer('out', 'deepflow.io.Output'),
        ], [edge('query', component.id, 'head_q'), edge('key', component.id, 'head_k'), edge('value', component.id, 'head_v'), edge(component.id, 'out')]),
    };
}

test('head width comes from Q features, not score length or a stale instance binding', () => {
    const { root, templates } = headFixture();
    const report = validate(root, { templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.out, [null, 5, 6]);
    const context = resolveComponentContext(root.nodes['head-instance'], templates.head, root, report.shapes);
    assert.equal(context.scope.d_head, 8);
    assert.equal(context.scope.dropout, 0);
    const child = validate(templates.head.graph, { templates, scope: context.scope, externalInputs: context.inputs });
    const scores = Object.values(templates.head.graph.nodes).find((node) => node.label === 'Q × Kᵀ');
    assert.deepEqual(child.shapes[scores.id], [null, 5, 7]);
    assert.match(generateArchitectureModule(root, { templates }), /\* 0\.35355339059327373/);
    root.nodes.key.parameters.shape = '(7, 4)';
    assert.equal(validate(root, { templates }).ok, false, 'Q/K width disagreement must remain an error');
});

test('missing input dimensions stay unresolved instead of inheriting a default or parent width', () => {
    const parameters = [{ ...attentionHeadParameters()[0], default: 64 }];
    assert.equal(buildParameterScope(parameters, { d_head: 64 }, { d_head: 128 }).d_head, null);
    assert.equal(buildParameterScope(parameters, {}, {}, undefined, { head_q: [null, 5, null] }).d_head, null);
    const { root, templates } = headFixture();
    root.edges = root.edges.filter((wire) => wire.source !== 'query');
    const report = validate(root, { templates });
    assert.equal(report.ok, false);
    assert.match(report.errors['head-instance'].join(' '), /d_head cannot be inferred from Q/);
    assert.throws(() => generateArchitectureModule(root, { templates }), /invalid architecture/);
});

test('Auto instances follows input collections and retains manual parallel counts', () => {
    const inputs = [collection([[null, 5, 8], [null, 5, 8]]), collection([[null, 5, 8], [null, 5, 8]])];
    assert.equal(resolveRepeatCount('auto', {}, inputs).count, 2);
    assert.equal(resolveRepeatCount('auto', {}, [[null, 5, 8]]).count, null);
    assert.match(resolveRepeatCount('auto', {}, [...inputs, collection([[null, 5, 8]])]).error, /equally sized/);
    assert.equal(resolveRepeatCount(200).count, 200);
    const templates = { head: head(), mha: { id: 'mha', name: 'MHA', graph: buildMultiHeadAttentionGraph({ headTemplateId: 'head' }), freeParameters: attentionParameters() } };
    const root = graph([
        layer('in', 'deepflow.io.Input', { shape: '(5, 24)', requires_grad: true }),
        { id: 'mha', kind: 'supernode', templateId: 'mha', parameterValues: { H: 3, dropout: 0 } },
        layer('out', 'deepflow.io.Output'),
    ], [edge('in', 'mha'), edge('mha', 'out')]);
    for (const count of [3, 6]) {
        root.nodes.mha.parameterValues.H = count;
        const report = validate(root, { templates });
        assert.equal(report.ok, true, JSON.stringify(report.errors));
        assert.deepEqual(report.shapes.out, [null, 5, 24]);
        const ctx = resolveComponentContext(root.nodes.mha, templates.mha, root, report.shapes);
        assert.equal(ctx.scope.D, 24);
        const child = validate(templates.mha.graph, { templates, scope: ctx.scope, externalInputs: ctx.inputs });
        const repeat = Object.values(templates.mha.graph.nodes).find((node) => node.kind === 'repeat');
        const repeated = resolveComponentContext(repeat, templates.head, templates.mha.graph, child.shapes, ctx.scope, count - 1);
        assert.equal(repeated.count, count);
        assert.equal(repeated.scope.d_head, 24 / count);
        const gradients = analyzeGradientFlow(root, { templates });
        assert.equal(gradients.children.mha[0].children[repeat.id].length, count);
        assert.equal(gradients.children.mha[0].children[repeat.id][0].scope.d_head, 24 / count);
        assert.equal((generateArchitectureModule(root, { templates }).match(/torch\.matmul/g) ?? []).length, count * 2);
    }
    root.nodes.mha.parameterValues.bias = false;
    assert.equal((generateArchitectureModule(root, { templates }).match(/bias=False/g) ?? []).length, 4, 'Exposed bias reaches every native projection too');
    const repeat = Object.values(templates.mha.graph.nodes).find((node) => node.kind === 'repeat');
    repeat.mode = 'sequential';
    assert.match(JSON.stringify(validate(root, { templates }).errors), /only for parallel/);
});

test('sequential derived dimensions are recalculated from each previous output', () => {
    const shrink = { id: 'shrink', name: 'Shrink', freeParameters: [{ name: 'width', type: 'int', derived: { kind: 'input-dimension', inputId: 'in', axis: -1 } }], graph: graph([
        layer('in', 'deepflow.template.Input'), layer('projection', 'torch.nn.Linear', { in_features: '$width', out_features: 'width / 2', bias: false }), layer('out', 'deepflow.template.Output'),
    ], [edge('in', 'projection'), edge('projection', 'out')]) };
    const root = graph([layer('in', 'deepflow.io.Input', { shape: '(8,)' }), { id: 'repeat', kind: 'repeat', templateId: 'shrink', count: 2 }, layer('out', 'deepflow.io.Output')], [edge('in', 'repeat'), edge('repeat', 'out')]);
    assert.deepEqual(validate(root, { templates: { shrink } }).shapes.out, [null, 2]);
    const source = generateArchitectureModule(root, { templates: { shrink } });
    assert.match(source, /in_features=8, out_features=4/);
    assert.match(source, /in_features=4, out_features=2/);
});

test('new standard generations preserve saved definitions and reuse user edits within their generation', () => {
    const originals = {
        old: { id: 'old', name: 'My previous attention', standardKey: 'attention', graph: { nodes: {}, edges: [] }, freeParameters: [{ name: 'D', default: 32 }] },
        oldHead: { id: 'oldHead', standardKey: 'attention-head', graph: { nodes: {}, edges: [] }, freeParameters: [{ name: 'd_head', default: 64 }] },
    };
    const templates = { ...originals };
    let sequence = 0;
    const state = { templates, createTemplate: (template) => { const id = `new-${++sequence}`; templates[id] = { ...template, id }; return id; } };
    const store = { getState: () => state };
    const id = ensureStandardTemplate('attention', store);
    assert.notEqual(id, 'old');
    assert.equal(templates.old, originals.old);
    assert.equal(templates.oldHead, originals.oldHead);
    templates[id].name = 'Edited current attention';
    assert.equal(ensureStandardTemplate('attention', store), id);
    assert.equal(templates[id].freeParameters.find((parameter) => parameter.name === 'D').derived.inputId, 'attention_input');
    const repeat = Object.values(templates[id].graph.nodes).find((node) => node.kind === 'repeat');
    assert.equal(templates[repeat.templateId].standardVersion, 2);
});

const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
test('derived non-square head export matches PyTorch outputs and gradients with a distinct V width', { skip: !torchAvailable }, () => {
    const { root, templates } = headFixture();
    const source = generateArchitectureModule(root, { templates });
    const result = spawnSync('python3', ['-c', `${source}
torch.manual_seed(83)
model = Model().double().eval()
q = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)
k = torch.randn(2, 7, 8, dtype=torch.double, requires_grad=True)
v = torch.randn(2, 7, 6, dtype=torch.double, requires_grad=True)
refs = [x.detach().clone().requires_grad_(True) for x in [q, k, v]]
actual = model(q, k, v)
expected = F.scaled_dot_product_attention(*refs, dropout_p=0, is_causal=True)
torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
actual.square().sum().backward()
expected.square().sum().backward()
for x, ref in zip([q, k, v], refs):
    torch.testing.assert_close(x.grad, ref.grad, rtol=1e-10, atol=1e-10)
print('derived attention equivalence passed')
`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /equivalence passed/);
});
