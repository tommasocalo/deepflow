import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { JSDOM } from 'jsdom';
import { buildAttentionHeadGraph, buildMultiHeadAttentionGraph } from '../src/storyboards/attention.js';
import { buildTransformerBlockGraph } from '../src/storyboards/gpt2.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';
import { validate } from '../src/validation/validator.js';

const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/' });
globalThis.window = dom.window;
globalThis.document = dom.window.document;
globalThis.IS_REACT_ACT_ENVIRONMENT = true;
const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');
const { ArchitectureStoreProvider, createArchitecturePaneStore, useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useValidationContext, useValidation } = await import('../src/hooks/useValidation.js');
const layer = (id, libraryKey, parameters = {}) => ({ id, kind: 'layer', libraryKey, parameters });
const edge = (source, target) => ({ id: `${source}_${target}`, source, target });
const nodesOf = (graph) => Object.values(graph.nodes);

function nestedFixture() {
    const transformer = buildTransformerBlockGraph({ attentionTemplateId: 'mha' });
    for (const node of nodesOf(transformer)) {
        if (node.libraryKey === 'torch.nn.Dropout') node.parameters.p = 0;
    }
    const attention = nodesOf(transformer).find((node) => node.kind === 'supernode');
    attention.parameterValues.dropout = 0;
    const mha = buildMultiHeadAttentionGraph({ headTemplateId: 'head' });
    const heads = nodesOf(mha).find((node) => node.kind === 'repeat');
    const templates = {
        head: { id: 'head', name: 'Head', graph: buildAttentionHeadGraph(), freeParameters: [
            { name: 'd_head', type: 'int', derived: { kind: 'input-dimension', inputId: 'head_q', axis: -1 } }, { name: 'dropout', type: 'float', default: 0 },
        ] },
        mha: { id: 'mha', name: 'MHA', graph: mha, freeParameters: [
            { name: 'D', type: 'int', default: 768 }, { name: 'H', type: 'int', default: 12 },
            { name: 'd_head', type: 'int', default: 'D / H' }, { name: 'dropout', type: 'float', default: 0 }, { name: 'bias', type: 'bool', default: true },
        ] },
        transformer: { id: 'transformer', name: 'Transformer', graph: transformer, freeParameters: [
            { name: 'd_model', type: 'int', default: 768 }, { name: 'n_heads', type: 'int', default: 12 },
        ] },
    };
    const input = layer('input', 'deepflow.io.Input', { shape: '(5, 12)' });
    const stack = { id: 'stack', kind: 'repeat', templateId: 'transformer', count: 2, mode: 'sequential', bindings: { d_model: 12, n_heads: 'i + 2' } };
    const output = layer('output', 'deepflow.io.Output');
    const root = { nodes: { input, stack, output }, edges: [edge('input', 'stack'), edge('stack', 'output')] };
    return { root, templates, path: ['stack', attention.id, heads.id], attention, heads };
}

let container, root;
const panes = [];
beforeEach(() => {
    templatesStore.setState({ templates: {} });
    architecture.setState({ graph: { nodes: {}, edges: [] }, currentPath: [], openPaths: [], inspectedIterations: {} });
    container = document.createElement('div');
    document.body.appendChild(container);
    root = createRoot(container);
});
afterEach(async () => {
    await act(() => root.unmount());
    container.remove();
    for (const pane of panes.splice(0)) pane.dispose();
});
after(() => dom.window.close());

test('nested head previews inherit outer iteration bindings and select the matching collection shapes per pane', async () => {
    const fixture = nestedFixture();
    templatesStore.setState({ templates: fixture.templates });
    architecture.setState({ graph: fixture.root });
    const first = createArchitecturePaneStore({ path: fixture.path });
    const second = createArchitecturePaneStore({ path: fixture.path });
    panes.push(first, second);
    first.getState().setInspectedIteration(['stack'], 1);
    first.getState().setInspectedIteration(fixture.path, 2);
    second.getState().setInspectedIteration(['stack'], 0);
    second.getState().setInspectedIteration(fixture.path, 1);
    const observed = {};
    function Probe({ name }) {
        observed[name] = { context: useValidationContext(), report: useValidation() };
        return null;
    }
    await act(() => root.render(h('div', null,
        h(ArchitectureStoreProvider, { store: first }, h(Probe, { name: 'first' })),
        h(ArchitectureStoreProvider, { store: second }, h(Probe, { name: 'second' })),
    )));
    const one = observed.first;
    const two = observed.second;
    assert.equal(one.context.scope.d_model, 12);
    assert.equal(one.context.scope.n_heads, 3);
    assert.equal(one.context.scope.D, 12);
    assert.equal(one.context.scope.H, 3);
    assert.equal(one.context.scope.d_head, 4);
    assert.equal(one.context.scope.i, 2);
    assert.equal(one.context.repeat.index, 2);
    assert.equal(two.context.scope.H, 2);
    assert.equal(two.context.scope.d_head, 6);
    assert.equal(two.context.repeat.index, 1);
    for (const id of ['head_q', 'head_k', 'head_v']) {
        assert.deepEqual(one.context.externalInputs[id], [null, 5, 4]);
        assert.deepEqual(two.context.externalInputs[id], [null, 5, 6]);
    }
    const idFor = (label) => nodesOf(fixture.templates.head.graph).find((node) => node.label === label).id;
    assert.equal(one.report.ok, true, JSON.stringify(one.report.errors));
    assert.equal(two.report.ok, true, JSON.stringify(two.report.errors));
    assert.deepEqual(one.report.shapes[idFor('K transpose')], [null, 4, 5]);
    assert.deepEqual(one.report.shapes[idFor('Q × Kᵀ')], [null, 5, 5]);
    assert.deepEqual(one.report.shapes[idFor('head output')], [null, 5, 4]);
    assert.deepEqual(two.report.shapes[idFor('head output')], [null, 5, 6]);

    // Selecting a previously valid head after the outer iteration changes is
    // clamped to the smaller head collection, without affecting the other pane.
    await act(() => first.getState().setInspectedIteration(['stack'], 0));
    assert.equal(observed.first.context.repeat.index, 1);
    assert.equal(observed.first.context.scope.d_head, 6);
    assert.equal(observed.second.context.repeat.index, 1);
    assert.equal(architecture.getState().graph, fixture.root);
});

test('nested sequential Transformer export resolves distinct per-block head counts without prebuilt attention calls', () => {
    const { root, templates } = nestedFixture();
    const report = validate(root, { templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.output, [null, 5, 12]);
    const source = generateArchitectureModule(root, { templates });
    assert.equal((source.match(/torch\.matmul/g) ?? []).length, 10, 'Two heads then three heads, each with two matrix products');
    assert.equal((source.match(/= torch.nn.Linear/g) ?? []).length, 12, 'Six independent learned projections per block');
    assert.doesNotMatch(source, /CausalSelfAttention|scaled_dot_product_attention|\$d_model|\$n_heads|\$H/);
});

const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
test('sequential repeats of explicit parallel MHA match an independently composed Transformer reference and gradients', { skip: !torchAvailable }, () => {
    const { root, templates } = nestedFixture();
    const source = generateArchitectureModule(root, { templates });
    const result = spawnSync('python3', ['-c', `${source}
import copy
torch.manual_seed(73)
model = Model().double().eval()
reference = copy.deepcopy(model)
x = torch.randn(2, 5, 12, dtype=torch.double, requires_grad=True)
x_ref = x.detach().clone().requires_grad_(True)
actual = model(x)
linears = [layer for layer in reference.children() if isinstance(layer, nn.Linear)]
norms = [layer for layer in reference.children() if isinstance(layer, nn.LayerNorm)]
expected = x_ref
for block_index, heads in enumerate([2, 3]):
    q_layer, k_layer, v_layer, out_layer, fc, proj = linears[6 * block_index:6 * (block_index + 1)]
    ln1, ln2 = norms[2 * block_index:2 * (block_index + 1)]
    normalized = ln1(expected)
    q = q_layer(normalized).reshape(2, 5, heads, 12 // heads).transpose(1, 2)
    k = k_layer(normalized).reshape(2, 5, heads, 12 // heads).transpose(1, 2)
    v = v_layer(normalized).reshape(2, 5, heads, 12 // heads).transpose(1, 2)
    context = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)
    residual = expected + out_layer(context.transpose(1, 2).reshape(2, 5, 12))
    expected = residual + proj(F.gelu(fc(ln2(residual))))
torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
actual.square().sum().backward()
expected.square().sum().backward()
torch.testing.assert_close(x.grad, x_ref.grad, rtol=1e-10, atol=1e-10)
for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
    torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-10, atol=1e-10)
assert len({layer.weight.data_ptr() for layer in linears}) == 12
print('Nested Transformer output and gradient equivalence passed')
`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /equivalence passed/);
});
