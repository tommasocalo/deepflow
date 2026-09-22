import assert from 'node:assert/strict';
import { test } from 'node:test';
import { availableParameters, exposeTemplateField, parameterConsumers, parameterFrames, parameterOrigin, parameterReferences, resolvedFieldValue } from '../src/domain/parameterBindings.js';

const graph = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const layer = (id, parameters) => ({ id, kind: 'layer', label: id, libraryKey: 'torch.nn.Dropout', parameters });

test('source provenance follows explicit aliases through nested components and respects local overrides', () => {
    const root = graph({ id: 'transformer', kind: 'supernode', label: 'Transformer', templateId: 'transformer-body', parameterValues: { n_heads: 8 } });
    const templates = {
        'transformer-body': { id: 'transformer-body', freeParameters: [{ name: 'n_heads', type: 'int', default: 12 }], graph: graph({ id: 'attention', kind: 'supernode', label: 'Attention', templateId: 'attention-body', parameterValues: { H: '$n_heads' } }) },
        'attention-body': { id: 'attention-body', freeParameters: [{ name: 'H', type: 'int', default: 12 }], graph: graph({ id: 'heads', kind: 'repeat', label: 'Heads', templateId: 'head-body', bindings: { H: '$H' } }) },
        'head-body': { id: 'head-body', freeParameters: [{ name: 'H', type: 'int', default: 12 }], graph: graph() },
    };
    let frames = parameterFrames(root, ['transformer', 'attention', 'heads'], templates);
    assert.deepEqual(parameterOrigin('H', frames).path, []);
    assert.equal(parameterOrigin('H', frames).name, 'n_heads');
    assert.equal(parameterOrigin('H', frames).nodeId, 'transformer');
    templates['attention-body'].graph.nodes.heads.bindings.H = 3;
    frames = parameterFrames(root, ['transformer', 'attention', 'heads'], templates);
    assert.equal(parameterOrigin('H', frames).nodeId, 'heads');
    assert.deepEqual(parameterOrigin('H', frames).path, ['transformer', 'attention']);
    assert.equal(availableParameters(frames).filter((p) => p.name === 'H').length, 1);
});

test('dimension references and dependent defaults appear among parameter consumers', () => {
    const template = { freeParameters: [{ name: 'head_width', default: 'D / H' }], graph: graph(layer('weights', { shape: ['$D', 'D * 4'], init: 'normal' }), { id: 'heads', kind: 'repeat', label: 'Heads', count: '$H', bindings: { dropout: '$dropout' } }) };
    assert.deepEqual(parameterConsumers(template, 'D').map((use) => [use.label, use.field]), [['weights', 'shape'], ['head_width', 'default']]);
    assert.deepEqual(parameterConsumers(template, 'H').map((use) => use.field), ['count', 'default']);
    assert.deepEqual(parameterReferences(['1e-5', '$width', 'width * 2', '.5']), ['width']);
    assert.deepEqual(resolvedFieldValue(['$D', 'D * 4'], { D: 64 }), [64, 256]);
    assert.equal(resolvedFieldValue('D / H', { D: null, H: 8 }), null);
});

test('exposing a field preserves its value and changes only the named field atomically', () => {
    const template = { id: 'head', freeParameters: [], graph: graph(layer('dropout', { p: 0.15, inplace: false })) };
    const result = exposeTemplateField(template, { nodeId: 'dropout', field: 'p', name: 'attention_dropout', spec: { name: 'p', type: 'float', required: true }, defaultValue: 0.15 });
    assert.equal(template.graph.nodes.dropout.parameters.p, 0.15, 'Original remains immutable');
    assert.deepEqual(result.graph.nodes.dropout.parameters, { p: '$attention_dropout', inplace: false });
    assert.deepEqual(result.freeParameters, [{ name: 'attention_dropout', type: 'float', default: 0.15 }]);
    assert.equal(exposeTemplateField(result, { nodeId: 'dropout', field: 'p', name: 'attention_dropout' }), null, 'No duplicate parameter');
    assert.equal(exposeTemplateField(template, { nodeId: 'dropout', field: 'p', name: 'bad name' }), null);
});

test('exposed count, nested bindings, and tuple dimensions preserve their types', () => {
    const template = { id: 'body', graph: graph({ id: 'repeat', kind: 'repeat', count: 4, bindings: { width: 64 } }, layer('weight', { shape: [12, 64] })), freeParameters: [] };
    const count = exposeTemplateField(template, { nodeId: 'repeat', field: 'count', name: 'copies', spec: { type: 'int' }, defaultValue: 4 });
    assert.equal(count.graph.nodes.repeat.count, '$copies');
    const width = exposeTemplateField(count, { nodeId: 'repeat', field: 'width', name: 'width', spec: { type: 'int' }, defaultValue: 64 });
    assert.equal(width.graph.nodes.repeat.bindings.width, '$width');
    const shape = exposeTemplateField(width, { nodeId: 'weight', field: 'shape', name: 'weight_shape', spec: { type: 'tuple' }, defaultValue: [12, 64] });
    assert.deepEqual(shape.freeParameters.at(-1).default, [12, 64]);
    assert.equal(shape.graph.nodes.weight.parameters.shape, '$weight_shape');
});

test('store exposure is a single undoable edit across definition and field', async () => {
    const saved = new Map();
    globalThis.window = { localStorage: { getItem: (key) => saved.get(key) ?? null, setItem: (key, value) => saved.set(key, value), removeItem: (key) => saved.delete(key) } };
    const { useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
    const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
    const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');
    const body = { id: 'body', name: 'Body', freeParameters: [], graph: graph(layer('drop', { p: .1 })) };
    templates.setState({ templates: { body } });
    architecture.setState({ graph: graph({ id: 'component', kind: 'supernode', templateId: 'body', parameterValues: {} }), currentPath: ['component'], openPaths: [['component']] });
    history.getState().clear();
    assert.equal(templates.getState().exposeField('body', { nodeId: 'drop', field: 'p', name: 'dropout', spec: { type: 'float' }, defaultValue: .1 }), true);
    assert.equal(history.getState().undoCount, 1);
    history.getState().undo();
    assert.deepEqual(templates.getState().templates.body, body);
    history.getState().redo();
    assert.equal(templates.getState().templates.body.graph.nodes.drop.parameters.p, '$dropout');
    assert.equal(templates.getState().templates.body.freeParameters[0].default, .1);
});
