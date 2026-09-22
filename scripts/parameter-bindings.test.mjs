import assert from 'node:assert/strict';
import { test } from 'node:test';
import { availableParameters, exposedFieldParameter, exposeTemplateField, parameterConsumers, parameterFrames, parameterOrigin, parameterReferences, resolvedFieldValue, toggleTemplateFieldExposure } from '../src/domain/parameterBindings.js';
import { buildParameterScope } from '../src/domain/expressions.js';

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
    assert.deepEqual(resolvedFieldValue([1024, '$'], {}), [1024, null], 'An unfinished reference cannot be converted to a local value');
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

test('field eyes expose a friendly control and hide it without disconnecting consumers or overrides', () => {
    const original = { id: 'head', freeParameters: [], graph: graph(layer('dropout', { p: 0.15, inplace: false })) };
    const options = { nodeId: 'dropout', field: 'p', spec: { name: 'p', type: 'float', default: .5 }, label: 'Dropout' };
    const exposed = toggleTemplateFieldExposure(original, options);
    assert.equal(exposed.graph.nodes.dropout.parameters.p, '$dropout');
    assert.equal(exposedFieldParameter(exposed, 'dropout', 'p').name, 'dropout');
    assert.equal(exposedFieldParameter(exposed, 'dropout', 'p').default, .15);
    assert.equal(exposedFieldParameter(exposed, 'dropout', 'p').hidden, false);
    const withConsumers = {
        ...exposed,
        graph: graph(exposed.graph.nodes.dropout, layer('other', { p: '$dropout' })),
        freeParameters: [...exposed.freeParameters, { name: 'doubled', type: 'float', default: '$dropout * 2' }],
    };
    const hidden = toggleTemplateFieldExposure(withConsumers, options);
    assert.equal(hidden.graph, withConsumers.graph, 'Visibility never changes fields or other consumers');
    assert.equal(hidden.freeParameters.length, 2, 'The parameter remains in the runtime scope');
    assert.equal(exposedFieldParameter(hidden, 'dropout', 'p').hidden, true);
    assert.deepEqual(buildParameterScope(hidden.freeParameters, { dropout: .27 }), { dropout: .27, doubled: .54 });
    assert.deepEqual(availableParameters([{ instance: { label: 'Head' }, template: hidden }]).map((parameter) => parameter.name), ['dropout', 'doubled'], 'Hidden controls remain available in expressions');
    const shown = toggleTemplateFieldExposure(hidden, options);
    assert.deepEqual(shown, withConsumers, 'Showing a control restores visibility without making a second alias');
    assert.equal(original.graph.nodes.dropout.parameters.p, .15);
    assert.deepEqual(original.freeParameters, []);
});

test('exposure avoids local, inherited and self-reference collisions without evaluating formulas', () => {
    const original = { freeParameters: [{ name: 'dropout', type: 'float', default: .1 }], graph: graph(layer('dropout', { p: '$dropout_3 * 0.5' })) };
    const exposed = toggleTemplateFieldExposure(original, { nodeId: 'dropout', field: 'p', spec: { type: 'float' }, reservedNames: ['dropout_2', '$dropout_4'] });
    assert.equal(exposed.graph.nodes.dropout.parameters.p, '$dropout_5');
    assert.equal(exposed.freeParameters.at(-1).default, '$dropout_3 * 0.5');
    assert.equal(buildParameterScope(exposed.freeParameters, {}, { dropout_3: .17 }).dropout_5, .085);
    assert.equal(exposedFieldParameter(original, 'dropout', 'p'), null, 'An arithmetic dependency is not a direct exposed control');
    const ancestorReference = { freeParameters: [], graph: graph(layer('dropout', { p: '$dropout' })) };
    assert.equal(exposedFieldParameter(ancestorReference, 'dropout', 'p'), null, 'An ancestor parameter is not exposed by this component yet');
    const forwarded = toggleTemplateFieldExposure(ancestorReference, { nodeId: 'dropout', field: 'p', spec: { type: 'float' } });
    assert.equal(forwarded.graph.nodes.dropout.parameters.p, '$dropout_2');
    assert.equal(forwarded.freeParameters[0].default, '$dropout');
});

test('visibility aliases preserve tuple formulas, explicit null, false, zero and nested component bindings', () => {
    const original = { freeParameters: [], graph: graph(
        layer('weight', { shape: ['$width', '2 * $width'], padding_idx: null, requires_grad: false, scale: 0, reset: null }),
        { id: 'child', kind: 'supernode', parameterValues: { dropout: .2 } },
        { id: 'repeat', kind: 'repeat', count: 3, bindings: { width: 64 } },
    ) };
    let current = original;
    const cases = [
        { nodeId: 'weight', field: 'shape', spec: { name: 'shape', type: 'tuple' }, expected: ['$width', '2 * $width'] },
        { nodeId: 'weight', field: 'padding_idx', spec: { name: 'padding_idx', type: 'int', nullable: true, default: 5 }, defaultValue: 5, expected: null },
        { nodeId: 'weight', field: 'requires_grad', spec: { type: 'bool', default: true }, expected: false },
        { nodeId: 'weight', field: 'scale', spec: { type: 'float', default: 1 }, expected: 0 },
        { nodeId: 'weight', field: 'reset', spec: { type: 'float', default: 1 }, defaultValue: null, expected: 1 },
        { nodeId: 'child', field: 'dropout', spec: { type: 'float' }, expected: .2 },
        { nodeId: 'repeat', field: 'count', spec: { type: 'int' }, expected: 3 },
        { nodeId: 'repeat', field: 'width', spec: { type: 'int' }, expected: 64 },
    ];
    for (const { expected, ...options } of cases) {
        current = toggleTemplateFieldExposure(current, options);
        assert.deepEqual(exposedFieldParameter(current, options.nodeId, options.field).default, expected);
    }
    assert.equal(current.freeParameters.find((parameter) => parameter.name === 'shape').editor, 'dimensions');
    assert.equal(current.graph.nodes.child.parameterValues.dropout, '$dropout');
    assert.equal(current.graph.nodes.repeat.count, '$count');
    assert.equal(current.graph.nodes.repeat.bindings.width, '$width');
    assert.equal(buildParameterScope(current.freeParameters).padding_idx, null);
    assert.equal(toggleTemplateFieldExposure(original, { nodeId: 'weight', field: 'shape', spec: { derived: { axis: -1 } } }), null);
    assert.equal(toggleTemplateFieldExposure({ ...original, graph: graph({ id: 'auto', kind: 'repeat', count: 'auto' }) }, { nodeId: 'auto', field: 'count', spec: { type: 'int' } }), null);
});

test('field visibility and initial exposure are each atomic undoable changes', async () => {
    const { useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
    const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
    const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');
    const body = { id: 'body', name: 'Body', freeParameters: [], graph: graph(layer('drop', { p: .1 })) };
    templates.setState({ templates: { body } });
    architecture.setState({ graph: graph({ id: 'component', kind: 'supernode', templateId: 'body', parameterValues: { dropout: .3 } }), currentPath: ['component'], openPaths: [['component']] });
    history.getState().clear();
    const options = { nodeId: 'drop', field: 'p', spec: { type: 'float' } };
    assert.equal(templates.getState().toggleFieldExposure('body', options), true);
    assert.equal(history.getState().undoCount, 1);
    const exposed = templates.getState().templates.body;
    assert.equal(templates.getState().toggleFieldExposure('body', options), true);
    assert.equal(history.getState().undoCount, 2);
    assert.equal(templates.getState().templates.body.freeParameters[0].hidden, true);
    assert.equal(architecture.getState().graph.nodes.component.parameterValues.dropout, .3);
    history.getState().undo();
    assert.deepEqual(templates.getState().templates.body, exposed);
    history.getState().undo();
    assert.deepEqual(templates.getState().templates.body, body);
    history.getState().redo();
    history.getState().redo();
    assert.equal(templates.getState().templates.body.freeParameters[0].hidden, true);
    assert.equal(templates.getState().templates.body.graph.nodes.drop.parameters.p, '$dropout');
    assert.equal(templates.getState().toggleFieldExposure('missing', options), false);
    assert.equal(history.getState().undoCount, 2, 'Rejected changes do not add history entries');
});
