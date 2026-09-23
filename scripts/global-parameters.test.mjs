import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createEdge, createLayerNode, createSuperNodeInstance, createTemplate } from '../src/domain/architecture/index.js';
import { resolveLevelContext } from '../src/domain/architecture/levelContext.js';
import { getEntry } from '../src/domain/library/index.js';
import { globalParameterUses } from '../src/domain/parameterBindings.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';
import { validate } from '../src/validation/validator.js';

const storage = new Map();
globalThis.window = { localStorage: { getItem: (key) => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value), removeItem: (key) => storage.delete(key) } };
const { useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');

const position = { x: 0, y: 0 };
const layer = (id, key, parameters) => createLayerNode({ id, libraryKey: key, entry: getEntry(key), position, parameters });
const chain = (...nodes) => ({
    nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
    edges: nodes.slice(1).map((node, index) => createEdge({ source: nodes[index].id, target: node.id })),
});

function model({ shadow = false } = {}) {
    const body = createTemplate({ id: 'body', name: 'Block', freeParameters: shadow ? [{ name: 'd_model', type: 'int', default: 32 }] : [], graph: chain(
        layer('in', 'deepflow.template.Input', {}),
        layer('proj', 'torch.nn.Linear', { in_features: '$d_model', out_features: '$d_model' }),
        layer('out', 'deepflow.template.Output', {}),
    ) });
    const root = chain(
        layer('x', 'deepflow.io.Input', { shape: '(16,)' }),
        layer('embed', 'torch.nn.Linear', { in_features: 16, out_features: '$d_model' }),
        createSuperNodeInstance({ id: 'block', templateId: 'body', label: 'Block', position }),
        layer('y', 'deepflow.io.Output', {}),
    );
    root.parameters = [{ name: 'd_model', type: 'int', default: 64 }, { name: 'd_ff', type: 'int', default: '4 * d_model' }];
    return { root, templates: { body } };
}

test('global parameters resolve at the root and inside components', () => {
    const { root, templates } = model();
    const report = validate(root, { templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.embed, [null, 64]);
    assert.deepEqual(report.shapes.block, [null, 64]);
    const inside = resolveLevelContext(root, ['block'], templates);
    assert.equal(inside.scope.d_model, 64);
    assert.equal(inside.scope.d_ff, 256, 'A global can be computed from another');
    assert.match(generateArchitectureModule(root, { templates }), /Linear\(in_features=64, out_features=64/);
});

test('a component parameter with the same name shadows the global inside it', () => {
    const { root, templates } = model({ shadow: true });
    assert.equal(resolveLevelContext(root, ['block'], templates).scope.d_model, 32);
    assert.equal(validate(root, { templates }).ok, false, 'The block now expects 32 features but receives 64');
    const uses = globalParameterUses(root, templates, 'd_model');
    assert.deepEqual(uses.map((use) => [use.template ?? 'root', use.label, use.field]), [['root', 'Linear', 'out_features'], ['root', 'd_ff', 'default']]);
    assert.equal(globalParameterUses(model().root, model().templates, 'd_model').filter((use) => use.template === 'Block').length, 2);
});

test('global parameters are one undoable edit each and live in the saved graph', () => {
    architecture.setState({ graph: { nodes: {}, edges: [] }, currentPath: [], openPaths: [] });
    history.getState().clear();
    assert.equal(architecture.getState().addGlobalParameter({ name: 'n_layers', default: 12 }), true);
    assert.equal(architecture.getState().addGlobalParameter({ name: 'n_layers', default: 6 }), false, 'Names are unique');
    assert.equal(architecture.getState().addGlobalParameter({ name: 'bad name' }), false);
    architecture.getState().patchGlobalParameter('n_layers', { default: 24 });
    assert.deepEqual(architecture.getState().graph.parameters, [{ type: 'int', default: 24, name: 'n_layers' }]);
    assert.equal(history.getState().undoCount, 2);
    history.getState().undo();
    assert.equal(architecture.getState().graph.parameters[0].default, 12);
    architecture.getState().removeGlobalParameter('n_layers');
    assert.deepEqual(architecture.getState().graph.parameters, []);
    history.getState().undo();
    assert.equal(architecture.getState().graph.parameters.length, 1);
});

test('removing a used parameter writes its value into the fields that used it', async () => {
    const { inlineParameter, removeTemplateParameter, removeGlobalParameter } = await import('../src/domain/parameterBindings.js');
    assert.equal(inlineParameter('$d_model', 'd_model', 768), 768);
    assert.equal(inlineParameter('4 * d_model', 'd_model', 768), '4 * 768');
    assert.equal(inlineParameter('d_model_2 + $d_model', 'd_model', 768), 'd_model_2 + 768', 'Longer names are left alone');
    assert.deepEqual(inlineParameter(['$d_model', 'D / H'], 'd_model', 768), [768, 'D / H']);
    assert.equal(inlineParameter('x * $w', 'w', 'D / H'), 'x * (D / H)', 'A formula value keeps its grouping');

    const { root, templates } = model();
    const withDefault = { ...templates.body, freeParameters: [{ name: 'width', type: 'int', default: 48 }, { name: 'wide', type: 'int', default: '2 * width' }],
        graph: { ...templates.body.graph, nodes: { ...templates.body.graph.nodes, proj: { ...templates.body.graph.nodes.proj, parameters: { in_features: '$width', out_features: '$wide' } } } } };
    const removed = removeTemplateParameter(withDefault, 'width');
    assert.deepEqual(removed.freeParameters, [{ name: 'wide', type: 'int', default: '2 * 48' }]);
    assert.equal(removed.graph.nodes.proj.parameters.in_features, 48);

    const result = removeGlobalParameter(root, templates, 'd_model');
    assert.deepEqual(result.graph.parameters, [{ name: 'd_ff', type: 'int', default: '4 * 64' }]);
    assert.equal(result.graph.nodes.embed.parameters.out_features, 64);
    assert.deepEqual(result.templates.body.graph.nodes.proj.parameters, { in_features: 64, out_features: 64 });
    assert.equal(validate(result.graph, { templates: result.templates }).ok, true);
    const shadowed = model({ shadow: true });
    assert.equal(removeGlobalParameter(shadowed.root, shadowed.templates, 'd_model').templates.body, shadowed.templates.body, 'A component with its own d_model is untouched');
});
