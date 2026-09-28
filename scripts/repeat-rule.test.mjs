import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate, resolveGraphLevel } from '../src/domain/architecture/index.js';
import { selectRepeatInstances, compileRepeatRule } from '../src/domain/architecture/repeatRule.js';
import { getEntry } from '../src/domain/library/index.js';

const saved = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => saved.get(key) ?? null,
    setItem: (key, value) => saved.set(key, value),
    removeItem: (key) => saved.delete(key),
} };
const { useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');

const position = { x: 100, y: 100 };
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const linear = () => createLayerNode({ id: 'linear', libraryKey: 'torch.nn.Linear', entry: getEntry('torch.nn.Linear'),
    position, parameters: { in_features: 16, out_features: 16, bias: true } });
const document = () => ({ graph: architecture.getState().graph, templates: templates.getState().templates });
const valueAt = (index) => resolveGraphLevel(architecture.getState().graph, ['blocks'], templates.getState().templates,
    { '["blocks"]': index }).graph.nodes.linear.parameters.out_features;

beforeEach(() => {
    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(), currentPath: [], openPaths: [], inspectedIterations: {},
        repeatSelections: {}, repeatEditModes: {}, repeatEditError: null });
    history.getState().clear();
    saved.clear();
});

function fixture() {
    const body = createTemplate({ id: 'body', name: 'Block', graph: graphOf(linear()),
        freeParameters: [{ name: 'width', type: 'int', default: 1 }] });
    templates.setState({ templates: { body } });
    const blocks = createRepeatNode({ id: 'blocks', templateId: body.id, label: 'Blocks', mode: 'sequential',
        count: 4, position, bindings: { width: [2, 4, 6, 8] } });
    const other = createSuperNodeInstance({ id: 'other', templateId: body.id, label: 'Other', position });
    architecture.setState({ graph: { ...graphOf(blocks, other), parameters: [{ name: 'threshold', type: 'int', default: 5 }] } });
    architecture.getState().openLevel(['blocks']);
}

test('rules evaluate 1-based position and numeric local/global parameters without changing arithmetic evaluator', () => {
    fixture();
    const state = architecture.getState();
    const result = selectRepeatInstances({ graph: state.graph, path: ['blocks'], templates: templates.getState().templates,
        inspectedIterations: state.inspectedIterations, count: 4,
        rule: 'position % 2 == 0 && local.width > global.threshold' });
    assert.deepEqual(result.indices, [3]);
    assert.equal(compileRepeatRule('position == 1')({ position: 1, local: {}, global: {} }), true);
    assert.equal(compileRepeatRule('position == 1 || local.missing > 0')({ position: 1, local: {}, global: {} }), true);
    assert.throws(() => compileRepeatRule('local.width = 2'), /Unexpected character/);
    assert.throws(() => compileRepeatRule('window.alert(1)'), /Unknown variable/);
    assert.throws(() => compileRepeatRule('position + 1')({ position: 1 }), /return true or false/);
});

test('selected settings edit clones only snapshot occurrences and undoes atomically', () => {
    fixture();
    const state = architecture.getState();
    const indices = selectRepeatInstances({ graph: state.graph, path: ['blocks'], templates: templates.getState().templates,
        count: 4, rule: 'local.width > global.threshold' }).indices;
    assert.deepEqual(indices, [2, 3]);
    assert.equal(state.setRepeatSelection(['blocks'], indices, 'local.width > global.threshold'), true);
    assert.deepEqual(architecture.getState().repeatSelections['["blocks"]'].indices, [2, 3]);
    const before = document();
    architecture.getState().patchLayerParameters('linear', { out_features: 32 });
    assert.deepEqual([0, 1, 2, 3].map(valueAt), [16, 16, 32, 32]);
    assert.equal(resolveGraphLevel(architecture.getState().graph, ['other'], templates.getState().templates).graph.nodes.linear.parameters.out_features, 16);
    assert.equal(history.getState().undoCount, 1);
    assert.deepEqual(architecture.getState().repeatSelections['["blocks"]'].indices, [2, 3], 'selection is a snapshot');
    history.getState().undo();
    assert.deepEqual(document(), before);
    history.getState().redo();
    assert.deepEqual([0, 1, 2, 3].map(valueAt), [16, 16, 32, 32]);
});

test('explicit Current edits just the viewed occurrence, and All reaches local variants too', () => {
    fixture();
    architecture.getState().setInspectedIteration(['blocks'], 1);
    architecture.getState().setRepeatEditMode(['blocks'], 'current');
    architecture.getState().patchLayerParameters('linear', { out_features: 24 });
    assert.deepEqual([0, 1, 2, 3].map(valueAt), [16, 24, 16, 16]);
    architecture.getState().setRepeatEditMode(['blocks'], 'all');
    architecture.getState().patchLayerParameters('linear', { out_features: 48 });
    assert.deepEqual([0, 1, 2, 3].map(valueAt), [48, 48, 48, 48]);
});

test('editing an exposed repeat parameter applies to the selected indices as one binding-array edit', () => {
    fixture();
    architecture.getState().setRepeatSelection(['blocks'], [1, 3], 'position % 2 == 0');
    architecture.getState().openLevel([]);
    const before = document();
    architecture.getState().patchRepeatBinding('blocks', 'width', 10);
    assert.deepEqual(architecture.getState().graph.nodes.blocks.bindings.width, [2, 10, 6, 10]);
    assert.equal(history.getState().undoCount, 1);
    history.getState().undo();
    assert.deepEqual(document(), before);
});

test('selected parameter mode does not bulk-edit a structure or node label', () => {
    fixture();
    architecture.getState().setRepeatSelection(['blocks'], [1, 3], 'position % 2 == 0');
    architecture.getState().patchNode('linear', { label: 'Renamed' });
    assert.equal(architecture.getState().graph.nodes.blocks.structureOverrides, undefined);
    assert.equal(resolveGraphLevel(architecture.getState().graph, ['blocks'], templates.getState().templates,
        { '["blocks"]': 0 }).graph.nodes.linear.label, 'Renamed');
    assert.equal(resolveGraphLevel(architecture.getState().graph, ['blocks'], templates.getState().templates,
        { '["blocks"]': 1 }).graph.nodes.linear.label, 'Renamed');
});

test('a selected variant without the edited node aborts without partial changes', () => {
    fixture();
    architecture.getState().setInspectedIteration(['blocks'], 1);
    architecture.getState().setStructureScope('local');
    architecture.getState().removeNode('linear');
    architecture.getState().setInspectedIteration(['blocks'], 0);
    architecture.getState().setRepeatSelection(['blocks'], [0, 1], 'position <= 2');
    const before = document();
    architecture.getState().patchLayerParameters('linear', { out_features: 99 });
    assert.deepEqual(document(), before);
    assert.match(architecture.getState().repeatEditError.message, /Instance 2/);
});
