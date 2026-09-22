import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import { createEdge, createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { createDragBatcher } from '../src/components/architecture/flowAdapter.js';
import { resolveDimensionTuple } from '../src/domain/expressions.js';
import { architectureHistoryCommand } from '../src/hooks/architectureHistoryKeys.js';

const saved = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => saved.get(key) ?? null,
    setItem: (key, value) => saved.set(key, value),
    removeItem: (key) => saved.delete(key),
} };
const { useArchitectureStore: architecture, createArchitecturePaneStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const { useArchitectureHistory: history, startArchitectureSession } = await import('../src/stores/architectureHistory.js');
const position = { x: 100, y: 100 };
const layer = (id, key = 'torch.nn.Linear') => createLayerNode({ id, libraryKey: key, position, entry: getEntry(key) });
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const panes = [];
const paneAt = (path = []) => {
    const pane = createArchitecturePaneStore({ path });
    panes.push(pane);
    return pane;
};
const document = () => ({ graph: architecture.getState().graph, templates: templates.getState().templates });

beforeEach(() => {
    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(layer('a'), layer('b'), layer('c')), currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null, inspectedIterations: {}, showDimensions: true, showGradientFlow: false });
    history.getState().clear();
    saved.clear();
});
afterEach(() => { for (const pane of panes.splice(0)) pane.dispose(); });

function repeatedFixture() {
    const template = createTemplate({ id: 'body', name: 'Shared body', graph: graphOf(layer('linear')) });
    const first = createRepeatNode({ id: 'repeat', templateId: 'body', count: 4, mode: 'parallel', position });
    const second = createSuperNodeInstance({ id: 'other', templateId: 'body', position });
    templates.setState({ templates: { body: template } });
    architecture.setState({ graph: graphOf(first, second) });
    return template;
}

test('repeat mode, bindings and operation type can be undone and redone in order', () => {
    repeatedFixture();
    const initial = document();
    architecture.getState().patchRepeatNode('repeat', { mode: 'sequential', count: 12 });
    architecture.getState().patchRepeatBinding('repeat', 'width', 'D / H');
    architecture.getState().openLevel(['repeat']);
    architecture.getState().patchNode('linear', { libraryKey: 'torch.nn.Dropout', parameters: { p: 0.2 } });
    assert.equal(history.getState().undoCount, 3);
    history.getState().undo();
    assert.equal(architecture.getState().getViewGraph().nodes.linear.libraryKey, 'torch.nn.Linear');
    history.getState().undo();
    assert.equal(architecture.getState().graph.nodes.repeat.bindings.width, undefined);
    history.getState().undo();
    assert.deepEqual(document(), initial);
    assert.deepEqual(architecture.getState().currentPath, ['repeat'], 'Undo does not jump to the level where an earlier edit happened');
    assert.equal(history.getState().canUndo, false);
    assert.equal(history.getState().redoCount, 3);
    history.getState().redo(); history.getState().redo(); history.getState().redo();
    assert.equal(architecture.getState().graph.nodes.repeat.mode, 'sequential');
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.p, 0.2);
});

test('shared template changes undo across panes while preserving each pane view and selection', () => {
    repeatedFixture();
    const left = paneAt(['repeat']);
    const right = paneAt(['other']);
    right.getState().selectOnly('linear');
    right.getState().setExpanded('linear');
    left.getState().patchLayerParameters('linear', { out_features: 987 });
    assert.equal(right.getState().getViewGraph().nodes.linear.parameters.out_features, 987);
    history.getState().undo();
    assert.notEqual(right.getState().getViewGraph().nodes.linear.parameters.out_features, 987);
    assert.deepEqual(right.getState().selectedNodeIds, ['linear']);
    assert.equal(right.getState().expandedNodeId, 'linear');
    assert.deepEqual(left.getState().currentPath, ['repeat']);
    assert.deepEqual(right.getState().currentPath, ['other']);
    history.getState().redo();
    assert.equal(left.getState().getViewGraph().nodes.linear.parameters.out_features, 987);
});

test('standard component creation and recursive copy each occupy one undo step', () => {
    const initial = document();
    const id = architecture.getState().addStandardComponent('learned-position', position);
    assert.equal(history.getState().undoCount, 1, 'All nested definitions are part of the insertion');
    const created = document();
    const sourceId = created.graph.nodes[id].templateId;
    const copyId = architecture.getState().cloneComponent(id, 'Independent positions');
    assert.notEqual(copyId, sourceId);
    const copied = document();
    assert.equal(history.getState().undoCount, 2);
    const side = paneAt([id]);
    history.getState().undo();
    assert.deepEqual(document(), created);
    assert.deepEqual(side.getState().currentPath, [id], 'Restore templates and graph before validating a surviving path');
    assert.equal(side.getState().getLevel().templateId, sourceId);
    history.getState().undo();
    assert.deepEqual(document(), initial);
    assert.deepEqual(side.getState().currentPath, []);
    assert.equal(side.getState().expandedNodeId, null);
    history.getState().redo(); history.getState().redo();
    assert.deepEqual(document(), copied);
});

test('grouping restores original nodes, edges and template catalog in one undo', () => {
    const edgeA = createEdge({ source: 'a', target: 'b' });
    const edgeB = createEdge({ source: 'b', target: 'c' });
    architecture.setState({ graph: { ...architecture.getState().graph, edges: [edgeA, edgeB] } });
    const before = document();
    const templateId = architecture.getState().groupSelection(['b'], 'Hidden layer');
    assert.ok(templateId);
    const grouped = document();
    assert.equal(history.getState().undoCount, 1);
    history.getState().undo();
    assert.deepEqual(document(), before);
    history.getState().redo();
    assert.deepEqual(document(), grouped);
});

test('connections preserve ids, handles and gradient metadata through undo/redo', () => {
    const initial = document();
    const edge = architecture.getState().connect({ source: 'a', target: 'b' });
    architecture.getState().setEdgeGradientOp(edge, 'reverse', 0.25);
    architecture.getState().reconnectEdge(edge, { source: 'a', target: 'c', targetHandle: 'weight' });
    const rewired = document();
    architecture.getState().removeEdge(edge);
    history.getState().undo();
    assert.deepEqual(document(), rewired);
    assert.equal(architecture.getState().graph.edges[0].gradientOpLambda, 0.25);
    history.getState().undo();
    assert.equal(architecture.getState().graph.edges[0].target, 'b');
    history.getState().undo(); history.getState().undo();
    assert.deepEqual(document(), initial);
    history.getState().redo(); history.getState().redo(); history.getState().redo();
    assert.deepEqual(document(), rewired);
});

test('a group drag commits once; cancelled or stale gestures leave no undo entries', () => {
    const batcher = createDragBatcher();
    const context = architecture.getState().getMutationContext();
    const initial = document();
    batcher.begin(context);
    for (const x of [120, 150, 180]) {
        assert.equal(batcher.consume([
            { id: 'a', type: 'position', position: { x, y: 100 }, dragging: true },
            { id: 'b', type: 'position', position: { x, y: 200 }, dragging: true },
        ], context), null);
    }
    assert.equal(history.getState().canUndo, false);
    const drag = batcher.consume([
        { id: 'a', type: 'position', dragging: false }, { id: 'b', type: 'position', dragging: false },
    ], context);
    architecture.getState().moveNodes(drag.moves, drag.context);
    assert.equal(history.getState().undoCount, 1);
    history.getState().undo();
    assert.deepEqual(document(), initial);
    history.getState().redo();
    assert.equal(architecture.getState().graph.nodes.a.position.x, 180);
    assert.equal(architecture.getState().graph.nodes.b.position.x, 180);
    assert.equal(architecture.getState().moveNodes([{ id: 'a', position: { x: 999, y: 999 } }], context), false);
    assert.equal(history.getState().undoCount, 1);
});

test('deleting a mixed multi-selection is one step and reconciles all panes', () => {
    repeatedFixture();
    const rootSide = paneAt();
    const nestedSide = paneAt(['repeat']);
    const before = document();
    rootSide.getState().selectOnly('other');
    architecture.getState().deleteElements({ nodeIds: ['repeat', 'other'], edgeIds: [] });
    assert.equal(history.getState().undoCount, 1);
    assert.deepEqual(rootSide.getState().selectedNodeIds, []);
    assert.deepEqual(nestedSide.getState().currentPath, []);
    history.getState().undo();
    assert.deepEqual(document(), before);
    assert.equal(rootSide.getState().graph, architecture.getState().graph);
});

test('template metadata and exposed parameter edits participate in the same history', () => {
    repeatedFixture();
    const before = document();
    history.getState().transaction('Configure shared component', () => {
        templates.getState().patchTemplate('body', { name: 'Shared projection' });
        templates.getState().addFreeParameter('body', { name: 'width', type: 'int', default: 64 });
        templates.getState().patchFreeParameter('body', 'width', { default: 128 });
    });
    assert.equal(history.getState().undoCount, 1);
    assert.equal(history.getState().undoLabel, 'Configure shared component');
    const after = document();
    history.getState().undo();
    assert.deepEqual(document(), before);
    history.getState().redo();
    assert.deepEqual(document(), after);
});

test('navigation, selection, display toggles and no-op patches do not consume or erase history', () => {
    repeatedFixture();
    architecture.getState().patchRepeatNode('repeat', { count: 8 });
    history.getState().undo();
    architecture.getState().selectOnly('repeat');
    architecture.getState().setExpanded('repeat');
    architecture.getState().setShowDimensions(false);
    architecture.getState().setShowGradientFlow(true);
    architecture.getState().openLevel(['repeat']);
    architecture.getState().setInspectedIteration(['repeat'], 2);
    architecture.getState().patchNode('linear', { label: architecture.getState().getViewGraph().nodes.linear.label });
    architecture.getState().moveNodes([{ id: 'linear', position }]);
    assert.equal(history.getState().undoCount, 0);
    assert.equal(history.getState().redoCount, 1);
    history.getState().redo();
    assert.equal(architecture.getState().graph.nodes.repeat.count, 8);
    assert.equal(architecture.getState().showDimensions, false);
    assert.equal(architecture.getState().showGradientFlow, true);
    assert.equal(architecture.getState().inspectedIterations['["repeat"]'], 2);
});

test('a new edit after Undo discards Redo; an external document replacement clears both', () => {
    architecture.getState().patchLayerParameters('a', { out_features: 17 });
    history.getState().undo();
    architecture.getState().patchLayerParameters('b', { out_features: 19 });
    assert.equal(history.getState().canRedo, false);
    assert.equal(history.getState().undoCount, 1);
    architecture.setState({ graph: graphOf(layer('loaded')) });
    assert.equal(history.getState().canUndo, false);
    assert.equal(history.getState().canRedo, false);
    assert.equal(history.getState().undo(), false);
    assert.ok(architecture.getState().graph.nodes.loaded);
});

test('example/initial document sessions suppress construction steps; persisted state contains no history', () => {
    architecture.getState().patchLayerParameters('a', { out_features: 17 });
    startArchitectureSession(() => {
        architecture.getState().resetCurrentGraph();
        architecture.getState().addStandardComponent('attention', position);
        architecture.getState().addLayerByLibraryKey('torch.nn.ReLU', position);
    });
    assert.equal(history.getState().canUndo, false);
    assert.equal(history.getState().canRedo, false);
    architecture.getState().addStandardComponent('linear', position);
    const architectureSaved = JSON.parse(saved.get('deepflow:architecture-graph')).state;
    const templatesSaved = JSON.parse(saved.get('deepflow:templates')).state;
    assert.deepEqual(Object.keys(architectureSaved).sort(), ['currentPath', 'graph', 'inspectedIterations', 'openPaths']);
    assert.deepEqual(Object.keys(templatesSaved), ['templates']);
});

test('history is bounded and disposed panes cannot mutate it or follow restored snapshots', () => {
    const pane = paneAt();
    const disposedState = pane.getState();
    pane.dispose();
    for (let index = 0; index < 110; index++) architecture.getState().patchNode('a', { label: `Layer ${index}` });
    assert.equal(history.getState().undoCount, 100);
    assert.equal(pane.getState(), disposedState);
    pane.getState().patchNode('a', { label: 'stale' });
    assert.equal(history.getState().undoCount, 100);
    for (let index = 0; index < 100; index++) history.getState().undo();
    assert.equal(architecture.getState().graph.nodes.a.label, 'Layer 9');
    assert.equal(history.getState().undo(), false);
    assert.equal(pane.getState(), disposedState);
});

test('shortcuts support Mac/Windows redo and preserve native text undo inside dialogs', () => {
    const dom = new JSDOM('<button></button><select></select><input type="checkbox"><dialog><input type="text"><textarea></textarea><span contenteditable="true"><b>edit</b></span></dialog>');
    const doc = dom.window.document;
    const event = (target, fields = {}) => ({ key: 'z', ctrlKey: true, target, ...fields });
    assert.equal(architectureHistoryCommand(event(doc.querySelector('button'))), 'undo');
    assert.equal(architectureHistoryCommand(event(doc.querySelector('select'))), 'undo');
    assert.equal(architectureHistoryCommand(event(doc.querySelector('[type="checkbox"]'))), 'undo');
    assert.equal(architectureHistoryCommand(event(doc.body, { ctrlKey: false, metaKey: true })), 'undo');
    assert.equal(architectureHistoryCommand(event(doc.body, { shiftKey: true })), 'redo');
    assert.equal(architectureHistoryCommand(event(doc.body, { key: 'y' })), 'redo');
    for (const selector of ['[type="text"]', 'textarea', 'b']) {
        assert.equal(architectureHistoryCommand(event(doc.querySelector(selector))), null, selector);
    }
    assert.equal(architectureHistoryCommand(event(doc.body, { altKey: true })), null);
    assert.equal(architectureHistoryCommand(event(doc.body, { isComposing: true })), null);
    assert.equal(architectureHistoryCommand(event(doc.body, { defaultPrevented: true })), null);
    dom.window.close();
});

test('repeat to Single round-trip remembers expression count/index, metadata, edges and bindings', () => {
    const template = repeatedFixture();
    const repeat = architecture.getState().graph.nodes.repeat;
    const configured = { ...repeat, count: '$heads', iterationVariable: 'head', label: 'Attention heads', annotation: { author: 'user' },
        bindings: { width: 'D * 2 ** head', chosen: [32, 64, 128], scale: '1e-3 * (head + 1)', untouched: '$head_dim' } };
    const edge = createEdge({ source: 'repeat', target: 'other', sourceHandle: 'output', targetHandle: 'features', gradientOp: 'reverse' });
    architecture.setState({ graph: { nodes: { ...architecture.getState().graph.nodes, repeat: configured }, edges: [edge] } });
    const before = document();
    assert.equal(architecture.getState().getSingleConversionIssue('repeat'), null);
    assert.equal(architecture.getState().convertRepeatToInstance('repeat'), 'repeat');
    const single = architecture.getState().graph.nodes.repeat;
    assert.equal(single.kind, 'supernode');
    assert.deepEqual(single.parameterValues, { width: 'D * 2 ** 0', chosen: 32, scale: 0.001, untouched: '$head_dim' });
    assert.equal(single.parameterValues.width.includes('D'), true, 'Upstream parameters are not frozen to their current numeric values');
    assert.equal(single.annotation, configured.annotation);
    assert.equal(single.position, configured.position);
    assert.equal(single.templateId, template.id);
    assert.equal(architecture.getState().graph.edges[0], edge);
    assert.equal(architecture.getState().convertInstanceToRepeat('repeat', undefined, 'sequential'), 'repeat');
    const restored = architecture.getState().graph.nodes.repeat;
    assert.deepEqual(restored, { ...configured, mode: 'sequential' });
    assert.equal(restored.count, '$heads');
    assert.equal(history.getState().undoCount, 2);
    history.getState().undo(); history.getState().undo();
    assert.deepEqual(document(), before);
    history.getState().redo(); history.getState().redo();
    assert.deepEqual(architecture.getState().graph.nodes.repeat, restored);
});

test('edited Single values override remembered formulas while untouched values and count survive', () => {
    repeatedFixture();
    architecture.getState().patchRepeatNode('repeat', { count: 12, iterationVariable: 'j', bindings: { width: '64 * 2 ** j', dropout: [0.1, 0.2], removed: 'j + 1' } });
    architecture.getState().convertRepeatToInstance('repeat');
    architecture.getState().patchNode('repeat', { parameterValues: { width: '$D', dropout: 0.1, extra: '$H' } });
    architecture.getState().convertInstanceToRepeat('repeat', undefined, 'sequential');
    let repeat = architecture.getState().graph.nodes.repeat;
    assert.equal(repeat.count, 12);
    assert.equal(repeat.iterationVariable, 'j');
    assert.deepEqual(repeat.bindings, { width: '$D', dropout: [0.1, 0.2], extra: '$H' });
    architecture.getState().convertRepeatToInstance('repeat');
    architecture.getState().convertInstanceToRepeat('repeat', '$new_count', 'parallel');
    repeat = architecture.getState().graph.nodes.repeat;
    assert.equal(repeat.count, '$new_count', 'An explicit count always overrides remembered settings');
    assert.equal(repeat.mode, 'parallel');
});

test('index-dependent defaults and tuple/array bindings project safely and return to inherited defaults', () => {
    repeatedFixture();
    templates.getState().patchTemplate('body', { freeParameters: [
        { name: 'width', type: 'int', default: 'D * (i + 1)' },
        { name: 'nullable', type: 'int', nullable: true, default: 'i + 1' },
    ] });
    architecture.getState().patchRepeatNode('repeat', { bindings: { nullable: null, shape: '(i, 3)', tupleSequence: [['i + 1', '$D'], [2, 3]] } });
    const before = architecture.getState().graph.nodes.repeat;
    architecture.getState().convertRepeatToInstance('repeat');
    assert.deepEqual(architecture.getState().graph.nodes.repeat.parameterValues, {
        width: 'D * (0 + 1)', nullable: null, shape: '(0, 3)', tupleSequence: [1, '$D'],
    });
    assert.deepEqual(resolveDimensionTuple(architecture.getState().graph.nodes.repeat.parameterValues.shape), [0, 3]);
    architecture.getState().convertInstanceToRepeat('repeat');
    assert.deepEqual(architecture.getState().graph.nodes.repeat, before, 'Unmodified inherited defaults remain inherited after restoring repeat');
});

test('Single conversion blocks free internal index references without mutating graph or history', () => {
    repeatedFixture();
    templates.getState().patchTemplateLayerParameters('body', 'linear', { out_features: '64 * 2 ** i' });
    history.getState().clear();
    const before = document();
    assert.match(architecture.getState().getSingleConversionIssue('repeat'), /uses the repeat index “i” directly/);
    assert.equal(architecture.getState().convertRepeatToInstance('repeat'), null);
    assert.deepEqual(document(), before);
    assert.equal(history.getState().canUndo, false);
    // An explicit template parameter gives those operations a valid Single scope.
    templates.getState().addFreeParameter('body', { name: 'i', type: 'int', default: 'i' });
    assert.equal(architecture.getState().getSingleConversionIssue('repeat'), null);
    architecture.getState().convertRepeatToInstance('repeat');
    assert.equal(architecture.getState().graph.nodes.repeat.parameterValues.i, 0);
});

test('free index checks respect nested repeat scopes and inspect each independently bound instance', () => {
    repeatedFixture();
    const child = createTemplate({ id: 'child', name: 'Child', graph: graphOf(layer('child-linear')), freeParameters: [{ name: 'width', type: 'int', default: '64 * (i + 1)' }] });
    templates.setState({ templates: { ...templates.getState().templates, child } });
    templates.getState().patchTemplate('body', { graph: graphOf(
        createSuperNodeInstance({ id: 'safe', templateId: 'child', parameterValues: { width: 64 }, position }),
        createSuperNodeInstance({ id: 'unsafe', templateId: 'child', position }),
    ) });
    assert.match(architecture.getState().getSingleConversionIssue('repeat'), /index/);
    templates.getState().patchTemplate('body', { graph: graphOf(createRepeatNode({ id: 'nested', templateId: 'child', count: 2, iterationVariable: 'i', position })) });
    assert.equal(architecture.getState().getSingleConversionIssue('repeat'), null, 'A nested repeat defines its own index');
    templates.getState().patchTemplateNode('body', 'nested', { count: 'i + 1' });
    assert.match(architecture.getState().getSingleConversionIssue('repeat'), /index/, 'Its count is still evaluated in the parent scope');
});

test('literal operation names do not block Single while numeric formulas still do', () => {
    repeatedFixture();
    const output = { ...layer('output', 'deepflow.io.Output'), parameters: { name: 'i' } };
    templates.getState().patchTemplate('body', { graph: graphOf(output) });
    assert.equal(architecture.getState().getSingleConversionIssue('repeat'), null);
    const scale = { ...layer('scale', 'deepflow.op.Scale'), parameters: { factor: 'i + 1' } };
    templates.getState().patchTemplate('body', { graph: graphOf(output, scale) });
    assert.match(architecture.getState().getSingleConversionIssue('repeat'), /index/);
});
