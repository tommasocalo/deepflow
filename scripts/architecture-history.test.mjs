import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import { createEdge, createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { createDragBatcher } from '../src/components/architecture/flowAdapter.js';
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
    assert.deepEqual(Object.keys(architectureSaved).sort(), ['currentPath', 'graph', 'openPaths']);
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
