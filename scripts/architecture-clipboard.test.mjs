import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import { copyGraphSelection, pasteGraphSelection, readGraphClipboard } from '../src/domain/architecture/clipboard.js';

globalThis.window = { localStorage: { getItem: () => null, setItem() {}, removeItem() {} } };
const { useArchitectureStore: store, createArchitecturePaneStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');
const layer = (id, x = 0) => ({ id, kind: 'layer', libraryKey: 'deepflow.op.Scale', label: id, position: { x, y: 50 }, parameters: { factor: '$width' } });
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const panes = [];
beforeEach(() => {
    templates.setState({ templates: {} });
    store.setState({ graph: graphOf(layer('a'), layer('b', 100), layer('c', 200)), currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null });
    history.getState().clear();
});
afterEach(() => { for (const pane of panes.splice(0)) pane.dispose(); });

test('copy is an immutable snapshot containing only selected nodes and their internal cables', () => {
    const graph = { ...store.getState().graph, edges: [
        { id: 'ab', source: 'a', target: 'b', sourceHandle: 'output', targetHandle: 'input', gradientOp: 'reverse', gradientOpLambda: 0.25 },
        { id: 'bc', source: 'b', target: 'c' },
    ] };
    const copied = copyGraphSelection(graph, ['a', 'b', 'a', 'missing']);
    assert.deepEqual(copied.nodes.map((node) => node.id), ['a', 'b']);
    assert.deepEqual(copied.edges, [graph.edges[0]]);
    graph.nodes.a.parameters.factor = 777;
    assert.equal(copied.nodes[0].parameters.factor, '$width');
    assert.equal(copyGraphSelection(graph, []), null);
    const pasted = pasteGraphSelection(graph, copied, { offset: { x: 36, y: 36 } });
    const [a, b] = pasted.nodeIds;
    assert.ok(a !== 'a' && b !== 'b' && a !== b);
    assert.deepEqual(pasted.graph.nodes[a].position, { x: 36, y: 86 });
    assert.equal(pasted.graph.nodes[b].position.x - pasted.graph.nodes[a].position.x, 100);
    assert.deepEqual(pasted.graph.edges.at(-1), { ...copied.edges[0], id: pasted.graph.edges.at(-1).id, source: a, target: b });
    assert.notEqual(pasted.graph.edges.at(-1).id, 'ab');
    assert.equal(graph.edges.length, 2);
});

test('paste duplicates repeat and component instances while retaining their shared definitions', () => {
    const component = { id: 'component', kind: 'supernode', templateId: 'body', parameterValues: { D: '$width' }, position: { x: 0, y: 0 } };
    const repeat = { id: 'repeat', kind: 'repeat', templateId: 'body', count: '$H', mode: 'parallel', bindings: { dropout: 0.2 }, position: { x: 0, y: 150 } };
    const graph = graphOf(component, repeat);
    const definitions = { body: { id: 'body', graph: graphOf(layer('inner')) } };
    const selection = copyGraphSelection(graph, ['component', 'repeat']);
    const result = pasteGraphSelection(graph, selection, { templates: definitions });
    for (const [index, id] of result.nodeIds.entries()) {
        const copy = result.graph.nodes[id];
        assert.equal(copy.templateId, 'body');
        assert.deepEqual({ ...copy, id: selection.nodes[index].id, position: selection.nodes[index].position }, selection.nodes[index]);
    }
    assert.equal(Object.keys(definitions).length, 1);
});

test('recursive component insertion and missing definitions reject the entire paste', () => {
    const root = { id: 'root', kind: 'supernode', templateId: 'outer', position: { x: 0, y: 0 } };
    const child = { ...root, id: 'child', templateId: 'inner' };
    const templates = { outer: { graph: graphOf(child) }, inner: { graph: graphOf(layer('n')) } };
    const selection = copyGraphSelection(graphOf(root, layer('other')), ['root', 'other']);
    assert.match(pasteGraphSelection(graphOf(), selection, { templates, templateId: 'inner' }).error, /cannot contain itself/);
    assert.match(pasteGraphSelection(graphOf(), selection, { templates: {} }).error, /no longer in this project/);
    assert.match(pasteGraphSelection(graphOf(), selection, { templates, templateId: 'outer' }).error, /cannot contain itself/);
});

test('unrelated clipboard text and malformed selections are ignored', () => {
    assert.equal(readGraphClipboard('hello world'), null);
    const selection = copyGraphSelection(store.getState().graph, ['a']);
    assert.deepEqual(readGraphClipboard(JSON.stringify(selection)), selection);
    assert.equal(readGraphClipboard(JSON.stringify({ ...selection, nodes: [...selection.nodes, ...selection.nodes] })), null);
    assert.equal(readGraphClipboard(JSON.stringify({ ...selection, edges: [{ source: 'a', target: 'outside' }] })), null);
    assert.equal(readGraphClipboard(JSON.stringify({ ...selection, nodes: [{ ...selection.nodes[0], position: { x: 'no', y: 0 } }] })), null);
});

test('paste is one undoable edit, preserves prior nodes and selects copies without opening settings', () => {
    const state = store.getState();
    state.selectOnly('a');
    state.setExpanded('a');
    const before = store.getState().graph;
    const copied = state.copySelection();
    const result = state.pasteSelection(copied, { x: 72, y: 72 }, state.getMutationContext());
    assert.equal(history.getState().undoCount, 1);
    assert.deepEqual(store.getState().selectedNodeIds, result.nodeIds);
    assert.equal(store.getState().expandedNodeId, null);
    assert.equal(store.getState().graph.nodes.a, before.nodes.a);
    const after = store.getState().graph;
    history.getState().undo();
    assert.deepEqual(store.getState().graph, before);
    history.getState().redo();
    assert.deepEqual(store.getState().graph, after);
});

test('paste targets the receiving pane and rejects stale levels and disposed panes', () => {
    const component = { id: 'component', kind: 'supernode', templateId: 'body', position: { x: 0, y: 0 } };
    templates.setState({ templates: { body: { id: 'body', graph: graphOf(layer('inner')) } } });
    store.setState({ graph: graphOf(component, layer('root-op')) });
    store.getState().selectOnly('root-op');
    const selection = store.getState().copySelection();
    const pane = createArchitecturePaneStore({ path: ['component'] });
    panes.push(pane);
    const rootBefore = store.getState().graph;
    const context = pane.getState().getMutationContext();
    const result = pane.getState().pasteSelection(selection, { x: 36, y: 36 }, context);
    assert.equal(store.getState().graph, rootBefore);
    assert.equal(templates.getState().templates.body.graph.nodes[result.nodeIds[0]].label, 'root-op');
    assert.equal(history.getState().undoCount, 1);
    history.getState().undo();
    assert.equal(Object.keys(templates.getState().templates.body.graph.nodes).length, 1);
    pane.getState().openLevel([]);
    assert.match(pane.getState().pasteSelection(selection, undefined, context).error, /destination changed/);
    assert.equal(history.getState().canUndo, false);
    pane.dispose();
    assert.match(pane.getState().pasteSelection(selection).error, /destination changed/);
});
