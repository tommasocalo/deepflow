import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { JSDOM } from 'jsdom';
import { createEdge, createLayerNode, createSuperNodeInstance, createTemplate, isValidConnection, reconnectEdge } from '../src/domain/architecture/index.js';
import { isBlankCanvasDrop } from '../src/components/architecture/connectionGesture.js';

const memory = new Map();
let writes = [];
globalThis.window = { localStorage: {
    getItem: (key) => memory.get(key) ?? null,
    setItem: (key, value) => { memory.set(key, value); writes.push(key); },
    removeItem: (key) => memory.delete(key),
} };
const { useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const layer = (id) => createLayerNode({ id, libraryKey: 'torch.nn.Linear', position: { x: 100, y: 100 } });
const makeGraph = () => ({
    nodes: Object.fromEntries(['a', 'b', 'c', 'd'].map((id) => [id, layer(id)])),
    edges: [createEdge({ id: 'wire', source: 'a', target: 'b', gradientOp: 'reverse', gradientOpLambda: .5 })],
});
const context = () => ({ graph: architecture.getState().getViewGraph(), levelKey: JSON.stringify(architecture.getState().currentPath) });

beforeEach(() => {
    memory.clear();
    templates.setState({ templates: {} });
    architecture.setState({ graph: makeGraph(), currentPath: [], selectedNodeIds: [], expandedNodeId: null });
    writes = [];
});

test('reconnecting either endpoint preserves identity and training metadata, persists only on drop', () => {
    const original = architecture.getState().graph.edges[0];
    assert.equal(architecture.getState().reconnectEdge('wire', { source: 'a', target: 'c' }, context()), true);
    assert.deepEqual(architecture.getState().graph.edges[0], { ...original, target: 'c' });
    assert.equal(architecture.getState().reconnectEdge('wire', { source: 'd', target: 'c' }, context()), true);
    assert.deepEqual(architecture.getState().graph.edges[0], { ...original, source: 'd', target: 'c' });
    assert.equal(writes.length, 2);
    assert.equal(architecture.getState().reconnectEdge('wire', { source: 'd', target: 'c', sourceHandle: null, targetHandle: null }, context()), true);
    assert.equal(writes.length, 2, 'Releasing back on the same endpoint is a no-op');
});

test('invalid endpoints, self-connections and duplicates preserve the original wire', () => {
    architecture.getState().connect({ source: 'a', target: 'c' });
    const graph = architecture.getState().graph;
    writes = [];
    for (const connection of [
        { source: 'a', target: 'a' }, { source: 'missing', target: 'b' },
        { source: 'a', target: 'missing' }, { source: 'a', target: 'c' },
    ]) {
        assert.equal(architecture.getState().reconnectEdge('wire', connection, context()), false);
        assert.equal(architecture.getState().connect(connection, context()), null);
        assert.equal(architecture.getState().graph, graph);
    }
    assert.equal(architecture.getState().reconnectEdge('missing', { source: 'd', target: 'b' }, context()), false);
    assert.deepEqual(writes, []);
    assert.equal(isValidConnection(graph, { source: 'a', target: 'b' }, 'wire'), true);
    assert.equal(reconnectEdge(graph, 'wire', { source: 'a', target: 'c' }), graph);
});

test('detach removes only the intended edge and rejects repeated removals', () => {
    const untouched = architecture.getState().connect({ source: 'b', target: 'c' });
    writes = [];
    assert.equal(architecture.getState().removeEdge('wire', context()), true);
    assert.deepEqual(architecture.getState().graph.edges.map((edge) => edge.id), [untouched]);
    assert.equal(architecture.getState().removeEdge('wire', context()), false);
    assert.deepEqual(writes, ['deepflow:architecture-graph']);
});

test('navigation and concurrent edits prevent a stale reconnect, detach or new connection', () => {
    const root = architecture.getState().graph;
    const instance = createSuperNodeInstance({ id: 'block', templateId: 'template' });
    const inner = makeGraph();
    templates.setState({ templates: { template: createTemplate({ id: 'template', name: 'Block', graph: inner }) } });
    architecture.setState({ graph: { ...root, nodes: { ...root.nodes, block: instance } } });
    const started = context();
    architecture.setState({ currentPath: ['block'] });
    writes = [];
    assert.equal(architecture.getState().reconnectEdge('wire', { source: 'a', target: 'c' }, started), false);
    assert.equal(architecture.getState().removeEdge('wire', started), false);
    assert.equal(architecture.getState().connect({ source: 'b', target: 'c' }, started), null);
    assert.equal(templates.getState().templates.template.graph, inner);
    assert.deepEqual(writes, []);
    assert.equal(architecture.getState().reconnectEdge('wire', { source: 'a', target: 'd' }, context()), true);
    assert.equal(templates.getState().templates.template.graph.edges[0].target, 'd');
    assert.deepEqual(writes, ['deepflow:templates']);
    const beforeEdit = context();
    architecture.getState().patchLayerParameters('a', { out_features: 768 });
    writes = [];
    assert.equal(architecture.getState().removeEdge('wire', beforeEdit), false);
    assert.equal(architecture.getState().reconnectEdge('wire', { source: 'b', target: 'd' }, beforeEdit), false);
    assert.equal(templates.getState().templates.template.graph.edges[0].target, 'd');
    assert.deepEqual(writes, []);
});

test('only an empty-canvas release detaches; invalid handles, nodes, panels and outside releases cancel', () => {
    const dom = new JSDOM('<main><div id="canvas"><div class="react-flow__pane"><div class="react-flow__node"><div class="react-flow__handle"></div></div></div><aside></aside></div></main>');
    const document = dom.window.document;
    const canvas = document.querySelector('#canvas');
    const eventAt = (selector) => ({ target: document.querySelector(selector), clientX: 10, clientY: 10 });
    assert.equal(isBlankCanvasDrop(eventAt('.react-flow__pane'), {}, canvas), true);
    for (const selector of ['.react-flow__node', '.react-flow__handle', 'aside', 'main']) {
        assert.equal(isBlankCanvasDrop(eventAt(selector), {}, canvas), false);
    }
    assert.equal(isBlankCanvasDrop(eventAt('.react-flow__pane'), { toHandle: { id: 'invalid' } }, canvas), false);
    assert.equal(isBlankCanvasDrop(eventAt('.react-flow__pane'), { toNode: { id: 'invalid' } }, canvas), false);
    assert.equal(isBlankCanvasDrop(eventAt('.react-flow__pane'), { isValid: true }, canvas), false);
    document.elementFromPoint = () => document.querySelector('.react-flow__pane');
    assert.equal(isBlankCanvasDrop({ ...eventAt('.react-flow__handle'), changedTouches: [{ clientX: 20, clientY: 20 }] }, {}, canvas), true, 'Touch release resolves the release location rather than the original target');
    dom.window.close();
});
