import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { applyNodeChanges } from '@xyflow/react';
import { createDragBatcher, syncFlowNodes } from '../src/components/architecture/flowAdapter.js';
import { createLayerNode, createSuperNodeInstance, createTemplate } from '../src/domain/architecture/index.js';

const memory = new Map();
let writes = [];
globalThis.window = { localStorage: {
    getItem: (key) => memory.get(key) ?? null,
    setItem: (key, value) => { memory.set(key, value); writes.push(key); },
    removeItem: (key) => memory.delete(key),
} };
const { useArchitectureStore: architecture } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const layer = (id, x = 100) => createLayerNode({ id, libraryKey: 'torch.nn.Linear', position: { x, y: 100 }, parameters: {} });
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const context = () => ({ graph: architecture.getState().getViewGraph(), levelKey: JSON.stringify(architecture.getState().currentPath) });
const change = (id, x, dragging) => ({ type: 'position', id, position: { x, y: 100 }, dragging });

beforeEach(() => {
    memory.clear();
    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(layer('a'), layer('b', 200), layer('c', 300)), currentPath: [], selectedNodeIds: [], expandedNodeId: null });
    writes = [];
});

test('sixty frames of a multi-node drag stay local and persist once at the final release', () => {
    const original = architecture.getState().graph;
    const start = context();
    const batcher = createDragBatcher();
    let visual = syncFlowNodes([], original, ['a', 'b'], new Set(), start.levelKey);
    const initialVisual = visual;
    batcher.begin(start);
    for (let frame = 1; frame <= 60; frame += 1) {
        const changes = [change('a', 100 + frame, true), change('b', 200 + frame, true)];
        visual = applyNodeChanges(changes, visual);
        assert.equal(batcher.consume(changes, start), null);
        assert.equal(architecture.getState().graph, original);
    }
    assert.deepEqual(writes, []);
    assert.equal(visual[2], initialVisual[2], 'Stationary nodes retain their ReactFlow identity');
    assert.equal(visual[0].data, initialVisual[0].data, 'Moving nodes keep stable card data during the gesture');
    assert.equal(syncFlowNodes(visual, original, ['a', 'b'], new Set(), start.levelKey), visual, 'Selection synchronization does not reset local drag positions');

    const firstRelease = [change('a', 160, false)];
    visual = applyNodeChanges(firstRelease, visual);
    assert.equal(batcher.consume(firstRelease, start), null, 'Wait for the other dragged node before committing');
    const lastRelease = [change('b', 260, false)];
    visual = applyNodeChanges(lastRelease, visual);
    const transaction = batcher.consume(lastRelease, start);
    assert.equal(architecture.getState().moveNodes(transaction.moves, transaction.context), true);
    assert.deepEqual(writes, ['deepflow:architecture-graph']);
    const saved = architecture.getState().graph;
    assert.equal(saved.nodes.a.position.x, 160);
    assert.equal(saved.nodes.b.position.x, 260);
    assert.equal(saved.nodes.c, original.nodes.c);
    const settled = syncFlowNodes(visual, saved, ['a', 'b'], new Set(), start.levelKey);
    assert.equal(settled[2], initialVisual[2]);
    assert.equal(settled[0].position.x, 160);
    assert.equal(settled[0].dragging, false);
});

test('keyboard position batches commit immediately, including consecutive events before a render', () => {
    const batcher = createDragBatcher();
    const start = context();
    for (const x of [110, 120]) {
        const transaction = batcher.consume([change('a', x, false), change('b', x + 100, false)], start);
        assert.equal(transaction.isDrag, false);
        assert.equal(architecture.getState().moveNodes(transaction.moves, { levelKey: start.levelKey }), true);
    }
    assert.equal(architecture.getState().graph.nodes.a.position.x, 120);
    assert.equal(architecture.getState().graph.nodes.b.position.x, 220);
    assert.equal(writes.length, 2);
    architecture.getState().moveNodes([{ id: 'a', position: { x: 120, y: 100 } }], context());
    assert.equal(writes.length, 2, 'A no-op position batch does not persist again');
});

test('cancellation restores local positions and a subsequent gesture can commit normally', () => {
    const start = context();
    const batcher = createDragBatcher();
    const originalVisual = syncFlowNodes([], start.graph, [], new Set(), start.levelKey);
    let visual = applyNodeChanges([change('a', 150, true)], originalVisual);
    batcher.consume([change('a', 150, true)], start);
    batcher.cancel();
    visual = syncFlowNodes(visual, start.graph, [], new Set(), start.levelKey, true);
    assert.equal(visual[0].position.x, 100);
    assert.equal(visual[0].dragging, false);
    assert.equal(visual[1], originalVisual[1]);
    const cancelled = batcher.consume([change('a', 150, false)], start);
    assert.equal(cancelled.cancelled, true);
    assert.deepEqual(cancelled.moves, []);
    assert.deepEqual(writes, []);
    batcher.begin(start);
    batcher.consume([change('a', 180, true)], start);
    const finished = batcher.consume([change('a', 180, false)], start);
    architecture.getState().moveNodes(finished.moves, finished.context);
    assert.equal(writes.length, 1);
    assert.equal(architecture.getState().graph.nodes.a.position.x, 180);
});

test('a delayed drag cannot write into a different template even when node IDs coincide', () => {
    const template = createTemplate({ id: 'template', name: 'Block', graph: graphOf(layer('a', 500)) });
    templates.setState({ templates: { template } });
    const root = graphOf(layer('a'), createSuperNodeInstance({ id: 'instance', templateId: 'template', position: { x: 400, y: 100 } }));
    architecture.setState({ graph: root });
    const start = context();
    const batcher = createDragBatcher();
    batcher.consume([change('a', 150, true)], start);
    architecture.getState().enterSupernode('instance');
    writes = [];
    const opened = context();
    assert.equal(batcher.reconcile(opened), true);
    const cancelled = batcher.consume([change('a', 160, false)], opened);
    assert.equal(cancelled.cancelled, true);
    assert.equal(architecture.getState().moveNodes([{ id: 'a', position: { x: 160, y: 100 } }], start), false);
    assert.equal(templates.getState().templates.template.graph.nodes.a.position.x, 500);
    assert.equal(architecture.getState().graph, root);
    assert.deepEqual(writes, []);
    batcher.begin(opened);
    batcher.consume([change('a', 550, true)], opened);
    const finished = batcher.consume([change('a', 550, false)], opened);
    architecture.getState().moveNodes(finished.moves, finished.context);
    assert.deepEqual(writes, ['deepflow:templates']);
    assert.equal(templates.getState().templates.template.graph.nodes.a.position.x, 550);
});

test('replacing or editing the graph during a drag rejects its stale final position', () => {
    const start = context();
    const batcher = createDragBatcher();
    batcher.consume([change('a', 150, true)], start);
    architecture.getState().patchLayerParameters('a', { out_features: 768 });
    writes = [];
    const edited = context();
    const cancelled = batcher.consume([change('a', 160, false)], edited);
    assert.equal(cancelled.cancelled, true);
    assert.equal(architecture.getState().moveNodes([{ id: 'a', position: { x: 160, y: 100 } }], start), false);
    assert.equal(architecture.getState().graph.nodes.a.position.x, 100);
    assert.equal(architecture.getState().graph.nodes.a.parameters.out_features, 768);
    assert.deepEqual(writes, []);
});
