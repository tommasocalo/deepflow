import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate } from '../src/domain/architecture/index.js';
import { containsTemplate } from '../src/domain/architecture/templateReferences.js';

const memory = new Map();
let writes = [];
globalThis.window = { localStorage: {
    getItem: (key) => memory.get(key) ?? null,
    setItem: (key, value) => { memory.set(key, value); writes.push(key); },
    removeItem: (key) => memory.delete(key),
} };
const {
    ArchitectureStoreProvider,
    createArchitecturePaneStore,
    useArchitectureStore: architecture,
    useArchitectureStoreApi,
} = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const position = { x: 100, y: 100 };
const layer = (id) => createLayerNode({ id, libraryKey: 'torch.nn.Linear', position });
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const panes = [];
const openPane = (path = []) => {
    const pane = createArchitecturePaneStore({ path });
    panes.push(pane);
    return pane;
};

beforeEach(() => {
    memory.clear();
    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(layer('a'), layer('b')), currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null });
    writes = [];
});
afterEach(() => {
    for (const pane of panes.splice(0)) pane.dispose();
});

function addNestedTemplate() {
    const leaf = createTemplate({ id: 'leaf-template', name: 'Leaf', graph: graphOf(layer('inside')) });
    const nested = createSuperNodeInstance({ id: 'nested', templateId: leaf.id, position });
    const block = createTemplate({ id: 'block-template', name: 'Block', graph: graphOf(nested, layer('middle')) });
    const instance = createSuperNodeInstance({ id: 'instance', templateId: block.id, position });
    templates.setState({ templates: { [leaf.id]: leaf, [block.id]: block } });
    architecture.setState({ graph: graphOf(layer('a'), layer('b'), instance) });
    return { leaf, block, instance };
}

test('root edits synchronize both ways with one main persistence write per edit', () => {
    const left = openPane();
    const right = openPane();
    assert.equal(left.getState().graph, architecture.getState().graph);
    left.getState().moveNodes([{ id: 'a', position: { x: 250, y: 100 } }], left.getState().getMutationContext());
    assert.equal(architecture.getState().graph.nodes.a.position.x, 250);
    assert.equal(right.getState().graph, left.getState().graph);
    assert.deepEqual(writes, ['deepflow:architecture-graph']);

    architecture.getState().patchLayerParameters('b', { out_features: 512 });
    assert.equal(left.getState().graph.nodes.b.parameters.out_features, 512);
    assert.equal(right.getState().graph, architecture.getState().graph);
    assert.equal(writes.length, 2);
    assert.equal(left.persist, undefined, 'A side pane has no independent persisted workspace');
    assert.equal(typeof architecture.persist.rehydrate, 'function', 'The main hook retains its persistence API');
});

test('each pane keeps independent tabs, active level, selection and expanded node', () => {
    addNestedTemplate();
    architecture.getState().selectOnly('a');
    architecture.getState().setExpanded('a');
    const pane = openPane(['instance']);
    writes = [];
    pane.getState().selectOnly('middle');
    pane.getState().setExpanded('middle');
    pane.getState().openLevel(['instance', 'nested']);
    pane.getState().selectOnly('inside');
    pane.getState().setExpanded('inside');
    assert.deepEqual(architecture.getState().currentPath, []);
    assert.deepEqual(architecture.getState().openPaths, []);
    assert.deepEqual(architecture.getState().selectedNodeIds, ['a']);
    assert.equal(architecture.getState().expandedNodeId, 'a');
    assert.deepEqual(pane.getState().openPaths, [['instance'], ['instance', 'nested']]);
    assert.deepEqual(pane.getState().selectedNodeIds, ['inside']);
    assert.equal(pane.getState().expandedNodeId, 'inside');
    assert.deepEqual(writes, [], 'Side navigation and selection never persist the main workspace');

    architecture.getState().openLevel(['instance']);
    assert.deepEqual(pane.getState().currentPath, ['instance', 'nested']);
    pane.getState().closeLevel(['instance', 'nested']);
    assert.deepEqual(pane.getState().currentPath, ['instance']);
    assert.deepEqual(architecture.getState().openPaths, [['instance']]);
});

test('template edits and structural pruning reach all panes with local navigation fallbacks', () => {
    const { leaf, block } = addNestedTemplate();
    const pane = openPane(['instance', 'nested']);
    pane.getState().openLevel(['instance']);
    pane.getState().openLevel(['instance', 'nested']);
    architecture.getState().openLevel(['instance', 'nested']);
    pane.getState().patchLayerParameters('inside', { out_features: 768 });
    assert.equal(architecture.getState().getViewGraph().nodes.inside.parameters.out_features, 768);
    architecture.getState().moveNode('inside', { x: 300, y: 100 });
    assert.equal(pane.getState().getViewGraph(), templates.getState().getTemplate(leaf.id).graph);
    assert.equal(pane.getState().getViewGraph().nodes.inside.position.x, 300);

    templates.getState().removeNodeFromTemplate(block.id, 'nested');
    assert.deepEqual(pane.getState().currentPath, ['instance']);
    assert.deepEqual(pane.getState().openPaths, [['instance']]);
    assert.deepEqual(architecture.getState().currentPath, []);
    assert.deepEqual(architecture.getState().openPaths, []);

    templates.getState().removeTemplate(block.id);
    assert.deepEqual(pane.getState().currentPath, []);
    assert.deepEqual(pane.getState().openPaths, []);
});

test('deleting a root instance from another pane prunes unreachable levels and inspectors', () => {
    addNestedTemplate();
    architecture.getState().openLevel(['instance', 'nested']);
    architecture.getState().selectOnly('inside');
    architecture.getState().setExpanded('inside');
    const side = openPane();
    side.getState().removeNode('instance');
    assert.deepEqual(architecture.getState().currentPath, []);
    assert.deepEqual(architecture.getState().openPaths, []);
    assert.deepEqual(architecture.getState().selectedNodeIds, []);
    assert.equal(architecture.getState().expandedNodeId, null);
    assert.equal(architecture.getState().graph, side.getState().graph);
    assert.equal(Object.keys(templates.getState().templates).length, 2, 'Deleting an instance preserves its reusable templates');
});

test('deleting a selected node in another pane clears only invalid local selection', () => {
    const side = openPane();
    side.getState().setNodeSelected('a', true);
    side.getState().setNodeSelected('b', true);
    side.getState().setExpanded('a');
    architecture.getState().removeNode('a');
    assert.deepEqual(side.getState().selectedNodeIds, ['b']);
    assert.equal(side.getState().expandedNodeId, null);
});

test('mutation contexts reject cross-pane gestures even when graph and level match', () => {
    const side = openPane();
    const mainContext = architecture.getState().getMutationContext();
    const sideContext = side.getState().getMutationContext();
    assert.equal(mainContext.graph, sideContext.graph);
    assert.equal(mainContext.levelKey, sideContext.levelKey);
    assert.notEqual(mainContext.storeId, sideContext.storeId);
    assert.equal(side.getState().moveNodes([{ id: 'a', position: { x: 999, y: 999 } }], mainContext), false);
    assert.equal(side.getState().connect({ source: 'a', target: 'b' }, mainContext), null);
    const edgeId = side.getState().connect({ source: 'a', target: 'b' }, sideContext);
    assert.ok(edgeId);
    const updatedMainContext = architecture.getState().getMutationContext();
    assert.equal(side.getState().removeEdge(edgeId, updatedMainContext), false);
    assert.equal(side.getState().reconnectEdge(edgeId, { source: 'b', target: 'a' }, updatedMainContext), false);
    assert.equal(side.getState().graph.nodes.a.position.x, 100);
    assert.equal(side.getState().graph.edges[0].source, 'a');
});

test('shared graph edits and pane navigation invalidate old drag contexts', () => {
    addNestedTemplate();
    const side = openPane();
    const context = side.getState().getMutationContext();
    architecture.getState().patchLayerParameters('a', { out_features: 256 });
    assert.equal(side.getState().moveNodes([{ id: 'a', position: { x: 999, y: 999 } }], context), false);
    const fresh = side.getState().getMutationContext();
    side.getState().openLevel(['instance']);
    assert.equal(side.getState().moveNodes([{ id: 'middle', position: { x: 999, y: 999 } }], fresh), false);
    assert.equal(side.getState().getViewGraph().nodes.middle.position.x, 100);
});

test('disposing a pane removes graph and template subscriptions and rejects delayed mutations', () => {
    const { block } = addNestedTemplate();
    const side = openPane(['instance']);
    const original = side.getState();
    const context = original.getMutationContext();
    side.dispose();
    side.dispose();
    architecture.getState().patchLayerParameters('a', { out_features: 512 });
    templates.getState().removeTemplate(block.id);
    assert.equal(side.getState(), original, 'The disposed pane no longer follows shared state changes');
    assert.equal(side.getState().moveNodes([{ id: 'middle', position: { x: 999, y: 999 } }], context), false);
    side.setState({ graph: graphOf(layer('detached')) });
    assert.equal(architecture.getState().graph.nodes.detached, undefined);
});

test('the provider scopes hook reads and imperative APIs while static calls stay on main', () => {
    addNestedTemplate();
    const side = openPane(['instance']);
    let scopedApi;
    function Probe() {
        const path = architecture((state) => state.currentPath);
        scopedApi = useArchitectureStoreApi();
        return createElement('span', null, path.join('/'));
    }
    assert.equal(renderToStaticMarkup(createElement(ArchitectureStoreProvider, { store: side }, createElement(Probe))), '<span>instance</span>');
    assert.equal(scopedApi, side);
    assert.deepEqual(architecture.getState().currentPath, []);
    assert.equal(renderToStaticMarkup(createElement(Probe)), '<span></span>');
    assert.equal(scopedApi.getState, architecture.getState);
});

test('invalid initial pane paths fall back to main without changing main navigation', () => {
    const side = openPane(['missing']);
    assert.deepEqual(side.getState().currentPath, []);
    assert.deepEqual(side.getState().openPaths, []);
    assert.deepEqual(architecture.getState().currentPath, []);
});

for (const action of ['addSuperNodeInstance', 'addRepeatNode']) {
    test(`${action} validates the receiving pane and rejects direct or transitive template cycles`, () => {
        const { leaf, block } = addNestedTemplate();
        const ancestor = createTemplate({
            id: 'ancestor-template', name: 'Ancestor',
            graph: graphOf(createRepeatNode({ id: 'repeated-block', templateId: block.id, position })),
        });
        const unrelated = createTemplate({ id: 'unrelated-template', name: 'Unrelated', graph: graphOf(layer('other')) });
        templates.setState({ templates: { ...templates.getState().templates, [ancestor.id]: ancestor, [unrelated.id]: unrelated } });
        const pane = openPane(['instance', 'nested']);

        const rootNodeId = architecture.getState()[action](ancestor.id, position);
        assert.ok(rootNodeId, 'The candidate remains valid in the root palette');
        assert.equal(architecture.getState().graph.nodes[rootNodeId].templateId, ancestor.id);
        const rootGraph = architecture.getState().graph;
        const templateGraph = pane.getState().getViewGraph();
        writes = [];

        assert.equal(pane.getState()[action](leaf.id, position), null, 'A pane cannot insert its own template');
        assert.equal(pane.getState()[action](block.id, position), null, 'A pane cannot insert a template that directly contains it');
        assert.equal(pane.getState()[action](ancestor.id, position), null, 'Nested repeats are included in the transitive cycle check');
        assert.equal(pane.getState().getViewGraph(), templateGraph);
        assert.equal(architecture.getState().graph, rootGraph);
        assert.deepEqual(writes, [], 'Rejected cross-pane drops never mutate or persist either graph');

        const validNodeId = pane.getState()[action](unrelated.id, position);
        assert.ok(validNodeId, 'Unrelated templates can still be inserted into the receiving pane');
        assert.equal(templates.getState().getTemplate(leaf.id).graph.nodes[validNodeId].templateId, unrelated.id);
    });
}

test('template containment terminates for existing cycles and missing references', () => {
    const cyclic = {
        first: createTemplate({ id: 'first', graph: graphOf(createSuperNodeInstance({ id: 'to-second', templateId: 'second', position })) }),
        second: createTemplate({ id: 'second', graph: graphOf(createRepeatNode({ id: 'to-first', templateId: 'first', position })) }),
    };
    assert.equal(containsTemplate('first', 'second', cyclic), true);
    assert.equal(containsTemplate('second', 'first', cyclic), true);
    assert.equal(containsTemplate('first', 'absent', cyclic), false);
    assert.equal(containsTemplate('missing', 'first', cyclic), false);
});
