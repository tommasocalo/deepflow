import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import {
    createLayerNode,
    createRepeatNode,
    createSuperNodeInstance,
    createTemplate,
    emptyGraph,
    removeNode,
    resolveGraphLevel,
} from '../src/domain/architecture/index.js';

// Give persisted Zustand stores isolated in-memory storage before importing
// them. These checks exercise public graph operations and store actions only.
const storage = new Map();
let storageWrites = [];
globalThis.localStorage = {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => { storageWrites.push(key); storage.set(key, String(value)); },
    removeItem: (key) => storage.delete(key),
};
globalThis.window = { localStorage: globalThis.localStorage };
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');

const position = { x: 20, y: 40 };
const layer = (id) => createLayerNode({ id, libraryKey: 'torch.nn.Linear', position });
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });

beforeEach(() => {
    storage.clear();
    useTemplatesStore.setState({ templates: {} });
    useArchitectureStore.setState({
        graph: emptyGraph(),
        currentPath: [],
        openPaths: [],
        selectedNodeIds: [],
        expandedNodeId: null,
        recentLibraryKeys: [],
    });
    storageWrites = [];
});

test('deletion removes connected edges while preserving remaining template instances', () => {
    const removed = layer('removed');
    const kept = layer('kept');
    const instance = createSuperNodeInstance({ id: 'instance', templateId: 'shared', position });
    const repeat = createRepeatNode({ id: 'repeat', templateId: 'shared', position });
    const graph = {
        ...graphOf(removed, kept, instance, repeat),
        edges: [
            { id: 'incoming', source: 'instance', target: 'removed' },
            { id: 'outgoing', source: 'removed', target: 'kept' },
            { id: 'untouched', source: 'kept', target: 'repeat' },
        ],
    };
    const result = removeNode(graph, 'removed');
    assert.deepEqual(Object.keys(result.nodes), ['kept', 'instance', 'repeat']);
    assert.deepEqual(result.edges.map((edge) => edge.id), ['untouched']);
    assert.equal(result.nodes.instance, instance);
    assert.equal(result.nodes.repeat, repeat);
    assert.equal(graph.nodes.removed, removed);
    assert.equal(graph.edges.length, 3);
});

test('repeat drill-in edits its shared template, supports nested instances, and exits cleanly', () => {
    const leaf = createTemplate({ id: 'leaf-template', name: 'Attention', graph: graphOf(layer('projection')) });
    const nested = createSuperNodeInstance({ id: 'nested', templateId: leaf.id, position, label: 'Attention' });
    const block = createTemplate({ id: 'block-template', name: 'Decoder block', graph: graphOf(nested) });
    const repeat = createRepeatNode({ id: 'stack', templateId: block.id, position, count: 12, label: 'Decoder stack' });
    const root = graphOf(repeat);
    useTemplatesStore.setState({ templates: { [leaf.id]: leaf, [block.id]: block } });
    useArchitectureStore.setState({ graph: root, selectedNodeIds: ['stack'], expandedNodeId: 'stack' });

    useArchitectureStore.getState().enterSupernode('stack');
    let state = useArchitectureStore.getState();
    assert.deepEqual(state.currentPath, ['stack']);
    assert.equal(state.getViewGraph(), block.graph);
    assert.equal(state.getLevel().templateId, block.id);
    assert.deepEqual(state.selectedNodeIds, []);
    assert.equal(state.expandedNodeId, null);

    const insertedId = state.addLayerByLibraryKey('torch.nn.Linear', position);
    assert.ok(useTemplatesStore.getState().getTemplate(block.id).graph.nodes[insertedId]);
    assert.equal(useArchitectureStore.getState().graph, root);

    state.enterSupernode('nested');
    state = useArchitectureStore.getState();
    assert.equal(state.getViewGraph(), leaf.graph);
    assert.deepEqual(state.getLevel().segments.map((segment) => segment.label), ['Decoder stack', 'Attention']);
    state.patchLayerParameters('projection', { out_features: 768 });
    assert.equal(useTemplatesStore.getState().getTemplate(leaf.id).graph.nodes.projection.parameters.out_features, 768);

    state.selectOnly('projection');
    state.setExpanded('projection');
    state.exitSupernode();
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['stack']);
    assert.equal(useArchitectureStore.getState().getLevel().templateId, block.id);
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, []);
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    state.goToLevel(0);
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
    assert.equal(useArchitectureStore.getState().getViewGraph(), root);
});

test('missing templates and ordinary layers cannot be entered', () => {
    const repeat = createRepeatNode({ id: 'missing', templateId: 'absent', position });
    const root = graphOf(repeat, layer('linear'));
    useArchitectureStore.setState({ graph: root });
    const state = useArchitectureStore.getState();
    state.enterSupernode('missing');
    state.enterSupernode('linear');
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
    const result = resolveGraphLevel(root, ['missing'], {});
    assert.equal(result.graph, root);
    assert.equal(result.isRoot, true);
    assert.deepEqual(result.segments, []);
});

test('deleting an expanded selection clears UI state without deleting shared templates', () => {
    const template = createTemplate({ id: 'shared', name: 'Shared', graph: graphOf(layer('internal')) });
    const instance = createSuperNodeInstance({ id: 'instance', templateId: template.id, position });
    const removed = layer('removed');
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    useArchitectureStore.setState({ graph: graphOf(instance, removed), selectedNodeIds: ['removed', 'instance'], expandedNodeId: 'removed' });
    useArchitectureStore.getState().removeNode('removed');
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, ['instance']);
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    assert.equal(useTemplatesStore.getState().getTemplate(template.id), template);
    useArchitectureStore.getState().removeNode('instance');
    assert.equal(useTemplatesStore.getState().getTemplate(template.id), template);
});


function tabFixture() {
    const leaf = createTemplate({ id: 'leaf', name: 'Leaf', graph: graphOf(layer('projection')) });
    const nested = createSuperNodeInstance({ id: 'nested', templateId: leaf.id, position, label: 'Nested' });
    const block = createTemplate({ id: 'block', name: 'Block', graph: graphOf(nested, layer('inner')) });
    const a = createRepeatNode({ id: 'a', templateId: block.id, position, label: 'A', count: 12 });
    const b = createSuperNodeInstance({ id: 'b', templateId: leaf.id, position, label: 'B' });
    useTemplatesStore.setState({ templates: { leaf, block } });
    useArchitectureStore.setState({ graph: graphOf(a, b, layer('ordinary')) });
    return { root: useArchitectureStore.getState().graph, leaf, block };
}

test('open levels survive a main round trip, retain insertion order and reopen without duplicates', () => {
    const { root } = tabFixture();
    const actions = useArchitectureStore.getState();
    assert.equal(actions.enterSupernode('a'), true);
    actions.goToLevel(0);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a']]);
    assert.equal(useArchitectureStore.getState().getViewGraph(), root);
    actions.enterSupernode('b');
    actions.openLevel(['a']);
    actions.openLevel(['b']);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a'], ['b']]);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['b']);
    actions.exitSupernode();
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a'], ['b']]);
});

test('opening a deep level adds just that tab and rejects incomplete, malformed and ordinary paths', () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    assert.equal(actions.openLevel(['a', 'nested']), true);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a', 'nested']]);
    const stable = useArchitectureStore.getState();
    for (const path of [['a', 'missing'], ['ordinary'], ['missing'], 'a', [null], ['a', 'nested', 'projection']]) {
        assert.equal(actions.openLevel(path), false);
        assert.equal(useArchitectureStore.getState(), stable);
    }
    actions.exitSupernode();
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['a']);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a', 'nested'], ['a']]);
});

test('closing an inactive parent preserves its nested tab, active level and inspector', () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    actions.enterSupernode('a');
    actions.enterSupernode('nested');
    actions.selectOnly('projection');
    actions.setExpanded('projection');
    assert.equal(actions.closeLevel(['a']), true);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a', 'nested']]);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['a', 'nested']);
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, ['projection']);
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'projection');
    assert.equal(actions.closeLevel(['a']), false);
    assert.equal(actions.closeLevel([]), false);
    actions.goToLevel(0);
    actions.openLevel(['a', 'nested']);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a', 'nested']]);
});

test('closing the active level falls back to its nearest open ancestor or main, never an unrelated sibling', () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    actions.openLevel(['b']);
    actions.openLevel(['a']);
    actions.openLevel(['a', 'nested']);
    actions.selectOnly('projection');
    actions.setExpanded('projection');
    actions.closeLevel(['a', 'nested']);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['a']);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['b'], ['a']]);
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, []);
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    actions.closeLevel(['a']);
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['b']]);
});

test('root deletion and reset prune unreachable tabs while keeping other levels', () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    actions.openLevel(['a']);
    actions.openLevel(['a', 'nested']);
    actions.openLevel(['b']);
    actions.goToLevel(0);
    actions.removeNode('a');
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['b']]);
    actions.resetCurrentGraph();
    assert.deepEqual(useArchitectureStore.getState().openPaths, []);
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
});

test('shared template deletion and nested reset prune affected tabs and choose a valid fallback', () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    actions.openLevel(['b']);
    actions.openLevel(['a']);
    actions.openLevel(['a', 'nested']);
    useTemplatesStore.getState().removeNodeFromTemplate('block', 'nested');
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['b'], ['a']]);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['a']);
    actions.resetCurrentGraph();
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['b'], ['a']]);
    assert.deepEqual(useArchitectureStore.getState().getViewGraph(), emptyGraph());
    useTemplatesStore.getState().removeTemplate('block');
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['b']]);
    actions.openLevel(['b']);
    useTemplatesStore.getState().removeTemplate('leaf');
    assert.deepEqual(useArchitectureStore.getState().openPaths, []);
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
});

test('root and template node moves do not rewrite unchanged tab navigation', () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    actions.openLevel(['a']);
    actions.openLevel(['a', 'nested']);
    const openPaths = useArchitectureStore.getState().openPaths;
    storageWrites = [];
    actions.moveNode('projection', { x: 200, y: 300 });
    assert.deepEqual(storageWrites, ['deepflow:templates']);
    assert.equal(useArchitectureStore.getState().openPaths, openPaths);
    actions.goToLevel(0);
    storageWrites = [];
    actions.moveNode('a', { x: 300, y: 400 });
    assert.deepEqual(storageWrites, ['deepflow:architecture-graph']);
    assert.equal(useArchitectureStore.getState().openPaths, openPaths);
});

test('open tabs and the active level survive persisted restoration without resurrecting closed tabs', async () => {
    tabFixture();
    const actions = useArchitectureStore.getState();
    actions.openLevel(['a']);
    actions.openLevel(['a', 'nested']);
    actions.openLevel(['b']);
    actions.closeLevel(['a']);
    const saved = storage.get('deepflow:architecture-graph');
    assert.deepEqual(JSON.parse(saved).state.openPaths, [['a', 'nested'], ['b']]);
    useArchitectureStore.setState({ graph: emptyGraph(), currentPath: [], openPaths: [] });
    storage.set('deepflow:architecture-graph', saved);
    await useArchitectureStore.persist.rehydrate();
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a', 'nested'], ['b']]);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['b']);
    actions.goToLevel(0);
    const mainSaved = storage.get('deepflow:architecture-graph');
    useArchitectureStore.setState({ openPaths: [] });
    storage.set('deepflow:architecture-graph', mainSaved);
    await useArchitectureStore.persist.rehydrate();
    assert.deepEqual(useArchitectureStore.getState().currentPath, []);
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a', 'nested'], ['b']]);
});

test('legacy saved breadcrumbs become valid tabs and malformed saved paths are pruned', async () => {
    const { root } = tabFixture();
    storage.set('deepflow:architecture-graph', JSON.stringify({ state: { graph: root, currentPath: ['a', 'nested'] }, version: 0 }));
    await useArchitectureStore.persist.rehydrate();
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a'], ['a', 'nested']]);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['a', 'nested']);
    storage.set('deepflow:architecture-graph', JSON.stringify({ state: {
        graph: root, currentPath: ['a', 'missing'],
        openPaths: [[], ['a'], ['a'], ['a', 'missing'], ['ordinary'], ['b'], null, 'invalid'],
    }, version: 0 }));
    await useArchitectureStore.persist.rehydrate();
    assert.deepEqual(useArchitectureStore.getState().openPaths, [['a'], ['b']]);
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['a']);
});
