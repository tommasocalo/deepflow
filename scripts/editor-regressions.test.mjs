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
globalThis.localStorage = {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, String(value)),
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
        selectedNodeIds: [],
        expandedNodeId: null,
        recentLibraryKeys: [],
    });
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
