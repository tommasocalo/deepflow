import assert from 'node:assert/strict';
import { afterEach, beforeEach, test } from 'node:test';
import {
    createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate, resolveGraphLevel,
} from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';

const saved = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => saved.get(key) ?? null,
    setItem: (key, value) => saved.set(key, value),
    removeItem: (key) => saved.delete(key),
} };
const { useArchitectureStore: architecture, createArchitecturePaneStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');

const position = { x: 100, y: 100 };
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const linear = (id = 'linear') => createLayerNode({
    id, libraryKey: 'torch.nn.Linear', entry: getEntry('torch.nn.Linear'), position,
    parameters: { in_features: 16, out_features: 16, bias: true },
});
const component = (id, templateId) => createSuperNodeInstance({ id, templateId, label: id, position });
const repeat = (id, templateId, count = 4) => createRepeatNode({ id, templateId, label: id, mode: 'parallel', count, position });
const document = () => ({ graph: architecture.getState().graph, templates: templates.getState().templates });
const iterations = (...entries) => Object.fromEntries(entries.map(([path, index]) => [JSON.stringify(path), index]));
const view = (path, inspectedIterations = {}) => resolveGraphLevel(
    architecture.getState().graph, path, templates.getState().templates, inspectedIterations,
);
const panes = [];
const paneAt = (path, inspectedIterations = {}) => {
    const pane = createArchitecturePaneStore({ path });
    panes.push(pane);
    for (const [key, index] of Object.entries(inspectedIterations)) pane.getState().setInspectedIteration(JSON.parse(key), index);
    return pane;
};

beforeEach(() => {
    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(), currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null, inspectedIterations: {} });
    history.getState().clear();
    saved.clear();
});
afterEach(() => { for (const pane of panes.splice(0)) pane.dispose(); });

function headsFixture() {
    const body = createTemplate({ id: 'head-body', name: 'Attention head', graph: graphOf(linear()) });
    templates.setState({ templates: { [body.id]: body } });
    architecture.setState({ graph: graphOf(repeat('heads', body.id), component('other-head', body.id)) });
    return body;
}

function nestedFixture() {
    const head = createTemplate({ id: 'head-body', name: 'Attention head', graph: graphOf(linear()) });
    const block = createTemplate({ id: 'block-body', name: 'Transformer block', graph: graphOf(repeat('heads', head.id)) });
    templates.setState({ templates: { [head.id]: head, [block.id]: block } });
    architecture.setState({ graph: graphOf(repeat('blocks', block.id, 3), component('other-block', block.id)) });
}

test('This instance isolates only the selected repeat occurrence and retains the shared definition', () => {
    const body = headsFixture();
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    assert.equal(architecture.getState().getStructureScope().available, true);
    assert.equal(architecture.getState().getStructureScope().local, false);
    assert.equal(architecture.getState().getStructureScope().index, 2);
    assert.equal(architecture.getState().setStructureScope('local'), true);
    architecture.getState().patchLayerParameters('linear', { out_features: 32 });

    const node = architecture.getState().graph.nodes.heads;
    assert.equal(node.templateId, body.id, 'The shared template is retained for all ordinary occurrences');
    assert.equal(node.structureOverrides['2'].enabled, true);
    assert.notEqual(node.structureOverrides['2'].templateId, body.id);
    assert.equal(architecture.getState().getStructureScope().local, true);
    assert.equal(view(['heads'], iterations([['heads'], 2])).graph.nodes.linear.parameters.out_features, 32);
    for (const index of [0, 1, 3]) {
        assert.equal(view(['heads'], iterations([['heads'], index])).graph.nodes.linear.parameters.out_features, 16);
    }
    assert.equal(view(['other-head']).graph.nodes.linear.parameters.out_features, 16);
    assert.equal(templates.getState().templates[body.id].graph.nodes.linear.parameters.out_features, 16);
});

test('switching back to Shared preserves the local variant and restores it without another clone', () => {
    headsFixture();
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 32 });
    const localId = architecture.getState().getLevel().templateId;
    const templateCount = Object.keys(templates.getState().templates).length;

    assert.equal(architecture.getState().setStructureScope('shared'), true);
    assert.equal(architecture.getState().getStructureScope().local, false);
    assert.equal(architecture.getState().getStructureScope().hasSavedLocal, true);
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.out_features, 16);
    architecture.getState().patchLayerParameters('linear', { out_features: 48 });
    assert.equal(view(['heads'], iterations([['heads'], 0])).graph.nodes.linear.parameters.out_features, 48);

    assert.equal(architecture.getState().setStructureScope('local'), true);
    assert.equal(architecture.getState().getLevel().templateId, localId);
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.out_features, 32);
    assert.equal(Object.keys(templates.getState().templates).length, templateCount);
});

test('ordinary component instances can opt out of shared editing independently', () => {
    const body = createTemplate({ id: 'body', name: 'Projection', graph: graphOf(linear()) });
    templates.setState({ templates: { body } });
    architecture.setState({ graph: graphOf(component('first', body.id), component('second', body.id)) });
    architecture.getState().openLevel(['first']);
    assert.equal(architecture.getState().getStructureScope().index, null);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 64 });
    assert.equal(architecture.getState().graph.nodes.first.structureOverrides.single.enabled, true);
    assert.equal(view(['first']).graph.nodes.linear.parameters.out_features, 64);
    assert.equal(view(['second']).graph.nodes.linear.parameters.out_features, 16);
    architecture.getState().setStructureScope('shared');
    assert.equal(view(['first']).graph.nodes.linear.parameters.out_features, 16);
});

test('isolating a nested head also isolates its selected outer block occurrence', () => {
    nestedFixture();
    architecture.getState().setInspectedIteration(['blocks'], 1);
    architecture.getState().setInspectedIteration(['blocks', 'heads'], 2);
    architecture.getState().openLevel(['blocks', 'heads']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 128 });

    const at = (blockIndex, headIndex) => view(['blocks', 'heads'], iterations(
        [['blocks'], blockIndex], [['blocks', 'heads'], headIndex],
    )).graph.nodes.linear.parameters.out_features;
    assert.equal(at(1, 2), 128);
    for (const blockIndex of [0, 1, 2]) {
        for (const headIndex of [0, 1, 2, 3]) {
            if (blockIndex !== 1 || headIndex !== 2) assert.equal(at(blockIndex, headIndex), 16);
        }
    }
    assert.equal(view(['other-block', 'heads']).graph.nodes.linear.parameters.out_features, 16);
    assert.equal(templates.getState().templates['block-body'].graph.nodes.heads.structureOverrides, undefined);
    assert.equal(templates.getState().templates['head-body'].graph.nodes.linear.parameters.out_features, 16);
});

test('editing deeper in a local component cannot change descendants of its shared siblings', () => {
    nestedFixture();
    architecture.getState().setInspectedIteration(['blocks'], 1);
    architecture.getState().openLevel(['blocks']);
    architecture.getState().setStructureScope('local');
    architecture.getState().openLevel(['blocks', 'heads']);
    architecture.getState().patchLayerParameters('linear', { out_features: 96 });
    assert.equal(view(['blocks', 'heads'], iterations([['blocks'], 1])).graph.nodes.linear.parameters.out_features, 96);
    assert.equal(view(['blocks', 'heads'], iterations([['blocks'], 0])).graph.nodes.linear.parameters.out_features, 16);
    assert.equal(view(['other-block', 'heads']).graph.nodes.linear.parameters.out_features, 16);
});

test('local clone creation and nested ancestor isolation form one atomic Undo step', () => {
    nestedFixture();
    architecture.getState().setInspectedIteration(['blocks'], 1);
    architecture.getState().setInspectedIteration(['blocks', 'heads'], 2);
    architecture.getState().openLevel(['blocks', 'heads']);
    const initial = document();
    architecture.getState().setStructureScope('local');
    const isolated = document();
    assert.equal(history.getState().undoCount, 1);
    architecture.getState().patchLayerParameters('linear', { out_features: 128 });
    const modified = document();
    assert.equal(history.getState().undoCount, 2);
    architecture.getState().setStructureScope('shared');
    const sharedAgain = document();
    assert.equal(history.getState().undoCount, 3);

    history.getState().undo();
    assert.deepEqual(document(), modified);
    history.getState().undo();
    assert.deepEqual(document(), isolated);
    history.getState().undo();
    assert.deepEqual(document(), initial);
    assert.deepEqual(architecture.getState().currentPath, ['blocks', 'heads']);
    history.getState().redo(); history.getState().redo(); history.getState().redo();
    assert.deepEqual(document(), sharedAgain);
});

test('two panes viewing different repeat occurrences edit and undo the right structure', () => {
    headsFixture();
    const left = paneAt(['heads'], iterations([['heads'], 2]));
    const right = paneAt(['heads'], iterations([['heads'], 0]));
    right.getState().selectOnly('linear');
    left.getState().setStructureScope('local');
    left.getState().patchLayerParameters('linear', { out_features: 24 });
    assert.equal(left.getState().getViewGraph().nodes.linear.parameters.out_features, 24);
    assert.equal(right.getState().getViewGraph().nodes.linear.parameters.out_features, 16);
    assert.equal(right.getState().getStructureScope().local, false);
    assert.equal(left.getState().getStructureScope().local, true);

    right.getState().patchLayerParameters('linear', { out_features: 48 });
    assert.equal(left.getState().getViewGraph().nodes.linear.parameters.out_features, 24);
    assert.equal(right.getState().getViewGraph().nodes.linear.parameters.out_features, 48);
    history.getState().undo();
    assert.equal(right.getState().getViewGraph().nodes.linear.parameters.out_features, 16);
    assert.equal(left.getState().getViewGraph().nodes.linear.parameters.out_features, 24);
    assert.deepEqual(right.getState().selectedNodeIds, ['linear']);
    assert.equal(left.getState().inspectedIterations['["heads"]'], 2);
});

test('inspecting another occurrence updates scope and graph without altering the document or history', () => {
    headsFixture();
    architecture.getState().openLevel(['heads']);
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 32 });
    const modified = document();
    const steps = history.getState().undoCount;
    architecture.getState().setInspectedIteration(['heads'], 0);
    assert.equal(architecture.getState().getStructureScope().local, false);
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.out_features, 16);
    architecture.getState().setInspectedIteration(['heads'], 2);
    assert.equal(architecture.getState().getStructureScope().local, true);
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.out_features, 32);
    assert.deepEqual(document(), modified);
    assert.equal(history.getState().undoCount, steps);
});

test('components added only to a local variant remain navigable, including after Undo and Redo', () => {
    headsFixture();
    const detail = createTemplate({ id: 'detail', name: 'Detail', graph: graphOf(linear('detail-linear')) });
    templates.setState({ templates: { ...templates.getState().templates, detail } });
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('local');
    const id = architecture.getState().addSuperNodeInstance('detail', position, 'Local detail');
    assert.equal(architecture.getState().openLevel(['heads', id]), true);
    assert.ok(architecture.getState().getViewGraph().nodes['detail-linear']);
    architecture.getState().patchLayerParameters('detail-linear', { out_features: 48 });
    history.getState().undo();
    assert.deepEqual(architecture.getState().currentPath, ['heads', id]);
    assert.equal(architecture.getState().getViewGraph().nodes['detail-linear'].parameters.out_features, 16);
    history.getState().redo();
    assert.equal(architecture.getState().getViewGraph().nodes['detail-linear'].parameters.out_features, 48);
});

test('persisted documents retain disabled local variants and restore them after reload', () => {
    headsFixture();
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 72 });
    architecture.getState().setStructureScope('shared');
    const persistedGraph = JSON.parse(saved.get('deepflow:architecture-graph')).state.graph;
    const persistedTemplates = JSON.parse(saved.get('deepflow:templates')).state.templates;
    assert.equal(persistedGraph.nodes.heads.structureOverrides['2'].enabled, false);

    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(), currentPath: [], openPaths: [], inspectedIterations: {} });
    templates.setState({ templates: persistedTemplates });
    architecture.setState({ graph: persistedGraph });
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.out_features, 16);
    assert.equal(architecture.getState().getStructureScope().hasSavedLocal, true);
    architecture.getState().setStructureScope('local');
    assert.equal(architecture.getState().getViewGraph().nodes.linear.parameters.out_features, 72);
});

test('rehydration reopens a local-only descendant at its saved nonzero repeat occurrence', async () => {
    headsFixture();
    const detail = createTemplate({ id: 'detail', name: 'Detail', graph: graphOf(linear('detail-linear')) });
    templates.setState({ templates: { ...templates.getState().templates, detail } });
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('local');
    const id = architecture.getState().addSuperNodeInstance('detail', position, 'Local detail');
    architecture.getState().openLevel(['heads', id]);
    const savedArchitecture = saved.get('deepflow:architecture-graph');
    const savedTemplates = saved.get('deepflow:templates');

    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(), currentPath: [], openPaths: [], inspectedIterations: {} });
    saved.set('deepflow:templates', savedTemplates);
    await templates.persist.rehydrate();
    saved.set('deepflow:architecture-graph', savedArchitecture);
    await architecture.persist.rehydrate();

    assert.deepEqual(architecture.getState().currentPath, ['heads', id]);
    assert.equal(architecture.getState().inspectedIterations['["heads"]'], 2);
    assert.ok(architecture.getState().getViewGraph().nodes['detail-linear']);
    assert.equal(view(['heads'], iterations([['heads'], 0])).graph.nodes[id], undefined);
    assert.equal(architecture.getState().getStructureScope(['heads']).local, true);
});

test('root, invalid paths and repeating the active choice never mutate the document', () => {
    headsFixture();
    const initial = document();
    assert.equal(architecture.getState().getStructureScope().available, false);
    assert.equal(architecture.getState().setStructureScope('local'), false);
    assert.equal(architecture.getState().setStructureScope('local', ['missing']), false);
    assert.equal(architecture.getState().setStructureScope('local', ['heads', 'missing']), false);
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('shared');
    assert.deepEqual(document(), initial);
    assert.equal(history.getState().undoCount, 0);
    architecture.getState().setStructureScope('local');
    const local = document();
    const steps = history.getState().undoCount;
    architecture.getState().setStructureScope('local');
    assert.deepEqual(document(), local);
    assert.equal(history.getState().undoCount, steps);
});

test('switching scope invalidates an in-flight drag so it cannot move nodes in the newly shown definition', () => {
    headsFixture();
    architecture.getState().openLevel(['heads']);
    const oldContext = architecture.getState().getMutationContext();
    architecture.getState().setStructureScope('local');
    const isolated = document();
    assert.equal(architecture.getState().moveNodes([{ id: 'linear', position: { x: 900, y: 800 } }], oldContext), false);
    assert.deepEqual(document(), isolated);
    assert.equal(history.getState().undoCount, 1);
});

test('pasting local occurrences makes private copies while ordinary shared definitions remain linked', () => {
    headsFixture();
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 32 });
    architecture.getState().openLevel([]);
    architecture.getState().selectOnly('heads');
    const copied = architecture.getState().copySelection();
    const beforePaste = document();
    const undoCount = history.getState().undoCount;
    const pasted = architecture.getState().pasteSelection(copied);
    const pastedId = pasted.nodeIds[0];
    const afterPaste = document();
    const original = architecture.getState().graph.nodes.heads;
    const duplicate = architecture.getState().graph.nodes[pastedId];
    assert.equal(duplicate.templateId, original.templateId);
    assert.notEqual(duplicate.structureOverrides['2'].templateId, original.structureOverrides['2'].templateId);
    assert.equal(history.getState().undoCount, undoCount + 1, 'Creating the private definitions is part of the paste transaction');
    history.getState().undo();
    assert.deepEqual(document(), beforePaste);
    history.getState().redo();
    assert.deepEqual(document(), afterPaste);

    architecture.getState().setInspectedIteration([pastedId], 2);
    architecture.getState().openLevel([pastedId]);
    architecture.getState().patchLayerParameters('linear', { out_features: 64 });
    assert.equal(view([pastedId], iterations([[pastedId], 2])).graph.nodes.linear.parameters.out_features, 64);
    assert.equal(view(['heads'], iterations([['heads'], 2])).graph.nodes.linear.parameters.out_features, 32);
    assert.equal(view([pastedId], iterations([[pastedId], 0])).graph.nodes.linear.parameters.out_features, 16);
});

test('isolating a nested child reactivates a saved local ancestor without losing its previous edits', () => {
    nestedFixture();
    architecture.getState().setInspectedIteration(['blocks'], 1);
    architecture.getState().openLevel(['blocks']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchNode('heads', { label: 'Locally named heads' });
    const localParentId = architecture.getState().getLevel().templateId;
    architecture.getState().setStructureScope('shared');
    assert.equal(architecture.getState().getViewGraph().nodes.heads.label, 'heads');

    architecture.getState().setInspectedIteration(['blocks', 'heads'], 2);
    architecture.getState().openLevel(['blocks', 'heads']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 128 });
    const localParent = view(['blocks'], iterations([['blocks'], 1]));
    assert.equal(localParent.templateId, localParentId);
    assert.equal(localParent.graph.nodes.heads.label, 'Locally named heads');
    assert.equal(view(['blocks'], iterations([['blocks'], 0])).graph.nodes.heads.label, 'heads');
    assert.equal(view(['blocks', 'heads'], iterations([['blocks'], 1], [['blocks', 'heads'], 2])).graph.nodes.linear.parameters.out_features, 128);
});

test('Repeat to Single and back retains first-occurrence edits and other private occurrences', () => {
    headsFixture();
    architecture.getState().openLevel(['heads']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 24 });
    architecture.getState().setInspectedIteration(['heads'], 2);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 32 });
    const beforeConversion = document();
    architecture.getState().openLevel([]);
    architecture.getState().convertRepeatToInstance('heads');
    assert.equal(architecture.getState().graph.nodes.heads.kind, 'supernode');
    assert.equal(view(['heads']).graph.nodes.linear.parameters.out_features, 24);
    architecture.getState().openLevel(['heads']);
    assert.equal(architecture.getState().getStructureScope().local, true);
    architecture.getState().patchLayerParameters('linear', { out_features: 48 });
    architecture.getState().openLevel([]);
    architecture.getState().convertInstanceToRepeat('heads');
    assert.equal(architecture.getState().graph.nodes.heads.count, 4);
    assert.equal(architecture.getState().graph.nodes.heads.mode, 'parallel');
    assert.equal(view(['heads'], iterations([['heads'], 0])).graph.nodes.linear.parameters.out_features, 48);
    assert.equal(view(['heads'], iterations([['heads'], 1])).graph.nodes.linear.parameters.out_features, 16);
    assert.equal(view(['heads'], iterations([['heads'], 2])).graph.nodes.linear.parameters.out_features, 32);
    history.getState().undo(); history.getState().undo(); history.getState().undo();
    assert.deepEqual(document(), beforeConversion);
});

test('converting a local ordinary component to Repeat carries its structure into occurrence one', () => {
    headsFixture();
    architecture.getState().openLevel(['other-head']);
    architecture.getState().setStructureScope('local');
    architecture.getState().patchLayerParameters('linear', { out_features: 72 });
    architecture.getState().openLevel([]);
    architecture.getState().convertInstanceToRepeat('other-head', 3, 'parallel');
    assert.equal(view(['other-head'], iterations([['other-head'], 0])).graph.nodes.linear.parameters.out_features, 72);
    assert.equal(view(['other-head'], iterations([['other-head'], 1])).graph.nodes.linear.parameters.out_features, 16);
    architecture.getState().convertRepeatToInstance('other-head');
    assert.equal(view(['other-head']).graph.nodes.linear.parameters.out_features, 72);
});
