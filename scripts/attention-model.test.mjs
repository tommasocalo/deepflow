import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { createEdge, createLayerNode, createRepeatNode, isValidConnection, reconnectEdge } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { ATTENTION_HEAD_INPUT_IDS, attentionStoryboard, buildAttentionHeadGraph, buildMultiHeadAttentionGraph, registerAttentionTemplates } from '../src/storyboards/attention.js';
import { buildTransformerBlockGraph, gpt2Storyboard } from '../src/storyboards/gpt2.js';
import { STORYBOARDS } from '../src/storyboards/index.js';

const storage = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
} };
const { useArchitectureStore: architectureStore, createArchitecturePaneStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: trainingStore } = await import('../src/stores/trainingStore.js');
const stores = { architectureStore, templatesStore, trainingStore };
const nodesOf = (graph) => Object.values(graph.nodes);
const byLabel = (graph, label) => nodesOf(graph).find((node) => node.label === label);
const layer = (id) => createLayerNode({ id, libraryKey: 'torch.nn.Linear', entry: getEntry('torch.nn.Linear'), position: { x: 100, y: 100 } });

beforeEach(() => {
    for (const store of Object.values(stores)) store.setState(store.getInitialState(), true);
    storage.clear();
});

test('attention head exposes three ordered inputs and editable causal attention operations', () => {
    const graph = buildAttentionHeadGraph();
    assert.deepEqual(nodesOf(graph).filter((node) => node.libraryKey === 'deepflow.template.Input').map((node) => node.id), Object.values(ATTENTION_HEAD_INPUT_IDS));
    const scores = byLabel(graph, 'Q × Kᵀ');
    const weighted = byLabel(graph, 'weights × V');
    const inputFor = (node, port) => graph.nodes[graph.edges.find((edge) => edge.target === node.id && edge.targetHandle === port).source];
    assert.equal(inputFor(scores, 'a').label, 'Q');
    assert.equal(inputFor(scores, 'b').libraryKey, 'deepflow.op.Transpose');
    assert.equal(inputFor(weighted, 'a').libraryKey, 'torch.nn.Dropout');
    assert.equal(inputFor(weighted, 'b').label, 'V');
    assert.equal(byLabel(graph, 'scaled scores').parameters.factor, '1 / (d_head ** 0.5)');
    assert.equal(byLabel(graph, 'attention weights').parameters.dim, -1);
    assert.equal(byLabel(graph, 'attention dropout').parameters.p, '$dropout');
    assert.ok(nodesOf(graph).some((node) => node.libraryKey === 'deepflow.op.CausalMask'));
    assert.ok(nodesOf(graph).every((node) => !node.libraryKey.includes('Attention')));
    for (const edge of graph.edges) assert.ok(graph.nodes[edge.source] && graph.nodes[edge.target]);
});

test('MHA combines full-width projections, equal splits and a collection-count parallel repeat', () => {
    const graph = buildMultiHeadAttentionGraph({ headTemplateId: 'head' });
    const repeat = nodesOf(graph).find((node) => node.kind === 'repeat');
    assert.equal(repeat.templateId, 'head');
    assert.equal(repeat.mode, 'parallel');
    assert.equal(repeat.count, 'auto');
    assert.deepEqual(repeat.bindings, { dropout: '$dropout' });
    const incoming = graph.edges.filter((edge) => edge.target === repeat.id);
    assert.deepEqual(incoming.map((edge) => edge.targetHandle), Object.values(ATTENTION_HEAD_INPUT_IDS));
    for (const edge of incoming) assert.deepEqual(graph.nodes[edge.source].parameters, { chunks: '$H', dim: -1 });
    const projections = nodesOf(graph).filter((node) => node.libraryKey === 'torch.nn.Linear');
    assert.equal(projections.length, 4);
    for (const node of projections) assert.deepEqual(node.parameters, { in_features: '$D', out_features: '$D', bias: '$bias' });
    const concat = byLabel(graph, 'concatenate heads');
    assert.equal(concat.parameters.dim, -1);
    assert.ok(graph.edges.some((edge) => edge.source === repeat.id && edge.target === concat.id));
});

test('registered templates retain user parameters and head-count edits preserve the shared body', () => {
    const { headTemplateId, attentionTemplateId } = registerAttentionTemplates(templatesStore);
    const tpl = templatesStore.getState();
    const attention = tpl.getTemplate(attentionTemplateId);
    assert.deepEqual(attention.freeParameters.map(({ name, default: value }) => [name, value]), [
        ['D', undefined], ['H', 12], ['dropout', 0.1], ['bias', true],
    ]);
    const head = tpl.getTemplate(headTemplateId);
    tpl.patchFreeParameter(attentionTemplateId, 'H', { default: 8 });
    assert.equal(tpl.getTemplate(attentionTemplateId).graph, attention.graph, 'Count changes use one generic repeat node');
    assert.equal(tpl.getTemplate(headTemplateId), head, 'All heads continue to share one editable definition');
    assert.equal(nodesOf(attention.graph).find((node) => node.kind === 'repeat').count, 'auto');
});

test('the default GPT-2 and standalone example register and open explicit attention templates', () => {
    gpt2Storyboard(stores);
    const root = architectureStore.getState().graph;
    const block = nodesOf(root).find((node) => node.kind === 'repeat');
    assert.equal(block.mode, 'sequential');
    const transformer = templatesStore.getState().getTemplate(block.templateId);
    const attention = byLabel(transformer.graph, 'causal self-attention');
    assert.equal(attention.kind, 'supernode');
    assert.deepEqual(attention.parameterValues, { D: '$d_model', H: '$n_heads' });
    assert.equal(templatesStore.getState().getTemplate(attention.templateId).name, 'Multi-head attention');
    for (const template of Object.values(templatesStore.getState().templates)) {
        assert.ok(nodesOf(template.graph).every((node) => !['deepflow.nn.CausalSelfAttention', 'torch.nn.MultiheadAttention'].includes(node.libraryKey)));
    }
    assert.equal(byLabel(buildTransformerBlockGraph(), 'causal self-attention').libraryKey, 'deepflow.nn.CausalSelfAttention', 'Standalone legacy builder API remains available');
    assert.ok(STORYBOARDS.some((storyboard) => storyboard.id === 'attention' && storyboard.run === attentionStoryboard));
    attentionStoryboard(stores);
    const state = architectureStore.getState();
    assert.equal(state.currentPath.length, 1);
    assert.equal(state.getLevel().label, 'Multi-head attention');
    assert.equal(nodesOf(state.getViewGraph()).find((node) => node.kind === 'repeat').mode, 'parallel');
});

test('generic wrapping and conversion preserve graph identity and accept editable count expressions', () => {
    const original = layer('standalone');
    architectureStore.setState({ graph: { nodes: { standalone: original }, edges: [] } });
    const arch = architectureStore.getState();
    const templateId = arch.groupSelection([original.id], 'My block');
    assert.ok(templateId, 'An isolated block can be wrapped from the creation toolbar');
    const template = templatesStore.getState().getTemplate(templateId);
    const input = nodesOf(template.graph).find((node) => node.libraryKey === 'deepflow.template.Input');
    const output = nodesOf(template.graph).find((node) => node.libraryKey === 'deepflow.template.Output');
    assert.ok(template.graph.edges.some((edge) => edge.source === input.id && edge.target === original.id));
    assert.ok(template.graph.edges.some((edge) => edge.source === original.id && edge.target === output.id));
    const instance = nodesOf(architectureStore.getState().graph)[0];
    arch.patchNode(instance.id, { parameterValues: { width: 32 } });
    assert.equal(arch.convertInstanceToRepeat(instance.id, '$branches', 'parallel'), instance.id);
    let repeat = architectureStore.getState().graph.nodes[instance.id];
    assert.equal(repeat.count, '$branches');
    assert.equal(repeat.mode, 'parallel');
    assert.equal(repeat.templateId, templateId);
    assert.deepEqual(repeat.bindings, { width: 32 });
    arch.patchRepeatNode(repeat.id, { count: 5, mode: 'sequential' });
    repeat = architectureStore.getState().graph.nodes[repeat.id];
    assert.equal(repeat.count, 5);
    assert.equal(repeat.mode, 'sequential');
    assert.equal(templatesStore.getState().getTemplate(templateId).graph, template.graph);
    assert.equal(createRepeatNode({ templateId }).mode, 'sequential');
});

test('head inspection is local to a pane and never rewrites the shared graph or persisted project', () => {
    const { headTemplateId } = registerAttentionTemplates(templatesStore);
    const repeat = createRepeatNode({ id: 'heads', templateId: headTemplateId, mode: 'parallel', count: 12 });
    architectureStore.setState({ graph: { nodes: { heads: repeat }, edges: [] } });
    const main = architectureStore.getState();
    const pane = createArchitecturePaneStore();
    try {
        assert.equal(pane.getState().enterRepeatIteration(repeat.id, 7), true);
        assert.deepEqual(pane.getState().currentPath, ['heads']);
        assert.equal(pane.getState().inspectedIterations['["heads"]'], 7);
        assert.deepEqual(architectureStore.getState().inspectedIterations, {});
        assert.equal(main.graph, pane.getState().graph);
        assert.equal(main.enterRepeatIteration(repeat.id, 2), true);
        assert.equal(architectureStore.getState().inspectedIterations['["heads"]'], 2);
        assert.equal(pane.getState().inspectedIterations['["heads"]'], 7);
        assert.equal(pane.getState().setInspectedIteration(['heads'], -1), false);
        assert.equal(pane.getState().setInspectedIteration(['heads'], 1.5), false);
        assert.equal(pane.getState().setInspectedIteration(['missing'], 0), false);
        assert.equal(Object.hasOwn(JSON.parse(storage.get('deepflow:architecture-graph')).state, 'inspectedIterations'), false);
    } finally {
        pane.dispose();
    }
});

test('distinct named ports allow repeated source/target pairs while exact and legacy duplicates remain invalid', () => {
    const graph = { nodes: { source: layer('source'), target: layer('target') }, edges: [] };
    const a = createEdge({ id: 'a', source: 'source', target: 'target', targetHandle: 'a' });
    graph.edges.push(a);
    assert.equal(isValidConnection(graph, { source: 'source', target: 'target', targetHandle: 'b' }), true);
    assert.equal(isValidConnection(graph, { source: 'source', target: 'target', targetHandle: 'a' }), false);
    graph.edges.push(createEdge({ id: 'b', source: 'source', target: 'target', targetHandle: 'b' }));
    assert.equal(reconnectEdge(graph, 'a', { source: 'source', target: 'target', targetHandle: 'b' }), graph);
    assert.equal(isValidConnection(graph, { source: 'source', target: 'target', sourceHandle: 'piece-2', targetHandle: 'a' }), true);
    const legacy = { ...graph, edges: [createEdge({ source: 'source', target: 'target' })] };
    assert.equal(isValidConnection(legacy, { source: 'source', target: 'target', sourceHandle: null, targetHandle: null }), false);
});
