import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { createEdge, createLayerNode, createSuperNodeInstance } from '../src/domain/architecture/index.js';
import { STANDARD_COMPONENTS, ensureStandardTemplate, getStandardComponentForLayer } from '../src/domain/components/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { gpt2Storyboard } from '../src/storyboards/gpt2.js';
import { validate } from '../src/validation/validator.js';

const storage = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
} };
const { useArchitectureStore: architecture, createArchitecturePaneStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templates } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: training } = await import('../src/stores/trainingStore.js');
const position = { x: 200, y: 180 };
const nodesOf = (graph) => Object.values(graph.nodes);
const graphOf = (...nodes) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges: [] });
const layer = (key, parameters, id = 'layer') => createLayerNode({ id, libraryKey: key, entry: getEntry(key), position, parameters });

beforeEach(() => {
    templates.setState({ templates: {} });
    architecture.setState({ graph: graphOf(), currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null, inspectedIterations: {}, showGradientFlow: false });
    training.getState().resetGraph();
    storage.clear();
});

test('standard definitions register on demand, retain edits and never rewrite existing graph nodes', () => {
    const legacy = layer('torch.nn.Linear', { in_features: 4, out_features: 6, bias: true });
    const graph = graphOf(legacy);
    architecture.setState({ graph });
    assert.deepEqual(templates.getState().templates, {});
    const id = ensureStandardTemplate('linear', templates);
    assert.equal(architecture.getState().graph, graph);
    assert.equal(architecture.getState().graph.nodes.layer, legacy);
    templates.getState().patchTemplate(id, { name: 'My edited standard', description: 'User content' });
    const edited = templates.getState().getTemplate(id);
    assert.equal(ensureStandardTemplate('linear', templates), id);
    assert.equal(templates.getState().getTemplate(id), edited);
    assert.equal(edited.standardKey, 'linear');
    assert.equal(ensureStandardTemplate('missing', templates), null);
    assert.deepEqual(STANDARD_COMPONENTS.map((component) => component.key), ['linear', 'embedding', 'attention', 'layernorm', 'learned-position', 'sinusoidal-position']);
});

test('every standard component validates with concrete bindings, including disabled affine terms and odd sinusoidal width', () => {
    const cases = [
        ['linear', '(5, 4)', { in_features: 4, out_features: 6, bias: false }, [null, 5, 6]],
        ['embedding', '(5,)', { num_embeddings: 16, embedding_dim: 6, padding_idx: 0 }, [null, 5, 6]],
        ['layernorm', '(5, 6)', { normalized_shape: 6, eps: 1e-5, elementwise_affine: false }, [null, 5, 6]],
        ['learned-position', '(5,)', { max_len: 16, embedding_dim: 6 }, [null, 5, 6]],
        ['sinusoidal-position', '(5,)', { D: 5 }, [null, 5, 5]],
        ['attention', '(5, 12)', { D: 12, H: 3, dropout: 0, bias: false }, [null, 5, 12]],
    ];
    for (const [key, shape, parameterValues, expected] of cases) {
        const templateId = ensureStandardTemplate(key, templates);
        const input = layer('deepflow.io.Input', { shape }, 'input');
        const component = createSuperNodeInstance({ id: 'component', templateId, position, parameterValues });
        const graph = { ...graphOf(input, component), edges: [createEdge({ source: input.id, target: component.id })] };
        const report = validate(graph, { templates: templates.getState().templates });
        assert.equal(report.ok, true, `${key}: ${JSON.stringify(report.errors)}`);
        assert.deepEqual(report.shapes.component, expected, key);
    }
});

test('explicit layer conversion preserves identity, layout, metadata, bindings and edge metadata', () => {
    const original = { ...layer('torch.nn.Linear', { in_features: '$width', out_features: 6, bias: false }), label: 'Projection', visualization: 'tensor-flow', tensorRole: 'project' };
    const input = layer('deepflow.io.Input', { shape: '(5, 4)' }, 'input');
    const output = layer('deepflow.io.Output', {}, 'output');
    const wire = createEdge({ id: 'wire', source: input.id, target: original.id, targetHandle: 'legacy-input', gradientOp: 'reverse', gradientOpLambda: 0.25 });
    const outgoing = createEdge({ id: 'outgoing', source: original.id, target: output.id });
    architecture.setState({ graph: { ...graphOf(input, original, output), edges: [wire, outgoing] } });
    assert.equal(architecture.getState().convertLayerToComponent(original.id), original.id);
    const graph = architecture.getState().graph;
    const converted = graph.nodes[original.id];
    assert.equal(converted.kind, 'supernode');
    assert.equal(converted.position, original.position);
    assert.equal(converted.label, original.label);
    assert.equal(converted.visualization, original.visualization);
    assert.equal(converted.tensorRole, original.tensorRole);
    assert.deepEqual(converted.parameterValues, original.parameters);
    assert.equal(graph.nodes.input, input);
    assert.equal(graph.edges[1], outgoing);
    const boundary = nodesOf(templates.getState().getTemplate(converted.templateId).graph).find((node) => node.libraryKey === 'deepflow.template.Input');
    assert.deepEqual(graph.edges[0], { ...wire, targetHandle: boundary.id });
    assert.equal(architecture.getState().convertLayerToComponent(original.id), null, 'Conversion is not applied twice');
});

test('conversion declines unsupported semantics without creating templates or changing the graph', () => {
    const cases = [
        layer('torch.nn.Embedding', { num_embeddings: 20, embedding_dim: 6, sparse: true }),
        layer('torch.nn.Embedding', { num_embeddings: 20, embedding_dim: 6, max_norm: 1 }),
        layer('torch.nn.LayerNorm', { normalized_shape: [5, 6] }),
        layer('torch.nn.LayerNorm', { normalized_shape: '(5, 6)' }),
        layer('torch.nn.LayerNorm', { normalized_shape: 6, bias: false }),
        layer('torch.nn.MultiheadAttention', { embed_dim: 12, num_heads: 3, batch_first: true }),
    ];
    for (const original of cases) {
        const graph = graphOf(original);
        architecture.setState({ graph });
        assert.equal(getStandardComponentForLayer(original), null);
        assert.equal(architecture.getState().convertLayerToComponent(original.id), null);
        assert.equal(architecture.getState().graph, graph);
        assert.deepEqual(templates.getState().templates, {});
    }
    assert.equal(getStandardComponentForLayer(layer('torch.nn.Linear', { in_features: 4, out_features: 6, bias: false })).key, 'linear');
    assert.equal(getStandardComponentForLayer(layer('torch.nn.Embedding', { num_embeddings: 20, embedding_dim: 6, padding_idx: 0 })).key, 'embedding');
});

test('making a copy deep-copies nested definitions and rebinds only the chosen root instance', () => {
    const arch = architecture.getState();
    const left = arch.addStandardComponent('learned-position', position);
    const right = arch.addStandardComponent('learned-position', { x: 700, y: 180 });
    const originalId = architecture.getState().graph.nodes[left].templateId;
    const original = templates.getState().getTemplate(originalId);
    const oldEmbedding = nodesOf(original.graph).find((node) => node.kind === 'supernode');
    arch.patchNode(left, { parameterValues: { max_len: 32, embedding_dim: 8 } });
    const copiedId = arch.cloneComponent(left, 'My positional encoding');
    assert.notEqual(copiedId, originalId);
    const copied = templates.getState().getTemplate(copiedId);
    assert.equal(copied.name, 'My positional encoding');
    assert.equal(copied.standardKey, undefined);
    assert.equal(copied.sourceTemplateId, originalId);
    assert.deepEqual(copied.freeParameters, original.freeParameters);
    assert.notEqual(copied.freeParameters, original.freeParameters);
    assert.equal(architecture.getState().graph.nodes[right].templateId, originalId);
    assert.deepEqual(architecture.getState().graph.nodes[left].parameterValues, { max_len: 32, embedding_dim: 8 });
    const newEmbedding = copied.graph.nodes[oldEmbedding.id];
    assert.notEqual(newEmbedding.templateId, oldEmbedding.templateId);
    const newTable = nodesOf(templates.getState().getTemplate(newEmbedding.templateId).graph).find((node) => node.libraryKey === 'deepflow.op.Parameter');
    templates.getState().patchTemplateLayerParameters(newEmbedding.templateId, newTable.id, { init: 'zeros' });
    const oldTable = templates.getState().getTemplate(oldEmbedding.templateId).graph.nodes[newTable.id];
    assert.equal(oldTable.parameters.init, 'normal');
    assert.equal(templates.getState().getTemplate(originalId), original);
    assert.equal(ensureStandardTemplate('learned-position', templates), originalId);
});

test('copying an opened nested component forks shared ancestors while retaining its path and unrelated instances', () => {
    const arch = architecture.getState();
    const left = arch.addStandardComponent('learned-position', position);
    const right = arch.addStandardComponent('learned-position', { x: 700, y: 180 });
    const originalGraph = architecture.getState().graph;
    const originalParentId = originalGraph.nodes[left].templateId;
    const parent = templates.getState().getTemplate(originalParentId);
    const nested = nodesOf(parent.graph).find((node) => node.kind === 'supernode');
    const originalNested = templates.getState().getTemplate(nested.templateId);
    const pane = createArchitecturePaneStore({ path: [left, nested.id] });
    try {
        const copiedId = pane.getState().cloneCurrentComponent('Custom lookup');
        assert.notEqual(copiedId, nested.templateId);
        assert.notEqual(architecture.getState().graph.nodes[left].templateId, originalParentId);
        assert.equal(architecture.getState().graph.nodes[right], originalGraph.nodes[right]);
        assert.deepEqual(pane.getState().currentPath, [left, nested.id]);
        assert.deepEqual(architecture.getState().currentPath, []);
        assert.equal(pane.getState().getLevel().templateId, copiedId);
        assert.equal(templates.getState().getTemplate(copiedId).name, 'Custom lookup');
        assert.equal(templates.getState().getTemplate(originalParentId), parent);
        assert.equal(templates.getState().getTemplate(nested.templateId), originalNested);
        pane.getState().patchLayerParameters(nodesOf(pane.getState().getViewGraph()).find((node) => node.libraryKey === 'deepflow.op.Parameter').id, { requires_grad: false });
        assert.equal(nodesOf(originalNested.graph).find((node) => node.libraryKey === 'deepflow.op.Parameter').parameters.requires_grad, true);
    } finally {
        pane.dispose();
    }
});

test('gradient display state is local to a pane and excluded from persisted graph content', () => {
    const pane = createArchitecturePaneStore();
    try {
        pane.getState().setShowGradientFlow(true);
        assert.equal(pane.getState().showGradientFlow, true);
        assert.equal(architecture.getState().showGradientFlow, false);
        architecture.getState().setShowGradientFlow(true);
        assert.equal(Object.hasOwn(JSON.parse(storage.get('deepflow:architecture-graph')).state, 'showGradientFlow'), false);
    } finally {
        pane.dispose();
    }
});

test('new GPT-2 uses learned positions and inspectable standard definitions throughout', () => {
    gpt2Storyboard({ architectureStore: architecture, templatesStore: templates, trainingStore: training });
    const graph = architecture.getState().graph;
    const positions = nodesOf(graph).find((node) => node.label === 'positional embedding');
    const definition = templates.getState().getTemplate(positions.templateId);
    assert.equal(positions.kind, 'supernode');
    assert.equal(definition.standardKey, 'learned-position');
    assert.ok(nodesOf(definition.graph).some((node) => node.libraryKey === 'deepflow.op.PositionIndices'));
    assert.equal(nodesOf(graph).filter((node) => node.libraryKey === 'deepflow.op.Add').length, 1, 'Token and position branches still combine explicitly');
    for (const template of Object.values(templates.getState().templates)) {
        assert.ok(nodesOf(template.graph).every((node) => !['torch.nn.Linear', 'torch.nn.Embedding', 'torch.nn.LayerNorm', 'deepflow.nn.PositionalEmbedding'].includes(node.libraryKey)));
    }
    const report = validate(graph, { templates: templates.getState().templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    const head = nodesOf(graph).find((node) => node.label === 'lm_head');
    assert.deepEqual(report.shapes[head.id], [null, 1024, 50257]);
    // Scaling the whole model is a change to its global parameters.
    architecture.getState().patchGlobalParameter('d_model', { default: 1024 });
    architecture.getState().patchGlobalParameter('n_heads', { default: 16 });
    architecture.getState().patchGlobalParameter('n_layers', { default: 24 });
    const scaled = validate(architecture.getState().graph, { templates: templates.getState().templates });
    assert.equal(scaled.ok, true, JSON.stringify(scaled.errors));
    const blocks = nodesOf(architecture.getState().graph).find((node) => node.kind === 'repeat');
    assert.deepEqual(scaled.shapes[blocks.id], [null, 1024, 1024]);
});
