import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { fetchHubModelConfig, parseModelReference, searchHubModels } from '../src/domain/huggingFace/client.js';
import { importHuggingFaceArchitecture, inspectHubModel } from '../src/domain/huggingFace/index.js';
import { ensureStandardTemplate } from '../src/domain/components/index.js';
import { validate } from '../src/validation/validator.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';

const sha = '607a30d783dfa663caf39e06633721c8d4cfcd7e';
const config = {
    model_type: 'gpt2', architectures: ['GPT2LMHeadModel'], vocab_size: 50257,
    n_positions: 1024, n_embd: 768, n_layer: 6, n_head: 12, n_inner: null,
    activation_function: 'gelu_new', embd_pdrop: 0.05, attn_pdrop: 0.15,
    resid_pdrop: 0.2, layer_norm_epsilon: 1e-6,
};
const preview = { repoId: 'distilbert/distilgpt2', requestedRevision: 'main', commit: sha, config };
const bertConfig = {
    model_type: 'bert', architectures: ['BertForMaskedLM'], vocab_size: 30522,
    max_position_embeddings: 512, hidden_size: 768, num_hidden_layers: 12,
    num_attention_heads: 12, intermediate_size: 3072, type_vocab_size: 2,
    pad_token_id: 0, hidden_act: 'gelu', hidden_dropout_prob: 0.1,
    attention_probs_dropout_prob: 0.1, layer_norm_eps: 1e-12,
    position_embedding_type: 'absolute',
};
const bertPreview = { repoId: 'google-bert/bert-base-uncased', requestedRevision: 'main', commit: sha, config: bertConfig };
const storage = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
} };
const { useArchitectureStore: architectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: trainingStore } = await import('../src/stores/trainingStore.js');

beforeEach(() => {
    architectureStore.setState(architectureStore.getInitialState(), true);
    templatesStore.setState(templatesStore.getInitialState(), true);
    trainingStore.setState(trainingStore.getInitialState(), true);
    storage.clear();
});

test('only Hugging Face model ids and URLs become model references', () => {
    assert.deepEqual(parseModelReference('distilbert/distilgpt2'), { repoId: 'distilbert/distilgpt2', revision: 'main' });
    assert.deepEqual(parseModelReference('https://huggingface.co/distilbert/distilgpt2/tree/main'), { repoId: 'distilbert/distilgpt2', revision: 'main' });
    assert.equal(parseModelReference('https://evil.example/distilbert/distilgpt2'), null);
    assert.equal(parseModelReference('../../private'), null);
});

test('Hub search carries the pagination cursor and returns compact model metadata', async () => {
    let requestUrl;
    const fetchImpl = async (url) => {
        requestUrl = new URL(url);
        return { ok: true, json: async () => [{ id: 'distilbert/distilgpt2', downloads: 42, pipeline_tag: 'text-generation' }],
            headers: { get: () => '<https://huggingface.co/api/models?cursor=next-page>; rel="next"' } };
    };
    const result = await searchHubModels('distilgpt2', { cursor: 'first-page', fetchImpl });
    assert.equal(requestUrl.searchParams.get('cursor'), 'first-page');
    assert.equal(requestUrl.searchParams.get('search'), 'distilgpt2');
    assert.equal(result.nextCursor, 'next-page');
    assert.deepEqual(result.models, [{ id: 'distilbert/distilgpt2', downloads: 42, pipelineTag: 'text-generation', lastModified: null, private: false, gated: false }]);
});

test('config is fetched at the resolved immutable commit', async () => {
    const visited = [];
    const fetchImpl = async (url) => {
        visited.push(url);
        return { ok: true, json: async () => visited.length === 1
            ? { id: 'distilbert/distilgpt2', sha, downloads: 42, pipeline_tag: 'text-generation' } : config };
    };
    const result = await fetchHubModelConfig('distilbert/distilgpt2', { fetchImpl });
    assert.equal(visited.length, 2);
    assert.equal(visited[0], 'https://huggingface.co/api/models/distilbert/distilgpt2/revision/main');
    assert.equal(visited[1], `https://huggingface.co/distilbert/distilgpt2/resolve/${sha}/config.json`);
    assert.equal(result.commit, sha);
});

test('inspectors reject unsupported architecture variants explicitly', () => {
    assert.equal(inspectHubModel({ config: { model_type: 'bert' } }).inspection.supported, false);
    assert.equal(inspectHubModel({ config: { ...bertConfig, architectures: ['BertModel'] } }).inspection.supported, false);
    assert.equal(inspectHubModel({ config: { ...bertConfig, is_decoder: true } }).inspection.supported, false);
    assert.equal(inspectHubModel({ config: { ...bertConfig, position_embedding_type: 'relative_key' } }).inspection.supported, false);
    assert.equal(inspectHubModel({ config: { ...bertConfig, hidden_act: 'relu' } }).inspection.supported, false);
    assert.match(inspectHubModel({ config: { ...config, add_cross_attention: true } }).inspection.reason, /attention topology/);
    assert.match(inspectHubModel({ config: { ...config, activation_function: 'relu' } }).inspection.reason, /Activation/);
    assert.equal(inspectHubModel(preview).inspection.configuration.nLayers, 6);
});

test('BERT masked-LM config builds a bidirectional post-LN encoder and MLM head', () => {
    const inspection = inspectHubModel(bertPreview).inspection;
    assert.equal(inspection.supported, true);
    assert.match(inspection.limitations.join(' '), /padding attention mask/);
    const source = importHuggingFaceArchitecture(bertPreview, { architectureStore, templatesStore, trainingStore });
    const graph = architectureStore.getState().graph;
    const nodes = Object.values(graph.nodes);
    const blocks = nodes.find((node) => node.kind === 'repeat');
    const encoder = templatesStore.getState().getTemplate(blocks.templateId);
    const encoderNodes = Object.values(encoder.graph.nodes);
    const attention = encoderNodes.find((node) => node.label === 'bidirectional attention');
    const attentionTemplate = templatesStore.getState().getTemplate(attention.templateId);
    const head = Object.values(attentionTemplate.graph.nodes).find((node) => node.kind === 'repeat');
    const headTemplate = templatesStore.getState().getTemplate(head.templateId);
    assert.equal(Object.values(headTemplate.graph.nodes).some((node) => node.libraryKey === 'deepflow.op.CausalMask'), false);
    assert.equal(encoderNodes.filter((node) => node.label?.includes('LayerNorm')).length, 2);
    assert.equal(encoder.graph.edges.length, encoderNodes.length - 1 + 2);
    assert.equal(nodes.filter((node) => node.libraryKey === 'deepflow.io.Input').length, 2);
    assert.equal(nodes.filter((node) => node.label === 'word embedding' || node.label === 'segment embedding' || node.label === 'position embedding').length, 3);
    assert.equal(nodes.find((node) => node.label === 'vocabulary decoder').parameterValues.out_features, '$vocab_size');
    assert.equal(graph.parameters.find((parameter) => parameter.name === 'n_layers').default, 12);
    assert.equal(source.fidelity, 'editable-config-approximation');
    assert.equal(source.attentionMaskRepresented, false);
    assert.equal(graph.importSource.repoId, 'google-bert/bert-base-uncased');
    const report = validate(graph, { templates: templatesStore.getState().templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    const generated = generateArchitectureModule(graph, { templates: templatesStore.getState().templates });
    assert.match(generated, /torch\.matmul/);
    assert.doesNotMatch(generated, /triu\(/, 'BERT attention must not be causal');
});

test('distilgpt2 config builds a validated editable graph with pinned provenance and no invented training', () => {
    const source = importHuggingFaceArchitecture(preview, { architectureStore, templatesStore, trainingStore });
    const graph = architectureStore.getState().graph;
    const blocks = Object.values(graph.nodes).find((node) => node.kind === 'repeat');
    const template = templatesStore.getState().getTemplate(blocks.templateId);
    const attention = Object.values(template.graph.nodes).find((node) => node.label === 'causal self-attention');
    const dropout = Object.values(template.graph.nodes).find((node) => node.label === 'attention projection dropout');
    const norm = Object.values(template.graph.nodes).find((node) => node.label === 'ln_1');
    assert.equal(graph.parameters.find((item) => item.name === 'n_layers').default, 6);
    assert.equal(blocks.bindings.mlp_width, '4 * d_model');
    assert.equal(blocks.bindings.attn_dropout, 0.15);
    assert.equal(dropout.parameters.p, '$resid_dropout');
    assert.equal(attention.parameterValues.dropout, '$attn_dropout');
    assert.equal(norm.parameterValues.eps, '$norm_eps');
    assert.equal(template.importSource.commit, sha);
    assert.equal(graph.importSource.commit, sha);
    assert.equal(source.weightsLoaded, false);
    assert.equal(trainingStore.getState().graph.nodes && Object.keys(trainingStore.getState().graph.nodes).length, 0);
    const report = validate(graph, { templates: templatesStore.getState().templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
});

test('imported components do not inherit edits from a previously customized standard definition', () => {
    const editedId = ensureStandardTemplate('linear', templatesStore);
    templatesStore.getState().patchTemplate(editedId, { name: 'Customized Linear' });
    const edited = templatesStore.getState().getTemplate(editedId);
    importHuggingFaceArchitecture(preview, { architectureStore, templatesStore, trainingStore });
    const head = Object.values(architectureStore.getState().graph.nodes).find((node) => node.label === 'lm_head');
    const imported = templatesStore.getState().getTemplate(head.templateId);
    assert.notEqual(head.templateId, editedId);
    assert.equal(imported.name, 'Linear');
    assert.equal(imported.componentStyleKey, 'linear');
    assert.equal(imported.importSource.repoId, 'distilbert/distilgpt2');
    assert.equal(templatesStore.getState().getTemplate(editedId), edited);
    assert.equal(ensureStandardTemplate('linear', templatesStore), editedId);
});
