import { createEdge, createLayerNode, createSuperNodeInstance } from '../architecture/index.js';
import { getEntry } from '../library/index.js';
import { ensureStandardTemplate } from '../components/index.js';
import { attentionHeadParameters, attentionParameters, buildAttentionHeadGraph, buildMultiHeadAttentionGraph } from '../../storyboards/attention.js';

const positiveInteger = (value) => Number.isSafeInteger(value) && value > 0;
const probability = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value < 1;
const X = 420;
const GAP = 140;

function layer(key, x, y, parameters = {}, label) {
    const node = createLayerNode({ libraryKey: key, entry: getEntry(key), position: { x, y }, label });
    return { ...node, parameters: { ...node.parameters, ...parameters } };
}

function graphOf(nodes, links) {
    return {
        nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
        edges: links.map(([source, target]) => createEdge({ source: source.id, target: target.id })),
    };
}

/** BERT's post-LayerNorm encoder block; unlike GPT-2 its attention is bidirectional. */
export function buildBertEncoderLayerGraph({ attentionTemplateId, linearTemplateId, layerNormTemplateId }) {
    const component = (templateId, x, y, parameters, label) => createSuperNodeInstance({
        templateId, position: { x, y }, parameterValues: parameters, label,
    });
    const input = layer('deepflow.template.Input', X, 0, {}, 'layer input');
    const attention = component(attentionTemplateId, X, GAP, { D: '$d_model', H: '$n_heads', dropout: '$attention_dropout', bias: true }, 'bidirectional attention');
    const attentionDrop = layer('torch.nn.Dropout', X, GAP * 2, { p: '$hidden_dropout' }, 'attention output dropout');
    const attentionAdd = layer('deepflow.op.Add', X, GAP * 3, {}, 'attention residual');
    const attentionNorm = component(layerNormTemplateId, X, GAP * 4, { normalized_shape: '$d_model', eps: '$norm_eps' }, 'attention LayerNorm');
    const intermediate = component(linearTemplateId, X, GAP * 5, { in_features: '$d_model', out_features: '$intermediate_size' }, 'intermediate dense');
    const activation = layer('torch.nn.GELU', X, GAP * 6, { approximate: 'none' }, 'GELU');
    const outputDense = component(linearTemplateId, X, GAP * 7, { in_features: '$intermediate_size', out_features: '$d_model' }, 'output dense');
    const outputDrop = layer('torch.nn.Dropout', X, GAP * 8, { p: '$hidden_dropout' }, 'output dropout');
    const outputAdd = layer('deepflow.op.Add', X, GAP * 9, {}, 'feed-forward residual');
    const outputNorm = component(layerNormTemplateId, X, GAP * 10, { normalized_shape: '$d_model', eps: '$norm_eps' }, 'output LayerNorm');
    const output = layer('deepflow.template.Output', X, GAP * 11, {}, 'layer output');
    const chain = [input, attention, attentionDrop, attentionAdd, attentionNorm, intermediate, activation, outputDense, outputDrop, outputAdd, outputNorm, output];
    const links = chain.slice(0, -1).map((node, index) => [node, chain[index + 1]]);
    links.push([input, attentionAdd], [attentionNorm, outputAdd]);
    return graphOf(chain, links);
}

/** Deliberately narrow: only the standard masked-language-model architecture. */
export function inspectBertConfig(config) {
    if (config?.model_type !== 'bert') return null;
    const architecture = config.architectures?.[0];
    if (!Array.isArray(config.architectures) || !config.architectures.includes('BertForMaskedLM')) {
        return { supported: false, family: 'BERT', reason: `Only BertForMaskedLM is mapped; this config declares ${architecture || 'no architecture'}.` };
    }
    if (config.auto_map?.AutoModelForMaskedLM || config.is_decoder || config.is_encoder_decoder || config.add_cross_attention
        || (config.position_embedding_type ?? 'absolute') !== 'absolute' || (config.chunk_size_feed_forward ?? 0) !== 0) {
        return { supported: false, family: 'BERT', reason: 'This BERT variant changes encoder topology, positional embeddings, or uses custom model code.' };
    }
    const dimensions = [config.vocab_size, config.hidden_size, config.num_hidden_layers, config.num_attention_heads,
        config.intermediate_size, config.max_position_embeddings, config.type_vocab_size];
    if (!dimensions.every(positiveInteger) || config.hidden_size % config.num_attention_heads !== 0) {
        return { supported: false, family: 'BERT', reason: 'The BERT dimensions are missing, invalid, or incompatible.' };
    }
    if (!Number.isSafeInteger(config.pad_token_id ?? 0) || (config.pad_token_id ?? 0) < 0 || (config.pad_token_id ?? 0) >= config.vocab_size) {
        return { supported: false, family: 'BERT', reason: 'The BERT padding token index is invalid.' };
    }
    const activation = config.hidden_act ?? 'gelu';
    if (activation !== 'gelu') {
        return { supported: false, family: 'BERT', reason: `Activation “${activation}” is not represented by the BERT adapter.` };
    }
    const rates = [config.hidden_dropout_prob ?? 0.1, config.attention_probs_dropout_prob ?? 0.1];
    if (!rates.every(probability) || !(Number(config.layer_norm_eps ?? 1e-12) > 0)) {
        return { supported: false, family: 'BERT', reason: 'A dropout rate or LayerNorm epsilon is invalid.' };
    }
    const configuration = {
        vocabSize: config.vocab_size, context: config.max_position_embeddings, dModel: config.hidden_size,
        nHeads: config.num_attention_heads, nLayers: config.num_hidden_layers, intermediateSize: config.intermediate_size,
        typeVocabSize: config.type_vocab_size, padTokenId: config.pad_token_id ?? 0,
        hiddenDropout: rates[0], attentionDropout: rates[1], normEpsilon: config.layer_norm_eps ?? 1e-12,
    };
    return {
        supported: true, family: 'BERT', architecture: 'BertForMaskedLM', configuration,
        components: [
            { name: 'Token, position and segment embeddings', detail: `${configuration.context} positions · ${configuration.dModel} features` },
            { name: 'Bidirectional encoder', detail: `${configuration.nLayers} post-LN layers · ${configuration.nHeads} heads` },
            { name: 'Masked-language-model head', detail: `${configuration.dModel} → ${configuration.vocabSize.toLocaleString()} vocabulary logits` },
        ],
        limitations: [
            'Editable config-level approximation; pretrained weights are not loaded.',
            'The optional padding attention mask is not represented; the graph describes unpadded sequences.',
            'Segment IDs appear as a separate input; BERT uses zeros when they are omitted.',
            'The MLM decoder is a separate Linear projection; token-embedding weight tying is not represented.',
            'The prediction loss and tokenizer are not imported from config.json.',
        ],
    };
}

export function importBertArchitecture(preview, { architectureStore, templatesStore, trainingStore }) {
    const inspection = inspectBertConfig(preview?.config);
    if (!inspection?.supported) throw new Error(inspection?.reason || 'This BERT model is not mapped.');
    const model = inspection.configuration;
    const importSource = {
        provider: 'huggingface', repoId: preview.repoId, revision: preview.requestedRevision, commit: preview.commit,
        modelType: 'bert', architecture: inspection.architecture, configuration: model,
        fidelity: 'editable-config-approximation', weightsLoaded: false, weightTyingRepresented: false,
        attentionMaskRepresented: false,
    };
    const arch = architectureStore.getState();
    const templates = templatesStore.getState();
    const standardOptions = { fresh: true, cache: new Map(), importSource };
    const addComponent = (key, position, label, parameterValues) => {
        const id = arch.addSuperNodeInstance(ensureStandardTemplate(key, templatesStore, standardOptions), position, label);
        arch.patchNode(id, { parameterValues });
        return id;
    };
    arch.goToLevel(0);
    arch.resetCurrentGraph();
    trainingStore.getState().resetGraph();
    const root = architectureStore.getState().graph;
    const input = Object.values(root.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
    const output = Object.values(root.nodes).find((node) => node.libraryKey === 'deepflow.io.Output');
    for (const [name, value] of [
        ['vocab_size', model.vocabSize], ['context', model.context], ['d_model', model.dModel],
        ['n_heads', model.nHeads], ['n_layers', model.nLayers], ['intermediate_size', model.intermediateSize],
        ['type_vocab_size', model.typeVocabSize],
    ]) arch.addGlobalParameter({ name, type: 'int', default: value, description: `Imported BERT ${name.replaceAll('_', ' ')}` });
    arch.moveNode(input.id, { x: X, y: 0 });
    arch.patchNode(input.id, { label: 'token IDs' });
    arch.patchLayerParameters(input.id, { shape: `(${model.context},)`, dtype: 'int64' });
    const segmentInput = arch.addLayerByLibraryKey('deepflow.io.Input', { x: 860, y: 0 });
    arch.patchNode(segmentInput, { label: 'segment IDs' });
    arch.patchLayerParameters(segmentInput, { shape: `(${model.context},)`, dtype: 'int64' });

    const token = addComponent('embedding', { x: X, y: GAP }, 'word embedding', {
        num_embeddings: '$vocab_size', embedding_dim: '$d_model', padding_idx: model.padTokenId,
    });
    const segment = addComponent('embedding', { x: 860, y: GAP }, 'segment embedding', {
        num_embeddings: '$type_vocab_size', embedding_dim: '$d_model', padding_idx: null,
    });
    const position = addComponent('learned-position', { x: 1260, y: GAP }, 'position embedding', {
        max_len: '$context', embedding_dim: '$d_model',
    });
    const addSegment = arch.addLayerByLibraryKey('deepflow.op.Add', { x: X, y: GAP * 2 });
    arch.patchNode(addSegment, { label: 'word + segment' });
    const addPosition = arch.addLayerByLibraryKey('deepflow.op.Add', { x: X, y: GAP * 3 });
    arch.patchNode(addPosition, { label: 'add positions' });
    const embeddingNorm = addComponent('layernorm', { x: X, y: GAP * 4 }, 'embedding LayerNorm', {
        normalized_shape: '$d_model', eps: model.normEpsilon,
    });
    const embeddingDrop = arch.addLayerByLibraryKey('torch.nn.Dropout', { x: X, y: GAP * 5 });
    arch.patchLayerParameters(embeddingDrop, { p: model.hiddenDropout });

    // Register separate imported attention templates; the ordinary palette remains causal.
    const linearTemplateId = ensureStandardTemplate('linear', templatesStore, standardOptions);
    const layerNormTemplateId = ensureStandardTemplate('layernorm', templatesStore, standardOptions);
    const headTemplateId = templates.createTemplate({
        name: 'Bidirectional attention head', description: 'Scaled dot-product attention without a causal mask.',
        graph: buildAttentionHeadGraph({ causal: false }), freeParameters: attentionHeadParameters(),
        importSource: { ...importSource, component: 'bidirectional-head' },
    });
    const attentionTemplateId = templates.createTemplate({
        name: 'Bidirectional multi-head attention', description: 'Q/K/V projections, parallel heads, concatenation, and output projection.',
        graph: buildMultiHeadAttentionGraph({ headTemplateId, linearTemplateId }), freeParameters: attentionParameters(),
        importSource: { ...importSource, component: 'bidirectional-attention' },
    });
    const encoderTemplateId = templates.createTemplate({
        name: 'BERT encoder layer', description: 'Post-LayerNorm bidirectional attention and feed-forward network with residual paths.',
        graph: buildBertEncoderLayerGraph({ attentionTemplateId, linearTemplateId, layerNormTemplateId }),
        importSource: { ...importSource, component: 'encoder-layer' },
    });
    for (const [name, value] of [
        ['d_model', model.dModel], ['n_heads', model.nHeads], ['intermediate_size', model.intermediateSize],
        ['norm_eps', model.normEpsilon], ['attention_dropout', model.attentionDropout], ['hidden_dropout', model.hiddenDropout],
    ]) templates.addFreeParameter(encoderTemplateId, { name, type: Number.isInteger(value) && name !== 'norm_eps' && !name.includes('dropout') ? 'int' : 'float', default: value, description: `BERT ${name.replaceAll('_', ' ')}` });
    const blocks = arch.addRepeatNode(encoderTemplateId, { x: X, y: GAP * 6 }, { count: '$n_layers', label: 'BERT encoder layers', bindings: {
        d_model: '$d_model', n_heads: '$n_heads', intermediate_size: '$intermediate_size',
        norm_eps: model.normEpsilon, attention_dropout: model.attentionDropout, hidden_dropout: model.hiddenDropout,
    } });
    const predictionDense = addComponent('linear', { x: X, y: GAP * 7 + 15 }, 'MLM transform', {
        in_features: '$d_model', out_features: '$d_model', bias: true,
    });
    const gelu = arch.addLayerByLibraryKey('torch.nn.GELU', { x: X, y: GAP * 8 + 15 });
    arch.patchLayerParameters(gelu, { approximate: 'none' });
    const predictionNorm = addComponent('layernorm', { x: X, y: GAP * 9 + 15 }, 'MLM LayerNorm', {
        normalized_shape: '$d_model', eps: model.normEpsilon,
    });
    const decoder = addComponent('linear', { x: X, y: GAP * 10 + 15 }, 'vocabulary decoder', {
        in_features: '$d_model', out_features: '$vocab_size', bias: true,
    });
    arch.moveNode(output.id, { x: X, y: GAP * 11 + 15 });
    arch.patchNode(output.id, { label: 'masked-token logits' });
    for (const [source, target] of [
        [input.id, token], [segmentInput, segment], [input.id, position],
        [token, addSegment], [segment, addSegment], [addSegment, addPosition], [position, addPosition],
        [addPosition, embeddingNorm], [embeddingNorm, embeddingDrop], [embeddingDrop, blocks],
        [blocks, predictionDense], [predictionDense, gelu], [gelu, predictionNorm],
        [predictionNorm, decoder], [decoder, output.id],
    ]) arch.connect({ source, target });
    architectureStore.setState((state) => ({ graph: { ...state.graph, importSource } }));
    return importSource;
}
