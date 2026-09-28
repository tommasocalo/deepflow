import { gpt2Storyboard } from '../../storyboards/gpt2.js';

const positiveInteger = (value) => Number.isSafeInteger(value) && value > 0;
const probability = (value) => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value < 1;

/** An editable representation of the GPT2LMHeadModel topology, not a weight loader. */
export function inspectGpt2Config(config) {
    if (config?.model_type !== 'gpt2') return null;
    const architecture = config.architectures?.[0];
    if (!Array.isArray(config.architectures) || !config.architectures.includes('GPT2LMHeadModel')) {
        return { supported: false, family: 'GPT-2', reason: `Only GPT2LMHeadModel is supported; this config declares ${architecture || 'no architecture'}.` };
    }
    if (config.auto_map?.AutoModelForCausalLM || config.add_cross_attention || config.is_encoder_decoder
        || config.scale_attn_weights === false || config.scale_attn_by_inverse_layer_idx || config.reorder_and_upcast_attn) {
        return { supported: false, family: 'GPT-2', reason: 'This GPT-2 variant changes the attention topology or uses custom model code.' };
    }
    const sizes = [config.vocab_size, config.n_positions, config.n_embd, config.n_layer, config.n_head];
    if (!sizes.every(positiveInteger) || config.n_embd % config.n_head !== 0) {
        return { supported: false, family: 'GPT-2', reason: 'The GPT-2 dimensions are missing, invalid, or incompatible.' };
    }
    if (config.n_inner != null && !positiveInteger(config.n_inner)) {
        return { supported: false, family: 'GPT-2', reason: 'The GPT-2 MLP width is invalid.' };
    }
    const activation = config.activation_function ?? 'gelu_new';
    if (!['gelu_new', 'gelu'].includes(activation)) {
        return { supported: false, family: 'GPT-2', reason: `Activation “${activation}” is not represented by the GPT-2 adapter.` };
    }
    const rates = [config.embd_pdrop ?? 0.1, config.attn_pdrop ?? 0.1, config.resid_pdrop ?? 0.1];
    if (!rates.every(probability) || !(Number(config.layer_norm_epsilon ?? 1e-5) > 0)) {
        return { supported: false, family: 'GPT-2', reason: 'A dropout rate or LayerNorm epsilon is invalid.' };
    }
    const configuration = {
        vocabSize: config.vocab_size,
        context: config.n_positions,
        dModel: config.n_embd,
        nHeads: config.n_head,
        nLayers: config.n_layer,
        mlpWidth: config.n_inner ?? config.n_embd * 4,
        mlpWidthFollowsModel: config.n_inner == null,
        activation,
        embeddingDropout: rates[0],
        attentionDropout: rates[1],
        residualDropout: rates[2],
        normEpsilon: config.layer_norm_epsilon ?? 1e-5,
    };
    return {
        supported: true,
        family: 'GPT-2',
        architecture: 'GPT2LMHeadModel',
        configuration,
        components: [
            { name: 'Token embedding', detail: `${configuration.vocabSize.toLocaleString()} × ${configuration.dModel}` },
            { name: 'Position embedding', detail: `${configuration.context} × ${configuration.dModel}` },
            { name: 'Transformer block', detail: `${configuration.nLayers} sequential · ${configuration.nHeads} attention heads` },
            { name: 'Attention head', detail: `${configuration.dModel / configuration.nHeads} features per head` },
            { name: 'Language-model head', detail: `${configuration.dModel} → ${configuration.vocabSize.toLocaleString()}` },
        ],
        limitations: [
            'Imports an editable architecture from config.json; pretrained weights are not loaded.',
            'The language-model head is currently a separate Linear projection; token-embedding weight tying is not represented.',
            'GPT-2 implementation details such as packed QKV weights and KV cache are not imported.',
        ],
    };
}

export function importGpt2Architecture(preview, { architectureStore, templatesStore, trainingStore }) {
    const inspection = inspectGpt2Config(preview?.config);
    if (!inspection?.supported) throw new Error(inspection?.reason || 'This model is not supported by the GPT-2 adapter.');
    const importSource = {
        provider: 'huggingface',
        repoId: preview.repoId,
        revision: preview.requestedRevision,
        commit: preview.commit,
        modelType: 'gpt2',
        architecture: inspection.architecture,
        configuration: inspection.configuration,
        fidelity: 'editable-config-architecture',
        weightsLoaded: false,
        weightTyingRepresented: false,
    };
    gpt2Storyboard({ architectureStore, templatesStore, trainingStore,
        configuration: inspection.configuration, importSource, includeTraining: false });
    return importSource;
}
