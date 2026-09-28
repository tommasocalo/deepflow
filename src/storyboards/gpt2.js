/**
 * GPT-2 (small) storyboard — the default architecture DeepFlow opens with.
 *
 *   tokens (B, 1024) ─► token embedding ─┐
 *                     positional embed ──⊕─► dropout ─► [Transformer block] ×12 ─► ln_f ─► lm_head ─► logits
 *
 * The block is a parametric template (d_model, n_heads) with two residual
 * streams, so it exercises the Repeat primitive on a real modern network.
 */
import { createEdge, createLayerNode, createSuperNodeInstance } from '../domain/architecture/index.js';
import { getEntry } from '../domain/library/index.js';
import { ensureStandardTemplate } from '../domain/components/index.js';

const X = 400;       // main column (node centre)
const X_BRANCH = 820; // positional-embedding branch

// Leave room between the two external dimension labels on each wire.
const ROW_GAP = 130;

function layer(key, position, parameters = {}, label) {
    const entry = getEntry(key);
    const node = createLayerNode({ libraryKey: key, position, entry, label });
    node.parameters = { ...node.parameters, ...parameters };
    return node;
}

/** Pre-LN GPT-2 block, expressed as a template graph with boundary blocks. */
export function buildTransformerBlockGraph({ attentionTemplateId, linearTemplateId, layerNormTemplateId, activation = 'gelu_new' } = {}) {
    const componentOrLayer = (templateId, key, position, parameters, label) => templateId
        ? createSuperNodeInstance({ templateId, position, parameterValues: parameters, label })
        : layer(key, position, parameters, label);
    const tin  = layer('deepflow.template.Input',        { x: X, y: 0 },   {}, 'block input');
    const ln1  = componentOrLayer(layerNormTemplateId, 'torch.nn.LayerNorm', { x: X, y: ROW_GAP }, { normalized_shape: '$d_model', eps: '$norm_eps' }, 'ln_1');
    // Callers without registered templates retain the standalone legacy graph API.
    const attn = attentionTemplateId
        ? createSuperNodeInstance({ templateId: attentionTemplateId, position: { x: X, y: ROW_GAP * 2 },
            label: 'causal self-attention', parameterValues: { D: '$d_model', H: '$n_heads', dropout: '$attn_dropout' } })
        : layer('deepflow.nn.CausalSelfAttention', { x: X, y: ROW_GAP * 2 }, { embed_dim: '$d_model', num_heads: '$n_heads', dropout: '$attn_dropout' }, 'causal self-attention');
    const attnDrop = layer('torch.nn.Dropout',            { x: X, y: ROW_GAP * 3 }, { p: '$resid_dropout' }, 'attention projection dropout');
    const add1 = layer('deepflow.op.Add',                { x: X, y: ROW_GAP * 4 }, {}, 'residual');
    const ln2  = componentOrLayer(layerNormTemplateId, 'torch.nn.LayerNorm', { x: X, y: ROW_GAP * 5 }, { normalized_shape: '$d_model', eps: '$norm_eps' }, 'ln_2');
    const fc   = componentOrLayer(linearTemplateId, 'torch.nn.Linear', { x: X, y: ROW_GAP * 6 }, { in_features: '$d_model', out_features: '$mlp_width' }, 'mlp fc');
    const gelu = layer('torch.nn.GELU',                  { x: X, y: ROW_GAP * 7 }, { approximate: activation === 'gelu_new' ? 'tanh' : 'none' }, 'gelu');
    const proj = componentOrLayer(linearTemplateId, 'torch.nn.Linear', { x: X, y: ROW_GAP * 8 }, { in_features: '$mlp_width', out_features: '$d_model' }, 'mlp proj');
    const drop = layer('torch.nn.Dropout',               { x: X, y: ROW_GAP * 9 }, { p: '$resid_dropout' }, 'dropout');
    const add2 = layer('deepflow.op.Add',                { x: X, y: ROW_GAP * 10 }, {}, 'residual');
    const tout = layer('deepflow.template.Output',       { x: X, y: ROW_GAP * 11 }, {}, 'block output');

    // Presentation-only metadata: the experimental treatment is limited to
    // this MLP segment; parameters and generated PyTorch remain unchanged.
    for (const [node, role] of [[ln2, 'norm'], [fc, 'expand'], [gelu, 'activation'], [proj, 'project'], [drop, 'dropout'], [add2, 'residual']]) {
        node.visualization = 'tensor-flow';
        node.tensorRole = role;
    }

    const chain = [tin, ln1, attn, attnDrop, add1, ln2, fc, gelu, proj, drop, add2, tout];
    const nodes = Object.fromEntries(chain.map((n) => [n.id, n]));
    const edges = [];
    for (let i = 0; i < chain.length - 1; i++) {
        edges.push(createEdge({ source: chain[i].id, target: chain[i + 1].id }));
    }
    // Residual streams
    edges.push(createEdge({ source: tin.id,  target: add1.id }));
    edges.push(createEdge({ source: add1.id, target: add2.id }));
    return { nodes, edges };
}

export function gpt2Storyboard({ architectureStore, templatesStore, trainingStore, configuration = {}, importSource = null, includeTraining = true }) {
    const model = {
        vocabSize: 50257, context: 1024, dModel: 768, nHeads: 12, nLayers: 12,
        mlpWidth: 3072, mlpWidthFollowsModel: true, activation: 'gelu_new', embeddingDropout: 0.1,
        attentionDropout: 0.1, residualDropout: 0.1, normEpsilon: 1e-5,
        ...configuration,
    };
    const arch = architectureStore.getState();
    const tpl = templatesStore.getState();
    const train = trainingStore.getState();

    arch.goToLevel(0);
    arch.resetCurrentGraph();
    train.resetGraph();

    const root = architectureStore.getState().graph;
    const inputId  = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
    const outputId = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;

    // Model-wide sizes: change one and every layer that uses it follows.
    for (const [name, value, description] of [
        ['vocab_size', model.vocabSize, 'Vocabulary size'], ['context', model.context, 'Maximum sequence length'],
        ['d_model', model.dModel, 'Model width'], ['n_heads', model.nHeads, 'Attention heads'], ['n_layers', model.nLayers, 'Transformer blocks'],
    ]) arch.addGlobalParameter({ name, type: 'int', default: value, description });

    arch.moveNode(inputId, { x: X, y: 0 });
    arch.patchNode(inputId, { label: 'tokens' });
    arch.patchLayerParameters(inputId, { shape: `(${model.context},)`, dtype: 'int64' });

    const wte = arch.addStandardComponent('embedding', { x: X, y: ROW_GAP });
    arch.patchNode(wte, { label: 'token embedding', parameterValues: { num_embeddings: '$vocab_size', embedding_dim: '$d_model' } });

    const wpe = arch.addStandardComponent('learned-position', { x: X_BRANCH, y: ROW_GAP });
    arch.patchNode(wpe, { label: 'positional embedding', parameterValues: { max_len: '$context', embedding_dim: '$d_model' } });

    const add  = arch.addLayerByLibraryKey('deepflow.op.Add',   { x: X, y: ROW_GAP * 2 });
    const drop = arch.addLayerByLibraryKey('torch.nn.Dropout',  { x: X, y: ROW_GAP * 3 });
    arch.patchLayerParameters(drop, { p: model.embeddingDropout });

    const attentionTemplateId = ensureStandardTemplate('attention', templatesStore);
    const linearTemplateId = ensureStandardTemplate('linear', templatesStore);
    const layerNormTemplateId = ensureStandardTemplate('layernorm', templatesStore);
    const tplId = tpl.createTemplate({
        name: 'Transformer block',
        description: 'Pre-LN GPT-2 block: LayerNorm → causal self-attention → residual → LayerNorm → MLP (4×) → residual.',
        graph: buildTransformerBlockGraph({ attentionTemplateId, linearTemplateId, layerNormTemplateId, activation: model.activation }),
        ...(importSource ? { importSource: { ...importSource, component: 'transformer-block' } } : {}),
    });
    tpl.addFreeParameter(tplId, { name: 'd_model', type: 'int', description: 'Model width', default: model.dModel });
    tpl.addFreeParameter(tplId, { name: 'n_heads', type: 'int', description: 'Attention heads', default: model.nHeads });
    tpl.addFreeParameter(tplId, { name: 'mlp_width', type: 'int', description: 'MLP hidden width', default: model.mlpWidth });
    tpl.addFreeParameter(tplId, { name: 'norm_eps', type: 'float', description: 'LayerNorm epsilon', default: model.normEpsilon });
    tpl.addFreeParameter(tplId, { name: 'attn_dropout', type: 'float', description: 'Attention weights dropout', default: model.attentionDropout });
    tpl.addFreeParameter(tplId, { name: 'resid_dropout', type: 'float', description: 'Residual projection dropout', default: model.residualDropout });

    const blocks = arch.addRepeatNode(tplId, { x: X, y: ROW_GAP * 4 }, { count: '$n_layers', label: 'Transformer blocks', bindings: {
        d_model: '$d_model', n_heads: '$n_heads', mlp_width: model.mlpWidthFollowsModel ? '4 * d_model' : model.mlpWidth, norm_eps: model.normEpsilon,
        attn_dropout: model.attentionDropout, resid_dropout: model.residualDropout,
    } });

    // The repeated stack extends 15px below its face.
    const lnf = arch.addStandardComponent('layernorm', { x: X, y: ROW_GAP * 5 + 15 });
    arch.patchNode(lnf, { label: 'ln_f', parameterValues: { normalized_shape: '$d_model', eps: model.normEpsilon } });

    const head = arch.addStandardComponent('linear', { x: X, y: ROW_GAP * 6 + 15 });
    arch.patchNode(head, { label: 'lm_head', parameterValues: { in_features: '$d_model', out_features: '$vocab_size', bias: false } });

    arch.moveNode(outputId, { x: X, y: ROW_GAP * 7 + 15 });
    arch.patchNode(outputId, { label: 'logits' });

    arch.connect({ source: inputId, target: wte });
    arch.connect({ source: inputId, target: wpe });
    arch.connect({ source: wte, target: add });
    arch.connect({ source: wpe, target: add });
    arch.connect({ source: add, target: drop });
    arch.connect({ source: drop, target: blocks });
    arch.connect({ source: blocks, target: lnf });
    arch.connect({ source: lnf, target: head });
    arch.connect({ source: head, target: outputId });

    if (importSource) architectureStore.setState((state) => ({ graph: { ...state.graph, importSource } }));

    if (includeTraining) {
        // The demo defines training. A Hub config has no optimizer or loss recipe.
        const optId = train.addOptimizer('AdamW', { x: 60, y: 100 });
        train.patchHyperparameters(optId, { learning_rate: 6e-4, weight_decay: 0.1 });
        train.patchNode(optId, { label: 'AdamW' });
        const lossId = train.addLoss('cross_entropy', { x: 360, y: 100 }, outputId);
        train.patchNode(lossId, { label: 'next-token loss' });
        train.connect({ source: lossId, target: optId });
    }
}
