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
import { registerAttentionTemplates } from './attention.js';

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
export function buildTransformerBlockGraph({ attentionTemplateId } = {}) {
    const tin  = layer('deepflow.template.Input',        { x: X, y: 0 },   {}, 'block input');
    const ln1  = layer('torch.nn.LayerNorm',             { x: X, y: ROW_GAP },     { normalized_shape: '$d_model' }, 'ln_1');
    // Callers without registered templates retain the standalone legacy graph API.
    const attn = attentionTemplateId
        ? createSuperNodeInstance({ templateId: attentionTemplateId, position: { x: X, y: ROW_GAP * 2 },
            label: 'causal self-attention', parameterValues: { D: '$d_model', H: '$n_heads' } })
        : layer('deepflow.nn.CausalSelfAttention', { x: X, y: ROW_GAP * 2 }, { embed_dim: '$d_model', num_heads: '$n_heads', dropout: 0.1 }, 'causal self-attention');
    const add1 = layer('deepflow.op.Add',                { x: X, y: ROW_GAP * 3 }, {}, 'residual');
    const ln2  = layer('torch.nn.LayerNorm',             { x: X, y: ROW_GAP * 4 }, { normalized_shape: '$d_model' }, 'ln_2');
    const fc   = layer('torch.nn.Linear',                { x: X, y: ROW_GAP * 5 }, { in_features: '$d_model', out_features: '4 * d_model' }, 'mlp fc');
    const gelu = layer('torch.nn.GELU',                  { x: X, y: ROW_GAP * 6 }, {}, 'gelu');
    const proj = layer('torch.nn.Linear',                { x: X, y: ROW_GAP * 7 }, { in_features: '4 * d_model', out_features: '$d_model' }, 'mlp proj');
    const drop = layer('torch.nn.Dropout',               { x: X, y: ROW_GAP * 8 }, { p: 0.1 }, 'dropout');
    const add2 = layer('deepflow.op.Add',                { x: X, y: ROW_GAP * 9 }, {}, 'residual');
    const tout = layer('deepflow.template.Output',       { x: X, y: ROW_GAP * 10 }, {}, 'block output');

    // Presentation-only metadata: the experimental treatment is limited to
    // this MLP segment; parameters and generated PyTorch remain unchanged.
    for (const [node, role] of [[ln2, 'norm'], [fc, 'expand'], [gelu, 'activation'], [proj, 'project'], [drop, 'dropout'], [add2, 'residual']]) {
        node.visualization = 'tensor-flow';
        node.tensorRole = role;
    }

    const chain = [tin, ln1, attn, add1, ln2, fc, gelu, proj, drop, add2, tout];
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

export function gpt2Storyboard({ architectureStore, templatesStore, trainingStore }) {
    const arch = architectureStore.getState();
    const tpl = templatesStore.getState();
    const train = trainingStore.getState();

    arch.goToLevel(0);
    arch.resetCurrentGraph();
    train.resetGraph();

    const root = architectureStore.getState().graph;
    const inputId  = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
    const outputId = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;

    arch.moveNode(inputId, { x: X, y: 0 });
    arch.patchNode(inputId, { label: 'tokens' });
    arch.patchLayerParameters(inputId, { shape: '(1024,)', dtype: 'int64' });

    const wte = arch.addLayerByLibraryKey('torch.nn.Embedding', { x: X, y: ROW_GAP });
    arch.patchNode(wte, { label: 'token embedding' });
    arch.patchLayerParameters(wte, { num_embeddings: 50257, embedding_dim: 768 });

    const wpe = arch.addLayerByLibraryKey('deepflow.nn.PositionalEmbedding', { x: X_BRANCH, y: ROW_GAP });
    arch.patchNode(wpe, { label: 'positional embedding' });
    arch.patchLayerParameters(wpe, { max_len: 1024, embedding_dim: 768 });

    const add  = arch.addLayerByLibraryKey('deepflow.op.Add',   { x: X, y: ROW_GAP * 2 });
    const drop = arch.addLayerByLibraryKey('torch.nn.Dropout',  { x: X, y: ROW_GAP * 3 });
    arch.patchLayerParameters(drop, { p: 0.1 });

    const { attentionTemplateId } = registerAttentionTemplates(templatesStore);
    const tplId = tpl.createTemplate({
        name: 'Transformer block',
        description: 'Pre-LN GPT-2 block: LayerNorm → causal self-attention → residual → LayerNorm → MLP (4×) → residual.',
        graph: buildTransformerBlockGraph({ attentionTemplateId }),
    });
    tpl.addFreeParameter(tplId, { name: 'd_model', type: 'int', description: 'Model width', default: 768 });
    tpl.addFreeParameter(tplId, { name: 'n_heads', type: 'int', description: 'Attention heads', default: 12 });

    const blocks = arch.addRepeatNode(tplId, { x: X, y: ROW_GAP * 4 }, { count: 12, label: 'Transformer blocks' });

    // The repeated stack extends 15px below its face.
    const lnf = arch.addLayerByLibraryKey('torch.nn.LayerNorm', { x: X, y: ROW_GAP * 5 + 15 });
    arch.patchNode(lnf, { label: 'ln_f' });
    arch.patchLayerParameters(lnf, { normalized_shape: 768 });

    const head = arch.addLayerByLibraryKey('torch.nn.Linear', { x: X, y: ROW_GAP * 6 + 15 });
    arch.patchNode(head, { label: 'lm_head' });
    arch.patchLayerParameters(head, { in_features: 768, out_features: 50257, bias: false });

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

    // Training graph: AdamW + next-token cross-entropy
    const optId = train.addOptimizer('AdamW', { x: 60, y: 100 });
    train.patchHyperparameters(optId, { learning_rate: 6e-4, weight_decay: 0.1 });
    train.patchNode(optId, { label: 'AdamW' });
    const lossId = train.addLoss('cross_entropy', { x: 360, y: 100 }, outputId);
    train.patchNode(lossId, { label: 'next-token loss' });
    train.connect({ source: lossId, target: optId });
}
