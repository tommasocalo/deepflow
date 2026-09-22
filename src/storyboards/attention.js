import { createEdge, createLayerNode, createRepeatNode, createSuperNodeInstance } from '../domain/architecture/index.js';
import { getEntry } from '../domain/library/index.js';
import { ensureStandardTemplate } from '../domain/components/index.js';

export const ATTENTION_HEAD_INPUT_IDS = Object.freeze({ q: 'head_q', k: 'head_k', v: 'head_v' });
export const ATTENTION_INPUT_ID = 'attention_input';

export const attentionHeadParameters = () => [
    { name: 'd_head', type: 'int', description: 'Q input feature dimension', derived: { kind: 'input-dimension', inputId: ATTENTION_HEAD_INPUT_IDS.q, axis: -1 } },
    { name: 'dropout', type: 'float', description: 'Attention dropout probability', default: 0.1 },
];
export const attentionParameters = () => [
    { name: 'D', type: 'int', description: 'Input feature dimension', derived: { kind: 'input-dimension', inputId: ATTENTION_INPUT_ID, axis: -1 } },
    { name: 'H', type: 'int', description: 'Number of parallel heads', default: 12 },
    { name: 'dropout', type: 'float', description: 'Attention dropout probability', default: 0.1 },
    { name: 'bias', type: 'bool', description: 'Learned projection bias', default: true },
];

const layer = (key, x, y, parameters = {}, label, id) => {
    const node = createLayerNode({ libraryKey: key, entry: getEntry(key), position: { x, y }, label, id });
    return { ...node, parameters: { ...node.parameters, ...parameters } };
};
const graphOf = (nodes, connections) => ({
    nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
    edges: connections.map(([source, target, targetHandle]) => createEdge({ source: source.id, target: target.id, targetHandle })),
});

/** One shared, editable head body; each parallel iteration receives its own Q/K/V slice. */
export function buildAttentionHeadGraph() {
    const q = layer('deepflow.template.Input', 180, 0, {}, 'Q', ATTENTION_HEAD_INPUT_IDS.q);
    const k = layer('deepflow.template.Input', 560, 0, {}, 'K', ATTENTION_HEAD_INPUT_IDS.k);
    const v = layer('deepflow.template.Input', 940, 0, {}, 'V', ATTENTION_HEAD_INPUT_IDS.v);
    const transpose = layer('deepflow.op.Transpose', 560, 140, { dim0: -2, dim1: -1 }, 'K transpose');
    const scores = layer('deepflow.op.MatMul', 380, 280, {}, 'Q × Kᵀ');
    const scale = layer('deepflow.op.Scale', 380, 410, { factor: '1 / (d_head ** 0.5)' }, 'scaled scores');
    const mask = layer('deepflow.op.CausalMask', 380, 540, {}, 'causal mask');
    const softmax = layer('torch.nn.Softmax', 380, 670, { dim: -1 }, 'attention weights');
    const drop = layer('torch.nn.Dropout', 380, 800, { p: '$dropout' }, 'attention dropout');
    const weighted = layer('deepflow.op.MatMul', 560, 930, {}, 'weights × V');
    const output = layer('deepflow.template.Output', 560, 1060, {}, 'head output');
    return graphOf([q, k, v, transpose, scores, scale, mask, softmax, drop, weighted, output], [
        [q, scores, 'a'], [k, transpose], [transpose, scores, 'b'],
        [scores, scale], [scale, mask], [mask, softmax], [softmax, drop],
        [drop, weighted, 'a'], [v, weighted, 'b'], [weighted, output],
    ]);
}

/** Projection → split → generic parallel Repeat → concat → output projection. */
export function buildMultiHeadAttentionGraph({ headTemplateId, headInputIds = ATTENTION_HEAD_INPUT_IDS, linearTemplateId } = {}) {
    if (!headTemplateId) throw new Error('A head template is required to build multi-head attention.');
    const input = layer('deepflow.template.Input', 560, 0, {}, 'attention input', ATTENTION_INPUT_ID);
    const projectionNode = (x, y, label) => linearTemplateId
        ? createSuperNodeInstance({ templateId: linearTemplateId, position: { x, y }, label, parameterValues: { in_features: '$D', out_features: '$D', bias: '$bias' } })
        : layer('torch.nn.Linear', x, y, { in_features: '$D', out_features: '$D' }, label);
    const q = projectionNode(180, 150, 'Q projection');
    const k = projectionNode(560, 150, 'K projection');
    const v = projectionNode(940, 150, 'V projection');
    const qSplit = layer('deepflow.op.Split', 180, 310, { chunks: '$H', dim: -1 }, 'split Q');
    const kSplit = layer('deepflow.op.Split', 560, 310, { chunks: '$H', dim: -1 }, 'split K');
    const vSplit = layer('deepflow.op.Split', 940, 310, { chunks: '$H', dim: -1 }, 'split V');
    const heads = createRepeatNode({
        templateId: headTemplateId, position: { x: 560, y: 490 }, label: 'Attention heads',
        mode: 'parallel', count: 'auto', bindings: { dropout: '$dropout' },
    });
    const concat = layer('deepflow.op.Concat', 560, 760, { dim: -1 }, 'concatenate heads');
    const projection = projectionNode(560, 910, 'output projection');
    const output = layer('deepflow.template.Output', 560, 1060, {}, 'attention output');
    return graphOf([input, q, k, v, qSplit, kSplit, vSplit, heads, concat, projection, output], [
        [input, q], [input, k], [input, v], [q, qSplit], [k, kSplit], [v, vSplit],
        [qSplit, heads, headInputIds.q], [kSplit, heads, headInputIds.k], [vSplit, heads, headInputIds.v],
        [heads, concat], [concat, projection], [projection, output],
    ]);
}

/** Register ordinary templates; saved architectures are never rewritten. */
export function registerAttentionTemplates(templatesStore) {
    const templates = templatesStore.getState();
    const headTemplateId = templates.createTemplate({
        name: 'Attention head',
        description: 'Editable scaled dot-product attention: Q × Kᵀ, scale, causal mask, softmax, dropout, and weights × V.',
        graph: buildAttentionHeadGraph(),
        freeParameters: attentionHeadParameters(),
    });
    const attentionTemplateId = templates.createTemplate({
        name: 'Multi-head attention',
        description: 'Editable Q/K/V projections, equal splits, parallel attention heads, concatenation, and output projection.',
        graph: buildMultiHeadAttentionGraph({ headTemplateId }),
        freeParameters: attentionParameters(),
    });
    return { headTemplateId, attentionTemplateId };
}

/** A small standalone example that opens directly inside the editable MHA template. */
export function attentionStoryboard({ architectureStore, templatesStore, trainingStore }) {
    const arch = architectureStore.getState();
    arch.goToLevel(0);
    arch.resetCurrentGraph();
    trainingStore.getState().resetGraph();
    const attentionTemplateId = ensureStandardTemplate('attention', templatesStore);
    const root = architectureStore.getState().graph;
    const input = Object.values(root.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
    const output = Object.values(root.nodes).find((node) => node.libraryKey === 'deepflow.io.Output');
    arch.moveNode(input.id, { x: 560, y: 0 });
    arch.patchNode(input.id, { label: 'token features' });
    arch.patchLayerParameters(input.id, { shape: '(128, 768)' });
    const attention = arch.addSuperNodeInstance(attentionTemplateId, { x: 560, y: 180 }, 'Multi-head attention');
    arch.moveNode(output.id, { x: 560, y: 360 });
    arch.patchNode(output.id, { label: 'contextual features' });
    arch.connect({ source: input.id, target: attention });
    arch.connect({ source: attention, target: output.id });
    arch.enterSupernode(attention);
}
