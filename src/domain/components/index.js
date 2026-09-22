import { createEdge, createLayerNode, createSuperNodeInstance } from '../architecture/index.js';
import { getEntry } from '../library/index.js';
import { attentionHeadParameters, attentionParameters, buildAttentionHeadGraph, buildMultiHeadAttentionGraph } from '../../storyboards/attention.js';

/** Standard components are ordinary editable templates, registered only on demand. */
export const STANDARD_COMPONENTS = [
    { key: 'linear', label: 'Linear', libraryKey: 'torch.nn.Linear', libraryKeys: ['torch.nn.Linear'], description: 'Learned weight and bias, transpose, matrix product, and addition.', aliases: ['dense', 'affine', 'projection'] },
    { key: 'embedding', label: 'Embedding', libraryKey: 'torch.nn.Embedding', libraryKeys: ['torch.nn.Embedding'], description: 'A learned table and indexed lookup.', aliases: ['lookup', 'tokens', 'table'] },
    { key: 'attention', label: 'Attention', libraryKey: 'deepflow.nn.CausalSelfAttention', libraryKeys: ['deepflow.nn.CausalSelfAttention'], description: 'Editable causal attention with Q/K/V projections and parallel heads.', aliases: ['MHA', 'causal', 'multi-head'] },
    { key: 'layernorm', label: 'Norm', libraryKey: 'torch.nn.LayerNorm', libraryKeys: ['torch.nn.LayerNorm'], description: 'Last-axis LayerNorm from mean, variance, reciprocal square root, scale, and bias.', aliases: ['LayerNorm', 'normalization', 'variance'] },
    { key: 'learned-position', label: 'Learned positions', libraryKey: 'deepflow.nn.PositionalEmbedding', libraryKeys: ['deepflow.nn.PositionalEmbedding'], description: 'Position indices and an editable learned embedding table.', aliases: ['positional embedding', 'GPT-2', 'positions'] },
    { key: 'sinusoidal-position', label: 'Sinusoidal positions', libraryKey: 'deepflow.op.Sin', libraryKeys: [], description: 'Position indices, frequencies, sine and cosine, with no learned weights.', aliases: ['positional encoding', 'sine', 'cosine', 'positions'] },
].map((component) => ({ ...component, name: component.label }));

const spec = (name, type, value, description = name) => ({ name, type, default: value, description });
const node = (libraryKey, label, x, y, parameters = {}) => {
    const created = createLayerNode({ libraryKey, entry: getEntry(libraryKey), label, position: { x, y } });
    return { ...created, parameters: { ...created.parameters, ...parameters } };
};
const graphOf = (nodes, wires) => ({
    nodes: Object.fromEntries(nodes.map((item) => [item.id, item])),
    edges: wires.map(([source, target, targetHandle]) => createEdge({ source: source.id, target: target.id, targetHandle })),
});
const boundary = (output = false, x = 400, y = 0) => node(`deepflow.template.${output ? 'Output' : 'Input'}`, output ? 'output' : 'input', x, y);
const parameter = (label, x, y, shape, init, rest = {}) => node('deepflow.op.Parameter', label, x, y, { shape, init, requires_grad: true, ...rest });

export function buildLinearComponentGraph() {
    const input = boundary(false, 220);
    const weight = parameter('weight', 680, 0, ['$out_features', '$in_features'], 'kaiming_uniform');
    const transpose = node('deepflow.op.Transpose', 'weight transpose', 680, 140, { dim0: -2, dim1: -1 });
    const product = node('deepflow.op.MatMul', 'input × weightᵀ', 400, 280);
    const bias = parameter('bias', 820, 280, ['$out_features'], 'uniform', { scale: '1 / (in_features ** 0.5)', enabled: '$bias', disabled_init: 'zeros' });
    const add = node('deepflow.op.Add', 'add bias', 400, 420);
    const output = boundary(true, 400, 560);
    return graphOf([input, weight, transpose, product, bias, add, output], [
        [input, product, 'a'], [weight, transpose], [transpose, product, 'b'], [product, add], [bias, add], [add, output],
    ]);
}

export function buildEmbeddingComponentGraph() {
    const input = boundary(false, 200);
    const table = parameter('embedding table', 720, 0, ['$num_embeddings', '$embedding_dim'], 'normal', { padding_idx: '$padding_idx' });
    const gather = node('deepflow.op.Gather', 'lookup rows', 400, 180, { padding_idx: '$padding_idx' });
    const output = boundary(true, 400, 360);
    return graphOf([input, table, gather, output], [[input, gather, 'indices'], [table, gather, 'weight'], [gather, output]]);
}

export function buildLayerNormComponentGraph() {
    const input = boundary(false, 220);
    const mean = node('deepflow.op.Mean', 'mean', 620, 140, { dim: -1, keepdim: true });
    const center = node('deepflow.op.Subtract', 'center', 400, 280);
    const square = node('deepflow.op.Pow', 'squared deviation', 740, 420, { exponent: 2 });
    const variance = node('deepflow.op.Mean', 'variance', 740, 560, { dim: -1, keepdim: true });
    const epsilon = node('deepflow.op.AddScalar', 'add epsilon', 740, 700, { value: '$eps' });
    const inverse = node('deepflow.op.Rsqrt', 'inverse standard deviation', 740, 840);
    const normalize = node('deepflow.op.Multiply', 'normalize', 400, 980);
    const gamma = parameter('scale', 40, 980, ['$normalized_shape'], 'ones', { enabled: '$elementwise_affine', disabled_init: 'ones' });
    const affine = node('deepflow.op.Multiply', 'apply scale', 400, 1120);
    const beta = parameter('bias', 820, 1120, ['$normalized_shape'], 'zeros', { enabled: '$elementwise_affine', disabled_init: 'zeros' });
    const add = node('deepflow.op.Add', 'add bias', 400, 1260);
    const output = boundary(true, 400, 1400);
    return graphOf([input, mean, center, square, variance, epsilon, inverse, normalize, gamma, affine, beta, add, output], [
        [input, mean], [input, center, 'a'], [mean, center, 'b'], [center, square], [square, variance], [variance, epsilon],
        [epsilon, inverse], [center, normalize], [inverse, normalize], [normalize, affine], [gamma, affine], [affine, add], [beta, add], [add, output],
    ]);
}

export function buildLearnedPositionComponentGraph(embeddingTemplateId) {
    const input = boundary();
    const positions = node('deepflow.op.PositionIndices', 'position indices', 400, 140);
    const embedding = createSuperNodeInstance({ templateId: embeddingTemplateId, label: 'position table', position: { x: 400, y: 280 }, parameterValues: { num_embeddings: '$max_len', embedding_dim: '$embedding_dim', padding_idx: null } });
    const output = boundary(true, 400, 420);
    return graphOf([input, positions, embedding, output], [[input, positions], [positions, embedding], [embedding, output]]);
}

export function buildSinusoidalPositionComponentGraph() {
    const input = boundary(false, 220);
    const positions = node('deepflow.op.PositionIndices', 'position indices', 220, 140);
    const expanded = node('deepflow.op.Unsqueeze', 'position axis', 220, 280, { dim: -1 });
    const indices = node('deepflow.op.Arange', 'even feature indices', 760, 0, { start: 0, end: '$D', step: 2 });
    const exponent = node('deepflow.op.Scale', 'frequency exponent', 760, 140, { factor: '-9.210340371976184 / D' });
    const frequency = node('deepflow.op.Exp', 'frequencies', 760, 280);
    const angles = node('deepflow.op.Multiply', 'position × frequency', 400, 420);
    const sin = node('deepflow.op.Sin', 'sine', 180, 560);
    const cos = node('deepflow.op.Cos', 'cosine', 700, 560);
    const stack = node('deepflow.op.Stack', 'interleave sine and cosine', 400, 700, { dim: -1 });
    const flatten = node('torch.nn.Flatten', 'flatten feature pairs', 400, 840, { start_dim: -2, end_dim: -1 });
    const slice = node('deepflow.op.Slice', 'model width', 400, 980, { dim: -1, start: 0, end: '$D', step: 1 });
    const output = boundary(true, 400, 1120);
    return graphOf([input, positions, expanded, indices, exponent, frequency, angles, sin, cos, stack, flatten, slice, output], [
        [input, positions], [positions, expanded], [indices, exponent], [exponent, frequency], [expanded, angles], [frequency, angles],
        [angles, sin], [angles, cos], [sin, stack], [cos, stack], [stack, flatten], [flatten, slice], [slice, output],
    ]);
}

/** Return the user's existing standard definition, or register a new definition and its dependencies. */
export function ensureStandardTemplate(key, templatesStore) {
    const component = STANDARD_COMPONENTS.find((item) => item.key === key);
    if (!component) return null;
    // A new catalog generation never rewrites an older saved/custom definition.
    // Existing graphs keep their IDs; newly inserted examples use the new model.
    const existing = Object.values(templatesStore.getState().templates).find((template) => template.standardKey === key
        && (key !== 'attention' || template.standardVersion === 2));
    if (existing) return existing.id;
    let graph, freeParameters;
    if (key === 'linear') {
        graph = buildLinearComponentGraph();
        freeParameters = [spec('in_features', 'int', 768, 'Input features'), spec('out_features', 'int', 768, 'Output features'), spec('bias', 'bool', true, 'Learned additive bias')];
    } else if (key === 'embedding') {
        graph = buildEmbeddingComponentGraph();
        freeParameters = [spec('num_embeddings', 'int', 1024, 'Table rows'), spec('embedding_dim', 'int', 768, 'Features per row'), { ...spec('padding_idx', 'int', null, 'Optional padding row with zero gradient'), nullable: true }];
    } else if (key === 'layernorm') {
        graph = buildLayerNormComponentGraph();
        freeParameters = [spec('normalized_shape', 'int', 768, 'Last-axis feature count'), spec('eps', 'float', 1e-5, 'Numerical stability'), spec('elementwise_affine', 'bool', true, 'Learned scale and bias')];
    } else if (key === 'learned-position') {
        graph = buildLearnedPositionComponentGraph(ensureStandardTemplate('embedding', templatesStore));
        freeParameters = [spec('max_len', 'int', 1024, 'Position table rows'), spec('embedding_dim', 'int', 768, 'Features per position')];
    } else if (key === 'sinusoidal-position') {
        graph = buildSinusoidalPositionComponentGraph();
        freeParameters = [spec('D', 'int', 768, 'Output feature count')];
    } else if (key === 'attention') {
        const linearTemplateId = ensureStandardTemplate('linear', templatesStore);
        let headTemplateId = Object.values(templatesStore.getState().templates).find((template) => template.standardKey === 'attention-head' && template.standardVersion === 2)?.id;
        if (!headTemplateId) headTemplateId = templatesStore.getState().createTemplate({
            name: 'Attention head', standardKey: 'attention-head', standardVersion: 2, graph: buildAttentionHeadGraph(),
            description: 'Scaled causal dot-product attention from editable operations.',
            freeParameters: attentionHeadParameters(),
        });
        graph = buildMultiHeadAttentionGraph({ headTemplateId, linearTemplateId });
        freeParameters = attentionParameters();
    }
    return templatesStore.getState().createTemplate({ name: component.key === 'layernorm' ? 'LayerNorm' : component.key === 'attention' ? 'Multi-head attention' : component.label,
        description: component.description, graph, freeParameters, standardKey: key, ...(key === 'attention' ? { standardVersion: 2 } : {}) });
}

const unsupported = (parameters, defaults) => Object.entries(defaults).some(([key, value]) => parameters[key] !== undefined && parameters[key] !== value);
/** Converting is explicit and supported only when the component preserves the opaque module's options. */
export function getStandardComponentForLayer(layerNode) {
    if (layerNode?.kind !== 'layer') return null;
    const component = STANDARD_COMPONENTS.find((item) => item.libraryKeys.includes(layerNode.libraryKey));
    if (!component) return null;
    const p = layerNode.parameters ?? {};
    if (component.key === 'embedding' && unsupported(p, { max_norm: null, norm_type: 2, scale_grad_by_freq: false, sparse: false, _freeze: false })) return null;
    if (component.key === 'layernorm') {
        const shape = p.normalized_shape;
        if ((Array.isArray(shape) && shape.length !== 1) || (typeof shape === 'string' && shape.includes(',') && shape.replace(/[()[\]\s]/g, '').split(',').filter(Boolean).length !== 1)) return null;
        if (p.bias === false) return null;
    }
    return component;
}

/** Bind legacy layer parameters without changing any graph identity or wiring. */
export function componentBindingsForLayer(layerNode) {
    const component = getStandardComponentForLayer(layerNode);
    if (!component) return null;
    const p = { ...Object.fromEntries((getEntry(layerNode.libraryKey)?.parameters ?? []).map((item) => [item.name, item.default ?? null])), ...layerNode.parameters };
    if (component.key === 'attention') return { D: p.embed_dim, H: p.num_heads, dropout: p.dropout, bias: p.bias };
    if (component.key === 'layernorm') {
        if (Array.isArray(p.normalized_shape)) p.normalized_shape = p.normalized_shape[0];
        else if (typeof p.normalized_shape === 'string' && /[()[\],]/.test(p.normalized_shape)) p.normalized_shape = p.normalized_shape.replace(/[()[\],\s]/g, '');
    }
    const names = component.key === 'linear' ? ['in_features', 'out_features', 'bias']
        : component.key === 'embedding' ? ['num_embeddings', 'embedding_dim', 'padding_idx']
            : component.key === 'layernorm' ? ['normalized_shape', 'eps', 'elementwise_affine'] : ['max_len', 'embedding_dim'];
    return Object.fromEntries(names.map((name) => [name, p[name]]));
}
