import { getEntry } from '../library/index.js';
import { isShapeCollection } from '../shapeCollections.js';
import { kernelVolume, layerSize, RECURRENT_GATES, resolvedLayerParameters } from '../optimization/parameterCount.js';

export { layerSize };

/** fp32: every activation element takes four bytes. */
export const ACTIVATION_BYTES = 4;

const BOUNDARY_KEYS = new Set(['deepflow.io.Input', 'deepflow.io.Output', 'deepflow.template.Input', 'deepflow.template.Output']);
/** Views share storage with their input in eager PyTorch. */
const VIEW_KEYS = new Set(['deepflow.op.Split', 'deepflow.op.Transpose', 'deepflow.op.Unsqueeze', 'deepflow.op.Slice', 'deepflow.op.Broadcast', 'torch.nn.Flatten']);
/** Sources whose output is a weight or buffer, never a per-sample activation. */
export const CONSTANT_SOURCE_KEYS = new Set(['deepflow.op.Parameter', 'deepflow.op.Arange']);
/** Self-attention modules over one (B, T, E) stream; causal masking still computes T × T scores. */
const SELF_ATTENTION = new Set(['torch.nn.MultiheadAttention', 'deepflow.nn.CausalSelfAttention', 'torch.nn.TransformerEncoderLayer']);
/** Opaque modules keep internal activations (gates, scores) the table cannot see. */
const OPAQUE = /^(torch\.nn\.(LSTM|GRU|RNN|MultiheadAttention|TransformerEncoderLayer|TransformerDecoderLayer|Transformer)|deepflow\.nn\.CausalSelfAttention)$/;
/** Modules whose compute is not modelled at all. */
const UNMODELLED = /^torch\.nn\.(TransformerDecoderLayer|Transformer)$/;
/** Convolutions without a shape rule: validation passes their input shape through. */
const NO_SHAPE_RULE = /^torch\.nn\.(Conv1d|Conv3d|ConvTranspose1d|ConvTranspose3d)$/;
/** Any layer whose output shape is that placeholder rather than an inferred shape. */
const PLACEHOLDER_SHAPE = /^torch\.nn\.(Conv1d|Conv3d|ConvTranspose1d|ConvTranspose3d|MaxPool1d|AvgPool1d|LSTM|GRU|RNN)$/;
/** Layers whose multiply-accumulates the table counts. */
const MAC_LAYERS = /^(torch\.nn\.(Linear|Conv[123]d|ConvTranspose[123]d|LSTM|GRU|RNN|MultiheadAttention|TransformerEncoderLayer|TransformerDecoderLayer|Transformer)|deepflow\.op\.MatMul|deepflow\.nn\.CausalSelfAttention)$/;

const positive = (value) => Number.isSafeInteger(value) && value > 0 ? value : null;

/**
 * Elements of one sample (B = 1). A leading unknown axis is the batch; any
 * other unknown axis leaves the size unresolved. Collections add their items.
 */
export function numelAtBatchOne(shape) {
    if (isShapeCollection(shape)) {
        let total = 0;
        for (const item of shape.items) {
            const count = numelAtBatchOne(item);
            if (count === null) return null;
            total += count;
        }
        return total;
    }
    if (!Array.isArray(shape)) return null;
    const dims = shape[0] === null ? shape.slice(1) : shape;
    if (!dims.every((axis) => Number.isSafeInteger(axis) && axis >= 0)) return null;
    return dims.reduce((product, axis) => product * axis, 1);
}

/** Modules whose MACs come from a closed form rather than one product: 'attention', 'recurrent' or null. */
export function moduleMacsKind(libraryKey) {
    if (SELF_ATTENTION.has(libraryKey)) return 'attention';
    return RECURRENT_GATES[libraryKey] ? 'recurrent' : null;
}

/** Whether a layer's multiply-accumulates can be nonzero; every other layer counts 0 whatever its shapes. */
export function countsMacs(libraryKey) {
    return MAC_LAYERS.test(libraryKey ?? '');
}

/** Whether validation only passes the input shape through, so the output size is a placeholder. */
export function hasPlaceholderShape(libraryKey) {
    return PLACEHOLDER_SHAPE.test(libraryKey ?? '');
}

/** Contracted dimension of torch.matmul(a, b). */
function matmulInner(a, b) {
    const last = Array.isArray(a) && a.length ? a.at(-1) : null;
    if (positive(last)) return last;
    if (Array.isArray(b) && b.length >= 2 && positive(b.at(-2))) return b.at(-2);
    if (Array.isArray(b) && b.length === 1 && positive(b[0])) return b[0];
    return null;
}

/** Sequence length of a batch-first (B, T, E) input, or null. */
function sequenceLength(shape) {
    return Array.isArray(shape) && shape.length === 3 && shape[0] === null ? positive(shape[1]) : null;
}

/**
 * Multiply-accumulates of one forward pass at B = 1. Only matrix products and
 * convolutions count, including those inside attention and recurrent modules;
 * lookups, element-wise ops, norms and pooling are 0. Causal attention counts
 * the full T × T score product. Null means unknown.
 */
export function layerMacs(node, scope, inputShapes = [], output = null) {
    const key = node.libraryKey;
    if (key === 'torch.nn.Linear') {
        const outputs = numelAtBatchOne(output);
        const inFeatures = positive(resolvedLayerParameters(node, scope).in_features);
        return outputs === null || !inFeatures ? null : outputs * inFeatures;
    }
    if (key === 'deepflow.op.MatMul') {
        const outputs = numelAtBatchOne(output);
        const inner = matmulInner(inputShapes[0], inputShapes[1]);
        return outputs === null || inner === null ? null : outputs * inner;
    }
    if (key === 'torch.nn.Conv2d' || key === 'torch.nn.ConvTranspose2d') {
        const p = resolvedLayerParameters(node, scope);
        const kernel = kernelVolume(key, node.parameters?.kernel_size, scope);
        const groups = positive(p.groups) ?? 1;
        if (key === 'torch.nn.Conv2d') {
            // Each output element sums (in / groups) × kernel products.
            const outputs = numelAtBatchOne(output);
            const channels = positive(p.in_channels);
            return outputs === null || !kernel || !channels || channels % groups ? null : outputs * (channels / groups) * kernel;
        }
        // Each input element scatters into (out / groups) × kernel outputs.
        const inputs = numelAtBatchOne(inputShapes[0]);
        const channels = positive(p.out_channels);
        return inputs === null || !kernel || !channels || channels % groups ? null : inputs * (channels / groups) * kernel;
    }
    if (SELF_ATTENTION.has(key)) {
        const p = resolvedLayerParameters(node, scope);
        if (p.batch_first === false) return null;
        const tokens = sequenceLength(inputShapes[0]);
        const embed = positive(key === 'torch.nn.TransformerEncoderLayer' ? p.d_model : p.embed_dim);
        if (!tokens || !embed) return null;
        // Q, K, V and output projections, then Q × Kᵀ and weights × V over all heads.
        const attention = 4 * tokens * embed * embed + 2 * tokens * tokens * embed;
        if (key !== 'torch.nn.TransformerEncoderLayer') return attention;
        const feedForward = positive(p.dim_feedforward);
        return feedForward ? attention + 2 * tokens * embed * feedForward : null;
    }
    if (RECURRENT_GATES[key]) {
        const p = resolvedLayerParameters(node, scope);
        const tokens = p.batch_first === false ? null : sequenceLength(inputShapes[0]);
        const hidden = positive(p.hidden_size);
        const input = positive(p.input_size);
        const layers = positive(p.num_layers) ?? 1;
        if (!tokens || !hidden || !input) return null;
        const directions = p.bidirectional === true ? 2 : 1;
        let perStep = 0;
        for (let index = 0; index < layers; index++) {
            perStep += directions * RECURRENT_GATES[key] * hidden * ((index === 0 ? input : hidden * directions) + hidden);
        }
        return tokens * perStep;
    }
    if (NO_SHAPE_RULE.test(key) || UNMODELLED.test(key)) return null;
    return 0;
}

/** Whether a node's output is a separate tensor in eager PyTorch. */
export function producesActivation(libraryKey) {
    return !BOUNDARY_KEYS.has(libraryKey) && !VIEW_KEYS.has(libraryKey);
}

/**
 * Whether the output size is known: placeholder shapes would give a wrong
 * size, and opaque modules keep internal activations the table cannot see.
 */
export function activationKnown(libraryKey) {
    return !PLACEHOLDER_SHAPE.test(libraryKey) && !OPAQUE.test(libraryKey);
}

/** Why a layer metric is unknown when its inputs are fine, for tooltips. */
export function unresolvedReason(node, metric) {
    const key = node?.libraryKey;
    const name = getEntry(key)?.name ?? key ?? 'This node';
    if (metric === 'params') {
        if (UNMODELLED.test(key ?? '') || (/^(torch|deepflow)\.nn\./.test(key ?? '') && !getEntry(key))) return `${name} has internal weights the table does not count yet`;
        return 'a size setting is unresolved';
    }
    if (metric === 'macs') {
        if (NO_SHAPE_RULE.test(key)) return `no shape rule for ${name}`;
        if (UNMODELLED.test(key)) return `${name} computes internally; not counted yet`;
        if (SELF_ATTENTION.has(key) || RECURRENT_GATES[key]) return `${name} needs a batch-first (B, T, E) input with known sizes`;
    }
    if (metric === 'activation') {
        if (OPAQUE.test(key)) return `${name} keeps internal activations the table does not count yet`;
        if (PLACEHOLDER_SHAPE.test(key)) return `no shape rule for ${name}, so its output size is unknown`;
    }
    return 'the output shape is unknown';
}

/** Why a placeholder output shape is shown as unknown. */
export function placeholderReason(node) {
    return `no shape rule for ${getEntry(node?.libraryKey)?.name ?? node?.libraryKey ?? 'this layer'}: its output size is unknown`;
}
