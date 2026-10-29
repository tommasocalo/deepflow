import { getEntry } from '../library/index.js';
import { isShapeCollection } from '../shapeCollections.js';
import { kernelVolume, layerSize, resolvedLayerParameters } from '../optimization/parameterCount.js';

export { layerSize };

/** fp32: every activation element takes four bytes. */
export const ACTIVATION_BYTES = 4;

const BOUNDARY_KEYS = new Set(['deepflow.io.Input', 'deepflow.io.Output', 'deepflow.template.Input', 'deepflow.template.Output']);
/** Views share storage with their input in eager PyTorch. */
const VIEW_KEYS = new Set(['deepflow.op.Split', 'deepflow.op.Transpose', 'deepflow.op.Unsqueeze', 'deepflow.op.Slice', 'deepflow.op.Broadcast', 'torch.nn.Flatten']);
/** Sources whose output is a weight or buffer, never a per-sample activation. */
export const CONSTANT_SOURCE_KEYS = new Set(['deepflow.op.Parameter', 'deepflow.op.Arange']);
/** Opaque modules that compute internally; v1 counts neither their MACs nor their weights. */
const HIDDEN_COMPUTE = /^(torch\.nn\.(LSTM|GRU|RNN|MultiheadAttention|TransformerEncoderLayer|TransformerDecoderLayer|Transformer)|deepflow\.nn\.CausalSelfAttention)$/;
/** Convolutions without a shape rule: validation passes their input shape through. */
const NO_SHAPE_RULE = /^torch\.nn\.(Conv1d|Conv3d|ConvTranspose1d|ConvTranspose3d)$/;

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

/** Contracted dimension of torch.matmul(a, b). */
function matmulInner(a, b) {
    const last = Array.isArray(a) && a.length ? a.at(-1) : null;
    if (positive(last)) return last;
    if (Array.isArray(b) && b.length >= 2 && positive(b.at(-2))) return b.at(-2);
    if (Array.isArray(b) && b.length === 1 && positive(b[0])) return b[0];
    return null;
}

/**
 * Multiply-accumulates of one forward pass at B = 1. Only matrix products and
 * convolutions count; lookups, element-wise ops, norms and pooling are 0.
 * Causal attention counts the full T × T score product. Null means unknown.
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
    if (NO_SHAPE_RULE.test(key) || HIDDEN_COMPUTE.test(key)) return null;
    return 0;
}

/** Whether a node's output is a separate tensor in eager PyTorch. */
export function producesActivation(libraryKey) {
    return !BOUNDARY_KEYS.has(libraryKey) && !VIEW_KEYS.has(libraryKey);
}

/** Why a layer metric is unknown, for tooltips. */
export function unresolvedReason(node, metric) {
    const key = node?.libraryKey;
    const name = getEntry(key)?.name ?? key ?? 'This node';
    if (metric === 'params') {
        if (HIDDEN_COMPUTE.test(key) || (/^(torch|deepflow)\.nn\./.test(key ?? '') && !getEntry(key))) return `${name} has internal weights the table does not count yet`;
        return 'a size setting is unresolved';
    }
    if (metric === 'macs') {
        if (NO_SHAPE_RULE.test(key)) return `no shape rule for ${name}`;
        if (HIDDEN_COMPUTE.test(key)) return `${name} computes internally; not counted yet`;
    }
    return 'the output shape is unknown';
}
