/**
 * PyTorch block library for DeepFlow v2.
 *
 * Each entry is a {@link import('../architecture/types.js').LibraryEntry}.
 * Categories drive the palette grouping. Defaults are surfaced in the UI; an
 * explicit `null` means "use the PyTorch default at codegen time."
 */

/** @type {import('../architecture/types.js').LibraryEntry[]} */
export const PYTORCH_BLOCKS = [
    {
        key: 'deepflow.io.Input',
        name: 'Input',
        category: 'IO',
        description: 'Entry point of the architecture. Bound to a Data graph terminus.',
        pyTorchFn: null,
        parameters: [
            { name: 'shape', type: 'string', description: 'Tensor shape, e.g. (B, C, H, W) — leave blank to infer from the data graph.', default: null },
            { name: 'dtype', type: 'enum', description: 'Tensor dtype', default: 'float32', options: ['float32', 'float16', 'bfloat16', 'int64', 'int32'] },
        ],
    },
    {
        key: 'deepflow.io.Output',
        name: 'Output',
        category: 'IO',
        description: 'Terminal node of the architecture. Connects to loss nodes in the training graph.',
        pyTorchFn: null,
        parameters: [
            { name: 'name', type: 'string', description: 'Output name (used in training graph bindings)', default: 'logits' },
        ],
    },
    {
        key: 'deepflow.template.Input',
        name: 'Template In',
        category: 'IO',
        description: 'Boundary block inside a supernode template — represents the data flowing in from the parent graph.',
        pyTorchFn: null,
        parameters: [],
    },
    {
        key: 'deepflow.template.Output',
        name: 'Template Out',
        category: 'IO',
        description: 'Boundary block inside a supernode template — its input becomes the supernode\'s output in the parent graph.',
        pyTorchFn: null,
        parameters: [],
    },

    {
        key: 'torch.nn.Conv1d',
        name: 'Conv1d',
        category: 'Convolution',
        description: '1D convolution over an input signal of several channels.',
        pyTorchFn: 'torch.nn.Conv1d',
        parameters: [
            { name: 'in_channels', type: 'int', description: 'Input channels', required: true },
            { name: 'out_channels', type: 'int', description: 'Output channels', required: true },
            { name: 'kernel_size', type: 'int', description: 'Kernel size', default: 3 },
            { name: 'stride', type: 'int', description: 'Stride', default: 1 },
            { name: 'padding', type: 'int', description: 'Zero-padding on both sides', default: 0 },
            { name: 'dilation', type: 'int', description: 'Spacing between kernel elements', default: 1 },
            { name: 'bias', type: 'bool', description: 'If False, no additive bias is learned', default: true },
        ],
    },
    {
        key: 'torch.nn.Conv2d',
        name: 'Conv2d',
        category: 'Convolution',
        description: '2D convolution over an input signal of several channels.',
        pyTorchFn: 'torch.nn.Conv2d',
        parameters: [
            { name: 'in_channels', type: 'int', description: 'Input channels', required: true },
            { name: 'out_channels', type: 'int', description: 'Output channels', required: true },
            { name: 'kernel_size', type: 'tuple', description: 'Kernel size (int or 2-tuple)', default: 3 },
            { name: 'stride', type: 'tuple', description: 'Stride', default: 1 },
            { name: 'padding', type: 'tuple', description: 'Zero-padding', default: 0 },
            { name: 'dilation', type: 'tuple', description: 'Spacing between kernel elements', default: 1 },
            { name: 'groups', type: 'int', description: 'Connections from input to output channels', default: 1 },
            { name: 'bias', type: 'bool', description: 'Additive bias', default: true },
        ],
    },
    {
        key: 'torch.nn.ConvTranspose2d',
        name: 'ConvTranspose2d',
        category: 'Convolution',
        description: 'Transposed 2D convolution — the operator used in GAN generators and segmentation upsamplers.',
        pyTorchFn: 'torch.nn.ConvTranspose2d',
        parameters: [
            { name: 'in_channels', type: 'int', description: 'Input channels', required: true },
            { name: 'out_channels', type: 'int', description: 'Output channels', required: true },
            { name: 'kernel_size', type: 'tuple', description: 'Kernel size', default: 3 },
            { name: 'stride', type: 'tuple', description: 'Stride', default: 1 },
            { name: 'padding', type: 'tuple', description: 'Padding', default: 0 },
            { name: 'output_padding', type: 'tuple', description: 'Output padding', default: 0 },
            { name: 'bias', type: 'bool', description: 'Additive bias', default: true },
        ],
    },

    {
        key: 'torch.nn.MaxPool1d',
        name: 'MaxPool1d',
        category: 'Pooling',
        description: '1D max pooling.',
        pyTorchFn: 'torch.nn.MaxPool1d',
        parameters: [
            { name: 'kernel_size', type: 'int', description: 'Window size', required: true },
            { name: 'stride', type: 'int', description: 'Stride', default: null },
            { name: 'padding', type: 'int', description: 'Zero-padding', default: 0 },
            { name: 'ceil_mode', type: 'bool', description: 'Use ceil instead of floor for output shape', default: false },
        ],
    },
    {
        key: 'torch.nn.MaxPool2d',
        name: 'MaxPool2d',
        category: 'Pooling',
        description: '2D max pooling.',
        pyTorchFn: 'torch.nn.MaxPool2d',
        parameters: [
            { name: 'kernel_size', type: 'tuple', description: 'Window size', required: true },
            { name: 'stride', type: 'tuple', description: 'Stride', default: null },
            { name: 'padding', type: 'tuple', description: 'Padding', default: 0 },
            { name: 'ceil_mode', type: 'bool', description: 'Ceil mode', default: false },
        ],
    },
    {
        key: 'torch.nn.AvgPool2d',
        name: 'AvgPool2d',
        category: 'Pooling',
        description: '2D average pooling.',
        pyTorchFn: 'torch.nn.AvgPool2d',
        parameters: [
            { name: 'kernel_size', type: 'tuple', description: 'Window size', required: true },
            { name: 'stride', type: 'tuple', description: 'Stride', default: null },
            { name: 'padding', type: 'tuple', description: 'Padding', default: 0 },
            { name: 'ceil_mode', type: 'bool', description: 'Ceil mode', default: false },
            { name: 'count_include_pad', type: 'bool', description: 'Include padding in the average', default: true },
        ],
    },
    {
        key: 'torch.nn.AdaptiveAvgPool2d',
        name: 'AdaptiveAvgPool2d',
        category: 'Pooling',
        description: '2D adaptive average pooling — collapses spatial dims to a target size.',
        pyTorchFn: 'torch.nn.AdaptiveAvgPool2d',
        parameters: [
            { name: 'output_size', type: 'tuple', description: 'Target spatial size', default: 1 },
        ],
    },

    {
        key: 'torch.nn.Linear',
        name: 'Linear',
        category: 'Linear',
        description: 'Affine transform: y = xW^T + b.',
        pyTorchFn: 'torch.nn.Linear',
        parameters: [
            { name: 'in_features', type: 'int', description: 'Input dimensionality', required: true },
            { name: 'out_features', type: 'int', description: 'Output dimensionality', required: true },
            { name: 'bias', type: 'bool', description: 'Additive bias', default: true },
        ],
    },
    {
        key: 'torch.nn.Embedding',
        name: 'Embedding',
        category: 'Linear',
        description: 'Lookup table mapping indices to dense vectors.',
        pyTorchFn: 'torch.nn.Embedding',
        parameters: [
            { name: 'num_embeddings', type: 'int', description: 'Vocab size', required: true },
            { name: 'embedding_dim', type: 'int', description: 'Embedding dimension', required: true },
            { name: 'padding_idx', type: 'int', description: 'Index whose gradient is zero', default: null },
        ],
    },
    {
        key: 'torch.nn.Flatten',
        name: 'Flatten',
        category: 'Reshape',
        description: 'Flattens a contiguous range of dimensions into a single one.',
        pyTorchFn: 'torch.nn.Flatten',
        parameters: [
            { name: 'start_dim', type: 'int', description: 'First dim to flatten', default: 1 },
            { name: 'end_dim', type: 'int', description: 'Last dim to flatten (-1 = last)', default: -1 },
        ],
    },

    {
        key: 'torch.nn.ReLU',
        name: 'ReLU',
        category: 'Activation',
        description: 'max(0, x).',
        pyTorchFn: 'torch.nn.ReLU',
        parameters: [{ name: 'inplace', type: 'bool', description: 'Mutate input', default: false }],
    },
    {
        key: 'torch.nn.LeakyReLU',
        name: 'LeakyReLU',
        category: 'Activation',
        description: 'Allows a small negative slope for x < 0.',
        pyTorchFn: 'torch.nn.LeakyReLU',
        parameters: [
            { name: 'negative_slope', type: 'float', description: 'Slope for negatives', default: 0.01 },
            { name: 'inplace', type: 'bool', description: 'Mutate input', default: false },
        ],
    },
    {
        key: 'torch.nn.GELU',
        name: 'GELU',
        category: 'Activation',
        description: 'Gaussian Error Linear Unit — the activation used in most modern Transformers.',
        pyTorchFn: 'torch.nn.GELU',
        parameters: [
            { name: 'approximate', type: 'enum', description: 'Approximation strategy', default: 'none', options: ['none', 'tanh'] },
        ],
    },
    {
        key: 'torch.nn.Sigmoid',
        name: 'Sigmoid',
        category: 'Activation',
        description: 'Squashes input to (0, 1).',
        pyTorchFn: 'torch.nn.Sigmoid',
        parameters: [],
    },
    {
        key: 'torch.nn.Tanh',
        name: 'Tanh',
        category: 'Activation',
        description: 'Squashes input to (-1, 1).',
        pyTorchFn: 'torch.nn.Tanh',
        parameters: [],
    },
    {
        key: 'torch.nn.Softmax',
        name: 'Softmax',
        category: 'Activation',
        description: 'Probability distribution over a dimension.',
        pyTorchFn: 'torch.nn.Softmax',
        parameters: [
            { name: 'dim', type: 'int', description: 'Dimension to softmax over', default: -1 },
        ],
    },

    {
        key: 'torch.nn.BatchNorm1d',
        name: 'BatchNorm1d',
        category: 'Normalization',
        description: 'Batch norm over a 2D/3D input.',
        pyTorchFn: 'torch.nn.BatchNorm1d',
        parameters: [
            { name: 'num_features', type: 'int', description: 'Channels', required: true },
            { name: 'eps', type: 'float', description: 'Numerical stability', default: 1e-5 },
            { name: 'momentum', type: 'float', description: 'Running stat momentum', default: 0.1 },
            { name: 'affine', type: 'bool', description: 'Learnable affine params', default: true },
        ],
    },
    {
        key: 'torch.nn.BatchNorm2d',
        name: 'BatchNorm2d',
        category: 'Normalization',
        description: 'Batch norm over a 4D input.',
        pyTorchFn: 'torch.nn.BatchNorm2d',
        parameters: [
            { name: 'num_features', type: 'int', description: 'Channels', required: true },
            { name: 'eps', type: 'float', description: 'Numerical stability', default: 1e-5 },
            { name: 'momentum', type: 'float', description: 'Running stat momentum', default: 0.1 },
            { name: 'affine', type: 'bool', description: 'Learnable affine params', default: true },
        ],
    },
    {
        key: 'torch.nn.LayerNorm',
        name: 'LayerNorm',
        category: 'Normalization',
        description: 'Per-sample layer normalization.',
        pyTorchFn: 'torch.nn.LayerNorm',
        parameters: [
            { name: 'normalized_shape', type: 'tuple', description: 'Shape to normalize over', required: true },
            { name: 'eps', type: 'float', description: 'Numerical stability', default: 1e-5 },
            { name: 'elementwise_affine', type: 'bool', description: 'Learnable affine params', default: true },
        ],
    },
    {
        key: 'torch.nn.GroupNorm',
        name: 'GroupNorm',
        category: 'Normalization',
        description: 'Splits channels into groups and normalizes within each.',
        pyTorchFn: 'torch.nn.GroupNorm',
        parameters: [
            { name: 'num_groups', type: 'int', description: 'Groups', required: true },
            { name: 'num_channels', type: 'int', description: 'Channels', required: true },
            { name: 'eps', type: 'float', description: 'Numerical stability', default: 1e-5 },
            { name: 'affine', type: 'bool', description: 'Learnable affine params', default: true },
        ],
    },

    {
        key: 'torch.nn.Dropout',
        name: 'Dropout',
        category: 'Regularization',
        description: 'Zeros elements with probability p during training.',
        pyTorchFn: 'torch.nn.Dropout',
        parameters: [
            { name: 'p', type: 'float', description: 'Drop probability', default: 0.5 },
            { name: 'inplace', type: 'bool', description: 'Mutate input', default: false },
        ],
    },

    {
        key: 'torch.nn.MultiheadAttention',
        name: 'MultiheadAttention',
        category: 'Attention',
        description: 'Scaled dot-product multi-head attention over Q, K, V.',
        pyTorchFn: 'torch.nn.MultiheadAttention',
        parameters: [
            { name: 'embed_dim', type: 'int', description: 'Total model dim', required: true },
            { name: 'num_heads', type: 'int', description: 'Attention heads (must divide embed_dim)', required: true },
            { name: 'dropout', type: 'float', description: 'Attention dropout', default: 0.0 },
            { name: 'batch_first', type: 'bool', description: 'Inputs as (B, T, D)', default: true },
        ],
    },
    {
        key: 'torch.nn.TransformerEncoderLayer',
        name: 'TransformerEncoderLayer',
        category: 'Attention',
        description: 'One Transformer encoder block: MHA + FFN with residual connections.',
        pyTorchFn: 'torch.nn.TransformerEncoderLayer',
        parameters: [
            { name: 'd_model', type: 'int', description: 'Model dim', required: true },
            { name: 'nhead', type: 'int', description: 'Attention heads', required: true },
            { name: 'dim_feedforward', type: 'int', description: 'FFN hidden dim', default: 2048 },
            { name: 'dropout', type: 'float', description: 'Dropout', default: 0.1 },
            { name: 'activation', type: 'enum', description: 'FFN activation', default: 'relu', options: ['relu', 'gelu'] },
            { name: 'batch_first', type: 'bool', description: 'Batch-first inputs', default: true },
            { name: 'norm_first', type: 'bool', description: 'Pre-norm vs. post-norm', default: false },
        ],
    },

    {
        key: 'torch.nn.LSTM',
        name: 'LSTM',
        category: 'Recurrent',
        description: 'Long Short-Term Memory recurrent cell.',
        pyTorchFn: 'torch.nn.LSTM',
        parameters: [
            { name: 'input_size', type: 'int', description: 'Input feature dim', required: true },
            { name: 'hidden_size', type: 'int', description: 'Hidden state dim', required: true },
            { name: 'num_layers', type: 'int', description: 'Stacked layers', default: 1 },
            { name: 'dropout', type: 'float', description: 'Dropout between layers', default: 0.0 },
            { name: 'bidirectional', type: 'bool', description: 'BiLSTM', default: false },
            { name: 'batch_first', type: 'bool', description: 'Batch-first inputs', default: true },
        ],
    },
    {
        key: 'torch.nn.GRU',
        name: 'GRU',
        category: 'Recurrent',
        description: 'Gated Recurrent Unit.',
        pyTorchFn: 'torch.nn.GRU',
        parameters: [
            { name: 'input_size', type: 'int', description: 'Input feature dim', required: true },
            { name: 'hidden_size', type: 'int', description: 'Hidden state dim', required: true },
            { name: 'num_layers', type: 'int', description: 'Stacked layers', default: 1 },
            { name: 'dropout', type: 'float', description: 'Dropout between layers', default: 0.0 },
            { name: 'batch_first', type: 'bool', description: 'Batch-first inputs', default: true },
        ],
    },

    {
        key: 'deepflow.op.Add',
        name: 'Add',
        category: 'Element-wise',
        description: 'Element-wise sum of incoming tensors. Used for residual connections.',
        pyTorchFn: 'torch.add',
        parameters: [],
    },
    {
        key: 'deepflow.op.Multiply',
        name: 'Multiply',
        category: 'Element-wise',
        description: 'Element-wise product of incoming tensors.',
        pyTorchFn: 'torch.mul',
        parameters: [],
    },
    {
        key: 'deepflow.op.Concat',
        name: 'Concat',
        category: 'Combine',
        description: 'Concatenate incoming tensors along a dimension.',
        pyTorchFn: 'torch.cat',
        parameters: [{ name: 'dim', type: 'int', description: 'Concat dimension', default: 1 }],
    },
    {
        key: 'deepflow.op.Split',
        name: 'Split',
        category: 'Combine',
        description: 'Fan-out: routes input to multiple downstream nodes.',
        pyTorchFn: null,
        parameters: [],
    },
];

const BY_KEY = Object.fromEntries(PYTORCH_BLOCKS.map((b) => [b.key, b]));

/** @param {string} key @returns {import('../architecture/types.js').LibraryEntry | undefined} */
export function getEntry(key) {
    return BY_KEY[key];
}

/** @returns {string[]} */
export function listCategories() {
    const seen = new Set();
    const ordered = [];
    for (const b of PYTORCH_BLOCKS) {
        if (!seen.has(b.category)) {
            seen.add(b.category);
            ordered.push(b.category);
        }
    }
    return ordered;
}

/** @param {string} category */
export function entriesInCategory(category) {
    return PYTORCH_BLOCKS.filter((b) => b.category === category);
}

/**
 * @param {import('../architecture/types.js').LibraryEntry | undefined} entry
 * @returns {Object<string, *>}
 */
export function defaultParameterValues(entry) {
    if (!entry) return {};
    const out = {};
    for (const p of entry.parameters) {
        out[p.name] = p.default ?? null;
    }
    return out;
}
