import { PYTORCH_BLOCKS, defaultParameterValues, getEntry } from './pytorchBlocks.js';

// Families are a presentation choice: persisted nodes keep their precise
// library key, so validation, export and existing projects retain semantics.
export const NODE_FAMILIES = [
    {
        key: 'linear', name: 'Linear', defaultKey: 'torch.nn.Linear',
        variants: [
            { key: 'torch.nn.Linear', name: 'Linear' },
            { key: 'torch.nn.Embedding', name: 'Embedding' },
            { key: 'deepflow.nn.PositionalEmbedding', name: 'Positional embedding' },
        ],
    },
    {
        key: 'attention', name: 'Attention', defaultKey: 'deepflow.nn.CausalSelfAttention',
        variants: [
            { key: 'deepflow.nn.CausalSelfAttention', name: 'Causal multi-head' },
            { key: 'torch.nn.MultiheadAttention', name: 'Multi-head' },
        ],
    },
    {
        key: 'normalization', name: 'Normalization', defaultKey: 'torch.nn.LayerNorm',
        variants: [
            { key: 'torch.nn.LayerNorm', name: 'LayerNorm' },
            { key: 'torch.nn.BatchNorm1d', name: 'BatchNorm 1D' },
            { key: 'torch.nn.BatchNorm2d', name: 'BatchNorm 2D' },
            { key: 'torch.nn.GroupNorm', name: 'GroupNorm' },
        ],
    },
    {
        key: 'activation', name: 'Activation', defaultKey: 'torch.nn.ReLU',
        variants: [
            { key: 'torch.nn.ReLU', name: 'ReLU' },
            { key: 'torch.nn.LeakyReLU', name: 'Leaky ReLU' },
            { key: 'torch.nn.GELU', name: 'GELU' },
            { key: 'torch.nn.Sigmoid', name: 'Sigmoid' },
            { key: 'torch.nn.Tanh', name: 'Tanh' },
            { key: 'torch.nn.Softmax', name: 'Softmax' },
            { key: 'torch.nn.LogSoftmax', name: 'Log-softmax' },
        ],
    },
    {
        key: 'convolution', name: 'Convolution', defaultKey: 'torch.nn.Conv2d',
        variants: [
            { key: 'torch.nn.Conv2d', name: 'Conv 2D' },
            { key: 'torch.nn.Conv1d', name: 'Conv 1D' },
            { key: 'torch.nn.ConvTranspose2d', name: 'Transposed 2D' },
        ],
    },
    {
        key: 'pooling', name: 'Pooling', defaultKey: 'torch.nn.MaxPool2d',
        variants: [
            { key: 'torch.nn.MaxPool2d', name: 'Max Pool 2D' },
            { key: 'torch.nn.MaxPool1d', name: 'Max Pool 1D' },
            { key: 'torch.nn.AvgPool2d', name: 'Average 2D' },
            { key: 'torch.nn.AdaptiveAvgPool2d', name: 'Adaptive avg 2D' },
        ],
    },
    {
        key: 'recurrent', name: 'Recurrent', defaultKey: 'torch.nn.LSTM',
        variants: [
            { key: 'torch.nn.LSTM', name: 'LSTM' },
            { key: 'torch.nn.GRU', name: 'GRU' },
        ],
    },
];

export function getNodeFamily(libraryKey) {
    return NODE_FAMILIES.find((family) => family.variants.some((variant) => variant.key === libraryKey)) ?? null;
}

export function isOperationKey(libraryKey) {
    return libraryKey?.startsWith('deepflow.op.') || libraryKey === 'torch.nn.Flatten';
}

/** One palette entry per family, with variant names included in search. */
export function getPaletteEntries() {
    return PYTORCH_BLOCKS.flatMap((entry) => {
        if (entry.key.startsWith('deepflow.template.')) return [];
        const family = getNodeFamily(entry.key);
        if (family && entry.key !== family.defaultKey) return [];
        if (!family) return [{ ...entry, category: isOperationKey(entry.key) ? 'Operations' : entry.category }];
        return [{
            ...entry, name: family.name, family,
            description: `${family.variants.map((variant) => variant.name).join(' · ')}. Choose a variant in node settings.`,
            aliases: family.variants.flatMap((variant) => [variant.name, variant.key]),
        }];
    });
}

/** A patch for the existing graph action; never changes identity or edges. */
export function createVariantPatch(node, nextKey) {
    const current = getEntry(node?.libraryKey);
    const next = getEntry(nextKey);
    if (node?.kind !== 'layer' || !current || !next || nextKey === node.libraryKey || current.category !== next.category || current.category === 'IO') return null;
    const family = getNodeFamily(current.key);
    if (family?.key !== getNodeFamily(next.key)?.key) return null;
    const parameters = defaultParameterValues(next);
    for (const spec of next.parameters) {
        const previous = current.parameters.find((candidate) => candidate.name === spec.name);
        if (!previous || !Object.hasOwn(node.parameters ?? {}, spec.name)) continue;
        const value = node.parameters[spec.name];
        // Spatial tuple parameters also accept a scalar. Carry a 1D scalar
        // forward, but never reduce a 2D tuple to one arbitrarily chosen axis.
        const scalarInteger = Number.isInteger(value) || (typeof value === 'string' && value.trim() !== '' && Number.isInteger(Number(value)));
        if (previous.type === spec.type || (previous.type === 'int' && spec.type === 'tuple')
            || (previous.type === 'tuple' && spec.type === 'int' && scalarInteger)) parameters[spec.name] = value;
    }
    // These names refer to the same output-vector width. Vocabulary size and
    // maximum sequence length deliberately remain independent requirements.
    const previousWidth = current.key === 'torch.nn.Linear' ? 'out_features' : 'embedding_dim';
    const nextWidth = next.key === 'torch.nn.Linear' ? 'out_features' : 'embedding_dim';
    if (family?.key === 'linear'
        && Object.hasOwn(node.parameters ?? {}, previousWidth)) {
        parameters[nextWidth] = node.parameters[previousWidth];
    }
    // BatchNorm and GroupNorm name channel count differently; a LayerNorm
    // normalized shape may span several axes and must not be treated as one.
    if (family?.key === 'normalization') {
        const previousChannels = current.key === 'torch.nn.GroupNorm' ? 'num_channels' : 'num_features';
        const nextChannels = next.key === 'torch.nn.GroupNorm' ? 'num_channels' : 'num_features';
        if (Object.hasOwn(parameters, nextChannels) && Object.hasOwn(node.parameters ?? {}, previousChannels)) {
            parameters[nextChannels] = node.parameters[previousChannels];
        }
    }
    return {
        libraryKey: next.key,
        label: node.label === current.name || node.label === current.key ? next.name : node.label,
        parameters,
    };
}
