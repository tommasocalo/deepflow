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
    const parameters = defaultParameterValues(next);
    for (const spec of next.parameters) {
        const previous = current.parameters.find((candidate) => candidate.name === spec.name && candidate.type === spec.type);
        if (previous && Object.hasOwn(node.parameters ?? {}, spec.name)) parameters[spec.name] = node.parameters[spec.name];
    }
    // These names refer to the same output-vector width. Vocabulary size and
    // maximum sequence length deliberately remain independent requirements.
    const previousWidth = current.key === 'torch.nn.Linear' ? 'out_features' : 'embedding_dim';
    const nextWidth = next.key === 'torch.nn.Linear' ? 'out_features' : 'embedding_dim';
    if (getNodeFamily(current.key)?.key === 'linear' && getNodeFamily(next.key)?.key === 'linear'
        && Object.hasOwn(node.parameters ?? {}, previousWidth)) {
        parameters[nextWidth] = node.parameters[previousWidth];
    }
    return {
        libraryKey: next.key,
        label: node.label === current.name || node.label === current.key ? next.name : node.label,
        parameters,
    };
}
