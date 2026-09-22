import { STANDARD_COMPONENTS } from '../../domain/components/index.js';
import { getPaletteEntries } from '../../domain/library/nodeFamilies.js';
import { PYTORCH_BLOCKS } from '../../domain/library/pytorchBlocks.js';

export const STANDARD_ENTRIES = STANDARD_COMPONENTS.map((entry) => ({
    ...entry,
    key: `component:${entry.key}`,
    name: entry.label,
    standardComponentKey: entry.key,
    category: 'Component',
}));
const convertedFamilies = new Set(['linear', 'attention', 'normalization']);
export const MODULE_ENTRIES = [
    ...getPaletteEntries().filter((entry) => !['IO', 'Operations'].includes(entry.category) && !convertedFamilies.has(entry.family?.key)),
    ...PYTORCH_BLOCKS.filter((entry) => entry.category === 'Normalization' && entry.key !== 'torch.nn.LayerNorm'),
].map((entry) => entry.key === 'torch.nn.TransformerEncoderLayer' ? { ...entry, name: 'Transformer', aliases: [entry.name] } : entry);
export const PINNABLE_ENTRIES = [...STANDARD_ENTRIES, ...MODULE_ENTRIES.filter((entry) => entry.family || entry.key === 'torch.nn.Dropout')];
export const DEFAULT_PINS = ['component:linear', 'component:embedding', 'component:attention', 'component:layernorm'];
export const OPERATION_ENTRIES = [
    ...getPaletteEntries().filter((entry) => ['IO', 'Operations'].includes(entry.category)),
    ...PYTORCH_BLOCKS.filter((entry) => entry.key === 'torch.nn.Softmax').map((entry) => ({ ...entry, category: 'Operations' })),
];
export const normalize = (value) => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
export const matchesEntry = (entry, query) => query.trim().split(/\s+/).every((word) => normalize(`${entry.name} ${entry.category} ${entry.description} ${entry.key} ${(entry.aliases ?? []).join(' ')}`).includes(normalize(word)));

export function setInsertionDragData(event, entry, template = false) {
    const type = entry.standardComponentKey ? 'application/deepflow-component-key'
        : template ? 'application/deepflow-template-id' : 'application/deepflow-lib-key';
    event.dataTransfer.setData(type, entry.standardComponentKey ?? entry.key);
    event.dataTransfer.effectAllowed = 'copy';
}
