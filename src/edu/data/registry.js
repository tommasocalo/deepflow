/**
 * Dataset registry (docs/edu/SPEC.md §6.9.1).
 *
 * A DatasetSpec has the fields of §6.9.1. Stage B presets add optional fields that the Data tab uses:
 *   featureNames: string[] | null       names of the input features (toy and tabular sets)
 *   preview: 'scatter' | 'tabular' | 'images' | 'none'
 *   loadSync(options) → RawDataset | null   synchronous load where the data is already in memory
 * Export-only presets (Stage C) may add `schematic`, one sentence shown instead of a data preview.
 */
import PRESETS from './presets/index.js';

export { dataSeedOf, datasetSize, defaultOptions, resolveOptions, resolveSplit, SPLIT_LIMITS } from './options.js';

/** In display order: blobs, circle, xor, spiral2, spiral3, plane, iris, digits, then the export presets. */
export const DATASETS = Object.freeze([...PRESETS]);

const BY_ID = new Map(DATASETS.map((spec) => [spec.id, spec]));

export function getDataset(id) {
    return BY_ID.get(id) ?? null;
}

export function browserDatasets() {
    return DATASETS.filter((spec) => spec.training === 'browser');
}

export function exportDatasets() {
    return DATASETS.filter((spec) => spec.training === 'export');
}

/** The shape after the batch axis, e.g. '(1, 8, 8)' → [1, 8, 8]. */
export function parseInputShape(shape) {
    if (typeof shape !== 'string') return null;
    const parts = shape.replace(/[()[\]]/g, '').split(',').map((part) => part.trim()).filter(Boolean);
    const dims = parts.map(Number);
    return dims.length && dims.every((dim) => Number.isSafeInteger(dim) && dim > 0) ? dims : null;
}

/** The read-only summary the bridge exposes: { id, name, kind, training, input, target }. */
export function datasetSummary(spec) {
    if (!spec) return null;
    return {
        id: spec.id,
        name: spec.name,
        kind: spec.kind,
        training: spec.training,
        input: { ...spec.input },
        target: { ...spec.target, ...(spec.target.classes ? { classes: [...spec.target.classes] } : {}) },
    };
}
