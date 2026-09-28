import { inspectGpt2Config, importGpt2Architecture } from './gpt2.js';

/** New architecture families register a config inspector and converter here. */
export const MODEL_ADAPTERS = Object.freeze([
    { id: 'gpt2', inspect: inspectGpt2Config, importArchitecture: importGpt2Architecture },
]);

export function inspectHubModel(preview) {
    const config = preview?.config;
    for (const adapter of MODEL_ADAPTERS) {
        const inspection = adapter.inspect(config);
        if (inspection) return { ...preview, adapterId: adapter.id, inspection };
    }
    return { ...preview, adapterId: null, inspection: {
        supported: false,
        family: config?.model_type || 'Unknown architecture',
        reason: `DeepFlow cannot yet convert ${config?.model_type || 'this model type'} into an editable graph.`,
    } };
}

export function importHuggingFaceArchitecture(preview, stores) {
    const inspected = inspectHubModel(preview);
    const adapter = MODEL_ADAPTERS.find((entry) => entry.id === inspected.adapterId);
    if (!adapter || !inspected.inspection.supported) throw new Error(inspected.inspection.reason);
    return adapter.importArchitecture(inspected, stores);
}
