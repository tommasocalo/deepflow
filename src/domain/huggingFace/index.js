import { fetchHubModelConfig, fetchHubOnnxFile } from './client.js';
import { inspectHubOnnxPreview, importHubOnnxArchitecture } from './onnxBridge.js';
import { inspectGpt2Config, importGpt2Architecture } from './gpt2.js';
import { inspectBertConfig, importBertArchitecture } from './bert.js';

/** New architecture families register a config inspector and converter here. */
export const MODEL_ADAPTERS = Object.freeze([
    { id: 'gpt2', inspect: inspectGpt2Config, importArchitecture: importGpt2Architecture },
    { id: 'bert', inspect: inspectBertConfig, importArchitecture: importBertArchitecture },
]);

export function inspectHubModel(preview) {
    if (preview?.onnx?.operatorGraph) return inspectHubOnnxPreview(preview, preview.onnx.operatorGraph, preview.onnx.filePath);
    const config = preview?.config;
    for (const adapter of MODEL_ADAPTERS) {
        const inspection = adapter.inspect(config);
        if (inspection) return { ...preview, adapterId: adapter.id, inspection };
    }
    return { ...preview, adapterId: null, inspection: {
        supported: false,
        family: config?.model_type || 'Unknown architecture',
        reason: `No exact editable adapter is mapped for ${config?.model_type || 'this model type'} yet.`,
    } };
}

export function importHuggingFaceArchitecture(preview, stores) {
    const inspected = inspectHubModel(preview);
    if (inspected.adapterId === 'onnx') {
        if (!inspected.inspection.supported) throw new Error(inspected.inspection.reason);
        return importHubOnnxArchitecture(inspected, stores);
    }
    const adapter = MODEL_ADAPTERS.find((entry) => entry.id === inspected.adapterId);
    if (!adapter || !inspected.inspection.supported) throw new Error(inspected.inspection.reason);
    return adapter.importArchitecture(inspected, stores);
}

/** Prefer known component adapters; otherwise use a published ONNX graph as the intermediate. */
export async function prepareHubModel(reference, options = {}) {
    const preview = await fetchHubModelConfig(reference, { ...options, allowMissingConfig: true });
    const inspected = inspectHubModel(preview);
    const onnxFiles = preview.files.filter((file) => /\.onnx$/i.test(file.path));
    const result = { ...inspected, onnxFiles, onnx: null, requiresOnnxSelection: false };
    if (inspected.inspection.supported) return result;
    if (onnxFiles.length === 1) return selectHubOnnxFile(result, onnxFiles[0].path, options);
    if (onnxFiles.length > 1) return { ...result, requiresOnnxSelection: true, inspection: {
        ...inspected.inspection,
        reason: 'This repository publishes several ONNX graphs. Choose the graph or component to inspect.',
    } };
    return { ...result, inspection: { ...inspected.inspection,
        reason: `${inspected.inspection.reason} This revision has no published ONNX graph for the intermediate conversion.`,
    } };
}

/** A multi-file repository can contain separate encoder/decoder graphs: never choose one silently. */
export async function selectHubOnnxFile(preview, filePath, options = {}) {
    const bytes = await fetchHubOnnxFile(preview, filePath, options);
    const { readOnnxModel } = await import('../onnx/index.js');
    const operatorGraph = readOnnxModel(bytes);
    return inspectHubOnnxPreview(preview, operatorGraph, filePath);
}
