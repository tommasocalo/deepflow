import { buildEditableOnnxArchitecture, recognizeOnnxComponents } from '../onnx/editable.js';

/** Keep Hub provenance while exposing the exact operator/component correspondence. */
export function inspectHubOnnxPreview(preview, operatorGraph, filePath) {
    const recognition = recognizeOnnxComponents(operatorGraph);
    const isPart = (preview.onnxFiles?.length ?? 0) > 1;
    const inspection = {
        supported: recognition.supported,
        family: preview.config?.model_type || 'ONNX architecture',
        architecture: operatorGraph.name,
        reason: recognition.supported ? null : `${recognition.issues[0]?.message || 'This graph has no editable mapping.'}${recognition.issues.length > 1 ? ` (${recognition.issues.length - 1} more mapping issues.)` : ''}`,
        components: recognition.components.map((component) => ({
            name: component.label || component.kind,
            detail: `${component.kind} · ${component.source.nodeIds.length} source operator${component.source.nodeIds.length === 1 ? '' : 's'}`,
        })),
        issues: recognition.issues,
        limitations: [
            'Editable topology from the published operator graph; pretrained weights are not loaded.',
            ...(isPart ? [`Imports only ${filePath}. Other graph files in this repository are not joined automatically.`] : []),
            ...operatorGraph.limitations,
        ],
    };
    return { ...preview, adapterId: 'onnx', requiresOnnxSelection: false,
        onnx: { filePath, operatorGraph, recognition }, inspection };
}

export function importHubOnnxArchitecture(preview, { architectureStore, trainingStore }) {
    if (!preview.onnx?.operatorGraph || !preview.onnx?.filePath) throw new Error('Choose and inspect a published ONNX graph first.');
    const result = buildEditableOnnxArchitecture(preview.onnx.operatorGraph, { fileName: preview.onnx.filePath });
    const source = {
        ...result.source,
        provider: 'huggingface',
        intermediateFormat: 'onnx',
        repoId: preview.repoId,
        revision: preview.requestedRevision,
        commit: preview.commit,
        filePath: preview.onnx.filePath,
        displayName: preview.repoId.split('/').at(-1),
        modelType: preview.config?.model_type ?? null,
        architecture: preview.onnx.operatorGraph.name,
    };
    architectureStore.setState({
        graph: { ...result.graph, importSource: source },
        selectedNodeIds: [], expandedNodeId: null, currentPath: [], openPaths: [],
        inspectedIterations: {}, repeatSelections: {}, repeatEditModes: {}, repeatEditError: null,
    });
    trainingStore.getState().resetGraph();
    return source;
}
