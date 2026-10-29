import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useTrainingStore } from '../../stores/trainingStore.js';
import { workspaceSlices } from '../bridge.js';

/**
 * Save and Load with Edu slices (docs/edu/SPEC.md §6.12). A saved file adds `edu.slices` (data binding, trainer
 * settings, …); a file without `edu` loads exactly as in DeepFlow.
 */
export function captureEduSlices() {
    const slices = {};
    for (const [name, slice] of workspaceSlices()) {
        try {
            const value = slice.capture();
            if (value !== undefined) slices[name] = value;
        } catch (error) {
            console.warn(`[edu] slice "${name}" could not be saved`, error);
        }
    }
    return slices;
}

export function buildSavePayload({ architecture, templates, training } = {}) {
    return {
        version: 'deepflow-v2',
        architecture: architecture ?? useArchitectureStore.getState().graph,
        templates: templates ?? useTemplatesStore.getState().templates,
        training: training ?? useTrainingStore.getState().graph,
        edu: { version: 1, slices: captureEduSlices() },
    };
}

/** Restore every registered slice from `values` (missing names get their defaults). */
export function restoreEduSlices(values = {}) {
    for (const [name, slice] of workspaceSlices()) {
        try {
            slice.restore(values?.[name]);
        } catch (error) {
            console.warn(`[edu] slice "${name}" could not be restored`, error);
        }
    }
}

/** Apply a parsed DeepFlow JSON file to the live stores, as DeepFlow's Load does, plus Edu slices when present. */
export function applyLoadedPayload(data) {
    useArchitectureStore.getState().goToLevel(0);
    if (data?.architecture) useArchitectureStore.setState({ graph: data.architecture, currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null });
    if (data?.templates) useTemplatesStore.setState((state) => ({ ...state, templates: data.templates }));
    if (data?.training) useTrainingStore.setState((state) => ({ ...state, graph: data.training }));
    if (data?.edu && typeof data.edu === 'object') restoreEduSlices(data.edu.slices ?? {});
}
