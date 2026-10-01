import { useArchitectureStore } from './architectureStore.js';
import { startArchitectureSession } from './architectureHistory.js';
import { useTemplatesStore } from './templatesStore.js';
import { useTrainingStore } from './trainingStore.js';

/** Commit a prepared architecture as a new session, preserving the document on failure. */
export function importEditableArchitecture({ graph, templates }, { paneStore = useArchitectureStore } = {}) {
    const previous = {
        architecture: useArchitectureStore.getState(),
        templates: useTemplatesStore.getState(),
        training: useTrainingStore.getState(),
        pane: paneStore.getState(),
    };
    return startArchitectureSession(() => {
        try {
            const navigation = {
                currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null,
                inspectedIterations: {}, repeatSelections: {}, repeatEditModes: {}, repeatEditError: null,
            };
            if (templates) useTemplatesStore.setState((state) => ({ templates: { ...state.templates, ...templates } }));
            useArchitectureStore.setState({ graph, ...navigation });
            if (paneStore !== useArchitectureStore) paneStore.setState(navigation);
            useTrainingStore.getState().resetGraph();
            return graph.importSource;
        } catch (error) {
            useTemplatesStore.setState(previous.templates, true);
            useArchitectureStore.setState(previous.architecture, true);
            useTrainingStore.setState(previous.training, true);
            paneStore.setState(previous.pane, true);
            throw error;
        }
    });
}
