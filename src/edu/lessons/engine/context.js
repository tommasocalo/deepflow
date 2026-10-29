import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';
import { onSourceRegistered, readSource, subscribeSource } from '../../bridge.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { createCheckContext } from './checks.js';

/** Bridge sources the checks read (registered by Stage B). */
export const CHECK_SOURCES = Object.freeze(['data', 'trainer']);

/** The main pane's UI state as checks see it. */
export function readUiState() {
    const architecture = useArchitectureStore.getState();
    const ui = useEduUiStore.getState();
    return {
        tab: ui.activeTab,
        showDimensions: !!architecture.showDimensions,
        showGradientFlow: !!architecture.showGradientFlow,
        codeOpen: !!ui.codeOpen,
        codeTab: ui.codeTab,
        currentPath: architecture.currentPath ?? [],
        expandedNodeId: architecture.expandedNodeId ?? null,
    };
}

/**
 * Read every store synchronously into a CheckContext (docs/edu/SPEC.md §6.4.1). The step part (answers, entry
 * sequence number, visited tabs, run tags) comes from the lesson session, which passes it in.
 */
export function buildCheckContext({ module = null, step = {}, facts } = {}) {
    const architecture = useArchitectureStore.getState();
    return createCheckContext({
        graph: architecture.graph,
        templates: useTemplatesStore.getState().templates,
        training: useTrainingStore.getState().graph,
        ui: readUiState(),
        data: readSource('data'),
        trainer: readSource('trainer'),
        step,
        facts: facts ?? factValues(module),
        module,
        inspectedIterations: architecture.inspectedIterations ?? {},
    });
}

export function factValues(module) {
    return Object.fromEntries(Object.entries(module?.facts ?? {}).map(([name, fact]) => [name, fact?.value ?? fact]));
}

/**
 * Call `listener` after any change a check may depend on: architecture, templates, training, the Edu UI store and
 * every bridge source, including sources registered later. Returns an unsubscribe function.
 */
export function subscribeCheckContext(listener) {
    const unsubscribers = [
        useArchitectureStore.subscribe((state, previous) => {
            if (state.graph !== previous.graph || state.showDimensions !== previous.showDimensions
                || state.showGradientFlow !== previous.showGradientFlow || state.currentPath !== previous.currentPath
                || state.expandedNodeId !== previous.expandedNodeId || state.inspectedIterations !== previous.inspectedIterations) listener('architecture');
        }),
        useTemplatesStore.subscribe((state, previous) => { if (state.templates !== previous.templates) listener('templates'); }),
        useTrainingStore.subscribe((state, previous) => { if (state.graph !== previous.graph) listener('training'); }),
        useEduUiStore.subscribe(() => listener('ui')),
    ];
    const sourceSubscriptions = new Map();
    const watch = (name) => {
        if (!CHECK_SOURCES.includes(name)) return;
        sourceSubscriptions.get(name)?.();
        sourceSubscriptions.set(name, subscribeSource(name, () => listener(name)));
    };
    for (const name of CHECK_SOURCES) watch(name);
    const stopWatching = onSourceRegistered((name) => { watch(name); listener(name); });
    return () => {
        for (const unsubscribe of unsubscribers) unsubscribe();
        for (const unsubscribe of sourceSubscriptions.values()) unsubscribe();
        stopWatching();
    };
}
