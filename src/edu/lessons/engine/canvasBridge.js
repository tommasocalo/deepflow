import { useArchitectureStore } from '../../../stores/architectureStore.js';

/**
 * Canvas bridge (docs/edu/SPEC.md §6.12). The architecture canvas keeps its React Flow capabilities inside its own
 * provider; Edu code reaches them through one synchronous window event, so `result` is set right after dispatch.
 * Without a mounted canvas every call is a harmless no-op.
 */
export const CANVAS_EVENT = 'deepflow-edu:canvas';
export const TRAINING_CANVAS_EVENT = 'deepflow-edu:training-canvas';

function dispatch(detail, name = CANVAS_EVENT) {
    const target = globalThis.window;
    if (!target?.dispatchEvent || typeof globalThis.CustomEvent !== 'function') return detail;
    target.dispatchEvent(new globalThis.CustomEvent(name, { detail }));
    return detail;
}

/** Returns true when a mounted canvas fitted its view. */
export function requestFit() {
    return dispatch({ command: 'fit', result: false }).result === true;
}

/** Fit the Training tab's canvas beside its palette; true when it was mounted. */
export function requestTrainingFit() {
    return dispatch({ command: 'fit', result: false }, TRAINING_CANVAS_EVENT).result === true;
}

/** Insert a palette entry ({ key } or { standardComponentKey }) with DeepFlow's own placement; returns the node id. */
export function requestInsert(entry, isTemplate = false) {
    return dispatch({ command: 'insert', entry, isTemplate, result: null }).result ?? null;
}

/** Select a root node, open its settings and centre the view on it. */
export function focusNode(id) {
    if (!id) return;
    const store = useArchitectureStore.getState();
    if (store.currentPath.length) {
        store.goToLevel(0);
        // Let the canvas render the root level before centring on one of its nodes.
        const frame = globalThis.requestAnimationFrame ?? ((callback) => setTimeout(callback, 16));
        frame(() => frame(() => dispatch({ command: 'focusNode', id })));
        return;
    }
    dispatch({ command: 'focusNode', id });
}
