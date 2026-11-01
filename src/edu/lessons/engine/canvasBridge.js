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

/** Select a wire of the root level and centre the view on it (wiring feedback); true when the canvas did. */
export function selectEdge(id) {
    if (!id) return false;
    const store = useArchitectureStore.getState();
    if (store.currentPath.length) store.goToLevel(0);
    return dispatch({ command: 'selectEdge', id, result: false }).result === true;
}

/**
 * Select a root node, open its settings and centre the view on it. With `stay`, the node is in the level the canvas
 * shows (a block the step asked to open): centre on it there instead of going back to the root.
 */
export function focusNode(id, { stay = false } = {}) {
    if (!id) return;
    const store = useArchitectureStore.getState();
    if (store.currentPath.length && !stay) {
        store.goToLevel(0);
        // Let the canvas render the root level before centring on one of its nodes.
        const frame = globalThis.requestAnimationFrame ?? ((callback) => setTimeout(callback, 16));
        frame(() => frame(() => dispatch({ command: 'focusNode', id })));
        return;
    }
    dispatch({ command: 'focusNode', id });
}
