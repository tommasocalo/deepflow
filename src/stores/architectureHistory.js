import { create } from 'zustand';

const HISTORY_LIMIT = 100;
let adapter = null;
let transactionDepth = 0;
let restoring = false;
const past = [];
const future = [];

// Immutable references skip almost all branches; ignore no-op patches from
// controlled fields without serializing the complete model on every keystroke.
function sameValue(left, right) {
    if (left === right) return true;
    if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false;
    if (Array.isArray(left) !== Array.isArray(right)) return false;
    const keys = Object.keys(left);
    return keys.length === Object.keys(right).length && keys.every((key) => Object.hasOwn(right, key) && sameValue(left[key], right[key]));
}
const differs = (left, right) => !sameValue(left.graph, right.graph) || !sameValue(left.templates, right.templates)
    || !sameValue(left.training, right.training);
const publish = () => useArchitectureHistory.setState({
    canUndo: past.length > 0,
    canRedo: future.length > 0,
    undoCount: past.length,
    redoCount: future.length,
    undoLabel: past.at(-1)?.label ?? '',
    redoLabel: future.at(-1)?.label ?? '',
});

/**
 * One document history for all panes. Snapshots retain immutable graph/template
 * references, not selection, navigation, measured layout, or display settings.
 * The training graph (loss, optimizer) is part of the same document.
 */
export const useArchitectureHistory = create(() => ({
    canUndo: false,
    canRedo: false,
    undoCount: 0,
    redoCount: 0,
    undoLabel: '',
    redoLabel: '',
    undo: () => restoreFrom(past, future, 'before'),
    redo: () => restoreFrom(future, past, 'after'),
    clear: () => {
        past.length = 0;
        future.length = 0;
        publish();
    },
    transaction: (label, callback) => recordArchitectureEdit(label, callback),
}));

function restoreFrom(source, destination, key) {
    if (!adapter || restoring || transactionDepth || !source.length) return false;
    const entry = source.at(-1);
    restoring = true;
    try {
        adapter.restore(entry[key]);
        source.pop();
        destination.push(entry);
    } finally {
        restoring = false;
        publish();
    }
    return true;
}

/** Install only after both persisted stores have hydrated. */
export function configureArchitectureHistory(nextAdapter) {
    adapter = nextAdapter;
    useArchitectureHistory.getState().clear();
}

export function isRestoringArchitectureHistory() {
    return restoring;
}

/** Direct replacements (load/example/rehydrate) start a new editing session. */
export function architectureDocumentReplaced() {
    if (!transactionDepth && !restoring && (past.length || future.length)) {
        useArchitectureHistory.getState().clear();
    }
}

/** Nesting makes composite actions, including newly-created templates, atomic. */
export function recordArchitectureEdit(label, callback) {
    if (!adapter || transactionDepth || restoring) return callback();
    const before = adapter.capture();
    transactionDepth++;
    try {
        return callback();
    } finally {
        transactionDepth--;
        const after = adapter.capture();
        if (differs(before, after)) {
            past.push({ label, before, after });
            if (past.length > HISTORY_LIMIT) past.shift();
            future.length = 0;
            publish();
        }
    }
}

/** Wrap explicit document mutators; view state actions remain outside history. */
export function withArchitectureHistory(state, labels) {
    for (const [name, label] of Object.entries(labels)) {
        const action = state[name];
        state[name] = (...args) => recordArchitectureEdit(label, () => action(...args));
    }
    return state;
}

/** Example loading/initial seeding replaces both architecture and training. */
export function startArchitectureSession(callback) {
    transactionDepth++;
    try {
        const result = callback();
        useArchitectureHistory.getState().clear();
        return result;
    } finally {
        transactionDepth--;
    }
}
