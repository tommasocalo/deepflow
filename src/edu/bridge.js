/**
 * Cross-stage registry (docs/edu/SPEC.md §6.2). Stores register read-only sources and workspace slices here so the
 * lesson engine can check and snapshot app state without importing other stages' modules. Do not edit; Stage E only.
 */
const sources = new Map();
const slices = new Map();
const listeners = new Set();

/** api: { read(): object, subscribe?(listener): () => void, actions?: Object<string, Function> } */
export function registerSource(name, api) {
    sources.set(name, api);
    for (const listener of listeners) listener(name);
    return () => { if (sources.get(name) === api) sources.delete(name); };
}
export function readSource(name) {
    const api = sources.get(name);
    try { return api ? api.read() : null; } catch { return null; }
}
export function subscribeSource(name, listener) {
    const api = sources.get(name);
    return api?.subscribe ? api.subscribe(listener) : () => {};
}
export function callSource(name, action, ...args) {
    const fn = sources.get(name)?.actions?.[action];
    if (!fn) throw new Error(`Source "${name}" has no action "${action}"`);
    return fn(...args);
}
export function onSourceRegistered(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** slice: { capture(): json-serialisable, restore(value | undefined): void } */
export function registerSlice(name, slice) {
    slices.set(name, slice);
    return () => { if (slices.get(name) === slice) slices.delete(name); };
}
export function workspaceSlices() {
    return [...slices.entries()];
}

/** Development-only handles for browser checks: window.__deepflow.edu.<name>. */
export const eduDevHandles = {};
export function exposeEduHandle(name, value) {
    eduDevHandles[name] = value;
    return value;
}
