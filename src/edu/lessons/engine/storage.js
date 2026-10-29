import { STORAGE_PREFIX } from '../../config.js';

/**
 * Every Edu localStorage access goes through here (docs/edu/SPEC.md §6.8). Reads and writes never throw: when
 * storage is missing, throws or is full, values live in an in-memory map for the rest of the session and
 * `storageDegraded` turns on, so the Learn home can say that progress is not being saved.
 */
const memory = new Map();
const listeners = new Set();
let degraded = false;

/** `deepflow-edu:<name>`; DeepFlow's own `deepflow:*` keys are never written here. */
export const eduKey = (name) => `${STORAGE_PREFIX}:${name}`;

function browserStorage() {
    try {
        return globalThis.localStorage ?? globalThis.window?.localStorage ?? null;
    } catch {
        return null;
    }
}

function markDegraded() {
    if (degraded) return;
    degraded = true;
    for (const listener of listeners) listener(true);
}

export function readJSON(key, fallback = null) {
    if (memory.has(key)) {
        try { return JSON.parse(memory.get(key)); } catch { return fallback; }
    }
    try {
        const raw = browserStorage()?.getItem(key);
        if (raw === null || raw === undefined) return fallback;
        return JSON.parse(raw);
    } catch {
        return fallback;
    }
}

/** Returns true when the value reached real storage. */
export function writeJSON(key, value) {
    let text;
    try {
        text = JSON.stringify(value);
    } catch {
        return false;
    }
    const storage = browserStorage();
    if (storage) {
        try {
            storage.setItem(key, text);
            memory.delete(key);
            return true;
        } catch {
            // Quota exceeded or access denied: fall through to memory.
        }
    }
    memory.set(key, text);
    markDegraded();
    return false;
}

export function removeKey(key) {
    memory.delete(key);
    try { browserStorage()?.removeItem(key); } catch { /* nothing to remove */ }
}

/** Every Edu key currently stored (memory and browser), for clearing slots. */
export function eduKeys() {
    const keys = new Set([...memory.keys()].filter((key) => key.startsWith(`${STORAGE_PREFIX}:`)));
    try {
        const storage = browserStorage();
        for (let index = 0; index < (storage?.length ?? 0); index++) {
            const key = storage.key(index);
            if (key?.startsWith(`${STORAGE_PREFIX}:`)) keys.add(key);
        }
    } catch { /* storage unavailable */ }
    return [...keys];
}

export function isStorageDegraded() {
    return degraded;
}

export function subscribeStorageDegraded(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** Tests only: forget the in-memory fallback and the degraded flag. */
export function resetStorageForTests() {
    memory.clear();
    degraded = false;
}
