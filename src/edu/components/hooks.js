import { useEffect, useState, useSyncExternalStore } from 'react';
import { getModules, subscribeModuleRegistry } from '../lessons/engine/registry.js';
import { subscribeLessonContext, useLessonSession } from '../lessons/engine/session.js';
import { isStorageDegraded, subscribeStorageDegraded } from '../lessons/engine/storage.js';

/** Checks re-run 150 ms after the last change (docs/edu/SPEC.md §3.3). */
export const CHECK_DELAY = 150;

export function useModules() {
    return useSyncExternalStore(subscribeModuleRegistry, getModules, getModules);
}

export function useStorageDegraded() {
    return useSyncExternalStore(subscribeStorageDegraded, isStorageDegraded, isStorageDegraded);
}

/**
 * The current step's evaluation, refreshed `delay` ms after any change to the stores, the bridge sources or the
 * session. `key` identifies the step; a new key evaluates at once.
 */
export function useStepEvaluation(key, { enabled = true, delay = CHECK_DELAY } = {}) {
    const [state, setState] = useState(() => ({ key, result: enabled ? useLessonSession.getState().evaluate() : null }));
    useEffect(() => {
        if (!enabled) return undefined;
        let timer = null;
        const run = () => setState({ key, result: useLessonSession.getState().evaluate() });
        const schedule = () => {
            clearTimeout(timer);
            timer = setTimeout(run, delay);
        };
        // Evaluate the new step right away, outside the effect body.
        const first = setTimeout(run, 0);
        const unsubscribe = subscribeLessonContext(schedule);
        return () => {
            clearTimeout(first);
            clearTimeout(timer);
            unsubscribe();
        };
    }, [key, enabled, delay]);
    if (!enabled) return null;
    // Until the first evaluation of a new step lands, evaluate synchronously so the panel never shows stale status.
    return state.key === key ? state.result : useLessonSession.getState().evaluate();
}

export function useMediaQuery(query) {
    const subscribe = (listener) => {
        const media = globalThis.matchMedia?.(query);
        if (!media) return () => {};
        media.addEventListener?.('change', listener);
        return () => media.removeEventListener?.('change', listener);
    };
    const read = () => !!globalThis.matchMedia?.(query)?.matches;
    return useSyncExternalStore(subscribe, read, () => false);
}
