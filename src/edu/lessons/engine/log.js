import { CURRICULUM_VERSION, PRODUCT_NAME } from '../../config.js';
import { eduKey, readJSON, removeKey, writeJSON } from './storage.js';

/**
 * Local learning log (docs/edu/SPEC.md §6.8.4): a ring buffer of the last 2,000 events. Nothing leaves the
 * browser unless the learner downloads it.
 */
export const LOG_LIMIT = 2000;
export const LOG_KEY = eduKey('log:v1');
export const EVENT_TYPES = Object.freeze([
    'module-start', 'step-enter', 'step-pass', 'step-skip', 'hint', 'show-me', 'reset-step', 'predict', 'warmup',
    'quiz', 'run', 'code-open', 'module-complete', 'palette-insert',
]);

let events = null;
const listeners = new Set();

function load() {
    if (!events) {
        const saved = readJSON(LOG_KEY, []);
        events = Array.isArray(saved) ? saved.slice(-LOG_LIMIT) : [];
    }
    return events;
}

export function logEvent(type, { module = null, step, data } = {}, now = Date.now()) {
    const event = { t: now, type, module };
    if (step !== undefined && step !== null) event.step = step;
    if (data !== undefined) event.data = data;
    const next = [...load(), event];
    events = next.length > LOG_LIMIT ? next.slice(next.length - LOG_LIMIT) : next;
    writeJSON(LOG_KEY, events);
    for (const listener of listeners) listener(events);
    return event;
}

export function readLog() {
    return load();
}

export function clearLog() {
    events = [];
    removeKey(LOG_KEY);
    for (const listener of listeners) listener(events);
}

export function subscribeLog(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** The "Download learning log" payload. */
export function learningLogExport(progress, now = Date.now()) {
    return {
        product: PRODUCT_NAME,
        curriculumVersion: CURRICULUM_VERSION,
        exportedAt: new Date(now).toISOString(),
        progress,
        events: load(),
    };
}

/** Tests only: drop the cached buffer so the next read comes from storage. */
export function resetLogForTests() {
    events = null;
}
