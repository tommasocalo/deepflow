import { MODULES, TEXT } from '../modules/index.js';
import { setTextBundles } from './i18n.js';

/**
 * The module registry the engine runs. By default it is Stage D's `modules/index.js`; tests and browser checks can
 * swap in other modules (for example scripts/fixtures/edu-test-module.js) with `setModuleRegistry`.
 */
let registry = { modules: [], text: {} };
const listeners = new Set();

export function setModuleRegistry({ modules = [], text = {} } = {}) {
    registry = {
        modules: [...modules].sort((left, right) => (left.order ?? 0) - (right.order ?? 0)),
        text: text ?? {},
    };
    setTextBundles(registry.text);
    for (const listener of listeners) listener(registry);
}

export function resetModuleRegistry() {
    setModuleRegistry({ modules: MODULES ?? [], text: TEXT ?? {} });
}

export function getModules() {
    return registry.modules;
}

export function getModule(id) {
    return registry.modules.find((module) => module.id === id) ?? null;
}

export function stepIndexById(module, stepId) {
    return module?.steps?.findIndex((step) => step.id === stepId) ?? -1;
}

export function nextModule(id) {
    const index = registry.modules.findIndex((module) => module.id === id);
    return index >= 0 ? registry.modules[index + 1] ?? null : null;
}

/** "1", "2", …: the module's order, used in titles such as "2 · Going non-linear". */
export function moduleNumber(module) {
    return module?.order ?? (registry.modules.indexOf(module) + 1);
}

export function subscribeModuleRegistry(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

resetModuleRegistry();
