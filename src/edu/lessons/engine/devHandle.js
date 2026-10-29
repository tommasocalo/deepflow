/**
 * Development handle window.__deepflow.edu.engine (docs/edu/SPEC.md §6.2): the engine API for browser checks,
 * for example swapping in scripts/fixtures/edu-test-module.js with setModuleRegistry.
 */
export { setModuleRegistry, resetModuleRegistry, getModules, getModule } from './registry.js';
export { evaluateCheck, evaluateStep, createCheckContext } from './checks.js';
export { buildCheckContext } from './context.js';
export { runActions } from './actions.js';
export { captureWorkspace, restoreWorkspace, loadSlot, saveSlot } from './workspace.js';
export { resolveTarget, resolveFirstTarget } from './targets.js';
export { friendlyIssues } from './feedback.js';
export { validateModule, moduleTextKeys } from './schema.js';
export { readLog, learningLogExport } from './log.js';
export { isStorageDegraded } from './storage.js';
export { t } from './i18n.js';
export { createFixtureHelpers } from './fixtures.js';
