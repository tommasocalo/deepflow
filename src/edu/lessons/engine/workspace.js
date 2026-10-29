import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { startArchitectureSession, useArchitectureHistory } from '../../../stores/architectureHistory.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';
import { STORYBOARDS } from '../../../storyboards/index.js';
import { workspaceSlices } from '../../bridge.js';
import { createFixtureHelpers } from './fixtures.js';
import { eduKey, readJSON, removeKey, writeJSON } from './storage.js';

/**
 * Workspace snapshots and slots (docs/edu/SPEC.md §3.7, §6.8.3). A snapshot holds the root architecture graph, all
 * templates, the training graph, every registered workspace slice (data binding, trainer settings) and two view
 * flags. Restoring one replaces the live stores and starts a fresh undo history, as loading an example does.
 */
export const SLOT_IDS = Object.freeze(['sandbox', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7']);
export const slotKey = (id) => eduKey(`slot:${id}:v1`);
export const STEP_ENTRY_KEY = eduKey('step-entry:v1');

const NAVIGATION_RESET = {
    currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null, inspectedIterations: {},
    repeatSelections: {}, repeatEditModes: {}, repeatEditError: null,
};

function captureSlices() {
    const slices = {};
    for (const [name, slice] of workspaceSlices()) {
        try {
            const value = slice.capture();
            if (value !== undefined) slices[name] = structuredClone(value);
        } catch (error) {
            console.warn(`[edu workspace] slice "${name}" could not be captured`, error);
        }
    }
    return slices;
}

function restoreSlices(values = {}) {
    for (const [name, slice] of workspaceSlices()) {
        try {
            slice.restore(values?.[name]);
        } catch (error) {
            console.warn(`[edu workspace] slice "${name}" could not be restored`, error);
        }
    }
}

/** @returns {import('./schema.js').WorkspaceSnapshot} */
export function captureWorkspace(now = Date.now()) {
    const architecture = useArchitectureStore.getState();
    return {
        v: 1,
        savedAt: now,
        graph: architecture.graph,
        templates: useTemplatesStore.getState().templates,
        training: useTrainingStore.getState().graph,
        slices: captureSlices(),
        ui: { showDimensions: !!architecture.showDimensions, showGradientFlow: !!architecture.showGradientFlow },
    };
}

/** Replace the live workspace. The undo stack never crosses workspaces. */
export function restoreWorkspace(snapshot) {
    if (!snapshot?.graph) throw new Error('Cannot restore an empty workspace snapshot');
    useArchitectureStore.getState().goToLevel(0);
    startArchitectureSession(() => {
        useTemplatesStore.setState({ templates: snapshot.templates ?? {} });
        useArchitectureStore.setState({
            graph: snapshot.graph,
            ...NAVIGATION_RESET,
            showDimensions: snapshot.ui?.showDimensions ?? true,
            showGradientFlow: snapshot.ui?.showGradientFlow ?? false,
        });
        useTrainingStore.setState({ graph: snapshot.training ?? { nodes: {}, edges: [] }, selectedNodeIds: [], expandedNodeId: null });
    });
    restoreSlices(snapshot.slices);
    useArchitectureHistory.getState().clear();
}

export function saveSlot(id, snapshot = captureWorkspace()) {
    return writeJSON(slotKey(id), snapshot);
}

export function loadSlot(id) {
    const snapshot = readJSON(slotKey(id), null);
    return snapshot?.v === 1 && snapshot.graph ? snapshot : null;
}

export function deleteSlot(id) {
    removeKey(slotKey(id));
}

export function saveStepEntry(moduleId, stepId, snapshot = captureWorkspace()) {
    writeJSON(STEP_ENTRY_KEY, { moduleId, stepId, snapshot });
    return snapshot;
}

export function loadStepEntry(moduleId, stepId) {
    const entry = readJSON(STEP_ENTRY_KEY, null);
    return entry && entry.moduleId === moduleId && entry.stepId === stepId && entry.snapshot?.graph ? entry.snapshot : null;
}

export function clearStepEntry() {
    removeKey(STEP_ENTRY_KEY);
}

/** Build a 'snapshot' fixture of a module into a WorkspaceSnapshot (pure: never touches the stores). */
export function buildFixtureSnapshot(module, fixtureId) {
    const fixture = module?.fixtures?.[fixtureId];
    if (!fixture) throw new Error(`Module ${module?.id} has no fixture "${fixtureId}"`);
    if ((fixture.kind ?? 'snapshot') !== 'snapshot') return null;
    const snapshot = fixture.build(createFixtureHelpers());
    if (!snapshot?.graph) throw new Error(`Fixture ${module.id}.${fixtureId} did not return a snapshot`);
    return { v: 1, savedAt: 0, templates: {}, training: { nodes: {}, edges: [] }, slices: {}, ...snapshot };
}

/**
 * Load a fixture into the live stores. 'storyboard' fixtures run a STORYBOARDS entry on fresh templates and then
 * their `after` actions (through `runAfter`, the engine's action runner), with history cleared at the end.
 */
export async function loadFixtureIntoWorkspace(module, fixtureId, { runAfter } = {}) {
    const fixture = module?.fixtures?.[fixtureId];
    if (!fixture) throw new Error(`Module ${module?.id} has no fixture "${fixtureId}"`);
    if ((fixture.kind ?? 'snapshot') === 'snapshot') {
        restoreWorkspace(buildFixtureSnapshot(module, fixtureId));
        return;
    }
    runStoryboard(fixture.storyboard, { configuration: fixture.configuration, includeTraining: fixture.includeTraining ?? true, freshTemplates: true });
    restoreSlices(fixture.slices ?? {});
    if (fixture.ui) useArchitectureStore.setState({ showDimensions: fixture.ui.showDimensions ?? true, showGradientFlow: fixture.ui.showGradientFlow ?? false });
    if (fixture.after?.length && runAfter) await runAfter(fixture.after);
    useArchitectureHistory.getState().clear();
}

/** Run an Examples storyboard on the live stores as one new editing session. */
export function runStoryboard(id, { configuration, includeTraining = true, freshTemplates = false } = {}) {
    const storyboard = STORYBOARDS.find((item) => item.id === id);
    if (!storyboard) throw new Error(`Unknown storyboard "${id}"`);
    const architectureStore = useArchitectureStore;
    architectureStore.getState().goToLevel(0);
    startArchitectureSession(() => {
        if (freshTemplates) useTemplatesStore.setState({ templates: {} });
        architectureStore.setState(NAVIGATION_RESET);
        storyboard.run({ architectureStore, templatesStore: useTemplatesStore, trainingStore: useTrainingStore, configuration, includeTraining });
    });
    useArchitectureHistory.getState().clear();
}
