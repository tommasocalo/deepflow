import assert from 'node:assert/strict';
import { test } from 'node:test';

// A module installed in a new version (review r2: M5 version 2 renumbered its steps, so a returning learner's records
// attached to the wrong steps). Its step records are dropped (engine/progress.js migrate), its saved workspace and
// step entry too, and a learner who left a lesson inside it reopens it at its first step, from its start fixture.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { MODULES } = await import('../src/edu/lessons/modules/index.js');
await import('../src/edu/data/store.js');
await import('../src/edu/trainer/store.js');
const { useLessonSession, PROGRESS_KEY } = await import('../src/edu/lessons/engine/session.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { buildFixtureSnapshot, captureWorkspace, loadSlot, loadStepEntry, saveSlot } = await import('../src/edu/lessons/engine/workspace.js');
const { readLog, resetLogForTests } = await import('../src/edu/lessons/engine/log.js');

const session = () => useLessonSession.getState();
const m2 = MODULES.find((module) => module.id === 'm2');
const labels = (graph) => Object.values(graph.nodes).map((node) => node.label ?? node.libraryKey).sort();

function freshProfile() {
    memory.clear();
    resetLogForTests();
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model' });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0 });
    session().hydrate();
    session().setHeadless(true);
}

/** Rewrite the stored progress as if module `id` had been recorded by an older version of it. */
function recordedByOlderVersion(id) {
    const stored = JSON.parse(memory.get(PROGRESS_KEY));
    stored.modules[id].moduleVersion = (MODULES.find((module) => module.id === id).version ?? 1) - 1;
    memory.set(PROGRESS_KEY, JSON.stringify(stored));
}

test('a learner left inside an updated module reopens it at its first step, from its start fixture', async () => {
    freshProfile();
    await session().startModule('m2', { stepId: 's5' });
    session().submitPrediction('m2.width', 'three');
    assert.ok(Object.keys(session().progress.modules.m2.steps).length >= 4);
    assert.ok(loadStepEntry('m2', 's5'), 'the step entry is stored');
    saveSlot('sandbox', captureWorkspace());
    const sandbox = JSON.stringify(loadSlot('sandbox').graph);
    recordedByOlderVersion('m2');

    session().hydrate();
    assert.equal(session().restart, 'm2');
    assert.deepEqual([session().mode, session().moduleId, session().phase, session().stepIndex], ['lesson', 'm2', 'steps', 0]);
    const record = session().progress.modules.m2;
    assert.deepEqual(record.steps, {}, 'no step record survives the renumbering');
    assert.equal(record.migratedFrom, m2.version - 1);
    assert.deepEqual(Object.keys(record.answers), ['m2.width'], 'an answer whose prediction still exists is kept');
    assert.equal(loadStepEntry('m2', 's5'), null, 'the step entry of the old steps is gone');
    assert.equal(JSON.parse(memory.get(PROGRESS_KEY)).modules.m2.moduleVersion, m2.version, 'the migrated progress is saved at once');
    assert.ok(readLog().some((event) => event.type === 'module-updated' && event.module === 'm2' && event.data.to === m2.version));

    // The lesson panel starts it at once: the module's start, not the old steps' workspace.
    assert.equal(await session().startModule('m2'), true);
    assert.equal(session().restart, null);
    assert.equal(session().currentStep().id, m2.steps[0].id);
    assert.deepEqual(labels(useArchitectureStore.getState().graph), labels(buildFixtureSnapshot(m2, m2.start).graph));
    assert.equal(JSON.stringify(loadSlot('sandbox').graph), sandbox, 'the sandbox slot is untouched');
    assert.equal(loadSlot('m2'), null, 'the old workspace was not saved as the module\'s');

    // A second hydrate finds nothing to update.
    session().hydrate();
    assert.equal(session().restart, null);
});

test('an updated module the learner is not in loses its saved workspace and step records, nothing else', async () => {
    freshProfile();
    await session().startModule('m2', { stepId: 's3' });
    await session().startModule('m1');
    assert.ok(loadSlot('m2'), 'leaving M2 saved its workspace');
    const m1 = structuredClone(session().progress.modules.m1);
    recordedByOlderVersion('m2');
    session().hydrate();
    assert.equal(session().restart, null);
    assert.deepEqual([session().mode, session().moduleId], ['lesson', 'm1']);
    assert.equal(loadSlot('m2'), null);
    assert.deepEqual(session().progress.modules.m2.steps, {});
    assert.deepEqual(session().progress.modules.m1, m1, 'M1 is untouched');
    // Starting M2 again opens its first step on its start fixture.
    await session().startModule('m2');
    assert.equal(session().currentStep().id, m2.steps[0].id);
    assert.deepEqual(labels(useArchitectureStore.getState().graph), labels(buildFixtureSnapshot(m2, m2.start).graph));
});
