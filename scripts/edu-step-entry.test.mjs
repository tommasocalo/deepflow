import assert from 'node:assert/strict';
import { test } from 'node:test';

// Entering any step directly (review r1: a deep link or a progress segment to M2 s3 opened on the one-neuron start and
// Show me failed with "No block matches lin2"). From a fresh profile, every step of every module is opened by deep
// link and by its progress segment; the steps before it are filled in, its own check fails, and its Show me passes it.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { MODULES } = await import('../src/edu/lessons/modules/index.js');
await import('../src/edu/data/store.js');
await import('../src/edu/trainer/store.js');
const { useLessonSession, workspaceEdits } = await import('../src/edu/lessons/engine/session.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { readLog, resetLogForTests } = await import('../src/edu/lessons/engine/log.js');

const session = () => useLessonSession.getState();

function freshProfile() {
    memory.clear();
    resetLogForTests();
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model' });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0 });
    session().hydrate();
    session().setHeadless(true);
}

async function assertSolvable(module, step, where) {
    assert.equal(session().mode, 'lesson', where);
    assert.equal(session().phase, 'steps', where);
    assert.equal(session().currentStep()?.id, step.id, `${where}: opens the step`);
    assert.equal(session().runtime.error, null, `${where}: the earlier steps were filled in without error`);
    assert.equal(session().runtime.busy, false, where);
    if (step.tab) assert.equal(useEduUiStore.getState().activeTab, step.tab, `${where}: opens tab ${step.tab}`);
    assert.equal(session().evaluate().pass, false, `${where}: the check fails on entry`);
    const shown = await session().showMe();
    assert.equal(shown.ok, true, `${where}: Show me failed: ${shown.error}`);
    const result = session().evaluate();
    assert.equal(result.pass, true, `${where}: the check passes after Show me (${JSON.stringify(result.detail)})`);
}

test('workspaceEdits keeps what a step leaves in the workspace and drops answers, runs and navigation', () => {
    assert.deepEqual(workspaceEdits([
        { do: 'answer', id: 'x', value: 1 }, { do: 'train' }, { do: 'evaluateInit' }, { do: 'openLevel', path: [{ id: 'a' }] },
        { do: 'setUi', codeOpen: true, codeTab: 'train' }, { do: 'setUi', tab: 'data', showDimensions: true },
        { do: 'setParam', node: { id: 'a' }, name: 'out_features', value: 3 }, { do: 'useDataset', id: 'iris', setOutput: false },
    ]), [
        { do: 'setUi', showDimensions: true }, { do: 'setParam', node: { id: 'a' }, name: 'out_features', value: 3 }, { do: 'useDataset', id: 'iris', setOutput: false },
    ]);
});

test('a deep link opens every step from a fresh profile, and its Show me passes it', async () => {
    for (const module of MODULES) {
        for (const step of module.steps) {
            const where = `#/learn/${module.id}/${step.id}`;
            freshProfile();
            assert.equal(await session().startModule(module.id, { stepId: step.id }), true, where);
            await assertSolvable(module, step, where);
            const statuses = module.steps.slice(0, module.steps.indexOf(step)).map((item) => session().progress.modules[module.id].steps[item.id]?.status);
            assert.ok(statuses.every((status) => status === 'visited'), `${where}: earlier steps are visited, not passed (${statuses})`);
        }
    }
});

test('a progress segment opens every step from the warm-up of a fresh module, and its Show me passes it', async () => {
    for (const module of MODULES) {
        for (const [index, step] of module.steps.entries()) {
            const where = `${module.id} segment ${index + 1}`;
            freshProfile();
            await session().startModule(module.id);
            assert.equal(session().phase, module.warmup.length ? 'warmup' : 'steps', where);
            assert.equal(await session().goToStep(index), true, where);
            await assertSolvable(module, step, where);
        }
    }
});

test('M2: jumping ahead fills in the steps between, Skip fills in the step, and a jump back only opens a step', async () => {
    const m2 = MODULES.find((module) => module.id === 'm2');
    const labels = () => Object.values(useArchitectureStore.getState().graph.nodes).map((node) => node.label ?? node.libraryKey).sort();
    freshProfile();
    await session().startModule('m2', { stepId: 's3' });
    assert.deepEqual(labels(), ['Linear 1', 'Linear 2', 'features', 'logit'], 's3 opens on the two Linear blocks of s2');
    assert.ok(session().runtime.skippedAhead > 0, 'the panel can say that earlier steps were filled in');
    assert.ok(readLog().some((event) => event.type === 'skip-ahead' && event.step === 's3'), 'the jump is logged');

    // Skip fills in s3 (the ReLU), so s4 can train the network.
    await session().skip();
    assert.equal(session().currentStep().id, 's4');
    assert.equal(session().progress.modules.m2.steps.s3.status, 'skipped');
    assert.ok(Object.values(useArchitectureStore.getState().graph.nodes).some((node) => node.libraryKey === 'torch.nn.ReLU'), 'Skip added the ReLU');
    assert.equal((await session().showMe()).ok, true);
    assert.equal(session().evaluate().pass, true, 's4 trains the network with its ReLU');

    // A segment back opens s2 read-only: its setup (the twoLinear fixture) does not run again.
    const before = JSON.stringify(useArchitectureStore.getState().graph.edges);
    const indexOf = (id) => m2.steps.findIndex((step) => step.id === id);
    assert.equal(await session().goToStep(indexOf('s2')), true);
    assert.equal(session().currentStep().id, 's2');
    assert.equal(JSON.stringify(useArchitectureStore.getState().graph.edges), before, 'the workspace is unchanged');

    // Forward again over filled-in and passed steps: nothing is replayed twice (one ReLU, still valid).
    assert.equal(await session().goToStep(indexOf('s6')), true);
    assert.equal(session().currentStep().id, 's6');
    const relus = Object.values(useArchitectureStore.getState().graph.nodes).filter((node) => node.libraryKey === 'torch.nn.ReLU');
    assert.equal(relus.length, 1, 'the ReLU is not inserted a second time');
    assert.equal(session().runtime.error, null);
});

test('a pass is kept while the learner explores, until Reset step', async () => {
    freshProfile();
    await session().startModule('m1', { stepId: 's8' });
    assert.equal((await session().showMe()).ok, true);
    assert.equal(session().notePass(), true, 'the pass is latched');
    useEduUiStore.getState().openCode('model');
    assert.equal(session().evaluate().pass, false, 'the live check no longer passes on model.py');
    assert.equal(session().progress.modules.m1.steps.s8.status, 'passed', 'but the step stays passed');
    session().resetStep();
    assert.equal(session().progress.modules.m1.steps.s8.status, 'visited', 'Reset step opens the step again');
});
