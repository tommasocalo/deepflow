import assert from 'node:assert/strict';
import { test } from 'node:test';

// What the status line says (review r2). A pass is kept while the learner explores, but once the model it was about
// no longer holds, the evaluation says so (`drift`), so the panel can show "Passed earlier · now: …" and drop the
// done sentence (M1 s5 back to one output, M2 s3 without its ReLU). A finished wrong model that a diagnose rule
// names (the M2 s7 Sigmoid trap) is `firm`: named at once; a half-done one (M1 s5 with only Input set) is not.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { MODULES } = await import('../src/edu/lessons/modules/index.js');
await import('../src/edu/data/store.js');
await import('../src/edu/trainer/store.js');
const { useLessonSession } = await import('../src/edu/lessons/engine/session.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useArchitectureHistory } = await import('../src/stores/architectureHistory.js');
const { runActions } = await import('../src/edu/lessons/engine/actions.js');
const { t } = await import('../src/edu/lessons/engine/i18n.js');
const { resetLogForTests } = await import('../src/edu/lessons/engine/log.js');

const session = () => useLessonSession.getState();
const store = () => useArchitectureStore.getState();
const moduleOf = (id) => MODULES.find((module) => module.id === id);
const stepOf = (moduleId, stepId) => moduleOf(moduleId).steps.find((step) => step.id === stepId);
const setParam = (node, name, value) => runActions([{ do: 'setParam', node: { id: node }, name, value }], { headless: true, label: 'learner' });

async function open(moduleId, stepId) {
    memory.clear();
    resetLogForTests();
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model' });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0 });
    session().hydrate();
    session().setHeadless(true);
    assert.equal(await session().startModule(moduleId, { stepId }), true);
    assert.equal(session().currentStep().id, stepId);
}

test('M1 s5: a kept pass whose model changed reports the drift, with the live status, until the model holds again', async () => {
    await open('m1', 's5');
    assert.equal((await session().showMe()).passed, true);
    assert.equal(session().notePass(), true);
    assert.equal(session().evaluate().drift, undefined, 'no drift while the step passes');
    await setParam('neuron', 'out_features', 1);
    const result = session().evaluate();
    assert.equal(result.pass, false);
    assert.equal(result.drift, true);
    assert.equal(session().progress.modules.m1.steps.s5.status, 'passed', 'the pass is kept');
    assert.match(t(result.detail.key, result.detail.params), /Output.*1.*3/);
    await setParam('neuron', 'out_features', 3);
    assert.deepEqual([session().evaluate().pass, session().evaluate().drift], [true, undefined]);
});

test('M2 s3: undoing the ReLU after the pass is a drift; a view or code change after a pass is not', async () => {
    await open('m2', 's3');
    assert.equal((await session().showMe()).passed, true);
    session().notePass();
    useArchitectureHistory.getState().undo();
    const result = session().evaluate();
    assert.deepEqual([result.pass, result.drift], [false, true]);

    // M1 s8 passes with train.py open; opening model.py afterwards changes nothing about the model.
    await open('m1', 's8');
    assert.equal((await session().showMe()).passed, true);
    session().notePass();
    useEduUiStore.getState().openCode('model');
    const moved = session().evaluate();
    assert.deepEqual([moved.pass, moved.drift], [false, undefined]);
});

test('M2 s4: going back to a trained step without a new run is no drift; removing its ReLU is', async () => {
    await open('m2', 's4');
    assert.equal((await session().showMe()).passed, true);
    session().notePass();
    await session().next();
    assert.equal(session().currentStep().id, 's5');
    await session().back();
    assert.equal(session().currentStep().id, 's4');
    const review = session().evaluate();
    assert.deepEqual([review.pass, review.drift], [false, undefined], 'no run since the step was reopened, the model unchanged');
    const relu = Object.values(store().graph.nodes).find((node) => node.libraryKey === 'torch.nn.ReLU');
    assert.equal((await runActions([{ do: 'remove', node: { id: relu.id } }], { headless: true, label: 'learner' })).ok, true);
    assert.equal(session().evaluate().drift, true);
});

test('the Sigmoid trap is named at once (firm); a half-done model and an unfinished build are not', async () => {
    await open('m2', 's7');
    const trap = stepOf('m2', 's7').diagnose.find((rule) => rule.misconception === 'sigmoid-before-bce').example;
    assert.equal((await runActions(trap, { module: moduleOf('m2'), headless: true, label: 'learner' })).ok, true);
    const result = session().evaluate();
    assert.deepEqual([result.misconception, result.predicate, result.firm], ['sigmoid-before-bce', 'chain', true]);

    // Halfway through the XOR build (one Linear and the ReLU so far): the chain message still guides the next block.
    await open('m2', 's7');
    const input = Object.values(store().graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
    const output = Object.values(store().graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Output');
    const half = [{ do: 'buildChain', from: { id: input.id }, to: { id: output.id }, blocks: [{ block: 'component:linear', params: { in_features: 2, out_features: 8 } }, { block: 'torch.nn.ReLU' }] }];
    assert.equal((await runActions(half, { module: moduleOf('m2'), headless: true, label: 'learner' })).ok, true);
    const building = session().evaluate();
    assert.equal(building.misconception, 'activation-at-output');
    assert.equal(building.firm, undefined, 'the palette still has a Linear to place');

    // M1 s5 with only Input set: the model 4 → 1 is valid and diagnosed, but the step's Output is still to come.
    await open('m1', 's5');
    await setParam('neuron', 'in_features', 4);
    const partial = session().evaluate();
    assert.equal(partial.misconception, 'one-output-multiclass');
    assert.equal(partial.firm, undefined);
});
