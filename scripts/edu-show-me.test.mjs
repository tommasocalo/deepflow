import assert from 'node:assert/strict';
import { test } from 'node:test';

// Show me after a wrong attempt (review r2): in M2 s7 the designed Sigmoid trap, and in M2 s3 a ribbon ReLU next to
// the chip's, left Show me splicing a second chain into the graph or toasting a change that did nothing. Show me now
// completes an attempt it can complete and otherwise starts the step over from its entry, as one undo step; a palette
// chip moves a block the learner already added instead of adding a second one. A generic test perturbs the entry state
// of every build step (an extra block without wires, a wire the model does not have) and runs Show me there.

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
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useArchitectureHistory } = await import('../src/stores/architectureHistory.js');
const { runActions } = await import('../src/edu/lessons/engine/actions.js');
const { workspaceCheck } = await import('../src/edu/lessons/engine/checks.js');
const { insertPaletteItem, paletteUsed, blocksLeft } = await import('../src/edu/lessons/engine/palette.js');
const { isTargetDone } = await import('../src/edu/lessons/engine/targets.js');
const { readLog, resetLogForTests } = await import('../src/edu/lessons/engine/log.js');

const session = () => useLessonSession.getState();
const store = () => useArchitectureStore.getState();
const history = () => useArchitectureHistory.getState();
const m2 = MODULES.find((module) => module.id === 'm2');
const stepOf = (module, id) => module.steps.find((step) => step.id === id);
const nodesOf = (key) => Object.values(store().graph.nodes).filter((node) => node.libraryKey === key);
const linears = () => Object.values(store().graph.nodes).filter((node) => node.kind === 'supernode');
const rootNode = (key) => Object.values(store().graph.nodes).find((node) => node.libraryKey === key);
const wire = (source, target) => assert.ok(store().connect({ source, target }), `wire ${source} → ${target}`);

function freshProfile() {
    memory.clear();
    resetLogForTests();
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model' });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0 });
    session().hydrate();
    session().setHeadless(true);
}

async function open(moduleId, stepId) {
    freshProfile();
    assert.equal(await session().startModule(moduleId, { stepId }), true);
    assert.equal(session().currentStep()?.id, stepId);
    assert.equal(session().runtime.error, null);
}

test('workspaceCheck keeps the model part of a check: runs keep their model condition, moves are left out', () => {
    const chain = { chain: [{ type: 'Input' }, { type: 'Output' }] };
    assert.equal(workspaceCheck(chain), chain, 'a graph check comes back as it is');
    assert.deepEqual(workspaceCheck({ all: [{ answered: 'x' }, chain, { run: { dataset: 'xor', minEpochs: 50 } }] }), { all: [chain] });
    assert.deepEqual(workspaceCheck({ run: { minEpochs: 50, model: { all: [{ valid: true }, { ui: { codeOpen: true } }] } } }), { all: [{ valid: true }] });
    assert.equal(workspaceCheck({ all: [{ answered: 'x' }, { ui: { showDimensions: true } }] }), null);
    assert.equal(workspaceCheck({ any: [chain, { tab: 'data' }] }), null, 'an alternative it cannot decide');
    assert.equal(workspaceCheck({ not: { all: [chain, { visited: 'data' }] } }), null, 'a weakened not would be stricter');
});

test('M2 s7: Show me after the Sigmoid trap starts the step over, in one undo step, and trains the right model', async () => {
    const s7 = stepOf(m2, 's7');
    const trap = s7.diagnose.find((rule) => rule.misconception === 'sigmoid-before-bce').example;
    await open('m2', 's7');
    // The trap as a learner builds it: features → Linear → ReLU → Linear → Sigmoid → logit.
    assert.equal((await runActions(trap, { module: m2, headless: true, label: 'learner' })).ok, true);
    const wrong = session().evaluate();
    assert.equal(wrong.pass, false);
    assert.equal(wrong.misconception, 'sigmoid-before-bce');
    const attempt = JSON.stringify(store().graph);
    const undoBefore = history().undoCount;

    const shown = await session().showMe();
    assert.deepEqual([shown.ok, shown.restarted, shown.passed], [true, true, true], shown.error);
    assert.equal(session().evaluate().pass, true, 'the step passes');
    assert.deepEqual([nodesOf('torch.nn.Sigmoid').length, nodesOf('torch.nn.ReLU').length, linears().length], [0, 1, 2], 'one chain, without the Sigmoid');
    assert.equal(history().undoCount, undoBefore + 1, 'one undo step');
    assert.equal(session().runtime.showMe.restarted, true, 'the toast says the step started over');
    assert.ok(readLog().some((event) => event.type === 'show-me' && event.data.restarted && event.data.passed));

    // The toast's Undo brings the learner's attempt back.
    assert.equal(session().undoShowMe(), true);
    assert.equal(JSON.stringify(store().graph), attempt);
    assert.equal(session().evaluate().misconception, 'sigmoid-before-bce');
});

test('M2 s7: the reviewer\'s Linear → ReLU → Sigmoid → logit attempt, and a correct but unfinished one', async () => {
    await open('m2', 's7');
    const input = rootNode('deepflow.io.Input').id;
    const output = rootNode('deepflow.io.Output').id;
    const blocks = [{ block: 'component:linear', params: { in_features: 2, out_features: 8 } }, { block: 'torch.nn.ReLU' }, { block: 'torch.nn.Sigmoid' }];
    assert.equal((await runActions([{ do: 'buildChain', from: { id: input }, to: { id: output }, blocks }], { module: m2, headless: true, label: 'learner' })).ok, true);
    const shown = await session().showMe();
    assert.deepEqual([shown.ok, shown.restarted, shown.passed], [true, true, true], shown.error);
    assert.equal(store().graph.edges.filter((edge) => edge.target === output).length, 1, 'the Output has one wire');
    assert.equal(nodesOf('torch.nn.Sigmoid').length, 0);

    // A correct start (Linear and ReLU added from the chips, not wired yet) is completed, not replaced.
    await open('m2', 's7');
    const entryNodeIds = session().runtime.entryNodeIds;
    const linearChip = stepOf(m2, 's7').palette.find((item) => item.block === 'component:linear');
    const reluChip = stepOf(m2, 's7').palette.find((item) => item.block === 'torch.nn.ReLU');
    const first = (await insertPaletteItem(linearChip, { entryNodeIds })).id;
    const relu = (await insertPaletteItem(reluChip, { entryNodeIds })).id;
    const done = await session().showMe();
    assert.deepEqual([done.ok, done.restarted, done.passed], [true, false, true], done.error);
    assert.ok(store().graph.nodes[first] && store().graph.nodes[relu], 'the learner\'s blocks are used');
    assert.equal(linears().length, 2);
});

test('M2 s3: the chip moves a ribbon ReLU into the wire instead of adding a second one, wherever it sits', async () => {
    const chip = stepOf(m2, 's3').palette[0];
    const relus = () => nodesOf('torch.nn.ReLU');
    // An unwired ReLU from the ribbon.
    await open('m2', 's3');
    let entryNodeIds = session().runtime.entryNodeIds;
    const { graph, templates } = { graph: store().graph, templates: useTemplatesStore.getState().templates };
    assert.equal(paletteUsed([chip], graph, templates, entryNodeIds), false);
    const ribbon = store().addLayerByLibraryKey('torch.nn.ReLU', { x: 600, y: 300 });
    assert.equal(paletteUsed([chip], store().graph, useTemplatesStore.getState().templates, entryNodeIds), true, 'the palette counts as used');
    assert.equal(isTargetDone('lesson.palette', undefined, { palette: [chip], entryNodeIds }), true, 'the ring moves off the chip');
    assert.equal(isTargetDone('ribbon.components', undefined, { palette: [chip], entryNodeIds }), true, 'and off Components');
    const inserted = await insertPaletteItem(chip, { entryNodeIds });
    assert.deepEqual([inserted.id, inserted.spliced, inserted.adopted], [ribbon, true, true]);
    assert.equal(relus().length, 1);
    assert.equal(session().evaluate().pass, true);

    // A ribbon ReLU wired between Linear 2 and the Output: lifted out, and that wire made whole again.
    await open('m2', 's3');
    entryNodeIds = session().runtime.entryNodeIds;
    const misplaced = store().addLayerByLibraryKey('torch.nn.ReLU', { x: 600, y: 300 });
    const outputWire = store().graph.edges.find((edge) => edge.source === 'lin2');
    store().removeEdge(outputWire.id);
    wire('lin2', misplaced);
    wire(misplaced, outputWire.target);
    assert.equal(session().evaluate().misconception, 'activation-at-output');
    const moved = await insertPaletteItem(chip, { entryNodeIds });
    assert.deepEqual([moved.id, moved.adopted], [misplaced, true]);
    assert.equal(relus().length, 1);
    assert.equal(session().evaluate().pass, true, 'Linear 1 → ReLU → Linear 2 → Output');
});

test('M2 s3: Show me after a double ReLU (chip, then a ribbon ReLU wired beside it) leaves one ReLU', async () => {
    await open('m2', 's3');
    const entryNodeIds = session().runtime.entryNodeIds;
    assert.equal((await insertPaletteItem(stepOf(m2, 's3').palette[0], { entryNodeIds })).spliced, true);
    const second = store().addLayerByLibraryKey('torch.nn.ReLU', { x: 600, y: 300 });
    wire('lin1', second);
    wire(second, 'lin2');
    assert.equal(session().evaluate().pass, false, 'Linear 2 has two incoming wires');
    const undoBefore = history().undoCount;
    const shown = await session().showMe();
    assert.deepEqual([shown.ok, shown.passed], [true, true], shown.error);
    assert.equal(nodesOf('torch.nn.ReLU').length, 1);
    assert.equal(session().evaluate().pass, true);
    assert.equal(history().undoCount, undoBefore + 1, 'one undo step');
    history().undo();
    assert.equal(nodesOf('torch.nn.ReLU').length, 2, 'Undo brings both ReLUs back');
});

test('M2 s7: a chip adds blocks until the step has as many as it asks for, then selects one still missing a wire', async () => {
    await open('m2', 's7');
    const entryNodeIds = session().runtime.entryNodeIds;
    const chip = stepOf(m2, 's7').palette.find((item) => item.block === 'component:linear');
    const left = () => blocksLeft(chip, store().graph, useTemplatesStore.getState().templates, entryNodeIds);
    assert.equal(left(), 2);
    const a = await insertPaletteItem(chip, { entryNodeIds });
    const b = await insertPaletteItem(chip, { entryNodeIds });
    assert.deepEqual([a.adopted, b.adopted, left()], [false, false, 0]);
    wire(rootNode('deepflow.io.Input').id, a.id);
    wire(a.id, b.id);
    const c = await insertPaletteItem(chip, { entryNodeIds });
    assert.deepEqual([c.id, c.adopted], [b.id, true], 'the third click selects the Linear still missing a wire');
    assert.equal(linears().length, 2);
    assert.deepEqual(store().selectedNodeIds, [b.id]);
});

/** Steps whose Show me edits the model: the build steps. */
const GRAPH_EDITS = new Set(['insertBetween', 'insertAfter', 'buildChain', 'connect', 'disconnect', 'remove', 'group', 'repeat', 'setParam']);
const buildSteps = MODULES.flatMap((module) => module.steps
    .filter((step) => (step.solution ?? []).some((action) => GRAPH_EDITS.has(action.do)))
    .map((step) => [module, step]));

test('Show me passes every build step after its entry state is perturbed (an extra block, a wrong wire)', async () => {
    assert.ok(buildSteps.length >= 10, `${buildSteps.length} build steps`);
    const perturbations = {
        'an extra ReLU without wires': () => { store().addLayerByLibraryKey('torch.nn.ReLU', { x: 900, y: 120 }); return true; },
        'a wire from the Input straight to the Output': () => {
            const input = rootNode('deepflow.io.Input');
            const output = rootNode('deepflow.io.Output');
            if (!input || !output || store().graph.edges.some((edge) => edge.source === input.id && edge.target === output.id)) return false;
            return !!store().connect({ source: input.id, target: output.id });
        },
    };
    let perturbed = 0;
    for (const [module, step] of buildSteps) {
        for (const [name, perturb] of Object.entries(perturbations)) {
            const where = `${module.id} ${step.id} with ${name}`;
            await open(module.id, step.id);
            if (!perturb()) continue;
            perturbed += 1;
            const shown = await session().showMe();
            assert.equal(shown.ok, true, `${where}: ${shown.error}`);
            const result = session().evaluate();
            assert.equal(result.pass, true, `${where}: ${JSON.stringify(result.detail)} ${JSON.stringify(result.issue ?? null)}`);
            assert.equal(shown.passed, true, where);
        }
    }
    assert.ok(perturbed >= buildSteps.length, `${perturbed} perturbed attempts`);
});
