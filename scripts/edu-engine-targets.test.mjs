import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

// Spotlight targets (docs/edu/SPEC.md §6.6): names, and when a nodeOpen target counts as reached.
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { TARGET_NAMES, isTargetDone, isValidTargetRef, reachedOpenTarget } = await import('../src/edu/lessons/engine/targets.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { runActions } = await import('../src/edu/lessons/engine/actions.js');
const { loadFixtureIntoWorkspace, restoreWorkspace } = await import('../src/edu/lessons/engine/workspace.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { MODULES } = await import('../src/edu/lessons/modules/index.js');

const m7 = MODULES.find((module) => module.id === 'm7');
const step = (module, id) => module.steps.find((item) => item.id === id);
const open = async (path) => assert.equal((await runActions([{ do: 'openLevel', path }])).ok, true);

beforeEach(() => memory.clear());

test('the code panel\'s notebook button and Colab note are spotlight targets', () => {
    for (const name of ['code.notebook', 'code.colabNote', 'code.tab:model', 'code.tab:train']) {
        assert.ok(TARGET_NAMES.includes(name), name);
        assert.equal(isValidTargetRef(name), true, name);
    }
});

test('a nodeOpen target is reached inside its block, not at the root or in another block', async () => {
    const h = createFixtureHelpers();
    const nodes = [h.input('in', [2]), h.linear('lin1', 2, 3), h.linear('lin2', 3, 1), h.output('out')];
    restoreWorkspace(h.snapshot({ nodes, edges: h.chain(...nodes) }));
    const refs = [{ nodeOpen: { id: 'lin1' } }];
    assert.equal(reachedOpenTarget(refs), null, 'at the root the arrow is still to press');
    await open([{ id: 'lin1' }]);
    assert.equal(reachedOpenTarget(refs)?.id, 'lin1');
    assert.equal(reachedOpenTarget([{ node: { id: 'lin1' } }]), null, 'a plain node target is never "reached" by opening it');
    useArchitectureStore.getState().goToLevel(0);
    await open([{ id: 'lin2' }]);
    assert.equal(reachedOpenTarget(refs), null, 'inside another Linear');
});

test('M7 s4: every level of the path to the attention head counts as reached', async () => {
    await loadFixtureIntoWorkspace(m7, m7.start, { runAfter: async (after) => assert.equal((await runActions(after, { module: m7 })).ok, true) });
    const refs = step(m7, 's4').target;
    assert.equal(reachedOpenTarget(refs), null);
    await open([{ label: 'Transformer blocks' }]);
    assert.equal(reachedOpenTarget(refs)?.label, 'Transformer blocks');
    useArchitectureStore.getState().goToLevel(0);
    await open([{ label: 'Transformer blocks' }, { type: 'Attention' }, { type: 'AttentionHead' }]);
    assert.ok(reachedOpenTarget(refs), 'at the head level the step only waits for the answer');
    useArchitectureStore.getState().goToLevel(0);
    await open([{ label: 'token embedding' }]);
    assert.equal(reachedOpenTarget(refs), null, 'the token embedding is not on the way: "Where is it?" stays useful');
});

test('a used target is done, so the ring moves on: the open tab, the open code panel and its tab, an opened block', async () => {
    useEduUiStore.setState({ activeTab: 'architecture', codeOpen: false, codeTab: 'model' });
    assert.equal(isTargetDone('tab.data'), false);
    useEduUiStore.getState().setTab('data');
    assert.equal(isTargetDone('tab.data'), true);
    assert.equal(isTargetDone('toolbar.code'), false);
    useEduUiStore.getState().openCode('model');
    assert.equal(isTargetDone('toolbar.code'), true);
    assert.equal(isTargetDone('code.tab:train'), false, 'the panel is open on model.py');
    useEduUiStore.getState().setCodeTab('train');
    assert.equal(isTargetDone('code.tab:train'), true);
    assert.equal(isTargetDone('ribbon.dimensions'), false, 'controls that stay to be used are never done');
    const h = createFixtureHelpers();
    const nodes = [h.input('in', [2]), h.linear('lin1', 2, 1), h.output('out')];
    restoreWorkspace(h.snapshot({ nodes, edges: h.chain(...nodes) }));
    assert.equal(isTargetDone({ nodeOpen: { id: 'lin1' } }), false);
    await open([{ id: 'lin1' }]);
    assert.equal(isTargetDone({ nodeOpen: { id: 'lin1' } }), true);
    assert.equal(isTargetDone({ node: { id: 'lin1' } }), false);
});
