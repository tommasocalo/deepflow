import assert from 'node:assert/strict';
import { test } from 'node:test';

// Editing a shared component definition (content request for M6): setParam with `in: [<component path>]` changes a
// block inside the definition, so every instance and every repeat copy follows; the param check reads it the same way.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { MODULES } = await import('../src/edu/lessons/modules/index.js');
await import('../src/edu/data/store.js');
await import('../src/edu/trainer/store.js');
const { useLessonSession } = await import('../src/edu/lessons/engine/session.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { runActions } = await import('../src/edu/lessons/engine/actions.js');
const { buildCheckContext } = await import('../src/edu/lessons/engine/context.js');
const { evaluateCheck } = await import('../src/edu/lessons/engine/checks.js');
const { validateActions, validateCheckSpec } = await import('../src/edu/lessons/engine/schema.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useArchitectureHistory } = await import('../src/stores/architectureHistory.js');
const { resolveGraphLevel } = await import('../src/domain/architecture/operations.js');
const { countArchitectureParameters } = await import('../src/domain/optimization/parameterCount.js');

const m6 = MODULES.find((module) => module.id === 'm6');
const session = () => useLessonSession.getState();
const graph = () => useArchitectureStore.getState().graph;
const templates = () => useTemplatesStore.getState().templates;
const BLOCK = [{ name: 'ResBlock' }];
const CONV1 = { label: 'conv1' };

test('setParam `in` edits the shared ResBlock definition, and all three repeat copies follow', async () => {
    memory.clear();
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false });
    session().hydrate();
    session().setHeadless(true);
    // M6 s5 starts after the block was grouped (s3) and repeated three times (s4).
    assert.equal(await session().startModule('m6', { stepId: 's5' }), true);
    const repeat = Object.values(graph().nodes).find((node) => node.kind === 'repeat');
    assert.ok(repeat, 'the ResBlock repeat');
    assert.equal(templates()[repeat.templateId].name, 'ResBlock');
    const before = countArchitectureParameters(graph(), templates()).total;
    const check = { param: { in: BLOCK, node: CONV1, name: 'bias', equals: true } };
    assert.equal(evaluateCheck(check, buildCheckContext({ module: m6 })).pass, false, 'conv1 has no bias yet');

    const undoBefore = useArchitectureHistory.getState().undoCount;
    const result = await runActions([{ do: 'setParam', in: BLOCK, node: CONV1, name: 'bias', value: true }], { module: m6, headless: true, label: 'test' });
    assert.equal(result.ok, true, result.error);
    assert.equal(useArchitectureHistory.getState().undoCount, undoBefore + 1, 'one undo step');
    assert.equal(evaluateCheck(check, buildCheckContext({ module: m6 })).pass, true, 'the param check reads the definition');

    // Every copy of the repeat resolves to the edited definition.
    for (let index = 0; index < 3; index++) {
        const level = resolveGraphLevel(graph(), [repeat.id], templates(), { [JSON.stringify([repeat.id])]: index });
        const conv = Object.values(level.graph.nodes).find((node) => node.label === 'conv1');
        assert.equal(conv?.parameters?.bias, true, `copy ${index + 1} sees the bias`);
    }
    // Three copies with separate weights: 32 more parameters each.
    assert.equal(countArchitectureParameters(graph(), templates()).total, before + 3 * 32);

    useArchitectureHistory.getState().undo();
    assert.equal(evaluateCheck(check, buildCheckContext({ module: m6 })).pass, false, 'Undo restores the definition');
});

test('a wrong component or block path fails with a readable error, and the schema checks `in`', async () => {
    const missing = await runActions([{ do: 'setParam', in: [{ name: 'NoSuchBlock' }], node: CONV1, name: 'bias', value: true }], { module: m6, headless: true });
    assert.equal(missing.ok, false);
    assert.match(missing.error, /No component matches NoSuchBlock/);
    const inner = await runActions([{ do: 'setParam', in: BLOCK, node: { label: 'nothing' }, name: 'bias', value: true }], { module: m6, headless: true });
    assert.match(inner.error, /No block matches nothing in ResBlock/);
    assert.deepEqual(validateActions([{ do: 'setParam', in: BLOCK, node: CONV1, name: 'bias', value: true }], 'a'), []);
    assert.equal(validateActions([{ do: 'setParam', in: [], node: CONV1, name: 'bias', value: true }], 'a').length, 1);
    assert.deepEqual(validateCheckSpec({ param: { in: BLOCK, node: CONV1, name: 'bias', equals: true } }), []);
    assert.equal(validateCheckSpec({ param: { in: [{ colour: 'red' }], node: CONV1, name: 'bias' } }).length, 1);
});
