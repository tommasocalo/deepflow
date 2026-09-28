import assert from 'node:assert/strict';
import { test } from 'node:test';
import { countArchitectureParameters } from '../src/domain/optimization/parameterCount.js';
import { bestArchitectureCandidate, evaluateArchitectureCandidate, parseSearchRange, parseSearchValues, prepareArchitectureSearch } from '../src/domain/optimization/search.js';
import { gpt2Storyboard } from '../src/storyboards/gpt2.js';

const storage = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
} };
const { useArchitectureStore: architectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: trainingStore } = await import('../src/stores/trainingStore.js');

function gpt2() {
    for (const store of [architectureStore, templatesStore, trainingStore]) store.setState(store.getInitialState(), true);
    storage.clear();
    gpt2Storyboard({ architectureStore, templatesStore, trainingStore });
    return { graph: architectureStore.getState().graph, templates: templatesStore.getState().templates };
}

test('architecture parameter estimate expands editable templates and repeated blocks', () => {
    const { graph, templates } = gpt2();
    const original = countArchitectureParameters(graph, templates);
    assert.equal(original.complete, true);
    assert.equal(original.total, 163037184, 'DeepFlow currently counts its separate, untied lm_head');
    const sixBlocks = { ...graph, parameters: graph.parameters.map((parameter) => parameter.name === 'n_layers' ? { ...parameter, default: 6 } : parameter) };
    const smaller = countArchitectureParameters(sixBlocks, templates);
    assert.equal(smaller.complete, true);
    assert.ok(smaller.total < original.total);
    assert.equal(original.total - smaller.total, 6 * 7_087_872, 'Each repeated transformer adds its own parameters');
});

test('search validates actual GPT-2 tensor dimensions and ranks only valid candidates', () => {
    const { graph, templates } = gpt2();
    const initial = JSON.stringify(graph);
    const prepared = prepareArchitectureSearch(graph, [
        { name: 'd_model', values: '384, 768' }, { name: 'n_heads', values: '8, 12' },
    ]);
    assert.equal(prepared.error, null);
    assert.equal(prepared.candidates.length, 4);
    const rows = prepared.candidates.map((values) => evaluateArchitectureCandidate(graph, templates, values));
    assert.deepEqual(rows.map((row) => [row.values.d_model, row.values.n_heads, row.valid]),
        [[384, 8, true], [384, 12, true], [768, 8, true], [768, 12, true]]);
    const invalid = evaluateArchitectureCandidate(graph, templates, { d_model: 768, n_heads: 5 });
    assert.equal(invalid.valid, false, 'Width 768 cannot split evenly into five heads');
    assert.ok(invalid.issues.length > 0);
    assert.equal(bestArchitectureCandidate([...rows, invalid]).values.d_model, 384);
    assert.equal(bestArchitectureCandidate(rows, { referenceValues: { d_model: 768, n_heads: 12 } }).values.n_heads, 12,
        'Equal-size candidates retain the global value closest to the current architecture');
    const budget = rows.find((row) => row.values.d_model === 384 && row.values.n_heads === 8).parameters;
    assert.equal(bestArchitectureCandidate(rows, { objective: 'largest-within-budget', budget }).parameters, budget);
    assert.equal(JSON.stringify(graph), initial, 'Search never changes the editor document');
});

test('search input rejects invalid and excessive candidate grids before validation', () => {
    const { graph } = gpt2();
    assert.deepEqual(parseSearchValues('4, 4, 8'), { values: [4, 8], error: null });
    assert.deepEqual(parseSearchRange('384', '1152', '384'), { values: [384, 768, 1152], error: null });
    assert.deepEqual(parseSearchRange('25128', '75386', '25129'), { values: [25128, 50257, 75386], error: null });
    assert.deepEqual(parseSearchRange('1234567890123', '1234567890125', '1'),
        { values: [1234567890123, 1234567890124, 1234567890125], error: null });
    assert.ok(parseSearchRange('1', '100', '0').error);
    assert.ok(parseSearchRange('0', '1000', '1').error);
    assert.ok(parseSearchRange('1.5', '3', '0.5', 'int').error);
    assert.ok(parseSearchValues('4, window.alert(1)').error);
    assert.ok(parseSearchValues('1.5', 'int').error);
    assert.ok(prepareArchitectureSearch(graph, [{ name: 'd_model', values: '384, 768' }, { name: 'd_model', values: '512' }]).error);
    assert.ok(prepareArchitectureSearch(graph, [{ name: 'd_model', values: '1,2,3,4,5,6,7' }, { name: 'n_heads', values: '1,2,3,4,5,6,7' }]).error);
    const threeAxes = prepareArchitectureSearch(graph, [
        { name: 'd_model', mode: 'range', start: 384, end: 768, step: 384 },
        { name: 'n_heads', values: '8, 12' },
        { name: 'n_layers', values: '6, 12' },
    ]);
    assert.equal(threeAxes.error, null);
    assert.equal(threeAxes.candidates.length, 8);
});
