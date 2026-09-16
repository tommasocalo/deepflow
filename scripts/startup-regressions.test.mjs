import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';
import { createDefaultModelSeeder } from '../src/storyboards/defaultModel.js';
import { gpt2Storyboard } from '../src/storyboards/gpt2.js';

const saved = new Map();
const writes = [];
const storage = {
    getItem: (key) => saved.get(key) ?? null,
    setItem: (key, value) => { writes.push(key); saved.set(key, String(value)); },
    removeItem: (key) => saved.delete(key),
};
globalThis.window = { localStorage: storage };

// Import the actual stores after installing storage. Their localStorage
// hydration runs synchronously, before Layout's mounting effect can seed.
const { useArchitectureStore: architectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: trainingStore } = await import('../src/stores/trainingStore.js');
const stores = { architectureStore, templatesStore, trainingStore };
const allStores = Object.values(stores);
const record = (state) => JSON.stringify({ state, version: 0 });
const emptyGraph = { nodes: {}, edges: [] };

beforeEach(() => {
    for (const store of allStores) store.setState(store.getInitialState(), true);
    saved.clear();
    writes.length = 0;
});

function hydrate(records) {
    for (const [key, value] of Object.entries(records)) saved.set(key, value);
    for (const store of allStores) store.persist.rehydrate();
}

function makeSeeder(options = {}) {
    return createDefaultModelSeeder({ seed: () => gpt2Storyboard(stores), ...options });
}

function snapshot() {
    return allStores.map((store) => store.getState());
}

function assertUnchanged(states, records) {
    allStores.forEach((store, index) => assert.equal(store.getState(), states[index]));
    assert.deepEqual(saved, records);
    assert.deepEqual(writes, [], 'startup must not write over an existing project');
}

test('fresh installation initializes GPT-2 once across repeated mount effects and remounts', () => {
    for (const store of allStores) assert.equal(store.persist.hasHydrated(), true);
    const seed = makeSeeder();
    assert.equal(seed(), true);
    const nodes = Object.values(architectureStore.getState().graph.nodes);
    assert.ok(nodes.some((node) => node.kind === 'repeat' && node.count === 12));
    assert.ok(Object.keys(templatesStore.getState().templates).length > 0);
    assert.ok(Object.values(trainingStore.getState().graph.nodes).some((node) => node.kind === 'optimizer'));
    assert.equal(saved.get('deepflow:seeded'), '1');

    const states = snapshot();
    const records = new Map(saved);
    writes.length = 0;
    assert.equal(seed(), false); // StrictMode's repeated effect.
    assert.equal(makeSeeder()(), false); // A new Layout/module instance.
    assertUnchanged(states, records);
});

test('a saved architecture without the new seed flag is hydrated and preserved', () => {
    const graph = {
        nodes: { custom: { id: 'custom', kind: 'layer', libraryKey: 'torch.nn.Linear', label: 'My saved layer', position: { x: 80, y: 90 }, parameters: { in_features: 3, out_features: 5 } } },
        edges: [],
    };
    hydrate({ 'deepflow:architecture-graph': record({ graph, currentPath: [] }) });
    assert.equal(architectureStore.persist.hasHydrated(), true);
    assert.deepEqual(architectureStore.getState().graph, graph);
    const states = snapshot();
    const records = new Map(saved);
    assert.equal(makeSeeder()(), false);
    assertUnchanged(states, records);
});

test('an intentionally empty saved architecture remains empty', () => {
    hydrate({ 'deepflow:architecture-graph': record({ graph: emptyGraph, currentPath: [] }) });
    const states = snapshot();
    const records = new Map(saved);
    assert.equal(makeSeeder()(), false);
    assert.deepEqual(architectureStore.getState().graph, emptyGraph);
    assertUnchanged(states, records);
});

for (const [name, records] of [
    ['training data alone', { 'deepflow:training-graph': record({ graph: { nodes: { savedOptimizer: { id: 'savedOptimizer', kind: 'optimizer', type: 'SGD' } }, edges: [] } }) }],
    ['templates alone', { 'deepflow:templates': record({ templates: { shared: { id: 'shared', name: 'My block', graph: emptyGraph } } }) }],
    ['an intentionally empty template library', { 'deepflow:templates': record({ templates: {} }) }],
    ['an intentionally empty training graph', { 'deepflow:training-graph': record({ graph: emptyGraph }) }],
    ['an existing seed flag alone', { 'deepflow:seeded': '1' }],
    ['malformed saved JSON', { 'deepflow:architecture-graph': '{broken' }],
    ['an empty saved record', { 'deepflow:templates': '' }],
]) {
    test(`startup preserves ${name}`, () => {
        hydrate(records);
        const states = snapshot();
        const before = new Map(saved);
        assert.equal(makeSeeder()(), false);
        assertUnchanged(states, before);
    });
}

test('viewport preferences alone do not prevent a fresh default model', () => {
    saved.set('deepflow:architecture-viewport:root', '{"x":0,"y":0,"zoom":1}');
    assert.equal(makeSeeder()(), true);
    assert.ok(Object.values(architectureStore.getState().graph.nodes).some((node) => node.kind === 'repeat'));
});

for (const [name, getStorage] of [
    ['storage access denied', () => { throw new Error('SecurityError'); }],
    ['storage reads denied', () => ({ ...storage, getItem: () => { throw new Error('SecurityError'); } })],
    ['storage writes denied', () => ({ ...storage, setItem: () => { throw new Error('QuotaExceededError'); } })],
]) {
    test(`${name} leaves all stores untouched`, () => {
        const states = snapshot();
        const records = new Map(saved);
        const seed = makeSeeder({ getStorage });
        assert.equal(seed(), false);
        assert.equal(seed(), false);
        assertUnchanged(states, records);
    });
}

test('an interrupted initialization stays usable and is not retried by StrictMode', () => {
    const failure = new Error('Storage became unavailable during initialization');
    const warnings = [];
    const originalWarn = console.warn;
    let calls = 0;
    console.warn = (...args) => warnings.push(args);
    try {
        const seed = makeSeeder({ seed: () => { calls++; throw failure; } });
        assert.equal(seed(), false);
        assert.equal(seed(), false);
        assert.equal(makeSeeder({ seed: () => { calls++; } })(), false);
        assert.equal(calls, 1);
        assert.equal(warnings.length, 1);
        assert.equal(warnings[0][1], failure);
    } finally {
        console.warn = originalWarn;
    }
});
