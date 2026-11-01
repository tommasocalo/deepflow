import assert from 'node:assert/strict';
import { beforeEach, test } from 'node:test';

// Training-graph edits are undo steps of the one document history (review r2: the M1 s6 loss type could not be
// undone from the lesson panel, and Ctrl+Z did nothing on the Training tab). Moving a training block is not an undo
// step, so an undo keeps the blocks where they are.

const saved = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => saved.get(key) ?? null,
    setItem: (key, value) => saved.set(key, value),
    removeItem: (key) => saved.delete(key),
} };
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTrainingStore: training } = await import('../src/stores/trainingStore.js');
const { useArchitectureHistory: history } = await import('../src/stores/architectureHistory.js');

const nodes = () => Object.values(training.getState().graph.nodes);
const loss = () => nodes().find((node) => node.kind === 'loss');

beforeEach(() => {
    training.getState().resetGraph();
    history.getState().clear();
});

test('adding and changing training blocks are undo steps; Undo and Redo restore them', () => {
    const lossId = training.getState().addLoss('bce', { x: 360, y: 100 }, null);
    const optimizerId = training.getState().addOptimizer('Adam', { x: 60, y: 100 });
    training.getState().connect({ source: lossId, target: optimizerId });
    assert.equal(history.getState().undoCount, 3);
    assert.equal(history.getState().undoLabel, 'Connect training blocks');
    training.getState().patchNode(lossId, { type: 'cross_entropy' });
    assert.equal(loss().type, 'cross_entropy');
    assert.equal(history.getState().undoLabel, 'Change training block');
    history.getState().undo();
    assert.equal(loss().type, 'bce', 'Undo restores the loss type');
    history.getState().redo();
    assert.equal(loss().type, 'cross_entropy');
    history.getState().undo();
    history.getState().undo();
    assert.equal(training.getState().graph.edges.length, 0, 'the connection is undone');
});

test('moving a training block is no undo step, and Undo keeps the block where it is', () => {
    const lossId = training.getState().addLoss('bce', { x: 360, y: 100 }, null);
    training.getState().patchNode(lossId, { type: 'mse' });
    const before = history.getState().undoCount;
    training.getState().moveNode(lossId, { x: 500, y: 240 });
    assert.equal(history.getState().undoCount, before);
    history.getState().undo();
    assert.equal(loss().type, 'bce');
    assert.deepEqual(loss().position, { x: 500, y: 240 });
});

test('a training edit and an architecture edit share one history, in order', () => {
    const lossId = training.getState().addLoss('bce', { x: 360, y: 100 }, null);
    const input = Object.values(useArchitectureStore.getState().graph.nodes)[0];
    useArchitectureStore.getState().moveNodes([{ id: input.id, position: { x: 999, y: 999 } }]);
    training.getState().patchNode(lossId, { type: 'cross_entropy' });
    history.getState().undo();
    assert.equal(loss().type, 'bce', 'the last edit, in the training graph, is undone first');
    assert.deepEqual(useArchitectureStore.getState().graph.nodes[input.id].position, { x: 999, y: 999 });
    history.getState().undo();
    assert.notDeepEqual(useArchitectureStore.getState().graph.nodes[input.id].position, { x: 999, y: 999 }, 'then the architecture edit');
});
