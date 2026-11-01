import assert from 'node:assert/strict';
import { test } from 'node:test';

// Answer-position bias (review r1): choice options are shown in a seeded per-learner order, stable across reloads;
// numbers ascend and yes/no reads yes/no. Across the curriculum the correct option no longer sits first by default.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { MODULES } = await import('../src/edu/lessons/modules/index.js');
await import('../src/edu/lessons/engine/registry.js');
const { optionOrder, quizOrder, predictionOrder, learnerSeed, setLearnerSeed, LEARNER_KEY } = await import('../src/edu/lessons/engine/order.js');
const { rawText } = await import('../src/edu/lessons/engine/i18n.js');

const items = () => MODULES.flatMap((module) => [
    ...[...module.warmup, ...module.quiz].map((item) => ({ id: item.id, options: item.options, correct: item.correct, order: (seed) => quizOrder(item, seed), text: (option) => rawText(`${item.id}.opt.${option}`) })),
    ...module.steps.filter((step) => step.predict?.kind === 'choice').map((step) => ({
        id: step.predict.id, options: step.predict.options, correct: step.predict.correct, order: (seed) => predictionOrder(step.predict, `${module.id}.${step.id}`, seed),
        text: (option) => rawText(`${module.id}.${step.id}.predict.opt.${option}`),
    })),
]);

test('the order is a permutation, the same for one learner, and differs between learners', () => {
    for (const item of items()) {
        const order = item.order(1234);
        assert.deepEqual([...order].sort(), [...item.options].sort(), `${item.id}: a permutation`);
        assert.deepEqual(item.order(1234), order, `${item.id}: stable for a seed`);
    }
    const shuffled = items().filter((item) => !item.options.every((option) => /^-?\d+(\.\d+)?$/.test(item.text(option) ?? '')) && item.options.length > 2);
    const differs = shuffled.filter((item) => JSON.stringify(item.order(1)) !== JSON.stringify(item.order(2)));
    assert.ok(differs.length >= shuffled.length / 2, `two learners see different orders (${differs.length} of ${shuffled.length})`);
});

test('numbers ascend and yes/no, true/false keep their reading order', () => {
    const m2 = MODULES.find((module) => module.id === 'm2');
    const width = m2.steps.find((step) => step.id === 's5').predict;
    assert.deepEqual(predictionOrder(width, 'm2.s5', 7), ['one', 'two', 'three', 'eight']);
    const stacked = m2.steps.find((step) => step.id === 's2').predict;
    assert.deepEqual(predictionOrder(stacked, 'm2.s2', 7), ['yes', 'no']);
    const m1 = MODULES.find((module) => module.id === 'm1');
    assert.deepEqual(quizOrder(m1.warmup[0], 7), ['true', 'false']);
    assert.deepEqual(optionOrder('x', ['b', 'a']), optionOrder('x', ['b', 'a']), 'two options are shuffled too, deterministically');
});

test('across the curriculum the correct option is no longer first by default', () => {
    for (const seed of [11, 222, 3333]) {
        const all = items();
        const first = all.filter((item) => item.order(seed)[0] === item.correct).length;
        // Authored: 50 of 62 first. A position-free order puts about one in three first.
        assert.ok(first <= all.length * 0.5, `seed ${seed}: correct first in ${first} of ${all.length}`);
    }
    const authored = items().filter((item) => item.options[0] === item.correct).length;
    assert.ok(authored > items().length * 0.6, `the authored order was biased (${authored} of ${items().length})`);
});

test('the learner seed is created once and kept across reloads', () => {
    memory.delete(LEARNER_KEY);
    setLearnerSeed(null);
    const seed = learnerSeed();
    assert.ok(Number.isInteger(seed));
    assert.equal(JSON.parse(memory.get(LEARNER_KEY)).seed, seed, 'stored in the profile');
    setLearnerSeed(null);
    assert.equal(learnerSeed(), seed, 'read back after a reload');
});
