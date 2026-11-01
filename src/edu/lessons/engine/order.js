import { rawText } from './i18n.js';
import { eduKey, readJSON, writeJSON } from './storage.js';

/**
 * The order in which a choice's options are shown (review r1: the correct option came first in 37 of 42 quiz and
 * warm-up items and in 13 of 20 choice predictions). Each learner gets a seeded shuffle that is stable across reloads:
 * the seed is drawn once per browser profile and kept under deepflow-edu:learner:v1. Two kinds of option sets keep a
 * natural order, which carries no answer: numbers ascend (1, 2, 3, 8; 0.0001, 0.01, 1.0), and yes/no and true/false
 * read in that order. The shown order is logged with every answer (session.js).
 */
export const LEARNER_KEY = eduKey('learner:v1');
const NATURAL_PAIRS = [['yes', 'no'], ['true', 'false']];

let cachedSeed = null;

/** This profile's shuffle seed (a 31-bit integer), created on first use. */
export function learnerSeed() {
    if (cachedSeed !== null) return cachedSeed;
    const saved = readJSON(LEARNER_KEY, null);
    if (Number.isInteger(saved?.seed)) {
        cachedSeed = saved.seed;
    } else {
        cachedSeed = Math.floor(Math.random() * 2 ** 31);
        writeJSON(LEARNER_KEY, { seed: cachedSeed, createdAt: Date.now() });
    }
    return cachedSeed;
}

/** Tests and the learning-log tooling: use a fixed seed (null forgets the cached one). */
export function setLearnerSeed(seed) {
    cachedSeed = Number.isInteger(seed) ? seed : null;
    if (cachedSeed !== null) writeJSON(LEARNER_KEY, { seed: cachedSeed, createdAt: Date.now() });
}

/** FNV-1a over the text, mixed with the seed: one stream per learner and item. */
function hash(seed, text) {
    let value = (2166136261 ^ seed) >>> 0;
    for (let index = 0; index < text.length; index++) {
        value ^= text.charCodeAt(index);
        value = Math.imul(value, 16777619) >>> 0;
    }
    return value;
}

/** mulberry32: a small deterministic PRNG in [0, 1). */
function random(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6D2B79F5) >>> 0;
        let mixed = Math.imul(state ^ (state >>> 15), 1 | state);
        mixed = (mixed + Math.imul(mixed ^ (mixed >>> 7), 61 | mixed)) ^ mixed;
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

/** A number written as an option ("8", "0.0001", "1.0"), or null for any other text. */
function numberOf(text) {
    if (typeof text !== 'string' || !/^\s*-?\d+(\.\d+)?\s*$/.test(text)) return null;
    return Number(text);
}

/**
 * Options in the order this learner sees them. `textKey(option)` names an option's text (numbers keep their order);
 * `seed` defaults to the profile's.
 * @param {string} itemId  the quiz item or prediction id, e.g. 'm2.q3' or 'm2.width'
 * @param {string[]} options
 */
export function optionOrder(itemId, options = [], textKey = null, seed = learnerSeed()) {
    const list = [...options];
    if (list.length < 2) return list;
    const pair = NATURAL_PAIRS.find((ids) => list.length === ids.length && list.every((option) => ids.includes(option)));
    if (pair) return [...pair];
    const numbers = textKey ? list.map((option) => numberOf(rawText(textKey(option)))) : [];
    if (numbers.length && numbers.every((number) => number !== null)) {
        return list.map((option, index) => ({ option, number: numbers[index] })).sort((left, right) => left.number - right.number).map((item) => item.option);
    }
    const next = random(hash(seed, itemId));
    for (let index = list.length - 1; index > 0; index--) {
        const swap = Math.floor(next() * (index + 1));
        [list[index], list[swap]] = [list[swap], list[index]];
    }
    return list;
}

/** The shown order of a quiz or warm-up item. */
export const quizOrder = (item, seed) => optionOrder(item.id, item.options, (option) => `${item.id}.opt.${option}`, seed);

/** The shown order of a choice prediction (`prefix` is the step's text prefix, e.g. 'm2.s5'). */
export const predictionOrder = (predict, prefix, seed) => optionOrder(predict.id, predict.options, (option) => `${prefix}.predict.opt.${option}`, seed);
