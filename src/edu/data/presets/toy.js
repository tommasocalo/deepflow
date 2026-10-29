/**
 * 2D toy datasets generated in the browser (docs/edu/SPEC.md §5.1–5.3, §6.9.3).
 */
import { GENERATORS } from '../generators.js';
import { resolveOptions } from '../options.js';
import { toyPythonLoader } from '../python/toy.js';

const PLAYGROUND = {
    licence: {
        name: 'Generated here; algorithm from TensorFlow Playground (Apache-2.0)',
        url: 'https://github.com/tensorflow/playground/blob/master/LICENSE',
        verified: true,
        note: 'The points are generated in your browser from the seed; nothing is downloaded.',
    },
    attribution: 'Generator adapted from TensorFlow Playground (Apache-2.0)',
    citation: ['D. Smilkov, S. Carter, D. Sculley, F. Viégas, M. Wattenberg (2016). TensorFlow Playground. https://playground.tensorflow.org'],
};

const CS231N = {
    licence: {
        name: 'Generated here; algorithm from CS231n (MIT)',
        url: 'https://github.com/cs231n/cs231n.github.io/blob/master/LICENSE',
        verified: true,
        note: 'The points are generated in your browser from the seed; nothing is downloaded.',
    },
    attribution: 'Generator adapted from CS231n (MIT)',
    citation: ['Stanford CS231n, "Neural Networks Case Study" (spiral data). https://cs231n.github.io/neural-networks-case-study/'],
};

const FEATURE_NAMES = ['x₁', 'x₂'];

const sizeOption = (value) => ({ default: value, min: 100, max: 1000, step: 50, label: 'Points' });
const noiseOption = (value) => ({ default: value, min: 0, max: 0.5, step: 0.05, label: 'Noise' });
const seedOption = { default: 1, min: 0, max: 9999, step: 1, label: 'Seed' };

function toySpec({ id, name, description, classes, regression = false, size, source, options }) {
    const spec = {
        id,
        name,
        description,
        kind: 'toy2d',
        training: 'browser',
        input: { shape: '(2,)', dtype: 'float32' },
        target: regression ? { kind: 'regression', size: 1 } : { kind: 'classes', classes, size: classes.length },
        size: { note: size },
        ...source,
        options,
        scaling: 'none',
        featureNames: FEATURE_NAMES,
        preview: 'scatter',
        /** Synchronous generation (toy sets take well under a millisecond). */
        loadSync(raw = {}) {
            const o = resolveOptions(spec, raw);
            const data = id === 'spiral3' ? GENERATORS[id](o.perClass, o.noise, o.seed) : GENERATORS[id](o.n, o.noise, o.seed);
            return { ...data, featureShape: [2], featureNames: FEATURE_NAMES };
        },
        async load(raw) {
            return spec.loadSync(raw);
        },
        python: (ctx = {}) => toyPythonLoader(spec, { ...ctx, options: resolveOptions(spec, ctx.options) }),
    };
    return spec;
}

export default [
    toySpec({
        id: 'blobs', name: 'Two blobs', classes: ['blue', 'orange'],
        description: 'Two round clouds of points, one per class. A straight line can separate them.',
        size: 'n points (you choose)', source: PLAYGROUND,
        options: { n: sizeOption(200), noise: noiseOption(0.1), seed: seedOption },
    }),
    toySpec({
        id: 'circle', name: 'Circle', classes: ['ring', 'disk'],
        description: 'An orange disk inside a blue ring. No straight line can separate them.',
        size: 'n points (you choose)', source: PLAYGROUND,
        options: { n: sizeOption(300), noise: noiseOption(0), seed: seedOption },
    }),
    toySpec({
        id: 'xor', name: 'XOR', classes: ['different signs', 'same sign'],
        description: 'A point is orange when x₁ and x₂ have the same sign, so the classes sit in opposite corners.',
        size: 'n points (you choose)', source: PLAYGROUND,
        options: { n: sizeOption(300), noise: noiseOption(0), seed: seedOption },
    }),
    toySpec({
        id: 'spiral2', name: 'Spiral (2 classes)', classes: ['arm A', 'arm B'],
        description: 'Two interleaved spiral arms. A small network needs many hidden units to separate them.',
        size: 'n points (you choose)', source: PLAYGROUND,
        options: { n: sizeOption(300), noise: noiseOption(0), seed: seedOption },
    }),
    toySpec({
        id: 'spiral3', name: 'Spiral (3 classes)', classes: ['arm A', 'arm B', 'arm C'],
        description: 'Three spiral arms from the CS231n course: a linear model gets about half right, a small MLP almost all.',
        size: '3 × points per class', source: CS231N,
        options: {
            perClass: { default: 100, min: 30, max: 300, step: 10, label: 'Points per class' },
            noise: noiseOption(0.2),
            seed: seedOption,
        },
    }),
    toySpec({
        id: 'plane', name: 'Plane (regression)', regression: true,
        description: 'Predict one number, (x₁ + x₂) / 10, for each point: a regression task.',
        size: 'n points (you choose)', source: PLAYGROUND,
        options: { n: sizeOption(300), noise: noiseOption(0), seed: seedOption },
    }),
];
