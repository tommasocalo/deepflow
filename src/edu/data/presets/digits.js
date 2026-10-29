/**
 * Digits 8×8 (docs/edu/SPEC.md §5.1, §5.2, §6.9.3). The pixel data is a separate chunk (dynamic import of
 * digitsData.js), so it stays out of the main bundle until the dataset is first used.
 */
import { digitsPythonLoader } from '../python/digits.js';

let decoded = null;
let pending = null;

/** Decode one base-36 character per pixel ('0'–'g' = 0–16); newlines in the source are not data. */
export function decodeDigits({ DIGITS_PIXELS, DIGITS_LABELS }) {
    const pixels = DIGITS_PIXELS.replace(/\s+/g, '');
    const n = DIGITS_LABELS.length;
    if (pixels.length !== n * 64) throw new Error(`Digits data is corrupt: ${pixels.length} pixels for ${n} labels`);
    const x = new Float64Array(n * 64);
    for (let i = 0; i < pixels.length; i++) x[i] = parseInt(pixels[i], 36);
    const y = new Int32Array(n);
    for (let i = 0; i < n; i++) y[i] = DIGITS_LABELS.charCodeAt(i) - 48;
    return { x, y, n, featureShape: [1, 8, 8], featureNames: null };
}

async function loadDigits() {
    if (decoded) return decoded;
    pending ??= import('./digitsData.js').then((module) => {
        decoded = decodeDigits(module);
        return decoded;
    }).finally(() => { pending = null; });
    return pending;
}

const digits = {
    id: 'digits',
    name: 'Digits 8×8',
    description: '1,797 tiny grey images of handwritten digits, 8 × 8 pixels each, in 10 classes.',
    kind: 'image',
    training: 'browser',
    input: { shape: '(1, 8, 8)', dtype: 'float32' },
    target: { kind: 'classes', classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], size: 10 },
    size: { total: 1797, note: '1,797 images, about 180 per digit' },
    licence: {
        name: 'BSD-3-Clause (copy distributed with scikit-learn)',
        url: 'https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits',
        verified: false,
        note: 'The UCI terms for the original data were not checked.',
    },
    attribution: 'E. Alpaydin, C. Kaynak (1998), UCI ML Repository; via scikit-learn',
    citation: [
        'Alpaydin, E. & Kaynak, C. (1998). Optical Recognition of Handwritten Digits. UCI Machine Learning Repository.',
        'scikit-learn, sklearn.datasets.load_digits (BSD-3-Clause).',
    ],
    options: { seed: { default: 1, min: 0, max: 9999, step: 1, label: 'Seed' } },
    scaling: 'divide16',
    featureNames: null,
    preview: 'images',
    /** Null until the data chunk has been loaded once. */
    loadSync: () => decoded,
    load: () => loadDigits(),
    python: (ctx = {}) => digitsPythonLoader(digits, ctx),
};

export default [digits];
