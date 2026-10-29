/**
 * Tabular datasets trained in the browser (docs/edu/SPEC.md §5.1, §5.2, §6.9.3): Iris.
 */
import { IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES, IRIS_ROWS } from './irisData.js';
import { irisPythonLoader } from '../python/iris.js';

let cached = null;
function irisRaw() {
    if (!cached) {
        const x = new Float64Array(IRIS_ROWS.length * 4);
        const y = new Int32Array(IRIS_ROWS.length);
        IRIS_ROWS.forEach((row, index) => {
            for (let f = 0; f < 4; f++) x[index * 4 + f] = row[f];
            y[index] = row[4];
        });
        cached = { x, y, n: IRIS_ROWS.length, featureShape: [4], featureNames: IRIS_FEATURE_NAMES };
    }
    return cached;
}

const iris = {
    id: 'iris',
    name: 'Iris',
    description: '150 flowers of 3 species. A botanist measured four lengths of each flower in centimetres.',
    kind: 'tabular',
    training: 'browser',
    input: { shape: '(4,)', dtype: 'float32' },
    target: { kind: 'classes', classes: IRIS_CLASS_NAMES, size: 3 },
    size: { total: 150, note: '150 flowers, 50 per species' },
    licence: {
        name: 'Public-domain measurements (Fisher, 1936); file from scikit-learn (BSD-3-Clause)',
        url: 'https://github.com/scikit-learn/scikit-learn/blob/main/COPYING',
        verified: true,
        note: 'Values as in scikit-learn load_iris(). Not the UCI copy, which has two wrong rows.',
    },
    attribution: 'Fisher, R.A. (1936) The use of multiple measurements in taxonomic problems. Values as in scikit-learn\'s load_iris().',
    citation: [
        'Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. Annals of Eugenics, 7(2), 179–188.',
        'scikit-learn, sklearn.datasets.load_iris (BSD-3-Clause).',
    ],
    options: { seed: { default: 1, min: 0, max: 9999, step: 1, label: 'Seed' } },
    scaling: 'standardize',
    featureNames: IRIS_FEATURE_NAMES,
    preview: 'tabular',
    loadSync: () => irisRaw(),
    load: async () => irisRaw(),
    python: (ctx = {}) => irisPythonLoader(iris, ctx),
};

export default [iris];
