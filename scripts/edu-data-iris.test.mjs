import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { IRIS_CLASS_NAMES, IRIS_FEATURE_NAMES, IRIS_ROWS } from '../src/edu/data/presets/irisData.js';
import { getDataset } from '../src/edu/data/registry.js';

const SKLEARN_IRIS_SHA256 = 'f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449';

test('Iris has 150 rows, 4 features and 50 flowers per species', () => {
    assert.equal(IRIS_ROWS.length, 150);
    assert.ok(IRIS_ROWS.every((row) => row.length === 5 && row.slice(0, 4).every((value) => value > 0 && value < 8)));
    assert.deepEqual([0, 1, 2].map((label) => IRIS_ROWS.filter((row) => row[4] === label).length), [50, 50, 50]);
    assert.deepEqual(IRIS_CLASS_NAMES, ['setosa', 'versicolor', 'virginica']);
    assert.equal(IRIS_FEATURE_NAMES.length, 4);
});

test('rows 35 and 38 hold the corrected values of scikit-learn, not the UCI typos', () => {
    assert.deepEqual(IRIS_ROWS[34], [4.9, 3.1, 1.5, 0.2, 0]); // UCI: 4.9, 3.1, 1.5, 0.1
    assert.deepEqual(IRIS_ROWS[37], [4.9, 3.6, 1.4, 0.1, 0]); // UCI: 4.9, 3.1, 1.5, 0.1
});

test('re-serialising the rows as scikit-learn iris.csv reproduces its sha256', () => {
    const csv = '150,4,setosa,versicolor,virginica\n'
        + IRIS_ROWS.map((row) => `${row.slice(0, 4).map((value) => value.toFixed(1)).join(',')},${row[4]}`).join('\n') + '\n';
    assert.equal(createHash('sha256').update(csv).digest('hex'), SKLEARN_IRIS_SHA256);
});

test('the Iris preset loads the rows and standardises with training statistics', () => {
    const iris = getDataset('iris');
    assert.equal(iris.input.shape, '(4,)');
    assert.equal(iris.scaling, 'standardize');
    assert.equal(iris.target.size, 3);
    const raw = iris.loadSync({});
    assert.equal(raw.n, 150);
    assert.equal(raw.x[34 * 4 + 3], 0.2);
    assert.match(iris.attribution, /Fisher, R\.A\. \(1936\)/);
    assert.match(iris.licence.note, /UCI/);
});
