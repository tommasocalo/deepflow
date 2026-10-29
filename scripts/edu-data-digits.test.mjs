import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { statSync } from 'node:fs';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';
import { decodeDigits } from '../src/edu/data/presets/digits.js';
import { getDataset } from '../src/edu/data/registry.js';

// sha256 of scikit-learn's sklearn/datasets/data/digits.csv.gz after decompression.
const SKLEARN_DIGITS_CSV_SHA256 = '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8';
const dataFile = fileURLToPath(new URL('../src/edu/data/presets/digitsData.js', import.meta.url));

test('the digits data chunk loads through a dynamic import and stays under 120 KB', async () => {
    const digits = getDataset('digits');
    assert.equal(digits.loadSync(), null, 'nothing is decoded before the first load');
    const raw = await digits.load();
    assert.equal(digits.loadSync(), raw, 'later synchronous loads reuse the decoded data');
    assert.equal(raw.n, 1797);
    assert.deepEqual(raw.featureShape, [1, 8, 8]);
    assert.ok(statSync(dataFile).size < 120 * 1024, `digitsData.js is ${statSync(dataFile).size} bytes`);
});

test('1,797 images × 64 values in 0–16 with the pinned class counts', async () => {
    const raw = await getDataset('digits').load();
    assert.equal(raw.x.length, 1797 * 64);
    assert.ok(raw.x.every((value) => Number.isInteger(value) && value >= 0 && value <= 16));
    assert.equal(Math.max(...raw.x), 16);
    const counts = Array(10).fill(0);
    for (const label of raw.y) counts[label] += 1;
    assert.deepEqual(counts, [178, 182, 177, 183, 181, 182, 181, 179, 174, 180]);
});

test('re-serialising the images as scikit-learn digits.csv reproduces its sha256', async () => {
    const raw = await getDataset('digits').load();
    const lines = [];
    for (let i = 0; i < raw.n; i++) lines.push([...raw.x.slice(i * 64, (i + 1) * 64), raw.y[i]].join(','));
    assert.equal(createHash('sha256').update(lines.join('\n') + '\n').digest('hex'), SKLEARN_DIGITS_CSV_SHA256);
});

test('decodeDigits rejects corrupt data', () => {
    assert.throws(() => decodeDigits({ DIGITS_PIXELS: '0'.repeat(63), DIGITS_LABELS: '1' }), /corrupt/);
    const one = decodeDigits({ DIGITS_PIXELS: 'g'.repeat(64), DIGITS_LABELS: '7' });
    assert.equal(one.x[0], 16);
    assert.equal(one.y[0], 7);
});
