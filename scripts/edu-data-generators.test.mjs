import assert from 'node:assert/strict';
import { test } from 'node:test';
import { blobs, circle, GENERATORS, plane, spiral2, spiral3, xor } from '../src/edu/data/generators.js';
import { browserDatasets, getDataset } from '../src/edu/data/registry.js';

const count = (y, label) => y.filter((value) => value === label).length;
const points = ({ x, n }) => Array.from({ length: n }, (_, i) => [x[2 * i], x[2 * i + 1]]);

test('every generator returns flat 2D coordinates and one label per point', () => {
    const cases = [[blobs, 200], [circle, 300], [xor, 300], [spiral2, 300], [plane, 300]];
    for (const [generate, n] of cases) {
        const data = generate(n, 0.1, 1);
        assert.equal(data.n, n, generate.name);
        assert.ok(data.x instanceof Float64Array);
        assert.equal(data.x.length, n * 2);
        assert.equal(data.y.length, n);
        assert.ok(generate === plane ? data.y instanceof Float64Array : data.y instanceof Int32Array, generate.name);
        assert.ok([...data.x].every(Number.isFinite));
    }
    const spiral = spiral3(100, 0.2, 1);
    assert.equal(spiral.n, 300);
    assert.equal(spiral.x.length, 600);
    assert.ok(spiral.y instanceof Int32Array);
    assert.deepEqual(Object.keys(GENERATORS), ['blobs', 'circle', 'xor', 'spiral2', 'spiral3', 'plane']);
});

test('generators are deterministic per seed and differ across seeds', () => {
    for (const id of ['blobs', 'circle', 'xor', 'spiral2', 'spiral3', 'plane']) {
        const generate = GENERATORS[id];
        const size = id === 'spiral3' ? 50 : 200;
        const a = generate(size, 0.2, 11);
        const b = generate(size, 0.2, 11);
        const c = generate(size, 0.2, 12);
        assert.deepEqual(a.x, b.x, id);
        assert.deepEqual(a.y, b.y, id);
        assert.notDeepEqual(a.x, c.x, `${id} changes with the seed`);
    }
});

test('blobs: first half label 1 around (2, 2), second half label 0 around (−2, −2)', () => {
    const data = blobs(400, 0.1, 3);
    const y = [...data.y];
    assert.deepEqual(y.slice(0, 200), Array(200).fill(1));
    assert.deepEqual(y.slice(200), Array(200).fill(0));
    const pts = points(data);
    const mean = (rows, axis) => rows.reduce((sum, p) => sum + p[axis], 0) / rows.length;
    assert.ok(Math.abs(mean(pts.slice(0, 200), 0) - 2) < 0.3 && Math.abs(mean(pts.slice(0, 200), 1) - 2) < 0.3);
    assert.ok(Math.abs(mean(pts.slice(200), 0) + 2) < 0.3 && Math.abs(mean(pts.slice(200), 1) + 2) < 0.3);
    // variance = 0.5 + 3.5 · noise / 0.5 = 1.2 for noise 0.1
    const variance = pts.slice(0, 200).reduce((sum, p) => sum + (p[0] - 2) ** 2, 0) / 200;
    assert.ok(Math.abs(variance - 1.2) < 0.35, `variance ${variance}`);
    assert.equal(blobs(201, 0, 1).n, 200, 'odd n gives ⌊n/2⌋ per class');
});

test('circle labels follow the radius rule: a disk below 2.5 inside a ring from 3.5 to 5', () => {
    const data = circle(300, 0, 2);
    const pts = points(data);
    assert.equal(count([...data.y], 1), 150);
    assert.equal(count([...data.y], 0), 150);
    pts.forEach(([a, b], i) => {
        const r = Math.sqrt(a * a + b * b);
        assert.equal(data.y[i], r < 2.5 ? 1 : 0);
        if (i < 150) assert.ok(r <= 2.5);
        else assert.ok(r >= 3.5 && r <= 5);
    });
    const noisy = circle(300, 0.3, 2);
    const flipped = [...noisy.y].filter((label, i) => label !== (i < 150 ? 1 : 0)).length;
    assert.ok(flipped > 0, 'noise moves the labelling point, so some labels change');
});

test('xor coordinates keep 0.3 away from the axes and label the same-sign quadrants', () => {
    const data = xor(300, 0, 3);
    for (const [i, [a, b]] of points(data).entries()) {
        assert.ok(Math.abs(a) >= 0.3 && Math.abs(a) <= 5.3);
        assert.ok(Math.abs(b) >= 0.3 && Math.abs(b) <= 5.3);
        assert.equal(data.y[i], a * b >= 0 ? 1 : 0);
    }
    const ones = count([...data.y], 1);
    assert.ok(ones > 120 && ones < 180, `balanced quadrants: ${ones}`);
});

test('spiral2 arms stay within radius 5 and alternate labels by half', () => {
    const data = spiral2(300, 0, 4);
    assert.deepEqual([...data.y.slice(0, 150)], Array(150).fill(1));
    assert.deepEqual([...data.y.slice(150)], Array(150).fill(0));
    for (const [a, b] of points(data)) assert.ok(Math.sqrt(a * a + b * b) <= 5 + 1e-9);
    // Arm B is arm A rotated by π.
    const pts = points(data);
    assert.ok(Math.abs(pts[10][0] + pts[160][0]) < 1e-12 && Math.abs(pts[10][1] + pts[160][1]) < 1e-12);
});

test('spiral3 matches the CS231n formula on its noiseless points', () => {
    const perClass = 100;
    const data = spiral3(perClass, 0, 5);
    const pts = points(data);
    for (let j = 0; j < 3; j++) {
        for (let i = 0; i < perClass; i++) {
            const r = i / (perClass - 1);
            const t = 4 * j + 4 * i / (perClass - 1);
            const [a, b] = pts[j * perClass + i];
            assert.ok(Math.abs(a - r * Math.sin(t)) < 1e-12 && Math.abs(b - r * Math.cos(t)) < 1e-12);
            assert.equal(data.y[j * perClass + i], j);
        }
    }
    const noisy = points(spiral3(perClass, 0.2, 5));
    for (const [a, b] of noisy) assert.ok(Math.sqrt(a * a + b * b) <= 1 + 1e-9, 'noise moves the angle, not the radius');
});

test('plane targets are (x1 + x2) / 10 without noise and unclamped with noise', () => {
    const data = plane(300, 0, 6);
    for (const [i, [a, b]] of points(data).entries()) {
        assert.ok(Math.abs(a) <= 6 && Math.abs(b) <= 6);
        assert.ok(Math.abs(data.y[i] - (a + b) / 10) < 1e-12);
    }
    const noisy = plane(1000, 0.5, 6);
    assert.ok(Math.max(...noisy.y.map(Math.abs)) > 1, 'noise can push targets past ±1');
});

test('toy presets expose option schemas, generate through loadSync and report balanced classes', () => {
    for (const spec of browserDatasets().filter((item) => item.kind === 'toy2d')) {
        assert.equal(spec.input.shape, '(2,)');
        assert.equal(spec.preview, 'scatter');
        assert.ok(spec.options.seed && spec.options.noise);
        assert.ok(spec.attribution.includes('(Apache-2.0)') || spec.attribution.includes('(MIT)'));
        const data = spec.loadSync({});
        assert.deepEqual(data.featureShape, [2]);
        if (spec.target.kind === 'classes') {
            for (let label = 0; label < spec.target.size; label++) assert.ok(count([...data.y], label) > 0, `${spec.id} has class ${label}`);
        }
    }
    const spiral = getDataset('spiral3');
    assert.equal(spiral.loadSync({ perClass: 50 }).n, 150);
    assert.equal(getDataset('blobs').loadSync({ n: 5000 }).n, 1000, 'options are clamped to the schema');
});
