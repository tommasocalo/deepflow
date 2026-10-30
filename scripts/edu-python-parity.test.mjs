import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { getDataset } from '../src/edu/data/registry.js';
import { pyGeneratorCall } from '../src/edu/data/python/toy.js';
import { splitIndices } from '../src/edu/data/split.js';
import { resolveOptions } from '../src/edu/data/options.js';

// The exported loaders must regenerate the browser's toy points and every split exactly (docs/edu/SPEC.md §5.3).
// Runs the real loader source with `python3 -I` (or EDU_PYTHON); torch is not needed because build_dataloaders is
// never called.
const PYTHON = process.env.EDU_PYTHON || 'python3';
const probe = spawnSync(PYTHON, ['-I', '-c', 'import numpy'], { encoding: 'utf8' });
const skip = probe.status === 0 ? false : 'needs python3 with numpy';

function runPython(source, body) {
    const program = `import json\nimport math\nimport numpy as np\n\n${source}\n${body}\n`;
    return JSON.parse(execFileSync(PYTHON, ['-I', '-c', program], { encoding: 'utf8', maxBuffer: 1 << 26 }));
}

const sameIndices = (js, py) => JSON.stringify([...js]) === JSON.stringify(py);

for (const id of ['blobs', 'circle', 'xor', 'spiral2', 'spiral3', 'plane']) {
    test(`${id}: the exported generator and split match the browser (2 seeds)`, { skip }, () => {
        const spec = getDataset(id);
        for (const seed of [1, 9]) {
            const options = resolveOptions(spec, { seed, noise: 0.25 });
            const split = { val: 0.2, test: 0.1 };
            const loader = spec.python({ options, split, batchSize: 32, dataSeed: seed, trainSeed: 1 });
            assert.match(loader.source, /def build_dataloaders\(batch_size, data_seed, train_seed\):/);
            const out = runPython(loader.source, `X, y = ${pyGeneratorCall(id, options, String(seed))}
train_idx, val_idx, test_idx = split_indices(len(y), ${split.val}, ${split.test}, ${seed})
print(json.dumps({"x": X.ravel().tolist(), "y": y.tolist(), "train": train_idx, "val": val_idx, "test": test_idx}))`);
            const js = spec.loadSync(options);
            assert.equal(out.x.length, js.x.length);
            let maxDelta = 0;
            for (let i = 0; i < js.x.length; i++) maxDelta = Math.max(maxDelta, Math.abs(js.x[i] - out.x[i]));
            assert.ok(maxDelta <= 1e-12, `${id} seed ${seed}: max |Δ| ${maxDelta}`);
            if (spec.target.kind === 'regression') {
                for (let i = 0; i < js.y.length; i++) assert.ok(Math.abs(js.y[i] - out.y[i]) <= 1e-12);
            } else {
                assert.deepEqual([...js.y], out.y, `${id} seed ${seed}: labels`);
            }
            const parts = splitIndices(js.n, split, seed);
            assert.ok(sameIndices(parts.train, out.train) && sameIndices(parts.val, out.val) && sameIndices(parts.test, out.test), `${id} seed ${seed}: split`);
        }
    });
}

for (const [id, n] of [['iris', 150], ['digits', 1797]]) {
    test(`${id}: the exported split_indices gives the browser's split`, { skip }, () => {
        const spec = getDataset(id);
        for (const [seed, split] of [[1, { val: 0.2, test: 0 }], [3, { val: 0.3, test: 0.1 }]]) {
            const loader = spec.python({ options: { seed }, split, batchSize: 32, dataSeed: seed, trainSeed: 1 });
            const out = runPython(loader.source, `train_idx, val_idx, test_idx = split_indices(${n}, ${split.val}, ${split.test}, ${seed})
print(json.dumps({"train": train_idx, "val": val_idx, "test": test_idx}))`);
            const parts = splitIndices(n, split, seed);
            assert.ok(sameIndices(parts.train, out.train), `${id} seed ${seed}: train`);
            assert.ok(sameIndices(parts.val, out.val), `${id} seed ${seed}: val`);
            assert.ok(sameIndices(parts.test, out.test), `${id} seed ${seed}: test`);
        }
    });
}

test('the Python PRNG port reproduces the pinned vectors', { skip }, () => {
    const loader = getDataset('iris').python({ split: { val: 0.2, test: 0 } });
    const out = runPython(loader.source, `r = mulberry32(42)
print(json.dumps({"r": [r() for _ in range(5)], "seeds": [derive_seed(1, "data"), derive_seed(1, "split"), derive_seed(5, "init")]}))`);
    assert.deepEqual(out.r, [0.6011037519201636, 0.44829055899754167, 0.8524657934904099, 0.6697340414393693, 0.17481389874592423]);
    assert.deepEqual(out.seeds, [2212395750, 3534800360, 2825648146]);
});
