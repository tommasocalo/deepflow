import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { beforeEach, test } from 'node:test';

// Stage C presets (docs/edu/SPEC.md §5.1, §5.2, §6.9.1, §8.3): MNIST, Fashion-MNIST, CIFAR-10 and Tiny Shakespeare are
// export-only metadata with a Python loader. They never load data in the browser and show no sample data.

// In-memory storage before the stores hydrate (as edu-data-usedataset.test.mjs does).
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { DATASETS, getDataset, exportDatasets, parseInputShape, datasetSummary } = await import('../src/edu/data/registry.js');
const { applyDataset } = await import('../src/edu/data/useDataset.js');
const { useEduDataStore } = await import('../src/edu/data/store.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { restoreWorkspace } = await import('../src/edu/lessons/engine/workspace.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { pyTuple4 } = await import('../src/edu/data/presets/images.js');
const { TINY_SHAKESPEARE_URL } = await import('../src/edu/data/presets/text.js');

const EXPORT_IDS = ['mnist', 'fashion-mnist', 'cifar10', 'tiny-shakespeare'];
const IMAGE_FACTS = {
    mnist: { shape: [1, 28, 28], torchvision: 'MNIST', classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], train: 60000, test: 10000, mean: [0.1307], std: [0.3081], defaults: { epochs: 5, batchSize: 64 } },
    'fashion-mnist': { shape: [1, 28, 28], torchvision: 'FashionMNIST', classes: ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'], train: 60000, test: 10000, mean: [0.2860], std: [0.3530], defaults: { epochs: 5, batchSize: 64 } },
    cifar10: { shape: [3, 32, 32], torchvision: 'CIFAR10', classes: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], train: 50000, test: 10000, mean: [0.4914, 0.4822, 0.4465], std: [0.2470, 0.2435, 0.2616], defaults: { epochs: 10, batchSize: 128 } },
};
const python = spawnSync('python3', ['--version']).status === 0 ? 'python3' : null;

const graph = () => useArchitectureStore.getState().graph;

beforeEach(() => {
    memory.clear();
    useEduDataStore.setState({ datasetId: null, options: {}, split: { val: 0.2, test: 0 }, lastUse: null });
});

test('the registry lists the four export presets last, in the order of §6.9.1', () => {
    assert.deepEqual(DATASETS.slice(-4).map((spec) => spec.id), EXPORT_IDS);
    assert.deepEqual(exportDatasets().map((spec) => spec.id), EXPORT_IDS);
    assert.equal(new Set(DATASETS.map((spec) => spec.id)).size, DATASETS.length, 'ids are unique');
});

test('every export preset is metadata only: shapes, target, sizes, licence, source, and no browser data', () => {
    for (const id of EXPORT_IDS) {
        const spec = getDataset(id);
        assert.equal(spec.training, 'export', id);
        assert.equal(spec.load, undefined, `${id} has no load()`);
        assert.equal(spec.loadSync, undefined, `${id} has no loadSync()`);
        assert.equal(spec.preview, 'none', `${id} shows no sample data`);
        for (const key of ['samples', 'examples', 'previewData', 'thumbnails']) assert.equal(spec[key], undefined, `${id}.${key}`);
        assert.ok(parseInputShape(spec.input.shape), `${id} input shape parses`);
        assert.match(spec.description, /^[A-Z0-9].+\.$/, `${id} description is one sentence`);
        assert.match(spec.schematic, /^[A-Z0-9].+\.$/, `${id} schematic is a sentence`);
        assert.equal(typeof spec.licence.name, 'string');
        assert.equal(typeof spec.licence.verified, 'boolean');
        assert.ok(spec.attribution.length > 10);
        assert.ok(Array.isArray(spec.citation) && spec.citation.length > 0);
        assert.equal(typeof spec.python, 'function');
        assert.deepEqual(datasetSummary(spec).input, spec.input);
    }
});

test('image presets carry the right shapes, classes, official sizes and normalisation constants', () => {
    for (const [id, facts] of Object.entries(IMAGE_FACTS)) {
        const spec = getDataset(id);
        assert.equal(spec.kind, 'image');
        assert.deepEqual(parseInputShape(spec.input.shape), facts.shape, id);
        assert.equal(spec.input.dtype, 'float32');
        assert.deepEqual(spec.target, { kind: 'classes', classes: facts.classes, size: 10 }, id);
        assert.equal(spec.size.train, facts.train);
        assert.equal(spec.size.test, facts.test);
        assert.deepEqual(spec.normalization, { mean: facts.mean, std: facts.std }, id);
        assert.equal(spec.normalization.mean.length, facts.shape[0], 'one mean per channel');
        assert.deepEqual(spec.exportDefaults, facts.defaults, id);
        assert.equal(spec.options.valSize.default, 5000);
        assert.equal(spec.valSize, 5000);
    }
    // CIFAR-10 uses the per-channel std over all training pixels, not the per-image mean (0.2023, 0.1994, 0.2010).
    assert.notDeepEqual(getDataset('cifar10').normalization.std, [0.2023, 0.1994, 0.2010]);
    assert.equal(getDataset('fashion-mnist').licence.verified, true);
    assert.equal(getDataset('mnist').licence.verified, false, 'the MNIST licence was not verified (§5.2)');
    assert.equal(getDataset('cifar10').licence.verified, false);
});

test('Tiny Shakespeare is next-token data over 65 characters with the measured 90/10 split', () => {
    const spec = getDataset('tiny-shakespeare');
    assert.equal(spec.kind, 'text');
    assert.deepEqual(spec.input, { shape: '(64,)', dtype: 'int64' });
    assert.deepEqual(spec.target, { kind: 'next-token', size: 65 });
    assert.equal(spec.size.total, 1115394);
    assert.equal(spec.size.train, 1003854);
    assert.equal(spec.size.val, 111540);
    assert.equal(spec.size.test, undefined, 'there is no test split');
    assert.equal(spec.options.blockSize.default, 64);
    assert.deepEqual(spec.exportDefaults, { epochs: 10, batchSize: 32, stepsPerEpoch: 200 });
    assert.equal(TINY_SHAKESPEARE_URL, 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt');
});

test('the image loaders use torchvision with the official splits, a seeded random_split and Normalize', () => {
    for (const [id, facts] of Object.entries(IMAGE_FACTS)) {
        const loader = getDataset(id).python({ options: {}, split: { val: 0.2, test: 0 }, batchSize: 64, dataSeed: 1, trainSeed: 1 });
        assert.equal(loader.task, 'classification');
        assert.equal(loader.numClasses, 10);
        assert.deepEqual(loader.classNames, facts.classes);
        assert.ok(loader.imports.includes('from torchvision import datasets, transforms'));
        assert.match(loader.source, /^def build_dataloaders\(batch_size, data_seed, train_seed\):$/m);
        assert.ok(loader.source.includes(`datasets.${facts.torchvision}(root, train=True, download=True, transform=transform)`), id);
        assert.ok(loader.source.includes(`datasets.${facts.torchvision}(root, train=False, download=True, transform=transform)`), id);
        assert.ok(loader.source.includes(`transforms.Normalize(${pyTuple4(facts.mean)}, ${pyTuple4(facts.std)})`), id);
        assert.ok(loader.source.includes('random_split(train_full, [len(train_full) - VAL_SIZE, VAL_SIZE],'));
        assert.ok(loader.source.includes('generator=torch.Generator().manual_seed(data_seed)'));
        assert.ok(loader.source.includes('generator=torch.Generator().manual_seed(train_seed)'));
        assert.match(loader.source, /^VAL_SIZE = 5000 /m);
        assert.ok(loader.comment.includes(getDataset(id).attribution));
    }
    // The constants M3 asks about, exactly as a learner reads them.
    assert.ok(getDataset('mnist').python({}).source.includes('transforms.Normalize((0.1307,), (0.3081,))'));
    assert.ok(getDataset('cifar10').python({}).source.includes('transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))'));
    // valSize follows the option and is clamped to its range.
    assert.match(getDataset('mnist').python({ options: { valSize: 2000 } }).source, /^VAL_SIZE = 2000 /m);
    assert.match(getDataset('mnist').python({ options: { valSize: 999999 } }).source, /^VAL_SIZE = 10000 /m);
});

test('the Tiny Shakespeare loader cuts seeded windows of block_size characters with targets shifted by one', () => {
    const spec = getDataset('tiny-shakespeare');
    const loader = spec.python({ options: { blockSize: 64 }, split: {}, batchSize: 32, dataSeed: 1, trainSeed: 1 });
    assert.equal(loader.task, 'next-token');
    assert.equal(loader.vocabNote, '65 characters');
    assert.ok(loader.imports.includes('import urllib.request'));
    assert.ok(loader.source.includes(`DATA_URL = "${TINY_SHAKESPEARE_URL}"`));
    assert.match(loader.source, /^BLOCK_SIZE = 64 /m);
    assert.match(loader.source, /^TRAIN_BATCHES = 200 /m);
    assert.match(loader.source, /^VAL_BATCHES = 50$/m);
    assert.ok(loader.source.includes('self.chars = sorted(set(text))'));
    assert.ok(loader.source.includes('x = torch.stack([self.data[i:i + T] for i in starts])'));
    assert.ok(loader.source.includes('y = torch.stack([self.data[i + 1:i + T + 1] for i in starts])'));
    assert.ok(loader.source.includes('n = int(0.9 * len(data))'));
    assert.ok(loader.source.includes('return train_loader, val_loader, None'));
    assert.match(spec.python({ options: { blockSize: 128 } }).source, /^BLOCK_SIZE = 128 /m);
});

test('every export loader is valid Python', { skip: python ? false : 'python3 is not installed' }, () => {
    const directory = mkdtempSync(join(tmpdir(), 'edu-presets-'));
    try {
        const files = EXPORT_IDS.map((id) => {
            const loader = getDataset(id).python({ options: {}, split: {}, batchSize: 32, dataSeed: 1, trainSeed: 1 });
            const file = join(directory, `${id.replace(/-/g, '_')}.py`);
            writeFileSync(file, `${loader.imports.join('\n')}\n\n\n${loader.source}`);
            return file;
        });
        const result = spawnSync(python, ['-I', '-c', 'import py_compile, sys\nfor path in sys.argv[1:]:\n    py_compile.compile(path, cfile=path + "c", doraise=True)', ...files], { encoding: 'utf8' });
        assert.equal(result.status, 0, result.stderr);
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
});

/** in → Flatten → Linear(features → 2) → out, as a learner might start; Tiny Shakespeare gets an Embedding first. */
function loadModel(spec) {
    const h = createFixtureHelpers();
    const text = spec.kind === 'text';
    const input = h.input('in', text ? [8] : [2], { label: 'input' });
    const middle = text
        ? [h.layer('emb', 'torch.nn.Embedding', { num_embeddings: 65, embedding_dim: 16 })]
        : [h.layer('flat', 'torch.nn.Flatten')];
    const linear = h.linear('head', text ? 16 : 2, 2, { label: 'head' });
    const output = h.output('out');
    const nodes = [input, ...middle, linear, output];
    restoreWorkspace(h.snapshot({ nodes, edges: h.chain(...nodes), training: [h.optimizer('Adam'), h.loss('cross_entropy', output)] }));
}

test('Use this dataset binds each export preset and sets the Input shape and the last Linear\'s Output', () => {
    for (const id of EXPORT_IDS) {
        const spec = getDataset(id);
        loadModel(spec);
        const result = applyDataset(id);
        assert.equal(useEduDataStore.getState().datasetId, id);
        assert.deepEqual(result.warnings, [], id);
        const nodes = graph().nodes;
        assert.equal(nodes.in.parameters.shape, spec.input.shape, id);
        assert.equal(nodes.in.parameters.dtype, spec.input.dtype, id);
        assert.equal(nodes.head.parameterValues.out_features, spec.target.size, id);
        assert.ok(result.changed.some((line) => line.startsWith(`Input → ${spec.input.shape}`)), result.changed.join('; '));
    }
});
