import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

// Stage C code export (docs/edu/SPEC.md §6.11, §8.3): train.py for every dataset, the legacy script without a dataset,
// the cross-entropy fix for (B, T, V) outputs, label keys, and the .ipynb.
//
// Python checks use python3 (or EDU_PYTHON). Scripts are compiled when Python exists; they are run only when torch,
// numpy and scikit-learn import, e.g. EDU_PYTHON=/path/to/venv/bin/python npm test.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { generateTrainingScript } = await import('../src/codegen/index.js');
const { COLAB_NOTE, PHASES_COMMENT, scriptSettings } = await import('../src/codegen/edu/trainScript.js');
const { stripMainGuard, trainingNotebook, trainingNotebookText } = await import('../src/codegen/edu/notebook.js');
const { DATASETS, getDataset, defaultOptions } = await import('../src/edu/data/registry.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { PRODUCT_NAME } = await import('../src/edu/config.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore } = await import('../src/stores/trainingStore.js');
const { STORYBOARDS } = await import('../src/storyboards/index.js');
const { gpt2Storyboard } = await import('../src/storyboards/gpt2.js');

const PYTHON = process.env.EDU_PYTHON || 'python3';
const hasPython = spawnSync(PYTHON, ['--version']).status === 0;
const hasTorch = hasPython && spawnSync(PYTHON, ['-I', '-c', 'import torch, numpy, sklearn']).status === 0;
const sha256 = (text) => createHash('sha256').update(text).digest('hex');

/**
 * sha256 of generateTrainingScript(graph, training, { templates }) for DeepFlow's storyboards on the fork point
 * 15c7c25 (built in this order, in one process, with `git archive 15c7c25 src`).
 */
const LEGACY_15C7C25 = {
    gpt2: '9db830bcbe2d546b93e13976b55faed7a40ef2d18fab70822125164a464f194b',
    resnet: '6235a58fdf35b200f38f339fcb08cf6603f25a97c465bcd484f0ec27377b0636',
    dann: '19c190b91e1b6534a153c025f974f01ce38554b297311bdfcb78875bf3d300bc',
    gan: '2156835cfd472938c79e545f002b16d6626ac240d8320f3b6b319b7a5572b341',
};
const GPT2_FIXED_LINE = 'loss_total = next_token_loss(logits.flatten(0, -2), y.flatten()) * 1';
const GPT2_OLD_LINE = 'loss_total = next_token_loss(logits, y) * 1';
const stores = { architectureStore: useArchitectureStore, templatesStore: useTemplatesStore, trainingStore: useTrainingStore };
const current = () => ({ graph: useArchitectureStore.getState().graph, training: useTrainingStore.getState().graph, templates: useTemplatesStore.getState().templates });
const legacyScript = ({ graph, training, templates }) => generateTrainingScript(graph, training, { templates });

/** A small model that fits the dataset, built with Stage A's fixture helpers (no stores). */
function modelFor(spec, { loss: lossOverride, outputs } = {}) {
    const h = createFixtureHelpers();
    const L = (id, key, params) => h.layer(id, key, params);
    let nodes;
    let loss = 'cross_entropy';
    if (spec.kind === 'toy2d') {
        loss = spec.target.kind === 'regression' ? 'mse' : spec.target.size === 2 ? 'bce' : 'cross_entropy';
        if (lossOverride) loss = lossOverride;
        const width = outputs ?? (loss === 'cross_entropy' ? spec.target.size : 1);
        nodes = [h.input('in', [2]), h.linear('lin1', 2, 8), L('relu', 'torch.nn.ReLU'), h.linear('lin2', 8, width), h.output('out')];
    } else if (spec.id === 'iris') {
        nodes = [h.input('in', [4]), h.linear('lin', 4, 3), h.output('out')];
    } else if (spec.id === 'digits' || spec.id === 'mnist') {
        const pixels = spec.id === 'digits' ? 64 : 784;
        nodes = [h.input('in', spec.id === 'digits' ? [1, 8, 8] : [1, 28, 28]), L('flat', 'torch.nn.Flatten'), h.linear('lin1', pixels, 32),
            L('relu', 'torch.nn.ReLU'), h.linear('lin2', 32, 10), h.output('out')];
    } else if (spec.id === 'fashion-mnist') {
        nodes = [h.input('in', [1, 28, 28]), L('c1', 'torch.nn.Conv2d', { in_channels: 1, out_channels: 8, kernel_size: 3, padding: 1 }), L('r1', 'torch.nn.ReLU'),
            L('p1', 'torch.nn.MaxPool2d', { kernel_size: 2 }), L('flat', 'torch.nn.Flatten'), L('fc', 'torch.nn.Linear', { in_features: 8 * 14 * 14, out_features: 10 }), h.output('out')];
    } else if (spec.id === 'cifar10') {
        nodes = [h.input('in', [3, 32, 32]), L('c1', 'torch.nn.Conv2d', { in_channels: 3, out_channels: 16, kernel_size: 3, padding: 1 }), L('r1', 'torch.nn.ReLU'),
            L('pool', 'torch.nn.AdaptiveAvgPool2d', { output_size: 1 }), L('flat', 'torch.nn.Flatten'), L('fc', 'torch.nn.Linear', { in_features: 16, out_features: 10 }), h.output('out')];
    } else if (spec.id === 'tiny-shakespeare') {
        nodes = [h.input('in', [64], { dtype: 'int64' }), L('emb', 'torch.nn.Embedding', { num_embeddings: 65, embedding_dim: 32 }),
            L('fc', 'torch.nn.Linear', { in_features: 32, out_features: 65 }), h.output('out')];
    } else {
        throw new Error(`No test model for ${spec.id}`);
    }
    const output = nodes.at(-1);
    return h.snapshot({ nodes, edges: h.chain(...nodes), training: [h.optimizer('Adam', { learning_rate: 0.01 }), h.loss(loss, output)] });
}

function dataFor(spec, overrides = {}) {
    return { spec, options: defaultOptions(spec), split: { val: 0.2, test: 0 }, dataSeed: 1, settings: { epochs: 3, batchSize: 16, seed: 2 }, ...overrides };
}

function eduScript(spec, { model = modelFor(spec), ...overrides } = {}) {
    return generateTrainingScript(model.graph, model.training, { templates: model.templates, data: dataFor(spec, overrides) });
}

/** Compile Python sources with py_compile; returns null or the error text. */
function compileAll(named) {
    const directory = mkdtempSync(join(tmpdir(), 'edu-codegen-'));
    try {
        const files = Object.entries(named).map(([name, source]) => {
            const file = join(directory, `${name.replace(/[^A-Za-z0-9_]/g, '_')}.py`);
            writeFileSync(file, source);
            return file;
        });
        const program = 'import py_compile, sys\nfor path in sys.argv[1:]:\n    py_compile.compile(path, cfile=path + "c", doraise=True)';
        const result = spawnSync(PYTHON, ['-I', '-c', program, ...files], { encoding: 'utf8' });
        return result.status === 0 ? null : result.stderr;
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
}

/** Run a script in a fresh working directory; `files` are written there first. */
function runPython(source, { files = {}, args = [], timeout = 600000 } = {}) {
    const directory = mkdtempSync(join(tmpdir(), 'edu-run-'));
    try {
        for (const [name, text] of Object.entries(files)) {
            mkdirSync(join(directory, name, '..'), { recursive: true });
            writeFileSync(join(directory, name), text);
        }
        writeFileSync(join(directory, 'script.py'), source);
        return spawnSync(PYTHON, ['-I', 'script.py', ...args], { cwd: directory, encoding: 'utf8', timeout });
    } finally {
        rmSync(directory, { recursive: true, force: true });
    }
}

const LOADER_MARKERS = {
    blobs: 'X, y = generate_blobs(200, 0.1, data_seed)',
    circle: 'X, y = generate_circle(300, 0, data_seed)',
    xor: 'X, y = generate_xor(300, 0, data_seed)',
    spiral2: 'X, y = generate_spiral2(300, 0, data_seed)',
    spiral3: 'X, y = generate_spiral3(100, 0.2, data_seed)',
    plane: 'X, y = generate_plane(300, 0, data_seed)',
    iris: 'X, y = load_iris(return_X_y=True)',
    digits: 'X, y = load_digits(return_X_y=True)',
    mnist: 'datasets.MNIST(root, train=True, download=True, transform=transform)',
    'fashion-mnist': 'datasets.FashionMNIST(root, train=True, download=True, transform=transform)',
    cifar10: 'datasets.CIFAR10(root, train=True, download=True, transform=transform)',
    'tiny-shakespeare': 'urllib.request.urlretrieve(DATA_URL, path)',
};
const COMPUTE_LOSS = {
    bce: 'return loss_fn(logits, y.float().view_as(logits))',
    mse: 'return loss_fn(logits, y.float().view_as(logits))',
    cross_entropy: 'return loss_fn(logits, y)',
    'next-token': 'return loss_fn(logits.flatten(0, -2), y.flatten())',
};

// ---------------------------------------------------------------------------------------------------------------
// Legacy output (no dataset bound)
// ---------------------------------------------------------------------------------------------------------------

test('without a dataset, train.py is the fork\'s output for ResNet, DANN and GAN, and GPT-2 differs only by the loss fix', () => {
    const outputs = {};
    for (const id of ['gpt2', 'resnet', 'dann', 'gan']) {
        STORYBOARDS.find((entry) => entry.id === id).run(stores);
        outputs[id] = legacyScript(current());
        assert.equal(generateTrainingScript(current().graph, current().training, { templates: current().templates, data: undefined }), outputs[id]);
        assert.equal(generateTrainingScript(current().graph, current().training, { templates: current().templates, data: null }), outputs[id]);
    }
    for (const id of ['resnet', 'dann', 'gan']) assert.equal(sha256(outputs[id]), LEGACY_15C7C25[id], `${id} is byte-identical to 15c7c25`);
    assert.equal(outputs.gpt2.split(GPT2_FIXED_LINE).length, 2, 'the GPT-2 step flattens (B, T, V) for cross-entropy');
    assert.equal(sha256(outputs.gpt2.replace(GPT2_FIXED_LINE, GPT2_OLD_LINE)), LEGACY_15C7C25.gpt2, 'and nothing else changed');
    // ResNet ends in a (B, 256, 32, 32) feature map: PyTorch's own per-pixel cross-entropy applies, unflattened.
    assert.ok(outputs.resnet.includes('loss_total = Cross_Entropy_loss(logits, y) * 1'));
});

test('cross-entropy flattens a language model\'s (B, T, V) scores in the legacy step and in phases', () => {
    gpt2Storyboard({ ...stores, configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4, nLayers: 2, mlpWidth: 256 }, includeTraining: true });
    assert.ok(legacyScript(current()).includes(GPT2_FIXED_LINE));

    // Scores from a Linear on the last axis of a float sequence: (B, 8, 5).
    const h = createFixtureHelpers();
    const nodes = [h.input('in', [8, 16]), h.linear('head', 16, 5), h.output('out')];
    const output = nodes.at(-1);
    const phased = {
        nodes: {
            ...h.training(h.optimizer('Adam', {}, { id: 'g' }), h.loss('cross_entropy', output, { id: 'l1', label: 'token loss' })).nodes,
            phase: { id: 'phase', kind: 'phase', label: 'warm-up', position: { x: 0, y: 0 }, optimizerIds: ['g'], lossIds: ['l1'], everyNSteps: 2 },
        },
        edges: [],
    };
    const snapshot = h.snapshot({ nodes, edges: h.chain(...nodes), training: phased });
    const script = generateTrainingScript(snapshot.graph, snapshot.training, { templates: snapshot.templates });
    assert.ok(script.includes('loss_total = token_loss(logits.flatten(0, -2), y.flatten()) * 1'), script.slice(-900));
    assert.ok(script.includes('if global_step % 2 == 0:'));
});

test('each loss reads its own label key from the batch', () => {
    const h = createFixtureHelpers();
    const nodes = [h.input('in', [4]), h.linear('lin', 4, 3), h.output('out')];
    const loss = { ...h.loss('cross_entropy', nodes.at(-1)), labelKey: 'species' };
    const snapshot = h.snapshot({ nodes, edges: h.chain(...nodes), training: [h.optimizer('SGD'), loss] });
    const script = generateTrainingScript(snapshot.graph, snapshot.training, { templates: snapshot.templates });
    assert.ok(script.includes('loss_total = Cross_Entropy_loss(logits, batch["species"].to(device)) * 1'), script.slice(-700));
    assert.ok(script.includes('y = batch.get("labels")'), 'the default key is read as before');
});

// ---------------------------------------------------------------------------------------------------------------
// train.py with a dataset
// ---------------------------------------------------------------------------------------------------------------

test('every registered dataset gets a complete train.py: loader, constants, loss, accuracy, evaluation, header', () => {
    assert.equal(DATASETS.length, 12);
    for (const spec of DATASETS) {
        const script = eduScript(spec);
        const id = spec.id;
        assert.ok(script.includes(LOADER_MARKERS[id]), `${id}: loader`);
        assert.match(script, /^def build_dataloaders\(batch_size, data_seed, train_seed\):$/m, id);
        assert.ok(script.includes('train_loader, val_loader, test_loader = build_dataloaders(BATCH_SIZE, DATA_SEED, SEED)'), id);
        for (const constant of ['SEED = ', 'DATA_SEED = ', 'EPOCHS = ', 'BATCH_SIZE = ', 'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"']) {
            assert.match(script, new RegExp(`^${constant.replace(/[().]/g, '\\$&')}`, 'm'), `${id}: ${constant}`);
        }
        assert.ok(script.includes('model.eval()'), `${id}: model.eval()`);
        assert.ok(script.includes('with torch.no_grad():'), `${id}: torch.no_grad()`);
        assert.ok(script.includes('model.train()'), `${id}: model.train()`);
        assert.ok(script.includes('def make_optimizer(model):') && script.includes('torch.optim.Adam(model.parameters(), lr=0.01'), `${id}: optimizer`);
        const kind = spec.kind === 'text' ? 'next-token' : spec.kind === 'toy2d' ? (spec.target.kind === 'regression' ? 'mse' : spec.target.size === 2 ? 'bce' : 'cross_entropy') : 'cross_entropy';
        assert.ok(script.includes(COMPUTE_LOSS[kind]), `${id}: compute_loss for ${kind}`);
        if (spec.target.kind === 'regression') {
            assert.ok(!script.includes('def accuracy('), `${id}: no accuracy for regression`);
            assert.ok(script.includes('return total_loss / count, None'));
        } else {
            assert.ok(script.includes('def accuracy(logits, y):'), `${id}: accuracy`);
            assert.ok(script.includes('val_acc {val_acc:.3f}'), `${id}: prints val_acc`);
        }
        assert.ok(script.includes('print(f"epoch {epoch:3d}  train_loss {total / count:.4f}  val_loss {val_loss:.4f}'), `${id}: epoch line`);
        assert.ok(script.includes('best epoch: {best_epoch}'), `${id}: best epoch`);
        assert.ok(script.includes('if test_loader is not None:'), `${id}: test once`);
        assert.ok(script.endsWith('\n\n\nif __name__ == "__main__":\n    train()\n'), `${id}: main guard`);
        // Header: dataset, licence, attribution, Colab.
        const header = script.slice(0, script.indexOf('import torch'));
        assert.ok(header.includes('Runs in Google Colab: upload this file and run `!python train.py`'), `${id}: Colab line`);
        assert.ok(header.replace(/\n#\s+/g, ' ').includes(COLAB_NOTE.slice(COLAB_NOTE.indexOf('(GPU'))), `${id}: GPU note`);
        assert.ok(header.includes(PRODUCT_NAME));
        assert.ok(header.replace(/\n#\s+/g, ' ').includes(spec.attribution), `${id}: attribution`);
        assert.ok(header.replace(/\n#\s+/g, ' ').includes(`Licence: ${spec.licence.name}`), `${id}: licence`);
        assert.ok(!header.includes('Check before running'), `${id}: the test model fits the dataset`);
        for (const line of script.split('\n')) assert.ok(!line.includes('\t'), `${id}: no tabs`);
    }
});

test('the initial validation loss is compared with ln C, and next-token data prints a sample', () => {
    assert.ok(eduScript(getDataset('iris')).includes('(guessing evenly among 3 classes gives ln 3 = {math.log(3):.4f})'));
    assert.ok(eduScript(getDataset('mnist')).includes('(guessing evenly among 10 classes gives ln 10 = {math.log(10):.4f})'));
    assert.ok(eduScript(getDataset('blobs')).includes('gives ln 2 = {math.log(2):.4f})'));
    const text = eduScript(getDataset('tiny-shakespeare'));
    assert.ok(text.includes('(guessing evenly among 65 tokens gives ln 65 = {math.log(65):.4f})'));
    assert.ok(text.includes('def generate(model, vocab, start="\\n", length=300, temperature=0.8):'));
    assert.ok(text.includes('scores = model(ids[:, -BLOCK_SIZE:])[:, -1, :]'), 'the context is cropped to the model\'s T');
    assert.ok(text.includes('print(generate(model, train_loader.vocab, start="\\n", length=300, temperature=0.8))'));
    assert.ok(text.includes('(logits.argmax(dim=-1) == y).float().mean().item()'), 'per-position accuracy');
    assert.ok(!eduScript(getDataset('plane')).includes(' gives ln '), 'no expected loss for regression');
});

test('export datasets train with their Colab defaults, browser datasets with the trainer settings', () => {
    const settings = { epochs: 7, batchSize: 16, seed: 3 };
    const toy = eduScript(getDataset('circle'), { settings });
    assert.match(toy, /^EPOCHS = 7$/m);
    assert.match(toy, /^BATCH_SIZE = 16$/m);
    assert.match(toy, /^SEED = 3 /m);
    for (const [id, epochs, batch] of [['mnist', 5, 64], ['fashion-mnist', 5, 64], ['cifar10', 10, 128], ['tiny-shakespeare', 10, 32]]) {
        const script = eduScript(getDataset(id), { settings });
        const name = getDataset(id).name;
        assert.ok(script.includes(`\nEPOCHS = ${epochs}  # ${name}'s preset for Colab (the Training tab says 7); change it freely\n`), id);
        assert.ok(script.includes(`\nBATCH_SIZE = ${batch}  # ${name}'s preset for Colab (the Training tab says 16)\n`), id);
        assert.match(script, /^SEED = 3 /m, 'the trainer seed still applies');
    }
    const same = eduScript(getDataset('mnist'), { settings: { epochs: 5, batchSize: 64, seed: 3 } });
    assert.ok(same.includes("\nEPOCHS = 5  # MNIST's preset for Colab; change it freely\n"), 'no Training tab value when it matches');
    assert.ok(same.includes("\nBATCH_SIZE = 64  # MNIST's preset for Colab\n"));
    assert.deepEqual(scriptSettings({ spec: getDataset('blobs') }), { epochs: 50, batchSize: 32, seed: 1 });
    assert.match(eduScript(getDataset('spiral3'), { dataSeed: 42, options: { perClass: 100, noise: 0.2, seed: 42 } }), /^DATA_SEED = 42 /m);
});

test('the header and the loop say what differs from the browser: weights, batch order, train_loss and export presets', () => {
    const unwrap = (script) => script.slice(0, script.indexOf('import torch')).replace(/\n#\s+/g, ' ');
    const loop = '        # train_loss: the running mean of this epoch\'s batch losses, in training mode (Dropout on).\n'
        + '        # val_loss: one pass after the epoch in evaluation mode, as the browser measures both losses.\n'
        + '        print(f"epoch {epoch:3d}  train_loss';
    for (const spec of DATASETS) {
        const script = eduScript(spec, { settings: { epochs: 7, batchSize: 16, seed: 3 } });
        const header = unwrap(script);
        assert.ok(header.includes('train_loss is the running mean of each epoch\'s batch losses, measured in training mode (Dropout on) while the weights still change.'), spec.id);
        assert.match(script, /^SEED = 3 {2}# seeds PyTorch's initial weights, Dropout and batch order \(its own draws, not the browser's\)$/m, spec.id);
        assert.ok(script.includes(loop), `${spec.id}: the epoch line explains train_loss`);
        if (spec.training === 'browser') {
            assert.ok(header.includes('# Compared with the browser: - Same data, split, loss and optimizer as in the browser, but PyTorch draws its own initial weights and batch order (from SEED), so the curves come out close to the browser\'s, not identical.'), spec.id);
            assert.ok(!header.includes('presets for Colab'), spec.id);
        } else {
            const { epochs, batchSize } = spec.exportDefaults;
            assert.ok(header.includes(`EPOCHS and BATCH_SIZE are ${spec.name}'s presets for Colab (${epochs} and ${batchSize}), not the Training tab's (7 and 16).`), spec.id);
        }
    }
});

test('loss choices: BCE, two-output cross-entropy, MSE on 0/1 labels, and the trainer defaults when the Training tab is empty', () => {
    const blobs = getDataset('blobs');
    const ce = eduScript(blobs, { model: modelFor(blobs, { loss: 'cross_entropy', outputs: 2 }) });
    assert.ok(ce.includes('return loss_fn(logits, y)') && ce.includes('logits.argmax(dim=-1) == y'));
    const bce = eduScript(blobs);
    assert.ok(bce.includes('loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")'));
    assert.ok(bce.includes('((logits.view_as(y) > 0).long() == y)'));
    const mse = eduScript(blobs, { model: modelFor(blobs, { loss: 'mse', outputs: 1 }) });
    assert.ok(mse.includes('loss_fn = torch.nn.MSELoss(reduction="mean")') && mse.includes('((logits.view_as(y) > 0.5).long() == y)'));

    const empty = modelFor(getDataset('iris'));
    const none = generateTrainingScript(empty.graph, { nodes: {}, edges: [] }, { templates: empty.templates, data: dataFor(getDataset('iris')) });
    assert.ok(none.includes('No optimizer in the Training tab: Adam with learning rate 0.01, as the browser trainer uses.'));
    assert.ok(none.includes('torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0, betas=(0.9, 0.999), eps=1e-8)'));
    assert.ok(none.includes('# No loss in the Training tab: Cross-entropy, as the browser trainer uses.'));
    const plane = modelFor(getDataset('plane'));
    assert.ok(generateTrainingScript(plane.graph, { nodes: {}, edges: [] }, { templates: plane.templates, data: dataFor(getDataset('plane')) }).includes('loss_fn = torch.nn.MSELoss()'));
});

test('a model that does not fit the dataset gets a "Check before running" list in the header', () => {
    const script = eduScript(getDataset('mnist'), { model: modelFor(getDataset('iris')) });
    const header = script.slice(0, script.indexOf('import torch')).replace(/\n#\s+/g, ' ');
    assert.ok(header.includes('Check before running:'));
    assert.ok(header.includes('Your Input is (4,), but MNIST examples are (1, 28, 28): use Data → Use this dataset.'), header);
    assert.ok(header.includes('Your model gives 3 outputs per example, but MNIST needs 10 for 10 classes'), header);
    const text = eduScript(getDataset('tiny-shakespeare'), { model: modelFor(getDataset('mnist')) });
    assert.ok(text.replace(/\n#\s+/g, ' ').includes('examples are windows of token ids, shaped (T,)'));
});

test('the M7 model (GPT-2 at lesson size) exports a Tiny Shakespeare script that trains on (B, T, V) scores', () => {
    gpt2Storyboard({ ...stores, configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4, nLayers: 2, mlpWidth: 256 }, includeTraining: true });
    const script = generateTrainingScript(current().graph, current().training, { templates: current().templates, data: dataFor(getDataset('tiny-shakespeare'), { options: { blockSize: 64 } }) });
    assert.ok(script.includes('return loss_fn(logits.flatten(0, -2), y.flatten())'));
    assert.match(script, /return torch\.optim\.AdamW\(model\.parameters\(\), lr=[0-9.e-]+, /);
    assert.match(script, /^BLOCK_SIZE = 64 /m);
    assert.ok(!script.includes('Check before running'), 'the lesson model fits Tiny Shakespeare');
    // A context of 32 positions: train.py reads 32-character windows, the length of the model's Input.
    gpt2Storyboard({ ...stores, configuration: { vocabSize: 65, context: 32, dModel: 32, nHeads: 2, nLayers: 1, mlpWidth: 64 }, includeTraining: true });
    const short = generateTrainingScript(current().graph, current().training, { templates: current().templates, data: dataFor(getDataset('tiny-shakespeare'), { options: { blockSize: 64 } }) });
    assert.match(short, /^BLOCK_SIZE = 32 /m);
    assert.ok(short.replace(/\n#\s+/g, ' ').includes('train.py reads windows of 32 characters, the length of your Input, instead of the dataset\'s 64.'));
});

test('multi-phase graphs keep the phase steps inside the new loop; extra losses are named, not trained', () => {
    STORYBOARDS.find((entry) => entry.id === 'gan').run(stores);
    const gan = generateTrainingScript(current().graph, current().training, { templates: current().templates, data: dataFor(getDataset('blobs')) });
    assert.ok(gan.includes(PHASES_COMMENT));
    assert.ok(gan.includes('G_optimizer = torch.optim.Adam('));
    assert.ok(gan.includes('loss = loss_total  # the last phase\'s loss'));
    assert.ok(gan.includes('global_step += 1'));
    STORYBOARDS.find((entry) => entry.id === 'dann').run(stores);
    const dann = generateTrainingScript(current().graph, current().training, { templates: current().templates, data: dataFor(getDataset('digits')) });
    assert.ok(dann.includes('def predict(model, x):'), 'two Outputs: the loss uses the bound one');
    assert.ok(dann.includes('loss = compute_loss(predict(model, x), y)'));
    assert.ok(dann.replace(/\n#\s+/g, ' ').includes('The Training tab has 2 losses.'));
    if (hasPython) assert.equal(compileAll({ gan, dann }), null);
});

test('every generated script is valid Python', { skip: hasPython ? false : 'python3 is not installed' }, () => {
    const named = {};
    for (const spec of DATASETS) named[spec.id] = eduScript(spec);
    named.blobsCe = eduScript(getDataset('blobs'), { model: modelFor(getDataset('blobs'), { loss: 'cross_entropy', outputs: 2 }) });
    named.blobsMse = eduScript(getDataset('blobs'), { model: modelFor(getDataset('blobs'), { loss: 'mse', outputs: 1 }) });
    named.mismatch = eduScript(getDataset('cifar10'), { model: modelFor(getDataset('iris')) });
    for (const id of ['gpt2', 'resnet', 'dann', 'gan']) {
        STORYBOARDS.find((entry) => entry.id === id).run(stores);
        named[`legacy_${id}`] = legacyScript(current());
        named[`edu_${id}`] = generateTrainingScript(current().graph, current().training, { templates: current().templates, data: dataFor(getDataset(id === 'gpt2' ? 'tiny-shakespeare' : 'cifar10')) });
    }
    named.notebook = trainingNotebook(named.iris).cells.map((cell) => cell.source.join('')).join('\n\n');
    assert.equal(compileAll(named), null);
});

// ---------------------------------------------------------------------------------------------------------------
// Notebook
// ---------------------------------------------------------------------------------------------------------------

test('the .ipynb is nbformat 4.5 JSON with the script (no main guard) and model = train()', () => {
    const script = eduScript(getDataset('digits'));
    const notebook = JSON.parse(trainingNotebookText(script, { name: 'train.ipynb' }));
    assert.equal(notebook.nbformat, 4);
    assert.equal(notebook.nbformat_minor, 5);
    assert.equal(notebook.metadata.kernelspec.name, 'python3');
    assert.equal(notebook.cells.length, 2);
    const [first, second] = notebook.cells;
    for (const cell of notebook.cells) {
        assert.equal(cell.cell_type, 'code');
        assert.match(cell.id, /^[A-Za-z0-9-_]{1,64}$/);
        assert.deepEqual(cell.outputs, []);
        assert.equal(cell.execution_count, null);
        assert.ok(cell.source.slice(0, -1).every((line) => line.endsWith('\n')));
    }
    const code = first.source.join('');
    assert.ok(!code.includes('__name__'), 'Jupyter would train twice');
    assert.equal(`${code}\n`, stripMainGuard(script));
    assert.ok(code.includes('def train():'));
    assert.deepEqual(second.source, ['model = train()']);
    assert.equal(stripMainGuard('x = 1\n\n\nif __name__ == "__main__":\n    train()\n'), 'x = 1\n');
});

// ---------------------------------------------------------------------------------------------------------------
// Running the scripts (needs torch, numpy and scikit-learn)
// ---------------------------------------------------------------------------------------------------------------

const oneEpoch = (script) => script.replace(/^EPOCHS = .*$/m, 'EPOCHS = 1');

for (const id of ['blobs', 'iris', 'digits']) {
    test(`the ${id} script trains for one epoch in PyTorch`, { skip: hasTorch ? false : 'needs python3 with torch, numpy and scikit-learn' }, () => {
        const result = runPython(oneEpoch(eduScript(getDataset(id), { split: { val: 0.2, test: id === 'iris' ? 0.2 : 0 } })));
        assert.equal(result.status, 0, result.stderr);
        assert.match(result.stdout, /^before training {2}val_loss \d+\.\d{4} {2}val_acc /m);
        assert.match(result.stdout, /^epoch {3}1 {2}train_loss \d+\.\d{4} {2}val_loss \d+\.\d{4} {2}val_acc \d\.\d{3}$/m);
        assert.match(result.stdout, /^best epoch: 1 /m);
        if (id === 'iris') assert.match(result.stdout, /^test {2}test_loss /m);
    });
}

test('the M7 Transformer export runs its first training steps and samples text in PyTorch', { skip: hasTorch ? false : 'needs python3 with torch' }, () => {
    gpt2Storyboard({ ...stores, configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4, nLayers: 2, mlpWidth: 256 }, includeTraining: true });
    const script = oneEpoch(generateTrainingScript(current().graph, current().training, { templates: current().templates, data: dataFor(getDataset('tiny-shakespeare'), { options: { blockSize: 64 } }) }))
        .replace(/^TRAIN_BATCHES = .*$/m, 'TRAIN_BATCHES = 3').replace(/^VAL_BATCHES = .*$/m, 'VAL_BATCHES = 2').replace('length=300', 'length=40');
    // A local stand-in for input.txt, so nothing is downloaded: 65 distinct characters as in the real file.
    const alphabet = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    const text = Array.from({ length: 4000 }, (_, i) => alphabet[(i * 7 + (i >> 3)) % alphabet.length]).join('');
    const result = runPython(script, { files: { 'data/tinyshakespeare.txt': text } });
    assert.equal(result.status, 0, result.stderr);
    assert.match(result.stdout, /^112,512 parameters/m);
    assert.match(result.stdout, /^epoch {3}1 {2}train_loss /m);
    assert.match(result.stdout, /A sample of generated text/);

    // The legacy GPT-2 script with dict batches {"inputs", "labels"} also runs its first step now.
    const harness = `import runpy, torch
module = runpy.run_path("legacy.py", run_name="legacy")
def build_dataloaders():
    data = torch.randint(0, 65, (4, 65))
    batch = {"inputs": data[:, :-1], "labels": data[:, 1:]}
    return [batch], [batch]
module["train"].__globals__["build_dataloaders"] = build_dataloaders
module["train"](num_epochs=1)
`;
    const legacy = runPython(harness, { files: { 'legacy.py': legacyScript(current()) } });
    assert.equal(legacy.status, 0, legacy.stderr);
    assert.match(legacy.stdout, /epoch 0 done/);
});
