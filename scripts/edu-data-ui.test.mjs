import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// Stage B UI (docs/edu/SPEC.md §8.2): the Data tab, the TrainerPanel and the TrainerCard with their real stores,
// rendered in jsdom. Layout and canvas drawing are not checked here (scripts/edu-data-browser-check.mjs does that).
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const jsdomErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => jsdomErrors.push(error));
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/', pretendToBeVisual: true, virtualConsole });
const installedGlobals = new Map();
for (const [key, value] of Object.entries({
    window: dom.window,
    document: dom.window.document,
    navigator: dom.window.navigator,
    HTMLElement: dom.window.HTMLElement,
    Element: dom.window.Element,
    SVGElement: dom.window.SVGElement,
    Node: dom.window.Node,
    CustomEvent: dom.window.CustomEvent,
    localStorage: dom.window.localStorage,
    getComputedStyle: dom.window.getComputedStyle.bind(dom.window),
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
    IS_REACT_ACT_ENVIRONMENT: true,
})) {
    installedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
}
// jsdom has no 2D canvas; the boundary view skips drawing when getContext returns null.
dom.window.HTMLCanvasElement.prototype.getContext = () => null;

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');
const temporaryDirectory = await mkdtemp(join(projectRoot, '.edu-ui-data-'));
let app;
try {
    const bundlePath = join(temporaryDirectory, 'components.mjs');
    await build({
        stdin: {
            contents: `
                export { default as DataWorkspace } from './src/edu/data/DataWorkspace.jsx';
                export { default as TrainerPanel } from './src/edu/trainer/components/TrainerPanel.jsx';
                export { default as TrainerCard } from './src/edu/trainer/components/TrainerCard.jsx';
                export { useEduDataStore } from './src/edu/data/store.js';
                export { useTrainerStore } from './src/edu/trainer/store.js';
                export { registerSource, workspaceSlices } from './src/edu/bridge.js';
                export { createFixtureHelpers } from './src/edu/lessons/engine/fixtures.js';
                export { restoreWorkspace } from './src/edu/lessons/engine/workspace.js';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useTrainingStore } from './src/stores/trainingStore.js';
                export { useEduUiStore } from './src/edu/shell/uiStore.js';
            `,
            resolveDir: projectRoot,
        },
        outfile: bundlePath,
        bundle: true,
        packages: 'external',
        format: 'esm',
        platform: 'node',
        jsx: 'automatic',
        loader: { '.css': 'empty' },
        logLevel: 'silent',
    });
    app = await import(pathToFileURL(bundlePath).href);
} finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
}
const {
    DataWorkspace, TrainerPanel, TrainerCard, useEduDataStore, useTrainerStore, registerSource, workspaceSlices,
    createFixtureHelpers, restoreWorkspace, useArchitectureStore, useTrainingStore, useEduUiStore,
} = app;

let container;
let root;
const render = (element) => act(() => root.render(element));
const click = (element) => {
    assert.ok(element, 'Expected an element to click');
    return act(() => element.click());
};
const $ = (selector) => container.querySelector(selector);
const $$ = (selector) => [...container.querySelectorAll(selector)];
const target = (name) => $(`[data-edu-target="${name}"]`);
const graph = () => useArchitectureStore.getState().graph;
const trainer = () => useTrainerStore.getState();

/** Wait until the trainer is idle with `count` runs (runs are sliced with setTimeout under Node), then flush React. */
async function untilRuns(count) {
    const started = Date.now();
    while (!(trainer().status === 'idle' && trainer().runs.length >= count)) {
        if (Date.now() - started > 20000) throw new Error(`Timed out waiting for ${count} runs (status ${trainer().status})`);
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    await act(async () => {});
}

/** One Linear between Input and Output; `dataset` bound with Stage A's fixture helpers. */
function load({ inputs = 2, outputs = 1, loss = 'bce', dataset = null, options = {}, trainerSettings = {} } = {}) {
    const fx = createFixtureHelpers();
    const input = fx.input('in', [inputs], { label: 'features' });
    const neuron = fx.linear('neuron', inputs, outputs, { label: 'neuron' });
    const output = fx.output('out', { label: 'logit' });
    restoreWorkspace(fx.snapshot({
        nodes: [input, neuron, output], edges: fx.chain(input, neuron, output),
        training: [fx.optimizer('Adam', { learning_rate: 0.03 }), fx.loss(loss, output)],
        data: dataset ? fx.data(dataset, options) : undefined,
        trainer: fx.trainer({ epochs: 30, batchSize: 16, seed: 1, ...trainerSettings }),
    }));
}

beforeEach(() => {
    jsdomErrors.length = 0;
    localStorage.clear();
    for (const [, slice] of workspaceSlices()) slice.restore(undefined);
    useEduDataStore.setState({ viewId: 'blobs', lastUse: null });
    useEduUiStore.setState({ activeTab: 'data' });
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
});
afterEach(async () => {
    await act(() => root.unmount());
    container.remove();
    assert.deepEqual(jsdomErrors, [], 'DOM event handlers must not throw uncaught errors');
});
after(() => {
    dom.window.close();
    for (const [key, descriptor] of installedGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
    }
});

test('the Data list groups datasets by where they train, each row with its data-edu-target', async () => {
    load();
    await render(h(DataWorkspace));
    const groups = $$('.edu-data-group');
    assert.ok(groups.length >= 1);
    assert.equal(groups[0].querySelector('h3').textContent, 'Train in the browser');
    const browserRows = [...groups[0].querySelectorAll('[data-edu-target^="dataset-"]')].map((row) => row.dataset.eduTarget);
    assert.deepEqual(browserRows, ['dataset-blobs', 'dataset-circle', 'dataset-xor', 'dataset-spiral2', 'dataset-spiral3', 'dataset-plane', 'dataset-iris', 'dataset-digits']);
    if (groups[1]) assert.equal(groups[1].querySelector('h3').textContent, 'Export to train (Colab)');
    assert.equal(target('dataset-blobs').getAttribute('aria-current'), 'true', 'the first dataset is shown');
    assert.ok(target('dataset-preview').querySelector('svg[role="img"] title').textContent.startsWith('Two blobs: scatter of 200'));
    assert.ok(target('dataset-options'));
    assert.ok(target('use-dataset'));
    assert.match($('.edu-data-title').textContent, /Two blobs — 200 points · 2 features · 2 classes/);
});

test('Use this dataset binds Iris and fits the Input and the last Linear in one step', async () => {
    load();
    await render(h(DataWorkspace));
    await click(target('dataset-iris'));
    assert.match($('.edu-data-title').textContent, /^Iris — 150 flowers · 4 features · 3 classes/);
    assert.equal(useEduDataStore.getState().datasetId, null, 'selecting a row only shows the dataset');
    // Scaling shows the training-split statistics (docs/edu/SPEC.md §11.4: mean 5.83 3.07 3.72 1.18).
    const means = $$('.edu-data-table-compact tbody tr').map((row) => row.children[1].textContent);
    assert.deepEqual(means, ['5.83', '3.07', '3.72', '1.18']);
    await click(target('use-dataset'));
    assert.equal(useEduDataStore.getState().datasetId, 'iris');
    assert.equal(graph().nodes.in.parameters.shape, '(4,)');
    assert.equal(graph().nodes.neuron.parameterValues.out_features, 3);
    assert.equal($('.edu-data-use-status').textContent, 'Done: Input → (4,); last Linear (neuron) Output → 3.');
    assert.ok(target('dataset-iris').querySelector('.edu-data-inuse'), 'the row says it is in use');
    // The table view lists the first 8 rows with their split.
    await click($$('.edu-data-segmented button').find((button) => button.textContent === 'Table'));
    assert.equal($$('.edu-data-table:not(.edu-data-table-compact) tbody tr').length, 8);
});

test('in a lesson step that leaves the output size to the learner, Use this dataset sets only the Input', async () => {
    load();
    const unregister = registerSource('lesson', { read: () => ({ mode: 'lesson', phase: 'steps', stepId: 's4', datasetUse: { setOutput: false } }) });
    try {
        await render(h(DataWorkspace));
        await click(target('dataset-iris'));
        assert.equal($('.edu-data-use-hint').textContent, 'Sets your model\'s Input to (4,). In this lesson you set the output size yourself.');
        await click(target('use-dataset'));
        assert.equal(graph().nodes.in.parameters.shape, '(4,)');
        assert.equal(graph().nodes.neuron.parameterValues.out_features, 1, 'the output size is the next exercise (M1 s5)');
        assert.equal($('.edu-data-use-status').textContent, 'Done: Input → (4,).');
    } finally {
        unregister();
    }
});

test('toy options and the split update the preview and the sizes', async () => {
    load();
    await render(h(DataWorkspace));
    const sizes = () => $('.edu-data-split-sizes').textContent;
    assert.equal(sizes(), 'train 160 · val 40 · test 0');
    const test = $$('.edu-data-option select').at(-1);
    await act(() => {
        test.value = '0.2';
        test.dispatchEvent(new window.Event('change', { bubbles: true }));
    });
    assert.equal(useEduDataStore.getState().split.test, 0.2);
    assert.equal(sizes(), 'train 120 · val 40 · test 40');
    assert.match(target('dataset-preview').querySelector('desc').textContent, /40 test points faint/);
});

test('digits loads its data chunk and shows two thumbnails per digit', async () => {
    load();
    await render(h(DataWorkspace));
    await click(target('dataset-digits'));
    const started = Date.now();
    while (!$('.edu-data-digit-grid') && Date.now() - started < 10000) await act(() => new Promise((resolve) => setTimeout(resolve, 10)));
    assert.equal($$('.edu-data-digit-grid li').length, 20);
    assert.equal($$('.edu-data-digit-grid svg')[0].getAttribute('aria-label'), 'Digit 0');
});

test('TrainerPanel: Train adds a run, Iris shows a confusion <table>, Check initial loss records epoch 0', async () => {
    load({ inputs: 4, outputs: 3, loss: 'cross_entropy', dataset: 'iris', options: { seed: 1 } });
    await render(h(TrainerPanel));
    assert.equal(target('trainer-confusion')?.tagName, 'DIV', 'before training the result column announces the confusion matrix');
    assert.match($('.edu-tr-facts').textContent, /Iris \(seed 1\)/);
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(1);
    assert.equal(trainer().runs.length, 1);
    const table = $('[data-edu-target="trainer-confusion"] table');
    assert.ok(table, 'a real table');
    assert.match(table.querySelector('caption').textContent, /Validation split, 30 examples/);
    assert.deepEqual([...table.querySelectorAll('tbody th')].map((cell) => cell.textContent), ['setosa', 'versicolor', 'virginica']);
    const counts = [...table.querySelectorAll('tbody td')].map((cell) => Number(cell.textContent));
    assert.equal(counts.length, 9);
    assert.equal(counts.reduce((a, b) => a + b, 0), 30);
    assert.match($('.edu-tr-status').textContent, /^Done · 30 epochs in/);
    const loss = $('[data-edu-target="trainer-curves"] svg[role="img"]');
    assert.match(loss.querySelector('desc').textContent, /^Epoch 30: training loss/);
    assert.ok($$('[data-edu-target="trainer-curves"] table tbody tr').length >= 2, 'a hidden data table backs the chart');
    await click($('[data-edu-target="trainer-init-loss"]'));
    await untilRuns(2);
    const init = trainer().runs.at(-1);
    assert.equal(init.initOnly, true);
    assert.equal(init.history.length, 1);
    assert.match($('.edu-tr-status').textContent, /^Initial loss \d/);
});

test('TrainerPanel on 2D data: the boundary view and the confusion toggle', async () => {
    load({ dataset: 'blobs', options: { n: 200, noise: 0.1, seed: 1 }, trainerSettings: { epochs: 20, batchSize: 32 } });
    await render(h(TrainerPanel));
    assert.ok(target('trainer-boundary'), 'the boundary column is there before training');
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(1);
    const figure = $('.edu-tr-boundary');
    assert.ok(figure.querySelector('canvas'));
    assert.ok(figure.querySelectorAll('svg circle, svg path').length >= 200, 'the 200 points are drawn over the boundary');
    assert.match(figure.querySelector('figcaption').textContent, /The model predicts blue in \d+%, orange in \d+% of the area/);
    await click($$('.edu-tr-segmented button').find((button) => button.textContent === 'Confusion'));
    assert.ok($('.edu-tr-confusion table'));
});

test('the learning-rate field writes the training graph, and earlier runs are labelled with what differed', async () => {
    load({ dataset: 'blobs', options: { n: 200, noise: 0.1, seed: 1 }, trainerSettings: { epochs: 5, batchSize: 32 } });
    await render(h(TrainerPanel));
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(1);
    const field = $('[data-edu-target="trainer-lr"]');
    await act(() => {
        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        setter.call(field, '0.0001');
        field.dispatchEvent(new window.Event('input', { bubbles: true }));
    });
    assert.equal(useTrainingStore.getState().graph.nodes.opt.hyperparameters.learning_rate, 0.0001);
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(2);
    assert.match($('.edu-tr-ghosts').textContent, /lr 0\.03/);
    assert.match($('[data-edu-target="trainer-curves"] desc').textContent, /Earlier runs \(dimmed\): lr 0\.03/);
});

test('preflight problems are listed, and a failed Train says why', async () => {
    load({ inputs: 2, outputs: 1, loss: 'bce', dataset: null });
    await render(h(TrainerPanel));
    await act(async () => { trainer().refreshPreflight(); });
    assert.match($('.edu-tr-issues.is-error').textContent, /No dataset yet/);
    assert.equal($('.edu-tr-facts dd').textContent.trim().startsWith('None yet'), true);
    await click($('[data-edu-target="trainer-run"]'));
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 0)); });
    assert.match($('.edu-tr-status').textContent, /^Not trained: No dataset yet/);
});

test('blocks the browser cannot train are guidance with a way to train.py, not errors to fix', async () => {
    const fx = createFixtureHelpers();
    const input = fx.input('in', [1, 8, 8], { label: 'images' });
    const conv = fx.layer('conv', 'torch.nn.Conv2d', { in_channels: 1, out_channels: 4, kernel_size: 3 }, 'conv1');
    const flat = fx.layer('flat', 'torch.nn.Flatten');
    const linear = fx.linear('fc', 144, 10, { label: 'classifier' });
    const output = fx.output('out');
    restoreWorkspace(fx.snapshot({
        nodes: [input, conv, flat, linear, output], edges: fx.chain(input, conv, flat, linear, output),
        training: [fx.optimizer('Adam', { learning_rate: 0.001 }), fx.loss('cross_entropy', output)],
        data: fx.data('digits', { seed: 1 }),
    }));
    await render(h(TrainerPanel));
    await act(async () => { trainer().refreshPreflight(); });
    assert.equal($('.edu-tr-issues.is-error'), null, 'nothing for the learner to fix');
    const colab = $('.edu-tr-issues.is-colab');
    assert.match(colab.textContent, /conv1 \(Conv2d\) is not supported by the browser trainer/);
    await click([...colab.querySelectorAll('button')].find((button) => button.textContent === 'Open train.py'));
    assert.equal(useEduUiStore.getState().codeOpen, true);
    assert.equal(useEduUiStore.getState().codeTab, 'train');
});

test('a locked TrainerCard disables Train and Check initial loss and says why; the panel follows the lesson lock', async () => {
    load({ dataset: 'blobs', options: { n: 200, noise: 0.1, seed: 1 } });
    await render(h(TrainerCard, { locked: true, lockedReason: 'Answer the question first, then train.' }));
    assert.equal($('[data-edu-target="trainer-run"]').disabled, true);
    assert.equal($('[data-edu-target="trainer-init-loss"]').disabled, true);
    assert.equal($('.edu-tr-locked').textContent, 'Answer the question first, then train.');
    assert.ok(target('trainer-lr'), 'the card has the learning-rate field');
    assert.ok(target('trainer-curves'), 'the card has the curve area');
    await render(h(TrainerCard, { locked: false }));
    assert.equal($('[data-edu-target="trainer-run"]').disabled, false);
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(1);
    assert.ok($('.edu-tr-card [data-edu-target="trainer-boundary"] canvas'), 'a boundary thumbnail for 2D data');
    assert.match($('.edu-tr-card-final').textContent, /^val acc \d+(\.\d)?% · val loss /, 'one decimal at most, as in the lesson reveals');
    await click($('.edu-tr-open'));
    assert.equal(useEduUiStore.getState().activeTab, 'training', 'Open Training switches the tab');

    let locked = true;
    const listeners = new Set();
    const unregister = registerSource('lesson', { read: () => ({ mode: 'lesson', trainLocked: locked }), subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); } });
    await render(h(TrainerPanel));
    await act(async () => {});
    assert.equal($('[data-edu-target="trainer-run"]').disabled, true, 'the Training tab respects the prediction gate');
    locked = false;
    await act(async () => { for (const listener of listeners) listener(); });
    assert.equal($('[data-edu-target="trainer-run"]').disabled, false);
    unregister();
});
