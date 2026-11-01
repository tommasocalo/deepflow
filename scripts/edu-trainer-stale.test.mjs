import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// Stale trainer results (review r1): after the model or the dataset changes, the TrainerCard and the Training tab
// must not present the previous run as this model's result, and "Recent runs" must tell architectures apart.
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
dom.window.HTMLCanvasElement.prototype.getContext = () => null;

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');
const temporaryDirectory = await mkdtemp(join(projectRoot, '.edu-ui-stale-'));
let app;
try {
    const bundlePath = join(temporaryDirectory, 'components.mjs');
    await build({
        stdin: {
            contents: `
                export { default as TrainerPanel } from './src/edu/trainer/components/TrainerPanel.jsx';
                export { default as TrainerCard } from './src/edu/trainer/components/TrainerCard.jsx';
                export { staleReason } from './src/edu/trainer/describe.js';
                export { runDifference, blockCounts } from './src/edu/trainer/charts.js';
                export { useEduDataStore } from './src/edu/data/store.js';
                export { useTrainerStore } from './src/edu/trainer/store.js';
                export { workspaceSlices } from './src/edu/bridge.js';
                export { createFixtureHelpers } from './src/edu/lessons/engine/fixtures.js';
                export { restoreWorkspace } from './src/edu/lessons/engine/workspace.js';
                export { runActions } from './src/edu/lessons/engine/actions.js';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useTemplatesStore } from './src/stores/templatesStore.js';
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
    TrainerPanel, TrainerCard, staleReason, runDifference, blockCounts, useEduDataStore, useTrainerStore, workspaceSlices,
    createFixtureHelpers, restoreWorkspace, runActions, useArchitectureStore, useTemplatesStore, useEduUiStore,
} = app;

let container;
let root;
const render = (element) => act(() => root.render(element));
const click = (element) => { assert.ok(element, 'Expected an element to click'); return act(() => element.click()); };
const $ = (selector) => container.querySelector(selector);
const trainer = () => useTrainerStore.getState();
const live = () => ({ graph: useArchitectureStore.getState().graph, templates: useTemplatesStore.getState().templates, binding: useEduDataStore.getState().current() });

async function untilRuns(count) {
    const started = Date.now();
    while (!(trainer().status === 'idle' && trainer().runs.length >= count)) {
        if (Date.now() - started > 20000) throw new Error(`Timed out waiting for ${count} runs (status ${trainer().status})`);
        await new Promise((resolve) => setTimeout(resolve, 5));
    }
    await act(async () => {});
}

/** Input → Linear (2 → hidden) [→ ReLU → Linear (hidden → 1)] → Output on the circle, as in M2. */
function load({ hidden = 0, relu = false, dataset = 'circle' } = {}) {
    const fx = createFixtureHelpers();
    const nodes = [fx.input('in', [2], { label: 'features' })];
    if (hidden) {
        nodes.push(fx.linear('lin1', 2, hidden, { label: 'Linear 1' }));
        if (relu) nodes.push(fx.layer('relu', 'torch.nn.ReLU', {}, 'ReLU'));
        nodes.push(fx.linear('lin2', hidden, 1, { label: 'Linear 2' }));
    } else {
        nodes.push(fx.linear('lin1', 2, 1, { label: 'neuron' }));
    }
    nodes.push(fx.output('out', { label: 'logit' }));
    restoreWorkspace(fx.snapshot({
        nodes, edges: fx.chain(...nodes),
        training: [fx.optimizer('Adam', { learning_rate: 0.03 }), fx.loss('bce', nodes.at(-1))],
        data: fx.data(dataset, { n: 200, noise: 0, seed: 2 }),
        trainer: fx.trainer({ epochs: 10, batchSize: 32, seed: 2 }),
    }));
}

beforeEach(() => {
    jsdomErrors.length = 0;
    localStorage.clear();
    for (const [, slice] of workspaceSlices()) slice.restore(undefined);
    useEduUiStore.setState({ activeTab: 'architecture' });
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

test('staleReason: moving a block is not a new model; a parameter, a block or the dataset is', async () => {
    load({ hidden: 8 });
    await trainer().start({ sync: true });
    const run = trainer().current;
    assert.equal(staleReason(run, live()), null, 'the run describes the live workspace');
    useArchitectureStore.getState().moveNodes([{ id: 'lin1', position: { x: 900, y: 40 } }]);
    assert.equal(staleReason(run, live()), null, 'a moved block is the same model');
    assert.equal(staleReason({ ...run, status: 'running' }, live()), null, 'a running run is never stale');
    await runActions([{ do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 3 }, { do: 'setParam', node: { id: 'lin2' }, name: 'in_features', value: 3 }], { headless: true });
    assert.equal(staleReason(run, live()), 'model');
    useEduDataStore.getState().setOptions({ seed: 7 });
    assert.equal(staleReason(run, live()), 'both', 'another data seed is another dataset');
    await runActions([{ do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 8 }, { do: 'setParam', node: { id: 'lin2' }, name: 'in_features', value: 8 }], { headless: true });
    assert.equal(staleReason(run, live()), 'dataset', 'the widths are back to the trained model');
    useEduDataStore.getState().setOptions({ seed: 2 });
    assert.equal(staleReason(run, live()), null, 'and so is the data');
    await runActions([{ do: 'useDataset', id: 'xor', setInput: false, setOutput: false }], { headless: true });
    assert.equal(staleReason(run, live()), 'dataset');
});

test('Recent runs tell a ReLU network from a linear stack with the same parameter count', async () => {
    load({ hidden: 8 });
    await trainer().start({ sync: true });
    const linear = trainer().current;
    load({ hidden: 8, relu: true });
    await trainer().start({ sync: true });
    const relu = trainer().current;
    load({ hidden: 3, relu: true });
    await trainer().start({ sync: true });
    const narrow = trainer().current;
    assert.deepEqual(Object.fromEntries(blockCounts(relu.graph, relu.templates)), { Linear: 2, ReLU: 1 });
    assert.equal(linear.paramCount, relu.paramCount, 'both 2-8-1 networks have 33 parameters');
    assert.equal(runDifference(relu, narrow), '33 params');
    assert.equal(runDifference(linear, narrow), 'no ReLU · 33 params');
    assert.equal(runDifference(linear, relu), 'no ReLU');
    assert.equal(runDifference(relu, linear), 'with ReLU');
});

test('the TrainerCard dims a run of another model, says so, and clears once the new model is trained', async () => {
    load({ hidden: 8 });
    await render(h(TrainerCard));
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(1);
    assert.equal($('.edu-tr-stale'), null);
    assert.match($('.edu-tr-card-epoch').textContent, /epoch 10\/10/);
    // M2 s3 → s4: a ReLU between the two Linear blocks.
    await act(async () => { await runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }], { headless: true }); });
    assert.equal($('.edu-tr-stale').textContent, 'This run used a different model: press Train to see results for this one.');
    assert.ok($('.edu-tr-card-body').classList.contains('is-stale'), 'the curve and boundary are dimmed');
    assert.equal($('.edu-tr-card-epoch'), null, 'no epoch count for a run of another model');
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(2);
    assert.equal($('.edu-tr-stale'), null, 'the new run is this model\'s result');
    assert.ok(!$('.edu-tr-card-body').classList.contains('is-stale'));
    assert.match($('.edu-tr-card-runs').textContent, /this run: val loss .*no ReLU: val loss/, 'the earlier run is the one without the ReLU');
});

test('the Training tab dims the curves and the result of a run on another dataset', async () => {
    load({ hidden: 8, relu: true });
    await render(h(TrainerPanel));
    await click($('[data-edu-target="trainer-run"]'));
    await untilRuns(1);
    assert.equal($('.edu-tr-stale'), null);
    await act(async () => { await runActions([{ do: 'useDataset', id: 'xor', setInput: false, setOutput: false }], { headless: true }); });
    assert.equal($('[data-edu-target="trainer-curves"] .edu-tr-stale').textContent, 'This run used a different dataset: press Train to see results for this one.');
    assert.ok($('[data-edu-target="trainer-curves"] .is-stale'), 'the loss curves are dimmed');
    assert.ok($('[data-edu-target="trainer-boundary"]').classList.contains('is-stale'), 'the boundary is dimmed');
});
