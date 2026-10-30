import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// Stage C CodePanel (docs/edu/SPEC.md §6.11.3, §8.3), rendered in jsdom with the real stores: controlled tabs with
// spotlight targets, the Edu train.py when a dataset is bound, DeepFlow's stub otherwise, and the .ipynb download.
// Layout at 390 px is checked in the browser (the scratch browser check), not here.
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
    localStorage: dom.window.localStorage,
    getComputedStyle: dom.window.getComputedStyle.bind(dom.window),
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    // An open MessagePort would keep this process alive (as in edu-shell.test.mjs).
    MessageChannel: undefined,
    IS_REACT_ACT_ENVIRONMENT: true,
})) {
    installedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
}

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');
const temporaryDirectory = await mkdtemp(join(projectRoot, '.edu-codepanel-'));
let app;
try {
    const bundlePath = join(temporaryDirectory, 'components.mjs');
    await build({
        stdin: {
            contents: `
                export { default as CodePanel } from './src/components/CodePanel.jsx';
                export { useEduDataStore } from './src/edu/data/store.js';
                export { useTrainerStore } from './src/edu/trainer/store.js';
                export { useEduUiStore } from './src/edu/shell/uiStore.js';
                export { createFixtureHelpers } from './src/edu/lessons/engine/fixtures.js';
                export { restoreWorkspace } from './src/edu/lessons/engine/workspace.js';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
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
const { CodePanel, useEduDataStore, useTrainerStore, useEduUiStore, createFixtureHelpers, restoreWorkspace, useArchitectureStore } = app;

let container;
let root;
const render = (element) => act(() => root.render(element));
const $ = (selector) => container.querySelector(selector);
const target = (name) => $(`[data-edu-target="${name}"]`);
const code = () => $('pre')?.textContent ?? '';
const click = (element) => {
    assert.ok(element, 'Expected an element to click');
    return act(() => element.click());
};
const buttonNamed = (text) => [...container.querySelectorAll('button')].find((button) => button.textContent.trim() === text);

/** The panel wired to the Edu UI store, as Layout does. */
function WiredPanel() {
    const open = useEduUiStore((state) => state.codeOpen);
    const tab = useEduUiStore((state) => state.codeTab);
    const closeCode = useEduUiStore((state) => state.closeCode);
    const setCodeTab = useEduUiStore((state) => state.setCodeTab);
    return h(CodePanel, { open, onClose: closeCode, tab, onTabChange: setCodeTab });
}

/** in (4,) → Linear 4→3 → out, Adam + cross-entropy, nothing bound. */
function loadIrisModel() {
    const fx = createFixtureHelpers();
    const nodes = [fx.input('in', [4]), fx.linear('lin', 4, 3), fx.output('out')];
    restoreWorkspace(fx.snapshot({
        nodes, edges: fx.chain(...nodes),
        training: [fx.optimizer('Adam', { learning_rate: 0.01 }), fx.loss('cross_entropy', nodes.at(-1))],
        data: fx.data(null),
        trainer: fx.trainer({ epochs: 30, batchSize: 16, seed: 1 }),
    }));
}

beforeEach(() => {
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    useEduUiStore.setState({ activeTab: 'architecture', codeOpen: false, codeTab: 'model', drawerOpen: true });
    loadIrisModel();
});

afterEach(async () => {
    await act(() => root.unmount());
    container.remove();
});

after(() => {
    for (const [key, descriptor] of installedGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
    }
    dom.window.close();
});

test('a closed panel renders nothing', async () => {
    await render(h(CodePanel, { open: false, onClose: () => {} }));
    assert.equal(container.innerHTML, '');
});

test('the tabs carry spotlight targets and follow the tab the UI store asks for', async () => {
    await render(h(WiredPanel));
    assert.equal(container.innerHTML, '');
    await act(() => useEduUiStore.getState().openCode('train'));
    assert.equal(target('code-tab-train').getAttribute('aria-selected'), 'true');
    assert.equal(target('code-tab-model').getAttribute('aria-selected'), 'false');
    assert.ok(code().includes('def train('), 'train.py is shown when a lesson opens it');

    await click(target('code-tab-model'));
    assert.equal(useEduUiStore.getState().codeTab, 'model', 'a learner\'s click reaches the store, so checks see it');
    assert.ok(code().startsWith('import torch'));
    assert.ok(!code().includes('def train('));

    await act(() => useEduUiStore.getState().setCodeTab('train'));
    assert.ok(code().includes('def train('), 'the panel is controlled by the store');
    await click($('[aria-label="Close code panel"]'));
    assert.equal(useEduUiStore.getState().codeOpen, false);
    assert.equal(container.innerHTML, '');
});

test('Esc closes the panel, and focus inside it returns to the Code button', async () => {
    const codeButton = document.createElement('button');
    codeButton.className = 'df-action-segment';
    codeButton.setAttribute('aria-label', 'Code');
    document.body.append(codeButton);
    try {
        await render(h(WiredPanel));
        await act(() => useEduUiStore.getState().openCode('model'));
        target('code-tab-train').focus();
        await act(async () => { document.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); });
        assert.equal(useEduUiStore.getState().codeOpen, false);
        assert.equal(container.innerHTML, '');
        assert.equal(document.activeElement, codeButton);
    } finally {
        codeButton.remove();
    }
});

test('without a controlling tab the panel keeps its own', async () => {
    await render(h(CodePanel, { open: true, onClose: () => {} }));
    assert.ok(code().startsWith('import torch'));
    await click(target('code-tab-train'));
    assert.ok(code().includes('def train('));
    assert.equal(target('code-tab-train').getAttribute('aria-selected'), 'true');
});

test('with no dataset bound, train.py is DeepFlow\'s stub without a notebook or Colab note', async () => {
    await act(() => useEduUiStore.getState().openCode('train'));
    await render(h(WiredPanel));
    assert.ok(code().includes('raise NotImplementedError("Wire up your datasets / dataloaders here.")'));
    assert.equal(target('code-notebook'), null);
    assert.equal(target('code-colab-note'), null);
});

test('binding a dataset switches train.py to the full script, and trainer settings flow into it', async () => {
    await act(() => useEduUiStore.getState().openCode('train'));
    await render(h(WiredPanel));
    await act(() => useEduDataStore.getState().setDataset('iris'));
    assert.ok(code().includes('def build_dataloaders(batch_size, data_seed, train_seed):'));
    assert.ok(code().includes('X, y = load_iris(return_X_y=True)'));
    assert.match(code(), /^EPOCHS = 30$/m);
    assert.match(target('code-colab-note').textContent, /Run in Colab:.*File → Upload notebook.*!python train\.py.*Iris/);
    assert.ok(target('code-notebook'));

    await act(() => useTrainerStore.getState().setSettings({ epochs: 12 }));
    assert.match(code(), /^EPOCHS = 12$/m);
    await act(() => useEduDataStore.getState().setOptions({ seed: 7 }, 'iris'));
    assert.match(code(), /^DATA_SEED = 7 /m);

    await click(target('code-tab-model'));
    assert.equal(target('code-colab-note'), null, 'the note belongs to train.py');
    assert.equal(target('code-notebook'), null);

    await act(() => useEduDataStore.getState().setDataset(null));
    await click(target('code-tab-train'));
    assert.ok(code().includes('raise NotImplementedError'), 'unbinding brings the stub back');
});

test('export datasets use their Colab defaults in the panel', async () => {
    await act(() => useEduUiStore.getState().openCode('train'));
    await render(h(WiredPanel));
    await act(() => useEduDataStore.getState().setDataset('mnist'));
    assert.ok(code().includes('datasets.MNIST(root, train=True, download=True, transform=transform)'));
    assert.match(code(), /^EPOCHS = 5 {2}#/m);
    assert.ok(code().includes('Check before running'), 'the Iris-sized model does not fit MNIST yet');
});

test('Download .ipynb saves a two-cell notebook of the shown script', async () => {
    const saved = [];
    const createObjectURL = dom.window.URL.createObjectURL;
    const revokeObjectURL = dom.window.URL.revokeObjectURL;
    const anchorClick = dom.window.HTMLAnchorElement.prototype.click;
    dom.window.URL.createObjectURL = (blob) => { saved.push({ blob }); return 'blob:test'; };
    dom.window.URL.revokeObjectURL = () => {};
    globalThis.URL.createObjectURL = dom.window.URL.createObjectURL;
    globalThis.URL.revokeObjectURL = dom.window.URL.revokeObjectURL;
    dom.window.HTMLAnchorElement.prototype.click = function captureDownload() { saved.at(-1).name = this.download; };
    try {
        await act(() => useEduUiStore.getState().openCode('train'));
        await render(h(WiredPanel));
        await act(() => useEduDataStore.getState().setDataset('iris'));
        await click(target('code-notebook'));
        assert.equal(saved.length, 1);
        assert.equal(saved[0].name, 'train.ipynb');
        const notebook = JSON.parse(await saved[0].blob.text());
        assert.equal(notebook.nbformat, 4);
        assert.equal(notebook.nbformat_minor, 5);
        assert.equal(notebook.cells.length, 2);
        assert.equal(`${notebook.cells[0].source.join('')}\n\n\nif __name__ == "__main__":\n    train()\n`, code());
        assert.deepEqual(notebook.cells[1].source, ['model = train()']);

        await click(buttonNamed('download'));
        assert.equal(saved[1].name, 'train.py');
        assert.equal(await saved[1].blob.text(), code());
    } finally {
        dom.window.URL.createObjectURL = createObjectURL;
        dom.window.URL.revokeObjectURL = revokeObjectURL;
        dom.window.HTMLAnchorElement.prototype.click = anchorClick;
    }
});

test('an invalid model shows the codegen error and offers no notebook', async () => {
    await act(() => useEduUiStore.getState().openCode('train'));
    await render(h(WiredPanel));
    await act(() => useEduDataStore.getState().setDataset('iris'));
    await act(() => useArchitectureStore.getState().patchLayerParameters('in', { shape: '(5,)' }));
    assert.ok(code().startsWith('# codegen error:'), code().slice(0, 120));
    assert.equal(target('code-notebook'), null);
    assert.deepEqual(jsdomErrors, []);
});
