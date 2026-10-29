import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// DeepFlow Edu shell (docs/edu/SPEC.md §6.12, §8.1): branding from one constant, the four top-level tabs, Save and
// Load with Edu slices, the lesson-mode confirmations and the canvas bridge, rendered with real stores in jsdom.
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const { PRODUCT_NAME, PRODUCT_TAGLINE } = await import('../src/edu/config.js');

async function* sourceFiles(directory) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) yield* sourceFiles(path);
        else if (/\.(js|jsx)$/.test(entry.name)) yield path;
    }
}

// ---- jsdom harness (as scripts/ui-regressions.test.mjs) -------------------------------------------------------

const jsdomErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => jsdomErrors.push(error));
const dom = new JSDOM('<!doctype html><html><head><title>x</title></head><body></body></html>', {
    url: 'http://localhost/', pretendToBeVisual: true, virtualConsole,
});
dom.window.DOMMatrixReadOnly = class { constructor() { this.m22 = 1; } };
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
    // jsdom has no MessageChannel; Node's would let a store's open port keep this test process alive.
    MessageChannel: undefined,
    IS_REACT_ACT_ENVIRONMENT: true,
})) {
    installedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
}

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');

// Stage B's mount points (SPEC §6.2) are replaced by stubs with the same props: these tests cover the shell that
// mounts them, and Stage B's own tests cover the Data tab and the trainer.
const MOUNT_POINT_STUBS = {
    'DataWorkspace.jsx': 'export default function DataWorkspace() { return <div data-stub="DataWorkspace">Datasets</div>; }',
    'TrainerPanel.jsx': 'export default function TrainerPanel() { return <div data-stub="TrainerPanel">Trainer</div>; }',
    'TrainerCard.jsx': `export default function TrainerCard({ locked = false, lockedReason = '' }) {
        return <div data-stub="TrainerCard"><button type="button" data-edu-target="trainer-run" disabled={locked}>Train</button>{lockedReason ? <p>{lockedReason}</p> : null}</div>;
    }`,
};
const stubMountPoints = {
    name: 'edu-mount-point-stubs',
    setup(builder) {
        builder.onResolve({ filter: /(DataWorkspace|TrainerPanel|TrainerCard)\.jsx$/ }, (args) => {
            const name = args.path.split('/').at(-1);
            return /[\\/]edu[\\/](data|trainer)/.test(join(args.resolveDir, args.path)) ? { path: name, namespace: 'edu-stub' } : undefined;
        });
        builder.onLoad({ filter: /.*/, namespace: 'edu-stub' }, (args) => ({ contents: MOUNT_POINT_STUBS[args.path], loader: 'jsx', resolveDir: projectRoot }));
    },
};
const temporaryDirectory = await mkdtemp(join(projectRoot, '.edu-ui-shell-'));
const bundlePath = join(temporaryDirectory, 'shell.mjs');
let app;
try {
    await build({
        stdin: {
            contents: `
                export { default as Layout } from './src/components/Layout.jsx';
                export { default as ToolbarStrip } from './src/components/ToolbarStrip.jsx';
                export { default as Navbar } from './src/components/Navbar.jsx';
                export { default as ArchitectureWorkspace } from './src/components/architecture/ArchitectureWorkspace.jsx';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useArchitectureHistory } from './src/stores/architectureHistory.js';
                export { useTemplatesStore } from './src/stores/templatesStore.js';
                export { useTrainingStore } from './src/stores/trainingStore.js';
                export { registerSlice } from './src/edu/bridge.js';
                export { useEduUiStore } from './src/edu/shell/uiStore.js';
                export { useLessonSession } from './src/edu/lessons/engine/session.js';
                export { setModuleRegistry } from './src/edu/lessons/engine/registry.js';
                export { restoreWorkspace, loadSlot } from './src/edu/lessons/engine/workspace.js';
                export { createFixtureHelpers } from './src/edu/lessons/engine/fixtures.js';
                export { applyLoadedPayload, buildSavePayload } from './src/edu/shell/workspaceFile.js';
                export { focusNode, requestFit, requestInsert } from './src/edu/lessons/engine/canvasBridge.js';
                export { TEST_MODULE, TEST_TEXT } from './scripts/fixtures/edu-test-module.js';
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
        plugins: [stubMountPoints],
        logLevel: 'silent',
    });
    app = await import(pathToFileURL(bundlePath).href);
} finally {
    // Remove the bundle at once (as ui-regressions.test.mjs does), so an interrupted run leaves nothing behind.
    await rm(temporaryDirectory, { recursive: true, force: true });
}
const {
    Layout, ToolbarStrip, Navbar, ArchitectureWorkspace, useArchitectureStore, useArchitectureHistory, useTemplatesStore, useTrainingStore,
    registerSlice, useEduUiStore, useLessonSession, setModuleRegistry, restoreWorkspace, loadSlot, createFixtureHelpers, applyLoadedPayload,
    buildSavePayload, focusNode, requestFit, requestInsert, TEST_MODULE, TEST_TEXT,
} = app;

// A stand-in Edu slice, as Stage B registers 'data' and 'trainer'.
let sliceValue = { datasetId: 'iris', note: 'saved with the graphs' };
const restored = [];
registerSlice('edu-shell-test', {
    capture: () => structuredClone(sliceValue),
    restore: (value) => { restored.push(value); sliceValue = value ?? { datasetId: null }; },
});

/** A small workspace: Input → Linear (standard component) → Output, Adam + BCE. */
function smallWorkspace() {
    const helpers = createFixtureHelpers();
    const input = helpers.input('in', [2], { label: 'features' });
    const linear = helpers.linear('lin1', 2, 1, { label: 'neuron' });
    const output = helpers.output('out', { label: 'logit' });
    return helpers.snapshot({
        nodes: [input, linear, output],
        edges: helpers.chain(input, linear, output),
        training: [helpers.optimizer('Adam', { learning_rate: 0.03 }), helpers.loss('bce', output)],
    });
}

let container;
let root;
const originals = {};
beforeEach(() => {
    jsdomErrors.length = 0;
    localStorage.clear();
    // The GPT-2 first-start seeding is covered by DeepFlow's own tests; start from the small workspace instead.
    localStorage.setItem('deepflow:seeded', '1');
    setModuleRegistry({ modules: [TEST_MODULE], text: TEST_TEXT });
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model', drawerOpen: true });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0 });
    useLessonSession.getState().hydrate();
    restoreWorkspace(smallWorkspace());
    restored.length = 0;
    sliceValue = { datasetId: 'iris', note: 'saved with the graphs' };
    originals.confirm = window.confirm;
    originals.createObjectURL = URL.createObjectURL;
    originals.revokeObjectURL = URL.revokeObjectURL;
    originals.anchorClick = dom.window.HTMLAnchorElement.prototype.click;
    originals.inputClick = dom.window.HTMLInputElement.prototype.click;
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
});
afterEach(async () => {
    if (root) await act(() => root.unmount());
    container?.remove();
    window.confirm = originals.confirm;
    URL.createObjectURL = originals.createObjectURL;
    URL.revokeObjectURL = originals.revokeObjectURL;
    dom.window.HTMLAnchorElement.prototype.click = originals.anchorClick;
    dom.window.HTMLInputElement.prototype.click = originals.inputClick;
    localStorage.clear();
    assert.deepEqual(jsdomErrors, [], 'DOM event handlers must not throw uncaught errors');
});
after(async () => {
    dom.window.close();
    for (const [key, descriptor] of installedGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
    }
});

const render = (element) => act(() => root.render(element));
const click = (element) => {
    assert.ok(element, 'Expected an element to click');
    return act(() => element.click());
};
const settle = () => act(() => new Promise((resolve) => setTimeout(resolve, 20)));
const byLabel = (label) => [...container.querySelectorAll('[aria-label]')].find((element) => element.getAttribute('aria-label') === label) ?? null;
const buttonByText = (text) => [...container.querySelectorAll('button')].find((element) => element.textContent.trim() === text) ?? null;

test('PRODUCT_NAME is the only place the product name is written in src/', async () => {
    const offenders = [];
    for await (const file of sourceFiles(join(projectRoot, 'src'))) {
        const name = relative(projectRoot, file).split(sep).join('/');
        if (name === 'src/edu/config.js') continue;
        if ((await readFile(file, 'utf8')).includes('DeepFlow Edu')) offenders.push(name);
    }
    assert.equal(PRODUCT_NAME, 'DeepFlow Edu');
    assert.deepEqual(offenders, [], 'import PRODUCT_NAME from src/edu/config.js instead of writing the name');
});

test('index.html is titled PRODUCT_NAME — PRODUCT_TAGLINE (static HTML cannot import the constant)', async () => {
    const html = await readFile(join(projectRoot, 'index.html'), 'utf8');
    assert.equal(html.match(/<title>([^<]*)<\/title>/)?.[1], `${PRODUCT_NAME} — ${PRODUCT_TAGLINE}`);
});

test('the toolbar shows Learn, Architecture, Data and Training in that order, all enabled, with data-tab', async () => {
    const changes = [];
    await render(h(ToolbarStrip, { activeTab: 'learn', onChange: (tab) => changes.push(tab), onShowCode: () => {} }));
    const tabs = [...container.querySelectorAll('.df-toolbar-tabs [data-tab]')];
    assert.deepEqual(tabs.map((tab) => tab.dataset.tab), ['learn', 'architecture', 'data', 'training']);
    assert.deepEqual(tabs.map((tab) => tab.textContent.trim()), ['Learn', 'Architecture', 'Data', 'Training']);
    assert.ok(tabs.every((tab) => tab.tagName === 'BUTTON' && !tab.disabled), 'Data is no longer "soon"');
    assert.equal(tabs[0].getAttribute('aria-current'), 'page');
    assert.ok(!container.querySelector('.df-tab-soon'));
    await click(tabs[2]);
    assert.deepEqual(changes, ['data']);
});

test('Save adds edu.slices to DeepFlow’s file; Load restores them only when the file has them', async () => {
    const blobs = [];
    URL.createObjectURL = (blob) => { blobs.push(blob); return 'blob:edu-shell-test'; };
    URL.revokeObjectURL = () => {};
    dom.window.HTMLAnchorElement.prototype.click = function captureDownload() {};
    await render(h(ToolbarStrip, { activeTab: 'architecture', onChange: () => {}, onShowCode: () => {} }));
    await click(byLabel('Save'));
    assert.equal(blobs.length, 1);
    const payload = JSON.parse(await blobs[0].text());
    assert.equal(payload.version, 'deepflow-v2');
    assert.deepEqual(Object.keys(payload.architecture.nodes).sort(), ['in', 'lin1', 'out']);
    assert.ok(payload.templates && payload.training, 'the DeepFlow parts are unchanged');
    assert.equal(payload.edu.version, 1);
    assert.deepEqual(payload.edu.slices['edu-shell-test'], { datasetId: 'iris', note: 'saved with the graphs' });
    assert.deepEqual(buildSavePayload().edu.slices['edu-shell-test'], payload.edu.slices['edu-shell-test']);

    // Load goes through a file input; drive its change handler with a fake file.
    let fileInput = null;
    dom.window.HTMLInputElement.prototype.click = function captureInput() { fileInput = this; };
    const load = async (content) => {
        await click(byLabel('Load'));
        assert.ok(fileInput, 'Load opens a file input');
        Object.defineProperty(fileInput, 'files', { configurable: true, value: [{ name: 'model.json', text: async () => JSON.stringify(content) }] });
        await act(async () => { await fileInput.onchange(); });
    };
    const legacy = { version: 'deepflow-v2', architecture: { nodes: {}, edges: [] }, templates: {}, training: { nodes: {}, edges: [] } };
    await load(legacy);
    assert.deepEqual(useArchitectureStore.getState().graph, legacy.architecture, 'a file without edu loads exactly as in DeepFlow');
    assert.deepEqual(restored, [], 'no slice is touched by a legacy file');

    await load(payload);
    assert.deepEqual(Object.keys(useArchitectureStore.getState().graph.nodes).sort(), ['in', 'lin1', 'out']);
    assert.deepEqual(restored.at(-1), { datasetId: 'iris', note: 'saved with the graphs' });
    await act(() => applyLoadedPayload({ ...legacy, edu: { version: 1, slices: {} } }));
    assert.equal(restored.at(-1), undefined, 'a file with edu but without this slice resets it to its default');
});

test('Reset empties the graphs, resets every slice and, in a lesson, says the progress is kept', async () => {
    const prompts = [];
    window.confirm = (message) => { prompts.push(message); return true; };
    await render(h(ToolbarStrip, { activeTab: 'architecture', onChange: () => {}, onShowCode: () => {} }));
    await click(byLabel('Reset'));
    assert.match(prompts[0], /Reset architecture and training graphs/);
    assert.deepEqual(restored, [undefined], 'the dataset binding and trainer settings return to their defaults');
    assert.ok(!Object.values(useArchitectureStore.getState().graph.nodes).some((node) => node.id === 'lin1'));
    assert.equal(useArchitectureHistory.getState().undoCount, 0);

    await act(() => useLessonSession.getState().startModule('demo'));
    await click(byLabel('Reset'));
    assert.match(prompts[1], /lesson.*progress is kept/i);
});

test('Code toggles the code panel and leaves Learn for the model', async () => {
    await render(h(Layout));
    const code = byLabel('Code');
    assert.equal(code.getAttribute('aria-expanded'), 'false');
    await click(code);
    assert.equal(useEduUiStore.getState().codeOpen, true);
    assert.equal(useEduUiStore.getState().activeTab, 'architecture', 'code belongs to the model, not the Learn home');
    assert.equal(byLabel('Code').getAttribute('aria-expanded'), 'true');
    await click(byLabel('Code'));
    assert.equal(useEduUiStore.getState().codeOpen, false);
    await click(byLabel('Code'));
    await click(container.querySelector('.df-toolbar-tabs [data-tab="learn"]'));
    assert.equal(useEduUiStore.getState().codeOpen, false, 'the code panel never floats over Learn');
});

test('Layout opens on Learn, brands with PRODUCT_NAME and docks the lesson panel beside the stage', async () => {
    await render(h(Layout));
    await settle();
    assert.equal(document.title, `${PRODUCT_NAME} — ${PRODUCT_TAGLINE}`);
    assert.equal(container.querySelector('.df-brand-name').textContent, PRODUCT_NAME);
    assert.ok(container.querySelector('.df-brand svg'), 'the Brain icon stays');
    assert.ok(container.querySelector('.edu-main > .edu-stage .edu-learn'), 'Learn is the first view');
    assert.ok(!container.querySelector('aside[aria-label="Lesson"]'));

    await act(() => useLessonSession.getState().startModule('demo'));
    await settle();
    const aside = container.querySelector('.edu-main > aside[aria-label="Lesson"]');
    assert.ok(aside, 'the lesson panel is a sibling of the stage, so it never covers the canvas');
    assert.ok(aside.previousElementSibling.classList.contains('edu-stage'));
    assert.ok(container.querySelector('.edu-stage [data-graph-pane="main"]'), 'the architecture canvas sits in the stage');

    await click(container.querySelector('.df-toolbar-tabs [data-tab="training"]'));
    await settle();
    const split = container.querySelector('.edu-stage .edu-training-split');
    assert.ok(split?.querySelector('.edu-training-canvas') && split.querySelector('.edu-trainer-dock [data-stub="TrainerPanel"]'), 'training canvas above the trainer');
    assert.ok(container.querySelector('aside[aria-label="Lesson"]'));

    await click(container.querySelector('.df-toolbar-tabs [data-tab="data"]'));
    await settle();
    assert.ok(container.querySelector('.edu-stage .df-tab-pane [data-stub="DataWorkspace"]'), 'the Data tab mounts DataWorkspace');
    assert.ok(container.querySelector('aside[aria-label="Lesson"]'));

    await click(container.querySelector('.df-toolbar-tabs [data-tab="learn"]'));
    assert.ok(!container.querySelector('aside[aria-label="Lesson"]'), 'no lesson panel on Learn');
    assert.equal(useLessonSession.getState().mode, 'lesson', 'visiting Learn keeps the lesson open');
});

test('an example chosen during a lesson asks first, then opens in the sandbox with the lesson saved', async () => {
    await act(() => useLessonSession.getState().startModule('demo'));
    await act(() => useLessonSession.getState().skipWarmup());
    useArchitectureStore.getState().patchNode('lin1', { label: 'my edit' });
    const prompts = [];
    let answer = false;
    window.confirm = (message) => { prompts.push(message); return answer; };
    await render(h(Navbar));
    assert.equal(container.querySelector('.df-brand-name').textContent, PRODUCT_NAME);
    const openExamples = () => click(buttonByText('Examples'));
    await openExamples();
    const firstExample = () => container.querySelector('[aria-label="Example architectures"] button');
    await click(firstExample());
    assert.equal(prompts[0], 'Open the example in your sandbox? Your lesson is saved.');
    assert.equal(useLessonSession.getState().mode, 'lesson', 'Cancel keeps the lesson');
    assert.equal(useArchitectureStore.getState().graph.nodes.lin1.label, 'my edit');

    answer = true;
    await click(firstExample());
    assert.equal(useLessonSession.getState().mode, 'sandbox');
    assert.equal(useEduUiStore.getState().activeTab, 'architecture');
    assert.ok(!useArchitectureStore.getState().graph.nodes.lin1, 'the example replaced the workspace');
    assert.equal(loadSlot('demo').graph.nodes.lin1.label, 'my edit', 'the lesson workspace was saved first');
});

test('the canvas bridge inserts with DeepFlow’s placement and returns the id; fit and focusNode reach the main pane', async () => {
    useEduUiStore.setState({ activeTab: 'architecture' });
    await render(h(ArchitectureWorkspace));
    await settle();
    let id = null;
    await act(() => { id = requestInsert({ key: 'torch.nn.ReLU' }); });
    const graph = useArchitectureStore.getState().graph;
    assert.ok(id && graph.nodes[id], 'insert returns the new node id synchronously');
    assert.equal(graph.nodes[id].libraryKey, 'torch.nn.ReLU');
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, [id], 'the new block is selected');

    let linear = null;
    await act(() => { linear = requestInsert({ standardComponentKey: 'linear' }); });
    const component = useArchitectureStore.getState().graph.nodes[linear];
    assert.equal(component?.kind, 'supernode');
    assert.equal(useTemplatesStore.getState().templates[component.templateId]?.standardKey, 'linear');

    // jsdom measures no node sizes, so the viewport math of fitView and setCenter yields NaN coordinates there.
    const consoleError = console.error;
    console.error = (...args) => { if (!String(args[0]).startsWith('Received NaN for the')) consoleError(...args); };
    try {
        let fitted = false;
        await act(() => { fitted = requestFit(); });
        assert.equal(fitted, true, 'fit reached the main pane’s fitView');
        await act(() => focusNode('lin1'));
        assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, ['lin1']);
        assert.equal(useArchitectureStore.getState().expandedNodeId, 'lin1', 'focusNode opens the block’s settings');
        // Let the 220 ms viewport transitions finish while jsdom still runs animation frames.
        await act(() => new Promise((resolve) => setTimeout(resolve, 450)));
    } finally {
        console.error = consoleError;
    }

    await act(() => root.unmount());
    root = createRoot(container);
    assert.equal(requestInsert({ key: 'torch.nn.ReLU' }), null, 'without a canvas the bridge is a no-op');
    assert.equal(requestFit(), false);
    assert.equal(useTrainingStore.getState().graph.nodes.loss.type, 'bce', 'the training graph is untouched');
});
