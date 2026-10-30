import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// Real React handlers and Zustand stores in jsdom. Layout, sticky headers,
// resizing and the bottom sheet are verified in the browser, not here.
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const jsdomErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => jsdomErrors.push(error));
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/', pretendToBeVisual: true, virtualConsole });
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

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');
const temporaryDirectory = await mkdtemp(join(projectRoot, '.model-table-ui-'));
let app;
try {
    const bundlePath = join(temporaryDirectory, 'workspace.mjs');
    await build({
        stdin: {
            contents: `
                export { default as ArchitectureWorkspace } from './src/components/architecture/ArchitectureWorkspace.jsx';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useTemplatesStore } from './src/stores/templatesStore.js';
                export { useTrainingStore } from './src/stores/trainingStore.js';
                export { useArchitectureHistory, startArchitectureSession } from './src/stores/architectureHistory.js';
                export { gpt2Storyboard } from './src/storyboards/gpt2.js';
                export { resnetStoryboard } from './src/storyboards/resnet.js';
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
const { ArchitectureWorkspace, useArchitectureStore, useTemplatesStore, useTrainingStore, useArchitectureHistory, startArchitectureSession, gpt2Storyboard, resnetStoryboard } = app;
let container;
let root;

beforeEach(async () => {
    jsdomErrors.length = 0;
    localStorage.clear();
    for (const store of [useArchitectureStore, useTemplatesStore, useTrainingStore]) store.setState(store.getInitialState(), true);
    startArchitectureSession(() => gpt2Storyboard({ architectureStore: useArchitectureStore, templatesStore: useTemplatesStore, trainingStore: useTrainingStore }));
    useArchitectureHistory.getState().clear();
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
    await act(() => root.render(h(ArchitectureWorkspace)));
});
afterEach(async () => {
    await act(() => root.unmount());
    container.remove();
    for (const element of document.body.querySelectorAll('.df-variable-suggestions')) element.remove();
    localStorage.clear();
    assert.deepEqual(jsdomErrors, [], 'DOM event handlers must not throw uncaught errors');
});
after(() => {
    dom.window.close();
    for (const [key, descriptor] of installedGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
    }
});

const byAria = (label, scope = container) => scope.querySelector(`[aria-label="${label}"]`);
const click = (element) => {
    assert.ok(element, 'Expected an element to click');
    return act(() => element.click());
};
const key = (element, name, options = {}) => act(() => {
    assert.ok(element, `Expected an element for ${name}`);
    element.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: name, code: name === ' ' ? 'Space' : name, bubbles: true, cancelable: true, ...options }));
});
const setText = (input, text) => act(() => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value').set.call(input, text);
    input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
});
async function typeText(input, text) {
    assert.ok(input, 'Expected a text input');
    await act(() => input.focus());
    await setText(input, '');
    for (const character of text) await setText(input, input.value + character);
}
const mainTable = () => container.querySelector('[data-graph-pane="main"] [aria-label="Model table"]');
const tableRows = (table = mainTable()) => [...table.querySelectorAll('tbody tr[role="row"]')];
const rowOf = (label, table = mainTable()) => {
    const row = tableRows(table).find((candidate) => candidate.querySelector('.df-mt-label')?.textContent === label);
    assert.ok(row, `Expected a table row labelled ${label}`);
    return row;
};
const cellOf = (label, column) => rowOf(label).querySelector(`[data-col="${column}"]`);
const radio = (name) => [...mainTable().querySelectorAll('[role="radio"]')].find((element) => element.textContent === name);
const graph = () => useArchitectureStore.getState().graph;
const templates = () => useTemplatesStore.getState().templates;
const rootNode = (label) => Object.values(graph().nodes).find((node) => node.label === label);
const blockTemplate = () => templates()[rootNode('Transformer blocks').templateId];
const blockNode = (label) => Object.values(blockTemplate().graph.nodes).find((node) => node.label === label);
const globalValue = (name) => graph().parameters.find((parameter) => parameter.name === name).default;
const history = () => useArchitectureHistory.getState();
async function openTable() {
    await click(byAria('Show model table'));
    assert.ok(mainTable(), 'The table opens in the main pane');
}
const nextFrame = () => act(() => new Promise((resolve) => requestAnimationFrame(() => resolve())));
/** Start again from another storyboard, optionally changed before the first render. */
async function seed(storyboard, prepare) {
    await act(() => root.unmount());
    for (const store of [useArchitectureStore, useTemplatesStore, useTrainingStore]) store.setState(store.getInitialState(), true);
    startArchitectureSession(() => storyboard({ architectureStore: useArchitectureStore, templatesStore: useTemplatesStore, trainingStore: useTrainingStore }));
    prepare?.();
    useArchitectureHistory.getState().clear();
    root = createRoot(container);
    await act(() => root.render(h(ArchitectureWorkspace)));
}
const footerStat = (label) => [...mainTable().querySelectorAll('.df-mt-stat')].find((stat) => stat.querySelector('dt').textContent === label);
async function editConfig(label) {
    const cell = cellOf(label, 3);
    await act(() => cell.focus());
    await key(cell, 'Enter');
    const editor = cell.querySelector('.df-mt-editor');
    assert.ok(editor, `${label} opens an inline editor`);
    return editor;
}

test('the Model table is an optional view of the active pane, off by default', async () => {
    assert.equal(mainTable(), null);
    const toggle = byAria('Show model table');
    assert.equal(toggle.getAttribute('aria-pressed'), 'false');
    await openTable();
    assert.equal(toggle.getAttribute('aria-pressed'), 'true');
    assert.equal(tableRows().length, 19, 'Blocks depth by default');
    assert.equal(mainTable().querySelector('[role="treegrid"]').getAttribute('aria-label'), 'Model layers');
    assert.match(mainTable().querySelector('thead').textContent, /% of 163\.0M/);
    assert.match(mainTable().querySelector('.df-mt-footer').textContent, /163,037,184/);
    assert.match(mainTable().querySelector('.df-mt-footer').textContent, /145\.8G/);
    assert.match(mainTable().querySelector('.df-mt-assumptions').textContent, /B = 1 · tokens B × 1024 · fp32/);
    await click(byAria('Close model table', mainTable()));
    assert.equal(mainTable(), null);
    assert.equal(toggle.getAttribute('aria-pressed'), 'false');
    assert.equal(history().undoCount, 0, 'Showing the table is view state, not an edit');
});

test('depth, disclosure and arrow keys move through the model tree', async () => {
    await openTable();
    await click(radio('Root'));
    assert.equal(tableRows().length, 9);
    assert.equal(radio('Root').getAttribute('aria-checked'), 'true');
    await click(radio('Ops'));
    assert.equal(tableRows().length, 109);
    assert.ok(tableRows().some((row) => row.querySelector('[data-col="1"]').textContent === '×144'), 'Attention heads repeat 144 times per pass');
    await click(radio('Blocks'));
    assert.equal(tableRows().length, 19);

    const blocks = rowOf('Transformer blocks');
    assert.equal(blocks.getAttribute('aria-expanded'), 'true');
    assert.equal(blocks.getAttribute('aria-level'), '1');
    assert.equal(rowOf('mlp fc').getAttribute('aria-level'), '2');
    await click(blocks.querySelector('.df-mt-chevron'));
    assert.equal(tableRows().length, 9);
    assert.equal(rowOf('Transformer blocks').getAttribute('aria-expanded'), 'false');

    const first = cellOf('tokens', 0);
    await act(() => first.focus());
    assert.equal(first.tabIndex, 0);
    await key(first, 'ArrowDown');
    assert.equal(document.activeElement, cellOf('token embedding', 0));
    assert.equal(cellOf('token embedding', 0).tabIndex, 0);
    assert.equal(first.tabIndex, -1, 'One roving tab stop');
    const stack = cellOf('Transformer blocks', 0);
    await act(() => stack.focus());
    await key(stack, 'ArrowRight');
    assert.equal(rowOf('Transformer blocks').getAttribute('aria-expanded'), 'true');
    assert.equal(tableRows().length, 19);
    await key(cellOf('Transformer blocks', 0), 'ArrowLeft');
    assert.equal(tableRows().length, 9);
    await key(cellOf('Transformer blocks', 0), 'ArrowRight');
    await act(() => cellOf('mlp fc', 0).focus());
    await key(cellOf('mlp fc', 0), 'ArrowLeft');
    assert.equal(document.activeElement, cellOf('Transformer blocks', 0), 'Left on a leaf moves to its parent row');
    await key(cellOf('Transformer blocks', 0), 'End');
    assert.equal(document.activeElement, cellOf('Transformer blocks', 6));
});

test('a root setting is edited in place as one undo step', async () => {
    await openTable();
    const editor = await editConfig('Dropout');
    const input = editor.querySelector('input[aria-label="Dropout p"]');
    assert.equal(input.value, '0.1');
    await typeText(input, '0.2');
    await key(input, 'Enter');
    assert.equal(cellOf('Dropout', 3).querySelector('.df-mt-editor'), null, 'Enter commits and closes the editor');
    assert.equal(rootNode('Dropout').parameters.p, 0.2);
    assert.equal(history().undoCount, 1);
    assert.equal(history().undoLabel, 'Set Dropout p');
    assert.match(cellOf('Dropout', 3).textContent, /0\.2/);
    await act(() => history().undo());
    assert.equal(rootNode('Dropout').parameters.p, 0.1);
    assert.match(cellOf('Dropout', 3).textContent, /0\.1/);
    await act(() => history().redo());
    assert.equal(rootNode('Dropout').parameters.p, 0.2);
});

test('Symbols mode edits a shared definition for every use and reports validation on the cells', async () => {
    await openTable();
    await click(radio('Symbols'));
    const before = graph();
    const dropoutCell = cellOf('attention projection dropout', 3);
    assert.match(dropoutCell.getAttribute('title'), /Transformer block definition · 12 uses/);
    assert.match(dropoutCell.textContent, /\$resid_dropout/);
    const editor = await editConfig('attention projection dropout');
    const input = editor.querySelector('input[aria-label="attention projection dropout p"]');
    assert.equal(input.value, '$resid_dropout');
    await typeText(input, '0.2');
    await key(input, 'Enter');
    assert.equal(blockNode('attention projection dropout').parameters.p, 0.2);
    assert.equal(graph(), before, 'The root graph is unchanged');
    await act(() => history().undo());
    assert.equal(blockNode('attention projection dropout').parameters.p, '$resid_dropout');

    const macsBefore = cellOf('mlp fc', 5).textContent;
    assert.match(macsBefore, /29\.0G/);
    const fcEditor = await editConfig('mlp fc');
    const out = fcEditor.querySelector('input[aria-label="mlp fc out_features"]');
    assert.equal(out.value, '$mlp_width');
    await typeText(out, '2 * d_model');
    await key(out, 'Enter');
    assert.equal(blockNode('mlp fc').parameterValues.out_features, '2 * d_model');
    assert.match(cellOf('mlp fc', 5).textContent, /14\.5G/, 'Half the MLP width halves its MACs');
    const projection = cellOf('mlp proj', 0);
    assert.equal(projection.getAttribute('aria-invalid'), 'true');
    assert.match(projection.getAttribute('title'), /MatMul inner dimensions mismatch: 1536 vs 3072/);
    const projectionMacs = cellOf('mlp proj', 5).querySelector('.df-mt-unknown');
    assert.equal(projectionMacs?.textContent, '?', 'MACs of the mismatched layer are unknown');
    assert.match(projectionMacs.getAttribute('title'), /MatMul inner dimensions mismatch: 1536 vs 3072/);
    assert.match(cellOf('lm_head', 5).textContent, /39\.5G/, 'Later layers keep their declared settings');
    const alert = mainTable().querySelector('.df-mt-alert');
    assert.match(alert.textContent, /^1 error · totals use declared settings · Transformer blocks › mlp proj › input × weightᵀ: MatMul inner dimensions mismatch/);
    assert.match(footerStat('MACs / sample').textContent, /≥ 102\.3G.*1 row unresolved/);
    await act(() => history().undo());
    assert.equal(cellOf('mlp proj', 0).getAttribute('aria-invalid'), null);
    assert.match(cellOf('mlp fc', 5).textContent, /29\.0G/);
    assert.equal(mainTable().querySelector('.df-mt-alert'), null, 'A valid model has no error line');
});

test('Values mode follows a reference to its source and keeps formulas read-only', async () => {
    await openTable();
    assert.equal(radio('Values').getAttribute('aria-checked'), 'true');
    const lnCell = cellOf('ln_1', 3);
    assert.match(lnCell.getAttribute('title'), /Sets global d_model/);
    const editor = await editConfig('ln_1');
    const input = editor.querySelector('input[aria-label="ln_1 normalized_shape"]');
    assert.equal(input.value, '768');
    await typeText(input, '1024');
    await key(input, 'Enter');
    assert.equal(globalValue('d_model'), 1024);
    assert.equal(history().undoCount, 1);
    assert.match(cellOf('mlp fc', 3).textContent, /1024\s*→\s*4096/, 'Every layer that uses the global follows');
    await act(() => history().undo());
    assert.equal(globalValue('d_model'), 768);

    const fcEditor = await editConfig('mlp fc');
    const readOnly = fcEditor.querySelector('[aria-label="mlp fc out_features"]');
    assert.equal(readOnly.tagName, 'SPAN');
    assert.equal(readOnly.getAttribute('aria-readonly'), 'true');
    assert.match(readOnly.getAttribute('title'), /4 \* d_model/);
    assert.ok(fcEditor.querySelector('input[aria-label="mlp fc in_features"]'), 'The reference beside it stays editable');
});

test('Escape cancels, invalid input stays open, and $ completion never commits', async () => {
    await openTable();
    let editor = await editConfig('Dropout');
    let input = editor.querySelector('input[aria-label="Dropout p"]');
    await typeText(input, '0.7');
    await key(input, 'Escape');
    assert.equal(cellOf('Dropout', 3).querySelector('.df-mt-editor'), null);
    assert.equal(rootNode('Dropout').parameters.p, 0.1);
    assert.equal(history().undoCount, 0);

    editor = await editConfig('Dropout');
    input = editor.querySelector('input[aria-label="Dropout p"]');
    await typeText(input, '$nope');
    await key(input, 'Enter');
    assert.ok(cellOf('Dropout', 3).querySelector('.df-mt-editor'), 'The editor stays open');
    assert.equal(input.getAttribute('aria-invalid'), 'true');
    assert.match(mainTable().querySelector('.df-mt-focus-text').textContent, /Unknown name \$nope/);
    assert.equal(rootNode('Dropout').parameters.p, 0.1);
    assert.equal(history().undoCount, 0);

    await typeText(input, '$');
    assert.equal(input.getAttribute('aria-expanded'), 'true', 'Typing $ offers the parameters in scope');
    await key(input, 'Enter');
    assert.equal(input.value, '$vocab_size', 'Enter chooses the highlighted parameter');
    assert.ok(cellOf('Dropout', 3).querySelector('.df-mt-editor'), 'Choosing a suggestion does not commit');
    assert.equal(rootNode('Dropout').parameters.p, 0.1);
    await key(input, 'Escape');
    await key(input, 'Escape');
    assert.equal(cellOf('Dropout', 3).querySelector('.df-mt-editor'), null);
    assert.equal(history().undoCount, 0);
});

test('keys typed in the table never reach the canvas', async () => {
    await openTable();
    const head = rootNode('lm_head');
    await act(() => useArchitectureStore.getState().selectOnly(head.id));
    const cell = cellOf('lm_head', 2);
    await act(() => cell.focus());
    for (const name of ['Delete', 'Backspace', ' ']) await key(cell, name);
    assert.ok(graph().nodes[head.id], 'Delete and Backspace in the table do not delete the selected node');
    assert.equal(container.querySelector('.df-space-hint'), null, 'Space in the table does not start canvas panning');
    const editor = await editConfig('Dropout');
    const input = editor.querySelector('input[aria-label="Dropout p"]');
    for (const name of ['Delete', 'Backspace', ' ']) await key(input, name);
    assert.ok(graph().nodes[head.id]);
    assert.equal(container.querySelector('.df-space-hint'), null);
    await key(input, 'Escape');
    await act(() => useArchitectureStore.getState().selectOnly(head.id));
    await act(() => document.activeElement?.blur());
    await key(document.body, 'Delete');
    assert.equal(graph().nodes[head.id], undefined, 'Control: Delete on the canvas removes the selected node');
});

test('selection follows the table and the canvas', async () => {
    await openTable();
    await click(cellOf('ln_f', 2));
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, [rootNode('ln_f').id]);
    assert.equal(rowOf('ln_f').getAttribute('aria-selected'), 'true');
    await act(() => useArchitectureStore.getState().selectOnly(rootNode('lm_head').id));
    assert.equal(rowOf('lm_head').getAttribute('aria-selected'), 'true');
    assert.equal(rowOf('ln_f').getAttribute('aria-selected'), 'false');

    // A row inside the blocks opens its level on Enter.
    await click(cellOf('mlp fc', 2));
    assert.deepEqual(useArchitectureStore.getState().currentPath, [], 'A click on another level only makes the row active');
    assert.match(mainTable().querySelector('.df-mt-focus-text').textContent, /main › Transformer blocks › mlp fc/);
    const layer = cellOf('mlp fc', 0);
    await act(() => layer.focus());
    await key(layer, 'Enter');
    const state = useArchitectureStore.getState();
    assert.deepEqual(state.currentPath, [rootNode('Transformer blocks').id]);
    assert.deepEqual(state.selectedNodeIds, [blockNode('mlp fc').id]);
    assert.equal(rowOf('mlp fc').getAttribute('aria-selected'), 'true');
    await act(() => useArchitectureStore.getState().goToLevel(0));
    await click(radio('Root'));
    assert.equal(tableRows().length, 9);
    await act(() => useArchitectureStore.getState().openLevel([rootNode('Transformer blocks').id]));
    await act(() => useArchitectureStore.getState().selectOnly(blockNode('gelu').id));
    await act(() => new Promise((resolve) => requestAnimationFrame(resolve)));
    assert.equal(rowOf('gelu').getAttribute('aria-selected'), 'true', 'Selecting on the canvas reveals the row inside a collapsed block');
});

test('large graphs render 300 rows at a time', async () => {
    const graph = useArchitectureStore.getState().graph;
    const input = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
    const nodes = { [input.id]: { ...input, parameters: { ...input.parameters, shape: '(64,)' } } };
    const edges = [];
    let previous = input.id;
    for (let index = 0; index < 400; index++) {
        const id = `relu-${index}`;
        nodes[id] = { id, kind: 'layer', libraryKey: 'torch.nn.ReLU', label: `ReLU ${index}`, position: { x: 0, y: 100 * (index + 1) }, parameters: { inplace: false } };
        edges.push({ id: `edge-${index}`, source: previous, target: id });
        previous = id;
    }
    await act(() => useArchitectureStore.setState({ graph: { nodes, edges, parameters: [] }, currentPath: [], openPaths: [] }));
    await openTable();
    assert.equal(tableRows().length, 300);
    const more = [...mainTable().querySelectorAll('.df-mt-more button')][0];
    assert.equal(more.textContent, 'Show 101 more');
    await click(more);
    assert.equal(tableRows().length, 401);
    assert.equal(mainTable().querySelector('[role="treegrid"]').getAttribute('aria-rowcount'), '402');
});

test('each pane has its own table', async () => {
    await click(byAria('Open to the side'));
    const side = container.querySelector('[data-graph-pane="side"]');
    assert.ok(side, 'A side pane opens');
    await click(byAria('Show model table'));
    assert.ok(side.querySelector('[aria-label="Model table"]'), 'The ribbon targets the active side pane');
    assert.equal(mainTable(), null, 'The main pane keeps its own view');
});

test('disclosure keeps focus on visible controls, and cells read as words', async () => {
    await openTable();
    const grid = mainTable().querySelector('[role="treegrid"]');
    assert.equal(grid.hasAttribute('aria-readonly'), false, 'The grid has editable cells');
    await click(rowOf('Transformer blocks').querySelector('.df-mt-chevron'));
    assert.equal(rowOf('Transformer blocks').getAttribute('aria-expanded'), 'false');
    assert.equal(document.activeElement, cellOf('Transformer blocks', 0), 'The chevron leaves focus on the row header, not on its hidden button');
    await click(rowOf('Transformer blocks').querySelector('.df-mt-chevron'));

    assert.equal(cellOf('mlp fc', 3).getAttribute('aria-label'), 'in_features 768 to out_features 3072');
    assert.equal(cellOf('mlp fc', 3).querySelector('.df-mt-config').getAttribute('aria-hidden'), 'true', 'The drawn digits are not read as 7683072');
    assert.equal(cellOf('lm_head', 3).getAttribute('aria-label'), 'in_features 768 to out_features 50257, no bias');
    assert.equal(cellOf('Transformer blocks', 3).getAttribute('aria-label'), 'count 12, mlp_width 3072');
    assert.equal(cellOf('mlp fc', 2).getAttribute('aria-label'), 'B × 1024 × 3072');
});

test('an invalid input shape keeps the editor open with its reason beside the field', async () => {
    await openTable();
    const shapeBefore = rootNode('tokens').parameters.shape;
    for (const [text, reason] of [
        ['(abc)', /numbers, not names \(abc\)/],
        ['(1024, $foo)', /numbers, not names \(\$foo\)/],
        ['()', /sizes such as/],
        ['(-3)', /positive whole numbers/],
        ['(1.5,)', /positive whole numbers/],
    ]) {
        const editor = cellOf('tokens', 3).querySelector('.df-mt-editor') ?? await editConfig('tokens');
        const input = editor.querySelector('input[aria-label="tokens shape"]');
        await typeText(input, text);
        await key(input, 'Enter');
        assert.ok(cellOf('tokens', 3).querySelector('.df-mt-editor'), `${text} keeps the editor open`);
        assert.equal(input.getAttribute('aria-invalid'), 'true', text);
        const bubble = document.getElementById(input.getAttribute('aria-describedby'));
        assert.ok(bubble && cellOf('tokens', 3).contains(bubble), 'The reason sits next to the field');
        assert.match(bubble.textContent, reason, text);
        assert.equal(rootNode('tokens').parameters.shape, shapeBefore, `${text} writes nothing`);
        assert.equal(history().undoCount, 0);
    }
    const input = cellOf('tokens', 3).querySelector('input[aria-label="tokens shape"]');
    await typeText(input, '(512,)');
    await key(input, 'Enter');
    assert.equal(rootNode('tokens').parameters.shape, '(512,)');
    assert.equal(history().undoCount, 1);
    assert.match(cellOf('token embedding', 2).textContent, /B×512×768/);
});

test('an exposed field bound on its instance is edited where the instance binds it', async () => {
    await seed(resnetStoryboard, () => {
        const block = Object.values(templates()).find((template) => template.name === 'ConvBlock');
        const conv = Object.values(block.graph.nodes).find((node) => node.libraryKey === 'torch.nn.Conv2d');
        useTemplatesStore.getState().toggleFieldExposure(block.id, { nodeId: conv.id, field: 'stride', spec: { name: 'stride', type: 'tuple', default: 1 }, defaultValue: 1, label: 'Stride' });
        const stages = rootNode('ResNet stages');
        useArchitectureStore.getState().patchRootNode(stages.id, { bindings: { ...stages.bindings, stride: 2 } });
    });
    await openTable();
    const cell = cellOf('Conv2d', 3);
    assert.match(cell.textContent, /s=2/);
    assert.match(cell.getAttribute('title'), /stride: Sets stride on ResNet stages · all 3 instances/);
    const editor = await editConfig('Conv2d');
    const input = editor.querySelector('input[aria-label="Conv2d stride"]');
    assert.equal(input.value, '2', 'The editor starts from the value the cell shows');
    await typeText(input, '3');
    await key(input, 'Enter');
    const block = Object.values(templates()).find((template) => template.name === 'ConvBlock');
    assert.equal(rootNode('ResNet stages').bindings.stride, 3, 'The binding the instances read changes');
    assert.equal(block.freeParameters.find((parameter) => parameter.name === 'stride').default, 1, 'The unused default stays');
    assert.match(cellOf('Conv2d', 3).textContent, /s=3/);
    assert.equal(history().undoLabel, 'Set Conv2d stride');
    await act(() => history().undo());
    assert.equal(rootNode('ResNet stages').bindings.stride, 2);
});

test('the Activations lens shows bytes per row, and partial totals say so', async () => {
    await openTable();
    await click(radio('Activations'));
    const header = mainTable().querySelectorAll('thead th');
    assert.equal(header[5].textContent.trim(), 'Act. mem', 'Bytes take the place of MACs');
    assert.match(header[5].getAttribute('title'), /sum of forward op outputs, eager, nothing freed/);
    assert.match(header[6].textContent, /% of 4\.03 GB/);
    assert.match(cellOf('lm_head', 5).textContent, /205\.9 MB/);
    assert.match(cellOf('Transformer blocks', 5).textContent, /3\.81 GB.*318 MB ea/);
    assert.match(mainTable().querySelector('.df-mt-assumptions').textContent, /activations: Σ op outputs, standard modules count one output/);
    assert.match(footerStat('Largest activation').textContent, /^Largest activation205\.9 MBlm_head · B×1024×50257/);

    const stack = rootNode('Transformer blocks');
    await act(() => useArchitectureStore.getState().patchRootNode(stack.id, { count: '$missing' }));
    assert.equal(cellOf('lm_head', 5).textContent, '?', 'Sizes after an unresolved count are unknown');
    assert.match(header[6].textContent, /% of ≥/, 'The share denominator is a lower bound');
    assert.match(footerStat('Largest activation').querySelector('dd').textContent, /^≥ /);
    await click(radio('MACs'));
    assert.equal(cellOf('lm_head', 5).textContent, '?', 'lm_head MACs are unknown, not computed from a placeholder');
    assert.match(mainTable().querySelector('.df-mt-alert').textContent, /^1 error · totals use declared settings · Transformer blocks: /);
});

test('read-only values explain themselves, and an input method keeps Enter', async () => {
    await openTable();
    const fc = cellOf('mlp fc', 3);
    await click(fc.querySelector('[data-field="out_features"]'));
    assert.equal(fc.querySelector('.df-mt-editor'), null, 'A computed value does not open an editor');
    assert.match(mainTable().querySelector('.df-mt-focus-text').textContent, /^out_features: Computed by 4 \* d_model/);

    const editor = await editConfig('Dropout');
    const input = editor.querySelector('input[aria-label="Dropout p"]');
    await typeText(input, '0.3');
    await key(input, 'Enter', { isComposing: true });
    assert.ok(cellOf('Dropout', 3).querySelector('.df-mt-editor'), 'Enter while composing picks the candidate');
    assert.equal(rootNode('Dropout').parameters.p, 0.1);
    await key(input, 'Enter');
    assert.equal(rootNode('Dropout').parameters.p, 0.3);
});
