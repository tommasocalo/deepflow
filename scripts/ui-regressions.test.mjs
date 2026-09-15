import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// Exercise real React event handlers and Zustand stores without pretending
// jsdom can verify canvas geometry, dragging, or visual layout.
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const jsdomErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => jsdomErrors.push(error));
const dom = new JSDOM('<!doctype html><html><body></body></html>', {
    url: 'http://localhost/',
    pretendToBeVisual: true,
    virtualConsole,
});
// jsdom has no DOMMatrix API. The graph harness has identity zoom and no
// measured layout; this supports ReactFlow's refresh without testing geometry.
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

// Import React DOM after installing document so its event system uses the DOM
// implementation. Keep packages external so tests and components share React.
const { act, createElement: h, useState } = await import('react');
const { createRoot } = await import('react-dom/client');
const { ReactFlow, ReactFlowProvider } = await import('@xyflow/react');
const temporaryDirectory = await mkdtemp(join(projectRoot, '.ui-regressions-'));
let app;
try {
    const bundlePath = join(temporaryDirectory, 'components.mjs');
    await build({
        stdin: {
            contents: `
                export { default as ParameterField } from './src/components/architecture/ParameterField.jsx';
                export { default as LayerNodeView } from './src/components/architecture/LayerNodeView.jsx';
                export { default as BlockPalette } from './src/components/architecture/BlockPalette.jsx';
                export { default as GradientEdge } from './src/components/architecture/GradientEdge.jsx';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useTemplatesStore } from './src/stores/templatesStore.js';
                export { createLayerNode, createTemplate, createEdge, emptyGraph } from './src/domain/architecture/index.js';
                export { getEntry } from './src/domain/library/index.js';
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
const { ParameterField, LayerNodeView, BlockPalette, GradientEdge, useArchitectureStore, useTemplatesStore, createLayerNode, createTemplate, createEdge, emptyGraph, getEntry } = app;
let container;
let root;

beforeEach(() => {
    jsdomErrors.length = 0;
    localStorage.clear();
    useTemplatesStore.setState({ templates: {} });
    useArchitectureStore.setState({ graph: emptyGraph(), currentPath: [], selectedNodeIds: [], expandedNodeId: null, recentLibraryKeys: [] });
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
});
afterEach(async () => {
    await act(() => root.unmount());
    container.remove();
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

const render = (element) => act(() => root.render(element));
const click = (element) => {
    assert.ok(element, 'Expected an element to click');
    return act(() => element.click());
};
const byAria = (label) => container.querySelector(`[aria-label="${label}"]`);
const byFieldLabel = (name) => {
    const label = [...container.querySelectorAll('label')].find((element) => element.textContent.replace(/\*/g, '').trim() === name);
    return label ? document.getElementById(label.htmlFor) : null;
};
const setText = (input, text) => act(() => {
    assert.ok(input, 'Expected a text input');
    // The native setter reproduces user input without bypassing React's
    // controlled-input change tracker with a direct JS property assignment.
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value').set.call(input, text);
    input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
});
async function typeText(input, text) {
    await act(() => input.focus());
    await setText(input, '');
    for (const character of text) await setText(input, input.value + character);
}
const blur = (input) => act(() => input.blur());
const graphNodes = () => Object.values(useArchitectureStore.getState().graph.nodes);

function ParameterHarness({ initial = null, type = 'float', onValue }) {
    const [value, setValue] = useState(initial);
    return h(ParameterField, {
        param: { name: 'width', type }, value,
        onChange(next) { setValue(next); onValue?.(next); },
    });
}

function PaletteHarness() {
    return h(ReactFlowProvider, null,
        h('div', null, h(BlockPalette), h('div', {
            'data-architecture-canvas': true,
            ref(element) {
                if (element) element.getBoundingClientRect = () => ({ left: 300, top: 100, width: 1000, height: 600, x: 300, y: 100, right: 1300, bottom: 700 });
            },
        })),
    );
}

const nodeTypes = { layer: LayerNodeView };
function GraphHarness() {
    const graph = useArchitectureStore((state) => state.graph);
    return h(ReactFlowProvider, null, h(ReactFlow, {
        width: 1000, height: 700,
        nodes: Object.values(graph.nodes).map((node) => ({ id: node.id, type: 'layer', position: node.position, data: { node }, width: 260, height: 160 })),
        edges: [], nodeTypes,
    }));
}

test('numeric editing retains decimal zeros and exponent text until blur', async () => {
    let latest;
    await render(h(ParameterHarness, { onValue: (value) => { latest = value; } }));
    const input = container.querySelector('input');
    await typeText(input, '1.05');
    assert.equal(input.value, '1.05');
    assert.equal(latest, '1.05');
    await blur(input);
    assert.equal(latest, 1.05);
    await typeText(input, '1.00e-2');
    assert.equal(input.value, '1.00e-2');
    assert.equal(latest, '1.00e-2');
    await blur(input);
    assert.equal(latest, 0.01);
});

test('numeric fields preserve expressions and unfinished literals without NaN or integer truncation', async () => {
    let latest;
    await render(h(ParameterHarness, { type: 'int', onValue: (value) => { latest = value; } }));
    const input = container.querySelector('input');
    for (const text of ['$width', '64 * 2 ** i', '1e-', '-']) {
        await typeText(input, text);
        await blur(input);
        assert.equal(input.value, text);
        assert.equal(latest, text);
    }
    await typeText(input, '1.05');
    await blur(input);
    assert.equal(latest, 1.05, 'Integer validation must see the invalid decimal instead of a silently truncated 1');
    await typeText(input, '');
    assert.equal(latest, null);
});

test('repeated parameter names have unique stable label associations', async () => {
    await render(h('div', null, h(ParameterHarness), h(ParameterHarness)));
    const inputs = [...container.querySelectorAll('input')];
    const ids = inputs.map((input) => input.id);
    assert.equal(new Set(ids).size, 2);
    assert.deepEqual([...container.querySelectorAll('label')].map((label) => label.htmlFor), ids);
    await typeText(inputs[0], '768');
    assert.deepEqual([...container.querySelectorAll('input')].map((input) => input.id), ids);
    assert.equal(inputs[1].value, '');
});

test('card clicks expand a layer, parameter editing stays open, and explicit controls toggle it', async () => {
    const entry = getEntry('torch.nn.Linear');
    const node = createLayerNode({ id: 'linear', libraryKey: entry.key, entry, position: { x: 10, y: 10 } });
    useArchitectureStore.setState({ graph: { nodes: { [node.id]: node }, edges: [] } });
    await render(h(GraphHarness));
    assert.equal(byAria('Expand Linear').getAttribute('aria-expanded'), 'false');
    await click(container.querySelector('.react-flow__node[data-id="linear"] h3'));
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'linear');
    const input = byFieldLabel('in_features');
    await click(input);
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'linear');
    await typeText(input, '768');
    await blur(input);
    assert.equal(useArchitectureStore.getState().graph.nodes.linear.parameters.in_features, 768);
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'linear');
    const collapse = byAria('Collapse Linear');
    assert.equal(collapse.getAttribute('aria-expanded'), 'true');
    await click(collapse);
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    await click(container.querySelector('.react-flow__node[data-id="linear"] .react-flow__handle'));
    assert.equal(useArchitectureStore.getState().expandedNodeId, null, 'A connector click must not resize its node');
    await click(byAria('Expand Linear'));
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'linear');
    assert.equal(byFieldLabel('in_features').value, '768');
});

test('opening a library category does not add a node; a search result adds exactly the chosen layer', async () => {
    await render(h(PaletteHarness));
    const category = [...container.querySelectorAll('.df-library-category')].find((element) => element.textContent.startsWith('Normalization'));
    assert.equal(category.getAttribute('aria-expanded'), 'false');
    await click(category);
    assert.equal(category.getAttribute('aria-expanded'), 'true');
    assert.equal(graphNodes().length, 0);
    await typeText(byAria('Search layers'), 'layer norm');
    const add = byAria('Add LayerNorm');
    assert.ok(add, 'Tokenized search should find LayerNorm');
    await click(add);
    const nodes = graphNodes();
    assert.equal(nodes.length, 1);
    assert.equal(nodes[0].libraryKey, 'torch.nn.LayerNorm');
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, [nodes[0].id]);
    assert.equal(useArchitectureStore.getState().expandedNodeId, nodes[0].id);
    assert.ok(Number.isFinite(nodes[0].position.x) && Number.isFinite(nodes[0].position.y));
});

test('Templates tab exposes its panel and adds a real reusable instance', async () => {
    const template = createTemplate({ id: 'decoder-template', name: 'Decoder block', graph: emptyGraph() });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    await render(h(PaletteHarness));
    const templatesTab = container.querySelector('#templates-tab');
    await click(templatesTab);
    assert.equal(templatesTab.getAttribute('aria-selected'), 'true');
    assert.equal(container.querySelector('[role="tabpanel"]').getAttribute('aria-labelledby'), templatesTab.id);
    assert.ok(byAria('Search templates'));
    assert.equal(graphNodes().length, 0);
    await click(byAria('Add Decoder block template'));
    assert.equal(graphNodes().length, 1);
    assert.equal(graphNodes()[0].kind, 'supernode');
    assert.equal(graphNodes()[0].templateId, template.id);
});

test('collapsed library reopens and focuses search with the keyboard shortcut', async () => {
    await render(h(PaletteHarness));
    await click(byAria('Collapse block library'));
    assert.ok(byAria('Open block library'));
    assert.equal(byAria('Search layers'), null);
    await act(() => window.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true })));
    await act(() => new Promise((resolve) => requestAnimationFrame(resolve)));
    const search = byAria('Search layers');
    assert.ok(search);
    assert.equal(document.activeElement, search);
    assert.equal(graphNodes().length, 0);
});


function EdgeHarness({ selected, menuVersion = 0 }) {
    const edge = useArchitectureStore((state) => state.graph.edges[0]);
    // ReactFlow supplies the real EdgeLabelRenderer portal. Coordinates are
    // fixed because this test verifies menu state and mutations, not geometry.
    return h(ReactFlowProvider, null, h(ReactFlow, {
        width: 1000, height: 700, nodes: [], edges: [],
    }, h('svg', null, h(GradientEdge, {
        id: edge.id, selected,
        sourceX: 100, sourceY: 100, targetX: 100, targetY: 300,
        sourcePosition: 'bottom', targetPosition: 'top',
        data: { gradientOp: edge.gradientOp, gradientOpLambda: edge.gradientOpLambda, menuVersion },
    }))));
}

test('gradient menu writes the edge operator, dismisses, and reopens on selection or a new click', async () => {
    const entry = getEntry('torch.nn.Linear');
    const source = createLayerNode({ id: 'source', libraryKey: entry.key, entry, position: { x: 100, y: 100 } });
    const target = createLayerNode({ id: 'target', libraryKey: entry.key, entry, position: { x: 100, y: 300 } });
    const edge = createEdge({ id: 'connection', source: source.id, target: target.id });
    useArchitectureStore.setState({ graph: { nodes: { source, target }, edges: [edge] } });
    const picker = () => container.querySelector('.df-gradient-picker');
    const option = (label) => [...picker().querySelectorAll('button')].find((button) => button.textContent === label);

    await render(h(EdgeHarness, { selected: false }));
    assert.equal(picker(), null);
    await render(h(EdgeHarness, { selected: true }));
    assert.ok(picker());
    await click(option('Detach'));
    assert.equal(useArchitectureStore.getState().graph.edges[0].gradientOp, 'detach');
    assert.equal(picker(), null);

    await render(h(EdgeHarness, { selected: false }));
    await render(h(EdgeHarness, { selected: true }));
    assert.ok(picker(), 'Reselecting the edge opens a fresh picker');
    assert.equal(option('Detach').getAttribute('aria-pressed'), 'true');
    await click(option('Normal'));
    assert.equal(useArchitectureStore.getState().graph.edges[0].gradientOp, 'none');
    assert.equal(picker(), null);
    await render(h(EdgeHarness, { selected: true, menuVersion: 1 }));
    assert.ok(picker(), 'Clicking an already selected edge opens the dismissed picker again');
});

test('library tabs move keyboard focus and their single tab stop with ArrowRight', async () => {
    await render(h(PaletteHarness));
    const layers = container.querySelector('#layers-tab');
    const templates = container.querySelector('#templates-tab');
    assert.equal(layers.tabIndex, 0);
    assert.equal(templates.tabIndex, -1);
    await act(() => layers.focus());
    await act(() => layers.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true })));
    assert.equal(document.activeElement, templates);
    assert.equal(templates.getAttribute('aria-selected'), 'true');
    assert.equal(templates.tabIndex, 0);
    assert.equal(layers.tabIndex, -1);
    assert.equal(container.querySelector('[role="tabpanel"]').getAttribute('aria-labelledby'), templates.id);
    assert.equal(graphNodes().length, 0);
});
