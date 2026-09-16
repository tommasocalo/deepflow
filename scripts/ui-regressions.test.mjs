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
const { act, createElement: h, useState, useEffect } = await import('react');
const { createRoot } = await import('react-dom/client');
const { ReactFlow, ReactFlowProvider, useStoreApi } = await import('@xyflow/react');
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
                export { default as ArchitectureCanvas } from './src/components/architecture/ArchitectureCanvas.jsx';
                export { default as ArchitectureEdge } from './src/components/architecture/ArchitectureEdge.jsx';
                export { default as TensorPreview } from './src/components/architecture/TensorPreview.jsx';
                export { default as NodeInspector } from './src/components/architecture/NodeInspector.jsx';
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
const { ParameterField, LayerNodeView, BlockPalette, ArchitectureCanvas, ArchitectureEdge, NodeInspector, TensorPreview, useArchitectureStore, useTemplatesStore, createLayerNode, createTemplate, createEdge, emptyGraph, getEntry } = app;
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
        h('div', {
            'data-architecture-canvas': true,
            style: { position: 'relative' },
            ref(element) {
                if (element) element.getBoundingClientRect = () => ({ left: 0, top: 100, width: 1300, height: 600, x: 0, y: 100, right: 1300, bottom: 700 });
            },
        }, h(BlockPalette)),
    );
}

const nodeTypes = { layer: LayerNodeView };
function GraphHarness() {
    const graph = useArchitectureStore((state) => state.graph);
    return h(ReactFlowProvider, null, h(ReactFlow, {
        width: 1000, height: 700,
        nodes: Object.values(graph.nodes).map((node) => ({ id: node.id, type: 'layer', position: node.position, data: { node }, width: 240, height: 44 })),
        edges: [], nodeTypes,
    }), h(NodeInspector));
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

test('compact nodes open the inspector and editing keeps node geometry unchanged', async () => {
    const entry = getEntry('torch.nn.Linear');
    const node = createLayerNode({ id: 'linear', libraryKey: entry.key, entry, position: { x: 10, y: 10 } });
    useArchitectureStore.setState({ graph: { nodes: { [node.id]: node }, edges: [] } });
    await render(h(GraphHarness));
    const block = container.querySelector('.react-flow__node[data-id="linear"] .df-block');
    const size = block.style.cssText;
    assert.equal(container.querySelector('.df-inspector'), null);
    assert.equal(block.querySelector('input'), null);
    await click(block);
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'linear');
    assert.ok(byAria('Linear inspector'));
    const input = byFieldLabel('in_features');
    await click(input);
    await typeText(input, '768');
    await blur(input);
    assert.equal(useArchitectureStore.getState().graph.nodes.linear.parameters.in_features, 768);
    assert.equal(useArchitectureStore.getState().expandedNodeId, 'linear');
    assert.equal(block.style.cssText, size, 'Opening and editing must not expand the block');
    await click(byAria('Close inspector'));
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    await click(block.querySelector('.react-flow__handle'));
    assert.equal(useArchitectureStore.getState().expandedNodeId, null, 'Connecting must not open the inspector');
    await click(byAria('Inspect Linear'));
    assert.equal(byAria('Inspect Linear').getAttribute('aria-expanded'), 'true');
    assert.equal(byFieldLabel('in_features').value, '768');
});

test('opening a library category does not add a node; a search result adds exactly the chosen layer', async () => {
    await render(h(PaletteHarness));
    const category = [...container.querySelectorAll('.df-library-category')].find((element) => element.textContent.startsWith('Normalization'));
    assert.equal(category.getAttribute('aria-expanded'), 'false');
    await click(category);
    assert.equal(category.getAttribute('aria-expanded'), 'true');
    assert.equal(graphNodes().length, 0);
    await typeText(byAria('Search blocks'), 'layer norm');
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

test('Templates category expands without adding nodes and exposes reusable instances to global search', async () => {
    const template = createTemplate({ id: 'decoder-template', name: 'Decoder block', graph: emptyGraph() });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    await render(h(PaletteHarness));
    const category = [...container.querySelectorAll('.df-library-category')].find((element) => element.textContent.startsWith('Templates'));
    assert.equal(category.getAttribute('aria-expanded'), 'false');
    assert.equal(byAria('Add Decoder block template'), null);
    await click(category);
    assert.equal(category.getAttribute('aria-expanded'), 'true');
    assert.ok(document.getElementById(category.getAttribute('aria-controls')).contains(byAria('Add Decoder block template')));
    assert.equal(graphNodes().length, 0);
    await click(category);
    assert.equal(byAria('Add Decoder block template'), null);
    await typeText(byAria('Search blocks'), 'decoder block');
    assert.ok(byAria('Add Decoder block template'), 'Global search includes templates without switching modes');
    await click(byAria('Add Decoder block template'));
    assert.equal(graphNodes().length, 1);
    assert.equal(graphNodes()[0].kind, 'supernode');
    assert.equal(graphNodes()[0].templateId, template.id);
});

test('collapsed library reopens and focuses search with the keyboard shortcut', async () => {
    await render(h(PaletteHarness));
    await click(byAria('Collapse block library'));
    assert.ok(byAria('Open block library'));
    assert.equal(byAria('Search blocks'), null);
    await act(() => window.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true })));
    await act(() => new Promise((resolve) => requestAnimationFrame(resolve)));
    const search = byAria('Search blocks');
    assert.ok(search);
    assert.equal(document.activeElement, search);
    assert.equal(graphNodes().length, 0);
});


function EdgeHarness({ selected, tensorFlow = false, shape = [null, 1024, 768] }) {
    const edge = useArchitectureStore((state) => state.graph.edges[0]);
    return h(ReactFlowProvider, null, h(ReactFlow, {
        width: 1000, height: 700, nodes: [], edges: [],
    }, h('svg', null, h(ArchitectureEdge, {
        id: edge.id, selected,
        sourceX: 100, sourceY: 100, targetX: 100, targetY: 300,
        sourcePosition: 'bottom', targetPosition: 'top',
        data: { activation: 'B × 1024 × 768', gradientOp: edge.gradientOp, tensorFlow, shape },
    }))));
}

test('architecture edges show tensor shapes without gradient controls or mutations', async () => {
    const edge = createEdge({ id: 'connection', source: 'a', target: 'b', gradientOp: 'reverse', gradientOpLambda: 0.5 });
    useArchitectureStore.setState({ graph: { nodes: {}, edges: [edge] } });
    await render(h(EdgeHarness, { selected: true }));
    assert.equal(container.querySelector('.df-activation').textContent, 'B × 1024 × 768');
    assert.equal(container.querySelector('.df-gradient-picker'), null);
    assert.equal(container.querySelector('.df-activation-op'), null);
    assert.equal(container.querySelector('.react-flow__edge-path').style.stroke, 'rgb(17, 17, 17)');
    assert.equal(useArchitectureStore.getState().graph.edges[0], edge, 'Existing training semantics must survive display changes');
    await render(h(EdgeHarness, { selected: false }));
    assert.equal(container.querySelector('.df-gradient-picker'), null);
});

test('search ArrowDown focuses a result and Enter adds the matching layer once', async () => {
    await render(h(PaletteHarness));
    const search = byAria('Search blocks');
    await typeText(search, 'layer norm');
    await act(() => search.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true })));
    assert.equal(document.activeElement, byAria('Add LayerNorm'));
    assert.equal(graphNodes().length, 0);
    await act(() => search.focus());
    await act(() => search.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })));
    assert.equal(graphNodes().length, 1);
    assert.equal(graphNodes()[0].libraryKey, 'torch.nn.LayerNorm');
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, [graphNodes()[0].id]);
});

// Real canvas / ReactFlow store integration; no pretend pointer geometry.
// triggerNodeChanges is the same update path used by ReactFlow's drag handler.
function CanvasBridge({ onReady }) {
    const store = useStoreApi();
    useEffect(() => { onReady(store); }, [store, onReady]);
    return null;
}

test('canvas keeps 60 multi-drag frames local, persists on release and recovers from cancellation', async () => {
    const entry = getEntry('torch.nn.Linear');
    const a = createLayerNode({ id: 'a', libraryKey: entry.key, entry, position: { x: 200, y: 100 } });
    const b = createLayerNode({ id: 'b', libraryKey: entry.key, entry, position: { x: 200, y: 250 } });
    const c = createLayerNode({ id: 'c', libraryKey: entry.key, entry, position: { x: 200, y: 400 } });
    const edge = createEdge({ id: 'skip', source: 'a', target: 'c' });
    const graph = { nodes: { a, b, c }, edges: [edge] };
    useArchitectureStore.setState({ graph });
    let flow;
    const onReady = (store) => { flow = store; };
    await render(h(ArchitectureCanvas, { overlay: h(CanvasBridge, { onReady }) }));
    const untouched = flow.getState().nodes.find((node) => node.id === 'c');
    const route = flow.getState().edges[0].data;
    assert.equal(route.detour, true);
    let writes = 0;
    const storageProto = dom.window.Storage.prototype;
    const setItem = storageProto.setItem;
    storageProto.setItem = function (key, value) {
        if (key === 'deepflow:architecture-graph') writes += 1;
        return setItem.call(this, key, value);
    };
    try {
        await act(() => flow.getState().onSelectionDragStart(null, flow.getState().nodes.filter((node) => node.id !== 'c')));
        for (let frame = 1; frame <= 60; frame += 1) {
            await act(() => flow.getState().triggerNodeChanges([
                { type: 'position', id: 'a', position: { x: 200 + frame, y: 100 + frame }, dragging: true },
                { type: 'position', id: 'b', position: { x: 200 + frame, y: 250 + frame }, dragging: true },
            ]));
        }
        assert.equal(useArchitectureStore.getState().graph, graph);
        assert.equal(writes, 0);
        assert.equal(flow.getState().nodes.find((node) => node.id === 'c'), untouched);
        assert.equal(flow.getState().edges[0].data, route, 'Routing classification stays stable during drag');
        assert.deepEqual(flow.getState().nodes.find((node) => node.id === 'a').position, { x: 260, y: 160 });
        await act(() => flow.getState().triggerNodeChanges([
            { type: 'position', id: 'a', dragging: false },
            { type: 'position', id: 'b', dragging: false },
        ]));
        assert.equal(writes, 1);
        assert.deepEqual(useArchitectureStore.getState().graph.nodes.b.position, { x: 260, y: 310 });

        await act(() => flow.getState().onNodeDragStart(null, flow.getState().nodes.find((node) => node.id === 'a')));
        await act(() => flow.getState().triggerNodeChanges([{ type: 'position', id: 'a', position: { x: 900, y: 160 }, dragging: true }]));
        await act(() => window.dispatchEvent(new dom.window.Event('blur')));
        assert.deepEqual(flow.getState().nodes.find((node) => node.id === 'a').position, { x: 260, y: 160 });
        // No trailing end event: a completely new gesture must still work.
        await act(() => flow.getState().onNodeDragStart(null, flow.getState().nodes.find((node) => node.id === 'b')));
        await act(() => flow.getState().triggerNodeChanges([{ type: 'position', id: 'b', position: { x: 500, y: 310 }, dragging: true }]));
        await act(() => flow.getState().triggerNodeChanges([{ type: 'position', id: 'b', dragging: false }]));
        assert.equal(writes, 2);
        assert.deepEqual(useArchitectureStore.getState().graph.nodes.a.position, { x: 260, y: 160 });
        assert.deepEqual(useArchitectureStore.getState().graph.nodes.b.position, { x: 500, y: 310 });
    } finally {
        storageProto.setItem = setItem;
    }
});

test('node feature previews use a common grouping scale and stay bounded', async () => {
    await render(h(TensorPreview, { shape: [null, 1024, 768] }));
    assert.equal(container.querySelectorAll('.df-tensor-grid rect').length, 48);
    assert.equal(container.querySelector('.df-tensor-features').textContent, '768');
    assert.match(container.querySelector('title').textContent, /16 adjacent features/);
    assert.match(container.querySelector('title').textContent, /One token/);
    await render(h(TensorPreview, { shape: [null, 1024, 3072] }));
    assert.equal(container.querySelectorAll('.df-tensor-grid rect').length, 192);
    await render(h(TensorPreview, { shape: [null, 1000000, 1000000] }));
    assert.ok(container.querySelectorAll('.df-tensor-grid rect').length <= 192, 'Large tensors stay aggregated');
    await render(h(TensorPreview, { shape: null }));
    assert.equal(container.querySelector('svg'), null);
});

test('feature previews live in operation nodes; connections show dimensions only on selection', async () => {
    const entry = getEntry('torch.nn.Linear');
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', position: { x: 100, y: 0 }, parameters: { shape: '(1024, 768)', dtype: 'float32' } });
    const linear = createLayerNode({ id: 'linear', libraryKey: entry.key, entry, position: { x: 100, y: 100 }, parameters: { in_features: 768, out_features: 3072, bias: true } });
    const edge = createEdge({ id: 'connection', source: 'input', target: 'linear' });
    const graph = { nodes: { input, linear }, edges: [edge] };
    useArchitectureStore.setState({ graph });
    await render(h(GraphHarness));
    const block = container.querySelector('.react-flow__node[data-id="linear"] .df-block');
    assert.equal(block.querySelectorAll('.df-tensor-grid rect').length, 192);
    assert.equal(block.querySelector('input'), null);
    await click(byAria('Inspect Linear'));
    assert.ok(byAria('Linear inspector'));
    assert.equal(block.querySelectorAll('.df-tensor-grid rect').length, 192);
    await render(h(EdgeHarness, { selected: false }));
    assert.equal(container.querySelector('.df-tensor-preview'), null);
    assert.equal(container.querySelector('.df-activation'), null);
    await render(h(EdgeHarness, { selected: true }));
    assert.ok(container.querySelector('.df-activation'));
    assert.equal(container.querySelector('.df-tensor-preview'), null);
    assert.equal(useArchitectureStore.getState().graph, graph);
});
