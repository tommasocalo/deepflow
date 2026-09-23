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
                export { default as SuperNodeInstanceView } from './src/components/architecture/SuperNodeInstanceView.jsx';
                export { default as RepeatNodeView } from './src/components/architecture/RepeatNodeView.jsx';
                export { default as RepeatIterationBar } from './src/components/architecture/RepeatIterationBar.jsx';
                export { default as BlockPalette } from './src/components/architecture/BlockPalette.jsx';
                export { default as ArchitectureRibbon } from './src/components/architecture/ArchitectureRibbon.jsx';
                export { default as ArchitectureCanvas } from './src/components/architecture/ArchitectureCanvas.jsx';
                export { default as ArchitectureEdge } from './src/components/architecture/ArchitectureEdge.jsx';
                export { default as NodeInspector } from './src/components/architecture/NodeInspector.jsx';
                export { default as LevelTabs } from './src/components/architecture/LevelTabs.jsx';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useTemplatesStore } from './src/stores/templatesStore.js';
                export { createLayerNode, createSuperNodeInstance, createRepeatNode, createTemplate, createEdge, emptyGraph } from './src/domain/architecture/index.js';
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
const { ParameterField, LayerNodeView, SuperNodeInstanceView, RepeatNodeView, RepeatIterationBar, BlockPalette, ArchitectureRibbon, ArchitectureCanvas, ArchitectureEdge, NodeInspector, LevelTabs, useArchitectureStore, useTemplatesStore, createLayerNode, createSuperNodeInstance, createRepeatNode, createTemplate, createEdge, emptyGraph, getEntry } = app;
let container;
let root;

beforeEach(() => {
    jsdomErrors.length = 0;
    localStorage.clear();
    useTemplatesStore.setState({ templates: {} });
    useArchitectureStore.setState({ graph: emptyGraph(), currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null, recentLibraryKeys: [] });
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
    const named = byAria(name);
    if (named) return named;
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
const selectValue = (select, value) => act(() => {
    assert.ok(select, 'Expected a select');
    Object.getOwnPropertyDescriptor(dom.window.HTMLSelectElement.prototype, 'value').set.call(select, value);
    select.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
});
async function typeText(input, text) {
    await act(() => input.focus());
    await setText(input, '');
    for (const character of text) await setText(input, input.value + character);
}
const blur = (input) => act(() => input.blur());
const graphNodes = () => Object.values(useArchitectureStore.getState().graph.nodes);

function ParameterHarness({ initial = null, type = 'float', name = 'width', editor, defaultValue, onValue }) {
    const [value, setValue] = useState(initial);
    return h(ParameterField, {
        param: { name, type, editor, default: defaultValue }, value,
        onChange(next) { setValue(next); onValue?.(next); },
    });
}

function PaletteHarness({ section = 'components' }) {
    // The library now delegates insertion to its host. Canvas coordinates and
    // routing to the focused pane are covered by the ribbon browser check.
    return h(BlockPalette, {
        section,
        onClose() {},
        onAdd(entry, isTemplate) {
            const state = useArchitectureStore.getState();
            const position = { x: 250, y: 200 };
            const id = entry.standardComponentKey ? state.addStandardComponent(entry.standardComponentKey, position) : isTemplate ? state.addSuperNodeInstance(entry.key, position, entry.name) : state.addLayerByLibraryKey(entry.key, position);
            if (id) { state.selectOnly(id); state.clearExpanded(); }
            return id;
        },
    });
}

const nodeTypes = { layer: LayerNodeView, supernode: SuperNodeInstanceView, repeat: RepeatNodeView };
function GraphHarness() {
    const graph = useArchitectureStore((state) => state.graph);
    return h(ReactFlowProvider, null, h(ReactFlow, {
        width: 1000, height: 700,
        nodes: Object.values(graph.nodes).map((node) => ({ id: node.id, type: node.kind, position: node.position, data: { node }, width: 240, height: 44 })),
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

test('boolean bindings show their default checkbox while expressions remain editable text', async () => {
    let latest;
    await render(h(ParameterField, { param: { name: 'bias', type: 'bool', default: true }, value: undefined, onChange: (value) => { latest = value; } }));
    const checkbox = container.querySelector('input[type="checkbox"]');
    assert.ok(checkbox, 'An unset boolean binding uses a checkbox');
    assert.equal(checkbox.checked, true, 'An unset binding reflects the true template default');
    assert.equal(container.querySelector('.df-param-bool span').textContent, 'Default');
    await click(checkbox);
    assert.equal(latest, false, 'The default can be overridden explicitly');

    await render(h(ParameterHarness, { initial: '$bias', type: 'bool', defaultValue: true, onValue: (value) => { latest = value; } }));
    const expression = container.querySelector('input[type="text"]');
    assert.ok(expression, 'A boolean expression stays text instead of appearing as checked True');
    assert.equal(expression.value, '$bias');
    assert.equal(container.querySelectorAll('input[type="checkbox"]').length, 0);
    assert.ok(expression.classList.contains('is-expression'));
    await setText(expression, '$other_bias');
    await blur(expression);
    assert.equal(expression.value, '$other_bias');
    assert.equal(latest, '$other_bias');
});

test('weight shapes edit ordered dimensions without parsing or truncating expressions', async () => {
    let latest;
    await render(h(ParameterHarness, { name: 'shape', type: 'tuple', initial: '($out_features, $in_features)', onValue: (value) => { latest = value; } }));
    assert.equal(byAria('shape dimension 1').value, '$out_features');
    assert.equal(byAria('shape dimension 2').value, '$in_features');
    assert.equal(container.querySelector('textarea'), null);
    await typeText(byAria('shape dimension 2'), '(width + 1) * (2)');
    await blur(byAria('shape dimension 2'));
    assert.deepEqual(latest, ['$out_features', '(width + 1) * (2)']);
    await click(byAria('Add shape dimension'));
    assert.deepEqual(latest, ['$out_features', '(width + 1) * (2)', 1]);
    await typeText(byAria('shape dimension 3'), '1.05');
    await blur(byAria('shape dimension 3'));
    assert.deepEqual(latest, ['$out_features', '(width + 1) * (2)', 1.05], 'Invalid dimension decimals remain visible to validation');
    await click(byAria('Remove shape dimension 2'));
    assert.deepEqual(latest, ['$out_features', 1.05]);
    assert.equal(byAria('shape dimension 2').value, '1.05');
    await click(byAria('Remove shape dimension 2'));
    assert.equal(byAria('Remove shape dimension 1').disabled, true);
    await typeText(byAria('shape dimension 1'), '');
    assert.deepEqual(latest, [''], 'An unfinished axis remains editable instead of disappearing');
});

test('whole-shape references remain a single binding until replaced by local dimensions', async () => {
    let latest;
    await render(h(ParameterHarness, { name: 'shape', type: 'tuple', initial: '$weight_shape', onValue: (value) => { latest = value; } }));
    assert.equal(byAria('shape').value, '$weight_shape');
    assert.equal(byAria('Add shape dimension'), null);
    assert.equal(byAria('shape dimension 1'), null);
    await typeText(byAria('shape'), '$other_shape');
    await blur(byAria('shape'));
    assert.equal(latest, '$other_shape', 'Editing a tuple reference never wraps it in a dimension array');
    assert.equal(byAria('shape').value, '$other_shape');

    await render(h(ParameterField, { param: { name: 'weight_shape', type: 'tuple', editor: 'dimensions' }, value: [32, 16], onChange: (value) => { latest = value; } }));
    assert.equal(byAria('weight_shape dimension 1').value, '32');
    assert.equal(byAria('weight_shape dimension 2').value, '16');
    await click(byAria('Add weight_shape dimension'));
    assert.deepEqual(latest, [32, 16, 1], 'Exposed shape aliases retain the dimension editor metadata');
});

test('shape visibility keeps symbolic axes in the cell and hiding puts them back', async () => {
    const weight = createLayerNode({ id: 'weight', libraryKey: 'deepflow.op.Parameter', entry: getEntry('deepflow.op.Parameter'), label: 'weight', position: { x: 0, y: 0 }, parameters: { shape: ['$out_features', '$in_features'] } });
    const template = createTemplate({ id: 'projection-body', name: 'Projection', graph: { nodes: { weight }, edges: [] }, freeParameters: [
        { name: 'out_features', type: 'int', default: 32 }, { name: 'in_features', type: 'int', default: 16 },
    ] });
    const projection = createSuperNodeInstance({ id: 'projection', templateId: template.id, label: 'Projection', position: { x: 0, y: 0 } });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    useArchitectureStore.setState({ graph: { nodes: { projection }, edges: [] }, currentPath: ['projection'], openPaths: [['projection']], expandedNodeId: 'weight' });
    await render(h(GraphHarness));
    const stored = () => useTemplatesStore.getState().templates[template.id];
    assert.equal(byAria('shape dimension 1').value, '$out_features');
    assert.equal(container.querySelector('.df-parameter-source'), null);
    assert.equal(container.querySelector('.df-expose-trigger'), null);
    const eye = () => byAria('Shape in Projection settings');
    assert.equal(eye().getAttribute('aria-pressed'), 'false');
    await click(eye());
    assert.equal(eye().getAttribute('aria-pressed'), 'true');
    assert.equal(stored().graph.nodes.weight.parameters.shape, '$shape');
    assert.deepEqual(stored().freeParameters.at(-1).default, ['$out_features', '$in_features'], 'Exposure preserves symbolic dimensions rather than freezing this instance');
    assert.equal(byAria('shape dimension 1').value, '$out_features', 'The cell keeps showing the same dimensions');
    assert.equal(byAria('shape dimension 2').value, '$in_features');
    await typeText(byAria('shape dimension 2'), '24');
    await blur(byAria('shape dimension 2'));
    assert.equal(String(stored().freeParameters.at(-1).default[1]), '24', 'Editing the cell edits the shared value');
    assert.equal(stored().graph.nodes.weight.parameters.shape, '$shape');
    await act(() => { useArchitectureStore.getState().openLevel([]); useArchitectureStore.getState().setExpanded('projection'); });
    assert.ok(byAria('shape') || byAria('shape dimension 1'), 'The parent presents the shown field');
    await act(() => { useArchitectureStore.getState().openLevel(['projection']); useArchitectureStore.getState().setExpanded('weight'); });
    await click(eye());
    assert.equal(eye().getAttribute('aria-pressed'), 'false');
    assert.equal(stored().graph.nodes.weight.parameters.shape[0], '$out_features', 'Hiding puts the dimensions back into the field');
    assert.equal(String(stored().graph.nodes.weight.parameters.shape[1]), '24');
    assert.equal(stored().freeParameters.some((parameter) => parameter.name === 'shape'), false);
    assert.equal(byAria('shape dimension 1').value, '$out_features');
    await act(() => { useArchitectureStore.getState().openLevel([]); useArchitectureStore.getState().setExpanded('projection'); });
    assert.equal(byAria('shape'), null, 'The parent no longer presents the hidden field');
    assert.equal(byAria('shape dimension 1'), null);
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

test('direct references stay in their inputs without duplicated values or unlink actions', async () => {
    const gather = createLayerNode({ id: 'gather', libraryKey: 'deepflow.op.Gather', entry: getEntry('deepflow.op.Gather'), label: 'lookup', position: { x: 0, y: 0 }, parameters: { padding_idx: '$padding_idx' } });
    const template = createTemplate({ id: 'table-body', name: 'Table', graph: { nodes: { gather }, edges: [] }, freeParameters: [{ name: 'padding_idx', type: 'int', nullable: true, default: null }] });
    const instance = createSuperNodeInstance({ id: 'table', templateId: template.id, label: 'Table', position: { x: 0, y: 0 } });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    useArchitectureStore.setState({ graph: { nodes: { table: instance }, edges: [] }, currentPath: ['table'], openPaths: [['table']], expandedNodeId: 'gather' });
    await render(h(GraphHarness));
    assert.equal(byAria('padding_idx').value, '$padding_idx');
    assert.equal(container.querySelector('.df-parameter-source'), null);
    assert.equal(container.querySelector('.df-parameter-result'), null);
    assert.equal(byAria('Unlink padding_idx'), null);
    const eye = () => byAria('Padding index in Table settings');
    assert.equal(eye().getAttribute('aria-pressed'), 'true');
    await click(eye());
    assert.equal(eye().getAttribute('aria-pressed'), 'false');
    assert.equal(byAria('padding_idx').value, '$padding_idx');
    assert.equal(useTemplatesStore.getState().templates[template.id].freeParameters[0].default, null);
    await act(() => useArchitectureStore.getState().patchLayerParameters('gather', { padding_idx: '$missing' }));
    assert.equal(byAria('padding_idx').value, '$missing');
    assert.equal(container.querySelector('.df-parameter-result'), null, 'A missing variable does not become a calculated result');
});

test('only formulas show a compact computed result while stored expressions retain their full precision', async () => {
    const dropout = createLayerNode({ id: 'dropout', libraryKey: 'torch.nn.Dropout', entry: getEntry('torch.nn.Dropout'), label: 'Dropout', position: { x: 0, y: 0 }, parameters: { p: '1 / 3' } });
    const template = createTemplate({ id: 'body', name: 'Block', graph: { nodes: { dropout }, edges: [] }, freeParameters: [{ name: 'dropout', type: 'float', default: 0.1 }] });
    const block = createSuperNodeInstance({ id: 'block', templateId: template.id, label: 'Block', position: { x: 0, y: 0 } });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    useArchitectureStore.setState({ graph: { nodes: { block }, edges: [] }, currentPath: ['block'], openPaths: [['block']], expandedNodeId: 'dropout' });
    await render(h(GraphHarness));
    assert.match(container.querySelector('.df-parameter-result').textContent, /Result:\s*≈\s*0\.333333/);
    assert.equal(container.querySelector('.df-parameter-result strong').title, `Exact value: ${1 / 3}`);
    assert.equal(byAria('p').value, '1 / 3');
    assert.equal(useTemplatesStore.getState().templates[template.id].graph.nodes.dropout.parameters.p, '1 / 3');
    await typeText(byAria('p'), '$dropout');
    await blur(byAria('p'));
    assert.equal(container.querySelector('.df-parameter-result'), null);
    await typeText(byAria('p'), '0.25');
    await blur(byAria('p'));
    assert.equal(container.querySelector('.df-parameter-result'), null);
    assert.equal(useTemplatesStore.getState().templates[template.id].graph.nodes.dropout.parameters.p, 0.25);
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

test('component menu inserts editable standard components and keeps less common modules available', async () => {
    await render(h(PaletteHarness));
    assert.equal(container.querySelector('.df-library-category'), null, 'Families are available without opening an accordion');
    for (const name of ['Linear', 'Embedding', 'Attention', 'Norm', 'Activation', 'Convolution', 'Pooling', 'Recurrent']) {
        assert.ok(byAria(`Add ${name}`)?.matches('.df-insert-item'), `${name} has a directly available tile`);
    }
    assert.equal(byAria('Add LayerNorm'), null);
    assert.ok(byAria('Add BatchNorm1d'), 'Other normalization modules stay available');
    assert.equal(graphNodes().length, 0);
    await typeText(byAria('Search components'), 'layer norm');
    const add = byAria('Add Norm');
    assert.ok(add, 'Searching LayerNorm finds the editable Norm component');
    assert.equal(container.querySelectorAll('.df-insert-item').length, 1);
    await click(add);
    const nodes = graphNodes();
    assert.equal(nodes.length, 1);
    assert.equal(nodes[0].kind, 'supernode');
    assert.equal(useTemplatesStore.getState().templates[nodes[0].templateId].standardKey, 'layernorm');
    assert.deepEqual(useArchitectureStore.getState().selectedNodeIds, [nodes[0].id]);
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    assert.ok(Number.isFinite(nodes[0].position.x) && Number.isFinite(nodes[0].position.y));
});

test('native component chevron preserves identity, cables and bias when opening its definition', async () => {
    const entry = getEntry('torch.nn.Linear');
    const node = createLayerNode({ id: 'projection', libraryKey: entry.key, entry, label: 'Q projection', position: { x: 120, y: 150 }, parameters: { in_features: 12, out_features: 6, bias: true } });
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', entry: getEntry('deepflow.io.Input'), position: { x: 120, y: 0 }, parameters: { shape: '(12,)' } });
    const edge = createEdge({ id: 'input-projection', source: input.id, target: node.id });
    useArchitectureStore.setState({ graph: { nodes: { input, projection: node }, edges: [edge] } });
    await render(h(GraphHarness));
    assert.ok(byAria('Open Q projection'));
    await click(byAria('Inspect Q projection'));
    assert.equal(container.querySelector('.df-inspector-options'), null, 'A lone Bias toggle needs no Options section');
    const bias = byAria('bias');
    assert.equal(bias.type, 'checkbox');
    assert.equal(bias.checked, true);
    assert.equal(bias.closest('.df-param-row').textContent, 'Bias', 'The checkbox itself communicates the boolean state');
    await click(bias);
    assert.equal(useArchitectureStore.getState().graph.nodes.projection.parameters.bias, false);
    assert.doesNotMatch(container.querySelector('.df-inspector').textContent, /Open operations|Open template/);

    await click(byAria('Open Q projection'));
    const state = useArchitectureStore.getState();
    assert.deepEqual(state.currentPath, ['projection']);
    assert.equal(state.graph.nodes.projection.kind, 'supernode');
    assert.equal(state.graph.nodes.projection.label, 'Q projection');
    assert.deepEqual(state.graph.nodes.projection.position, node.position);
    assert.deepEqual(state.graph.edges, [edge]);
    assert.equal(state.graph.nodes.projection.parameterValues.bias, false);
    assert.equal(useTemplatesStore.getState().templates[state.graph.nodes.projection.templateId].standardKey, 'linear');
    assert.ok(byAria('Open Q projection'), 'The composed node uses the same open affordance');
});

test('nodes without an equivalent editable decomposition do not offer a misleading open chevron', async () => {
    const entry = getEntry('torch.nn.LSTM');
    const node = createLayerNode({ id: 'recurrent', libraryKey: entry.key, entry, position: { x: 0, y: 0 }, parameters: { input_size: 12, hidden_size: 24 } });
    useArchitectureStore.setState({ graph: { nodes: { recurrent: node }, edges: [] } });
    await render(h(GraphHarness));
    assert.equal(byAria('Open LSTM'), null);
    await click(byAria('Inspect LSTM'));
    assert.ok(byAria('LSTM inspector'));
    assert.equal(useArchitectureStore.getState().graph.nodes.recurrent.kind, 'layer');
});

test('repeat inspection numbers instances from one while retaining zero-based formula indexes', async () => {
    const template = createTemplate({ id: 'body', name: 'Shared body', graph: emptyGraph() });
    const repeat = createRepeatNode({ id: 'heads', templateId: template.id, label: 'Attention heads', mode: 'parallel', count: 3, position: { x: 0, y: 0 } });
    useTemplatesStore.setState({ templates: { body: template } });
    useArchitectureStore.setState({ graph: { nodes: { heads: repeat }, edges: [] }, selectedNodeIds: ['heads'], expandedNodeId: 'heads', inspectedIterations: {} });
    await render(h('div', null, h(ArchitectureRibbon, { paneId: 'main', onInsert() {}, onFitView() {} }), h(GraphHarness)));
    const labels = [...container.querySelectorAll('.df-parallel-instance b')].map((element) => element.textContent);
    assert.deepEqual(labels, ['1', '2', '3']);
    const inspector = container.querySelector('.df-inspector');
    assert.equal(byAria('count').closest('.df-param-row').querySelector('label').textContent, 'Instances');
    assert.ok(byAria('iterationVariable').closest('details'));
    assert.equal(byAria('iterationVariable').closest('details').open, false);
    assert.doesNotMatch(inspector.textContent, /Open template|Index name/);
    assert.equal(byAria('Repeat mode'), null, 'Structure controls live in the ribbon');
    assert.equal([...inspector.querySelectorAll('button')].find((button) => button.textContent.trim() === 'Single'), undefined);
    await click(byAria('Repeat selection sequentially'));
    assert.equal(byAria('count').closest('.df-param-row').querySelector('label').textContent, 'Steps');
    await click(byAria('Repeat selection in parallel'));
    await click(byAria('Open Attention heads instance 2'));
    assert.equal(useArchitectureStore.getState().inspectedIterations['["heads"]'], 1);
    await render(h(RepeatIterationBar));
    const number = byAria('Inspected instance number');
    assert.equal(number.value, '2');
    assert.equal(number.min, '1');
    assert.equal(number.max, '3');
    assert.doesNotMatch(container.textContent, /index starts/);
    assert.doesNotMatch(container.textContent, /Editing shared structure/, 'Scope is controlled by the ribbon switch');
    await click(byAria('Next instance'));
    assert.equal(number.value, '3');
    assert.equal(useArchitectureStore.getState().inspectedIterations['["heads"]'], 2);
    assert.equal(byAria('Next instance').disabled, true);
    await setText(number, '1');
    assert.equal(useArchitectureStore.getState().inspectedIterations['["heads"]'], 0);
    assert.equal(byAria('Previous instance').disabled, true);
});

test('parallel previews keep a fixed footprint and retain access to every instance', async () => {
    const template = createTemplate({ id: 'body', name: 'Shared body', graph: emptyGraph() });
    const repeat = createRepeatNode({ id: 'heads', templateId: template.id, label: 'Attention heads', mode: 'parallel', count: 12, position: { x: 0, y: 0 } });
    useTemplatesStore.setState({ templates: { body: template } });
    useArchitectureStore.setState({ graph: { nodes: { heads: repeat }, edges: [] }, inspectedIterations: {} });
    await render(h(GraphHarness));
    const width = container.querySelector('.df-parallel').style.width;
    assert.deepEqual([...container.querySelectorAll('.df-parallel-instance b')].map((element) => element.textContent), ['1', '2', '12']);
    await act(() => useArchitectureStore.getState().patchRepeatNode('heads', { count: 200 }));
    assert.equal(container.querySelector('.df-parallel').style.width, width);
    assert.deepEqual([...container.querySelectorAll('.df-parallel-instance b')].map((element) => element.textContent), ['1', '2', '200']);
    assert.equal(container.querySelectorAll('.df-parallel-ellipsis').length, 1);
    await click(byAria('Open Attention heads instance 200'));
    assert.equal(useArchitectureStore.getState().inspectedIterations['["heads"]'], 199);
    await render(h(RepeatIterationBar));
    assert.equal(byAria('Inspected instance number').value, '200');
    await setText(byAria('Inspected instance number'), '137');
    assert.equal(useArchitectureStore.getState().inspectedIterations['["heads"]'], 136);
});

test('structure toggles return to a single component without replacing its bindings or cables', async () => {
    const template = createTemplate({ id: 'body', name: 'Shared projection', graph: emptyGraph(), freeParameters: [{ name: 'width', type: 'int', default: 8 }] });
    const instance = createSuperNodeInstance({ id: 'projection', templateId: template.id, label: 'Custom projection', position: { x: 80, y: 160 }, parameterValues: { width: 16 } });
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', entry: getEntry('deepflow.io.Input'), position: { x: 80, y: 0 }, parameters: { shape: '(16,)' } });
    const output = createLayerNode({ id: 'output', libraryKey: 'deepflow.io.Output', entry: getEntry('deepflow.io.Output'), position: { x: 80, y: 320 } });
    const edges = [
        createEdge({ id: 'incoming', source: input.id, sourceHandle: 'features', target: instance.id, targetHandle: 'body-input', gradientOp: 'stop' }),
        createEdge({ id: 'outgoing', source: instance.id, sourceHandle: 'body-output', target: output.id }),
    ];
    useTemplatesStore.setState({ templates: { body: template } });
    useArchitectureStore.setState({ graph: { nodes: { input, projection: instance, output }, edges }, selectedNodeIds: [instance.id], expandedNodeId: instance.id });
    await render(h('div', null, h(ArchitectureRibbon, { paneId: 'main', onInsert() {}, onFitView() {} }), h(GraphHarness)));
    const sequential = byAria('Repeat selection sequentially');
    const parallel = byAria('Repeat selection in parallel');
    assert.equal(byAria('Group selection into component').disabled, true, 'One selected component cannot be grouped alone');
    assert.equal(sequential.getAttribute('aria-pressed'), 'false');
    assert.equal(parallel.getAttribute('aria-pressed'), 'false');
    assert.equal([...container.querySelectorAll('.df-inspector button')].find((button) => button.textContent.trim() === 'Repeat'), undefined, 'The inspector has no duplicate repeat action');
    const readNode = () => useArchitectureStore.getState().graph.nodes.projection;
    const preserved = () => {
        const state = useArchitectureStore.getState();
        const node = readNode();
        for (const key of ['id', 'templateId', 'label', 'position']) assert.deepEqual(node[key], instance[key], `${key} survives structural changes`);
        assert.deepEqual(state.graph.edges, edges, 'Port and gradient metadata survive structural changes');
        assert.deepEqual(state.selectedNodeIds, [instance.id]);
        assert.deepEqual(node.kind === 'repeat' ? node.bindings : node.parameterValues, { width: 16 });
    };
    await click(sequential);
    assert.equal(readNode().mode, 'sequential');
    assert.equal(sequential.getAttribute('aria-pressed'), 'true');
    preserved();
    await act(() => useArchitectureStore.getState().patchRepeatNode(instance.id, { count: 7, iterationVariable: 'head' }));
    await click(parallel);
    assert.equal(readNode().mode, 'parallel');
    assert.equal(readNode().count, 7, 'Switching repetition direction preserves its count');
    assert.equal(readNode().iterationVariable, 'head');
    assert.equal(sequential.getAttribute('aria-pressed'), 'false');
    assert.equal(parallel.getAttribute('aria-pressed'), 'true');
    preserved();
    await click(parallel);
    assert.equal(readNode().kind, 'supernode', 'Clicking the active Parallel mode returns to single');
    assert.equal(sequential.getAttribute('aria-pressed'), 'false');
    assert.equal(parallel.getAttribute('aria-pressed'), 'false');
    assert.equal(byAria('count'), null);
    preserved();
    await click(sequential);
    assert.equal(readNode().count, 7, 'Re-enabling repetition restores its previous count');
    assert.equal(readNode().iterationVariable, 'head', 'Re-enabling repetition restores its formula index name');
    await click(sequential);
    assert.equal(readNode().kind, 'supernode', 'Clicking the active Sequential mode returns to single');
    assert.equal(sequential.getAttribute('aria-pressed'), 'false');
    assert.equal(parallel.getAttribute('aria-pressed'), 'false');
    preserved();
    assert.deepEqual(useTemplatesStore.getState().templates, { body: template }, 'Toggles reuse the existing definition');
});

test('template library searches reusable components and hides recursive insertions', async () => {
    const template = createTemplate({ id: 'decoder-template', name: 'Decoder block', graph: emptyGraph() });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    await render(h(PaletteHarness, { section: 'templates' }));
    assert.ok(byAria('Add Decoder block template')?.matches('.df-insert-item'));
    assert.equal(graphNodes().length, 0);
    await typeText(byAria('Search templates'), 'decoder block');
    assert.ok(byAria('Add Decoder block template'), 'Search filters the opened template library');
    await click(byAria('Add Decoder block template'));
    assert.equal(graphNodes().length, 1);
    assert.equal(graphNodes()[0].kind, 'supernode');
    assert.equal(graphNodes()[0].templateId, template.id);
    assert.equal(useArchitectureStore.getState().expandedNodeId, null);
    await act(() => useArchitectureStore.getState().enterSupernode(graphNodes()[0].id));
    assert.equal(byAria('Add Decoder block template'), null, 'An unavailable recursive template is hidden');
    assert.equal(container.querySelector('section[aria-label="Saved templates"]'), null, 'Empty template section is hidden too');
    await act(() => byAria('Search templates').dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })));
    assert.equal(Object.keys(useTemplatesStore.getState().templates[template.id].graph.nodes).length, 0);
});

test('normalization inspector switches variants, preserves a custom name and shows the new parameter schema', async () => {
    const entry = getEntry('torch.nn.LayerNorm');
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', entry: getEntry('deepflow.io.Input'), position: { x: 100, y: 0 }, parameters: { shape: '(64, 16, 16)', dtype: 'float32' } });
    const normalization = createLayerNode({ id: 'norm', libraryKey: entry.key, entry, label: 'Feature normalization', position: { x: 100, y: 100 }, parameters: { normalized_shape: [64, 16, 16], eps: 0.001, elementwise_affine: false } });
    const edge = createEdge({ id: 'normalization-input', source: input.id, target: normalization.id });
    const graph = { nodes: { input, norm: normalization }, edges: [edge] };
    useArchitectureStore.setState({ graph });
    await render(h(GraphHarness));
    await click(byAria('Inspect Feature normalization'));
    const variant = byAria('Operation type');
    assert.deepEqual([...variant.options].map((option) => option.value), ['torch.nn.LayerNorm', 'torch.nn.BatchNorm1d', 'torch.nn.BatchNorm2d', 'torch.nn.GroupNorm']);
    assert.equal(byFieldLabel('normalized_shape').value, '64, 16, 16');
    await selectValue(variant, 'torch.nn.GroupNorm');

    const state = useArchitectureStore.getState();
    assert.equal(state.graph.nodes.norm.libraryKey, 'torch.nn.GroupNorm');
    assert.equal(state.graph.nodes.norm.label, 'Feature normalization');
    assert.equal(state.graph.edges, graph.edges);
    assert.ok(byAria('Feature normalization inspector'));
    assert.equal(byFieldLabel('normalized_shape'), null);
    assert.equal(byFieldLabel('elementwise_affine'), null);
    assert.equal(byFieldLabel('num_groups').value, '');
    assert.equal(byFieldLabel('num_channels').value, '');
    assert.equal(byFieldLabel('eps').value, '0.001');
    await typeText(byFieldLabel('num_groups'), '8');
    await blur(byFieldLabel('num_groups'));
    await typeText(byFieldLabel('num_channels'), '64');
    await blur(byFieldLabel('num_channels'));
    assert.deepEqual(useArchitectureStore.getState().graph.nodes.norm.parameters, { num_groups: 8, num_channels: 64, eps: 0.001, affine: true });
    assert.equal(useArchitectureStore.getState().expandedNodeId, normalization.id);
});

test('ribbon keyboard shortcut opens library search and Escape restores trigger focus', async () => {
    await render(h(ArchitectureRibbon, { paneId: 'main', onInsert() { assert.fail('Opening a library must not insert a block'); }, onFitView() {} }));
    assert.ok(byAria('Insert operations'));
    assert.equal(byAria('Search operations'), null);
    await act(() => window.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'k', ctrlKey: true, bubbles: true, cancelable: true })));
    await act(() => new Promise((resolve) => requestAnimationFrame(resolve)));
    const search = byAria('Search operations');
    assert.ok(search);
    assert.equal(document.activeElement, search);
    assert.equal(graphNodes().length, 0);
    await act(() => search.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true, cancelable: true })));
    assert.equal(byAria('Search operations'), null);
    assert.equal(document.activeElement, byAria('Insert operations'));
    assert.equal(byAria('Insert operations').getAttribute('aria-expanded'), 'false');
});

test('inspector shows accessible field errors once and reveals invalid options', async () => {
    const entry = getEntry('torch.nn.MaxPool1d');
    const pool = createLayerNode({ id: 'pool', libraryKey: entry.key, entry, position: { x: 100, y: 100 } });
    useArchitectureStore.setState({ graph: { nodes: { pool }, edges: [] } });
    await render(h(GraphHarness));
    await click(byAria('Inspect MaxPool1d'));
    const kernel = byFieldLabel('kernel_size');
    assert.equal(kernel.getAttribute('aria-invalid'), 'true');
    const error = document.getElementById(kernel.getAttribute('aria-describedby'));
    assert.equal(error.textContent, 'Required');
    assert.ok(kernel.closest('.df-param-row').contains(error));
    assert.equal(container.querySelectorAll('.df-inspector-issue.is-error').length, 0, 'Field error is not repeated below');
    await typeText(kernel, '2');
    await blur(kernel);
    assert.equal(kernel.getAttribute('aria-invalid'), 'false');
    assert.equal(kernel.getAttribute('aria-describedby'), null);
    assert.equal(container.querySelector('.df-param-errors'), null);

    await act(() => useArchitectureStore.getState().patchLayerParameters('pool', { stride: 1.5 }));
    assert.equal(container.querySelector('.df-inspector-options').open, true);
    const stride = byFieldLabel('stride');
    assert.equal(stride.getAttribute('aria-invalid'), 'true');
    assert.equal(document.getElementById(stride.getAttribute('aria-describedby')).textContent, 'Must be an integer');
    await typeText(stride, '2');
    await blur(stride);
    assert.equal(stride.getAttribute('aria-invalid'), 'false');
});

test('inline inspector feedback respects resolved template parameters', async () => {
    const entry = getEntry('torch.nn.Linear');
    const linear = createLayerNode({ id: 'bound', libraryKey: entry.key, entry, position: { x: 100, y: 100 }, parameters: { in_features: 4, out_features: '$width' } });
    const template = createTemplate({ id: 'parameterized', name: 'Projection', graph: { nodes: { bound: linear }, edges: [] }, freeParameters: [{ name: 'width', default: 64 }] });
    const instance = createSuperNodeInstance({ id: 'instance', templateId: template.id, label: 'Projection' });
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    useArchitectureStore.setState({ graph: { nodes: { instance }, edges: [] }, currentPath: ['instance'], expandedNodeId: 'bound' });
    await render(h(ReactFlowProvider, null, h(NodeInspector)));
    assert.equal(byFieldLabel('out_features').value, '$width');
    assert.equal(byFieldLabel('out_features').getAttribute('aria-invalid'), 'false');
    assert.equal(byFieldLabel('out_features').getAttribute('aria-describedby'), null);
    assert.equal(container.querySelector('.df-param-errors'), null);
});


test('open level tabs survive main navigation and close independently of selection', async () => {
    const template = createTemplate({ id: 'shared', name: 'Shared block', graph: emptyGraph() });
    const first = createSuperNodeInstance({ id: 'first', templateId: template.id, label: 'Encoder' });
    const second = createSuperNodeInstance({ id: 'second', templateId: template.id, label: 'Decoder' });
    const graph = { nodes: { first, second }, edges: [] };
    useTemplatesStore.setState({ templates: { [template.id]: template } });
    useArchitectureStore.setState({ graph });
    await render(h(LevelTabs));
    const tab = (name) => [...container.querySelectorAll('[role="tab"]')].find((el) => el.textContent === name);
    const state = useArchitectureStore.getState();

    await act(() => state.enterSupernode('first'));
    await click(tab('main'));
    assert.ok(tab('Encoder'), 'Returning to main keeps the supernode tab open');
    assert.equal(tab('main').getAttribute('aria-selected'), 'true');
    assert.equal(tab('Encoder').getAttribute('aria-selected'), 'false');
    assert.equal(container.querySelector('.df-levels-hint'), null);

    await act(() => state.enterSupernode('second'));
    await click(tab('Encoder'));
    assert.deepEqual(useArchitectureStore.getState().currentPath, ['first']);
    assert.equal(container.querySelectorAll('[role="tab"]').length, 3);
    await click(byAria('Close Decoder'));
    assert.equal(tab('Decoder'), undefined);
    assert.equal(tab('Encoder').getAttribute('aria-selected'), 'true', 'Closing an inactive tab leaves the current view alone');

    await click(byAria('Close Encoder'));
    assert.equal(container.querySelectorAll('[role="tab"]').length, 1);
    assert.equal(tab('main').getAttribute('aria-selected'), 'true');
    assert.equal(useArchitectureStore.getState().graph, graph, 'Closing tabs never deletes architecture nodes');
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

test('architecture edges stay unlabeled and preserve training semantics', async () => {
    const edge = createEdge({ id: 'connection', source: 'a', target: 'b', gradientOp: 'reverse', gradientOpLambda: 0.5 });
    useArchitectureStore.setState({ graph: { nodes: {}, edges: [edge] } });
    await render(h(EdgeHarness, { selected: true }));
    assert.equal(container.querySelector('.df-activation'), null);
    assert.equal(container.querySelector('.df-gradient-picker'), null);
    assert.equal(container.querySelector('.df-activation-op'), null);
    assert.equal(useArchitectureStore.getState().graph.edges[0], edge, 'Existing training semantics must survive display changes');
    await render(h(EdgeHarness, { selected: false }));
    assert.equal(container.querySelector('.df-gradient-picker'), null);
});

test('operation search supports ArrowDown and adds its match exactly once on Enter', async () => {
    await render(h(PaletteHarness, { section: 'operations' }));
    const search = byAria('Search operations');
    await typeText(search, 'MatMul');
    await act(() => search.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true, cancelable: true })));
    assert.equal(document.activeElement, byAria('Add MatMul'));
    assert.equal(graphNodes().length, 0);
    await act(() => search.focus());
    await act(() => search.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })));
    assert.equal(graphNodes().length, 1);
    assert.equal(graphNodes()[0].libraryKey, 'deepflow.op.MatMul');
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
    const graph = { nodes: { a, b, c }, edges: [edge, createEdge({ source: 'a', target: 'b' }), createEdge({ source: 'b', target: 'c' })] };
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

test('operation nodes expose dimensions and inspector; selected connections stay unlabeled', async () => {
    const entry = getEntry('torch.nn.Linear');
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', position: { x: 100, y: 0 }, parameters: { shape: '(1024, 768)', dtype: 'float32' } });
    const linear = createLayerNode({ id: 'linear', libraryKey: entry.key, entry, position: { x: 100, y: 100 }, parameters: { in_features: 768, out_features: 3072, bias: true } });
    const edge = createEdge({ id: 'connection', source: 'input', target: 'linear' });
    const graph = { nodes: { input, linear }, edges: [edge] };
    useArchitectureStore.setState({ graph });
    await render(h(GraphHarness));
    const block = container.querySelector('.react-flow__node[data-id="linear"] .df-block');
    assert.equal(block.querySelector('.df-port-shape--output').getAttribute('aria-label'), 'Output dimensions: B × 1024 × 3072');
    assert.equal(block.querySelector('input'), null);
    await click(byAria('Inspect Linear'));
    assert.ok(byAria('Linear inspector'));
    assert.equal(block.querySelector('.df-port-shape--output').getAttribute('aria-label'), 'Output dimensions: B × 1024 × 3072');
    await render(h(EdgeHarness, { selected: false }));
    assert.equal(container.querySelector('.df-tensor-preview'), null);
    assert.equal(container.querySelector('.df-activation'), null);
    await render(h(EdgeHarness, { selected: true }));
    assert.equal(container.querySelector('.df-activation'), null);
    assert.equal(container.querySelector('.df-tensor-preview'), null);
    assert.equal(useArchitectureStore.getState().graph, graph);
});
