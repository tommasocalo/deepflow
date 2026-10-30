import assert from 'node:assert/strict';
import { test } from 'node:test';
import { computeModelMetrics, formatShare } from '../src/domain/metrics/index.js';
import { buildModelTableRows, readingOrder, rowKey } from '../src/domain/modelTable/rows.js';
import { configAccessibleText } from '../src/domain/modelTable/configFields.js';
import { cellStatus, eachText, failingText, knownShapeMark, metricLines, SHAPE_GUESS_NOTE, shareInfo } from '../src/domain/modelTable/cellText.js';
import { cutExpression, fitConfigParts, fitShapeText } from '../src/domain/modelTable/fitText.js';
import { describeTarget, followReference, parseCellValue, precheckCellValue, resolveWriteTarget, targetSummary, targetTag } from '../src/domain/modelTable/writeTargets.js';
import { createEdge, createLayerNode, createRepeatNode, createSuperNodeInstance, createTemplate } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { gpt2Storyboard } from '../src/storyboards/gpt2.js';
import { resnetStoryboard } from '../src/storyboards/resnet.js';
import { attentionStoryboard } from '../src/storyboards/attention.js';

const storage = new Map();
globalThis.window = { localStorage: {
    getItem: (key) => storage.get(key) ?? null,
    setItem: (key, value) => storage.set(key, value),
    removeItem: (key) => storage.delete(key),
} };
const { useArchitectureStore: architectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore: templatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore: trainingStore } = await import('../src/stores/trainingStore.js');

function load(storyboard) {
    for (const store of [architectureStore, templatesStore, trainingStore]) store.setState(store.getInitialState(), true);
    storage.clear();
    storyboard({ architectureStore, templatesStore, trainingStore });
    return model();
}
function model() {
    const graph = architectureStore.getState().graph;
    const templates = templatesStore.getState().templates;
    return { graph, templates, metrics: computeModelMetrics(graph, templates) };
}
const rowsAt = ({ graph, templates, metrics }, depth, overrides) => buildModelTableRows({ metrics, rootGraph: graph, templates, depth, overrides });
const find = (rows, label, parentLabel) => {
    const row = rows.find((candidate) => candidate.label === label
        && (!parentLabel || rows.find((parent) => parent.key === candidate.parentKey)?.label === parentLabel));
    assert.ok(row, `Expected a row labelled ${label}`);
    return row;
};
const fieldOf = (row, name) => {
    const field = row.config.find((part) => part.kind === 'field' && part.name === name);
    assert.ok(field, `${row.label} shows ${name}`);
    return field;
};
const cellText = (row, cells) => row.config.filter((part) => !part.concealed).map((part) => part.kind === 'field' ? `${part.prefix && part.prefix !== '×' ? `${part.prefix} ` : part.prefix ?? ''}${part[cells].text}` : part.text).join(' ');
const target = (ctx, row, name, cells) => resolveWriteTarget({ row, field: fieldOf(row, name), cells, rootGraph: ctx.graph, templates: ctx.templates });
/** The row of a single layer fed by a model input. */
const layerRow = (libraryKey, parameters) => {
    const input = createLayerNode({ libraryKey: 'deepflow.io.Input', entry: getEntry('deepflow.io.Input'), position: { x: 0, y: 0 } });
    const node = createLayerNode({ libraryKey, entry: getEntry(libraryKey), position: { x: 0, y: 100 } });
    const graph = {
        nodes: { [input.id]: { ...input, parameters: { ...input.parameters, shape: '(8, 16)' } }, [node.id]: { ...node, parameters: { ...node.parameters, ...parameters } } },
        edges: [createEdge({ source: input.id, target: node.id })],
    };
    return buildModelTableRows({ metrics: computeModelMetrics(graph, {}), rootGraph: graph, templates: {}, depth: 'ops' }).find((row) => row.nodeId === node.id);
};

test('depth selects root nodes, expanded blocks, or every operation', () => {
    const gpt2 = load(gpt2Storyboard);
    const root = rowsAt(gpt2, 'root');
    assert.deepEqual(root.map((row) => row.label), ['tokens', 'token embedding', 'positional embedding', 'Add', 'Dropout', 'Transformer blocks', 'ln_f', 'lm_head', 'logits']);
    assert.equal(rowsAt(gpt2, 'blocks').length, 19);
    const ops = rowsAt(gpt2, 'ops');
    assert.equal(ops.length, 109);
    for (const rows of [root, rowsAt(gpt2, 'blocks'), ops]) {
        assert.equal(rows.some((row) => row.libraryKey?.startsWith('deepflow.template.')), false, 'Template boundaries are never rows');
        assert.equal(new Set(rows.map((row) => row.key)).size, rows.length, 'Row keys are unique');
    }
    const resnet = load(resnetStoryboard);
    assert.deepEqual(['root', 'blocks', 'ops'].map((depth) => rowsAt(resnet, depth).length), [3, 6, 6]);
    const attention = load(attentionStoryboard);
    assert.deepEqual(['root', 'blocks', 'ops'].map((depth) => rowsAt(attention, depth).length), [3, 3, 39]);
});

test('rows merge every instance: ×12 inside blocks, ×144 for attention heads', () => {
    const gpt2 = load(gpt2Storyboard);
    const blocks = rowsAt(gpt2, 'blocks');
    const stack = find(blocks, 'Transformer blocks');
    assert.equal(stack.multiplicity, 12);
    assert.equal(stack.expanded, true);
    const children = blocks.filter((row) => row.parentKey === stack.key);
    assert.equal(children.length, 10);
    assert.ok(children.every((row) => row.multiplicity === 12));
    assert.deepEqual(children.map((row) => row.label), ['ln_1', 'causal self-attention', 'attention projection dropout', 'residual', 'ln_2', 'mlp fc', 'gelu', 'mlp proj', 'dropout', 'residual']);
    assert.equal(find(blocks, 'mlp fc').expanded, false, 'Standard components stay single rows at Blocks depth');
    assert.deepEqual(find(blocks, 'mlp fc').path, [stack.nodeId, find(blocks, 'mlp fc').nodeId]);
    assert.equal(find(blocks, 'mlp fc').key, rowKey(find(blocks, 'mlp fc').path));

    const ops = rowsAt(gpt2, 'ops');
    const heads = find(ops, 'Attention heads');
    assert.equal(heads.multiplicity, 144);
    assert.ok(ops.filter((row) => row.parentKey === heads.key).every((row) => row.multiplicity === 144));
    assert.equal(find(ops, 'Q × Kᵀ').macs.total, 805306368 * 12);
    assert.deepEqual(find(ops, 'Q × Kᵀ').macs.each, { min: 67108864, max: 67108864 }, 'One head computes T × T × d_head products');
    assert.equal(find(blocks, 'mlp fc').params.total, 28348416);
    assert.deepEqual(find(blocks, 'mlp fc').params.each, { min: 2362368, max: 2362368 });
});

test('row disclosure overrides the depth default for one row', () => {
    const gpt2 = load(gpt2Storyboard);
    const stack = find(rowsAt(gpt2, 'root'), 'Transformer blocks');
    assert.equal(stack.expandable, true);
    assert.equal(stack.expanded, false);
    assert.equal(rowsAt(gpt2, 'root', { [stack.key]: true }).length, 19);
    assert.equal(rowsAt(gpt2, 'blocks', { [stack.key]: false }).length, 9);
    const lmHead = find(rowsAt(gpt2, 'root'), 'lm_head');
    assert.equal(rowsAt(gpt2, 'root', { [lmHead.key]: true }).length, 9 + 5, 'Linear opens into weight, transpose, MatMul, bias and add');
});

test('shares are fractions of the whole model for each lens', () => {
    const gpt2 = load(gpt2Storyboard);
    const rows = rowsAt(gpt2, 'blocks');
    const stack = find(rows, 'Transformer blocks');
    assert.equal(stack.share.params.toFixed(4), '0.5217');
    assert.equal(stack.share.macs.toFixed(3), '0.729');
    assert.equal(stack.share.activation.toFixed(3), '0.945');
    assert.equal(find(rows, 'lm_head').share.params.toFixed(3), '0.237');
    assert.equal(formatShare(find(rows, 'ln_1').share.params), '<0.1%');
    assert.equal(find(rows, 'Dropout').share.params, 0);
});

test('config cells show bindings as written and their resolved values', () => {
    const gpt2 = load(gpt2Storyboard);
    const rows = rowsAt(gpt2, 'blocks');
    const fc = find(rows, 'mlp fc');
    assert.deepEqual(['in_features', 'out_features'].map((name) => fieldOf(fc, name).symbols.text), ['$d_model', '$mlp_width']);
    assert.deepEqual(['in_features', 'out_features'].map((name) => fieldOf(fc, name).values.text), ['768', '3072']);
    assert.equal(fieldOf(fc, 'out_features').symbols.tone, 'expression');
    assert.equal(cellText(find(rows, 'lm_head'), 'values'), '768 → 50257 no bias');
    assert.equal(cellText(find(rows, 'Transformer blocks'), 'symbols'), '×$n_layers · mlp_width 4 * d_model');
    assert.equal(cellText(find(rows, 'Transformer blocks'), 'values'), '×12 · mlp_width 3072');
    assert.equal(cellText(find(rows, 'causal self-attention'), 'values'), 'H 12');
    assert.equal(cellText(find(rows, 'Dropout'), 'values'), 'p 0.1');
    assert.equal(find(rows, 'mlp fc').shape.text, 'B × 1024 × 3072');

    const resnet = load(resnetStoryboard);
    const conv = find(rowsAt(resnet, 'blocks'), 'Conv2d');
    assert.equal(fieldOf(conv, 'in_channels').values.text, '3…128');
    assert.equal(fieldOf(conv, 'in_channels').values.varies, true);
    assert.equal(fieldOf(conv, 'in_channels').values.title, 'i=0: 3 · i=1: 64 · i=2: 128');
    assert.equal(fieldOf(conv, 'out_channels').values.text, '64…256');
    assert.equal(conv.shape.text, 'B × 64…256 × 32 × 32');
    assert.equal(conv.shape.varies, true);
    assert.equal(cellText(find(rowsAt(resnet, 'blocks'), 'ResNet stages'), 'values'), '×3 · in 3…128 · out 64…256');
});

test('Values mode follows a reference to the value it reads; Symbols edits where it is written', () => {
    const gpt2 = load(gpt2Storyboard);
    const rows = rowsAt(gpt2, 'ops');
    const stack = find(rows, 'Transformer blocks');
    const kinds = (row, name, cells) => {
        const result = target(gpt2, row, name, cells);
        return [result.kind, result.name ?? result.field ?? null];
    };
    assert.deepEqual(kinds(find(rows, 'ln_1'), 'normalized_shape', 'values'), ['global', 'd_model']);
    assert.deepEqual(kinds(find(rows, 'mlp fc'), 'in_features', 'values'), ['global', 'd_model']);
    const formula = target(gpt2, find(rows, 'mlp fc'), 'out_features', 'values');
    assert.equal(formula.kind, 'readonly');
    assert.match(formula.reason, /4 \* d_model/);
    const residual = target(gpt2, find(rows, 'attention projection dropout'), 'p', 'values');
    assert.deepEqual([residual.kind, residual.nodeId, residual.source, residual.field], ['rootNode', stack.nodeId, 'bindings', 'resid_dropout']);
    assert.deepEqual(kinds(find(rows, 'causal self-attention'), 'H', 'values'), ['global', 'n_heads']);
    assert.deepEqual(kinds(stack, 'count', 'values'), ['global', 'n_layers']);
    assert.deepEqual(kinds(find(rows, 'token embedding'), 'num_embeddings', 'values'), ['global', 'vocab_size']);
    assert.deepEqual(kinds(find(rows, 'position table'), 'num_embeddings', 'values'), ['global', 'context']);
    assert.equal(target(gpt2, find(rows, 'Q projection'), 'in_features', 'values').kind, 'readonly', 'D is derived from the input');
    assert.deepEqual(kinds(find(rows, 'split Q'), 'chunks', 'values'), ['global', 'n_heads']);
    assert.deepEqual(kinds(find(rows, 'add epsilon', 'ln_1'), 'value', 'values'), ['rootNode', 'norm_eps']);
    assert.deepEqual(kinds(find(rows, 'attention dropout'), 'p', 'values'), ['rootNode', 'attn_dropout']);

    // Q projection's bias is `$bias`, which the attention instance leaves unbound.
    const qProjection = find(rows, 'Q projection');
    const attentionTemplate = gpt2.templates[find(rows, 'causal self-attention').definitionIds[0]];
    const frames = [stack, find(rows, 'causal self-attention')].map((row, depth) => ({
        container: depth ? gpt2.templates[stack.definitionIds[0]].graph.nodes[row.nodeId] : gpt2.graph.nodes[row.nodeId],
        owner: depth ? stack.definitionIds[0] : null, templateId: row.definitionIds[0], template: gpt2.templates[row.definitionIds[0]],
    }));
    const bias = followReference('bias', frames, gpt2.graph);
    assert.deepEqual([bias.kind, bias.templateId, bias.name], ['templateDefault', attentionTemplate.id, 'bias']);
    assert.equal(qProjection.levelPath.length, 2);

    const shared = target(gpt2, find(rows, 'mlp fc'), 'out_features', 'symbols');
    assert.deepEqual([shared.kind, shared.templateId, shared.source], ['templateNode', stack.definitionIds[0], 'parameterValues']);
    assert.match(describeTarget(shared, { row: find(rows, 'mlp fc'), rootGraph: gpt2.graph, templates: gpt2.templates, templateUses: gpt2.metrics.templateUses }), /shared Transformer block definition · 12 uses/);
    assert.deepEqual(kinds(find(rows, 'Dropout'), 'p', 'symbols'), ['rootNode', 'p']);
    assert.equal(target(gpt2, find(rows, 'attention projection dropout'), 'p', 'symbols').kind, 'templateLayer');
    const auto = target(gpt2, find(rows, 'Attention heads'), 'count', 'symbols');
    assert.equal(auto.kind, 'readonly');
    assert.match(auto.reason, /Split/);
    assert.equal(target(gpt2, find(rows, 'weight', 'mlp fc'), 'shape', 'symbols').kind, 'readonly');

    const resnet = load(resnetStoryboard);
    const conv = find(rowsAt(resnet, 'blocks'), 'Conv2d');
    assert.equal(target(resnet, conv, 'in_channels', 'values').kind, 'readonly');
    assert.match(target(resnet, conv, 'in_channels', 'values').reason, /i === 0 \? 3 : 64 \* 2 \*\* \(i - 1\)/);
    assert.equal(target(resnet, conv, 'in_channels', 'symbols').kind, 'templateLayer');
});

test('local versions, per-instance arrays and eye aliases resolve to safe owners', () => {
    load(gpt2Storyboard);
    const stack = Object.values(architectureStore.getState().graph.nodes).find((node) => node.label === 'Transformer blocks');
    architectureStore.getState().setInspectedIteration([stack.id], 3);
    assert.equal(architectureStore.getState().setStructureScope('local', [stack.id]), true);
    const local = model();
    const rows = rowsAt(local, 'blocks');
    const fc = find(rows, 'mlp fc');
    assert.equal(fc.multiplicity, 12, 'Local versions keep node identities, so rows still merge');
    assert.equal(fc.ownerIds.length, 2);
    assert.equal(find(rows, 'Transformer blocks').definitionIds.length, 2);
    const versioned = target(local, fc, 'out_features', 'symbols');
    assert.equal(versioned.kind, 'readonly');
    assert.match(versioned.reason, /local versions/);

    const resnet = load(resnetStoryboard);
    const stages = find(rowsAt(resnet, 'blocks'), 'ResNet stages');
    architectureStore.getState().patchRootNode(stages.nodeId, { bindings: { ...resnet.graph.nodes[stages.nodeId].bindings, out: [64, 128, 256] } });
    const arrays = model();
    const perInstance = target(arrays, find(rowsAt(arrays, 'blocks'), 'ResNet stages'), 'out', 'symbols');
    assert.equal(perInstance.kind, 'readonly');
    assert.match(perInstance.reason, /per instance/);

    load(resnetStoryboard);
    const block = Object.values(templatesStore.getState().templates).find((template) => template.name === 'ConvBlock');
    const conv = Object.values(block.graph.nodes).find((node) => node.libraryKey === 'torch.nn.Conv2d');
    templatesStore.getState().toggleFieldExposure(block.id, { nodeId: conv.id, field: 'kernel_size', spec: { name: 'kernel_size', type: 'tuple', default: 3 }, defaultValue: 3, label: 'Kernel' });
    const aliased = model();
    const aliasRow = find(rowsAt(aliased, 'blocks'), 'Conv2d');
    assert.equal(fieldOf(aliasRow, 'kernel_size').symbols.text, '3', 'An eye alias keeps showing its value');
    const aliasTarget = target(aliased, aliasRow, 'kernel_size', 'symbols');
    assert.deepEqual([aliasTarget.kind, aliasTarget.templateId, aliasTarget.name], ['templateDefault', block.id, 'kernel']);
});

test('cell values parse like parameter fields and are checked before writing', () => {
    assert.equal(parseCellValue(' 0.2 ', { type: 'float' }), 0.2);
    assert.equal(parseCellValue('', { type: 'int' }), null);
    assert.equal(parseCellValue('$d_model', { type: 'int' }), '$d_model');
    assert.equal(parseCellValue('false', { type: 'bool' }), false);
    assert.equal(parseCellValue('tanh', { type: 'enum', options: ['none', 'tanh'] }), 'tanh');
    const scope = { d_model: 768, i: 0 };
    assert.equal(precheckCellValue('2 * d_model', { spec: { type: 'int' }, scope }), null);
    assert.equal(precheckCellValue('$nope', { spec: { type: 'int' }, scope }), 'No parameter named $nope');
    assert.equal(precheckCellValue('abc', { spec: { type: 'int' }, scope }), 'No parameter named abc', 'The name reads as it was typed, without an added $');
    assert.equal(precheckCellValue('abc', { spec: { type: 'int' }, scope, source: 'count' }), 'No parameter named abc');
    assert.equal(precheckCellValue('d_model / 5', { spec: { type: 'int' }, scope }), 'Must be an integer');
    assert.equal(precheckCellValue('$d_model', { spec: { type: 'int' }, scope, literalOnly: true }), 'Values mode sets a value here; switch to Symbols to write an expression.');
    assert.equal(precheckCellValue(null, { spec: { type: 'int' }, scope, source: 'count' }), 'Required');
    assert.match(precheckCellValue(0, { spec: { type: 'int' }, scope, source: 'count' }), /positive integer/);
    assert.equal(precheckCellValue('fast', { spec: { type: 'enum', options: ['none', 'tanh'] }, scope }), 'Choose one of none, tanh');
});

test('reading order follows dataflow and canvas position', () => {
    const graph = {
        nodes: {
            c: { id: 'c', position: { x: 0, y: 200 } },
            b: { id: 'b', position: { x: 300, y: 100 } },
            a: { id: 'a', position: { x: 0, y: 100 } },
            source: { id: 'source', position: { x: 0, y: 0 } },
        },
        edges: [{ source: 'source', target: 'a' }, { source: 'source', target: 'b' }, { source: 'a', target: 'c' }, { source: 'b', target: 'c' }],
    };
    assert.deepEqual(readingOrder(graph), ['source', 'a', 'b', 'c']);
    assert.deepEqual(readingOrder({ nodes: { x: { id: 'x', position: { x: 0, y: 9 } }, y: { id: 'y', position: { x: 0, y: 1 } } }, edges: [{ source: 'x', target: 'y' }, { source: 'y', target: 'x' }] }), ['y', 'x'], 'Cycles fall back to position');
});

const layerNode = (libraryKey, parameters = {}, label) => {
    const node = createLayerNode({ libraryKey, entry: getEntry(libraryKey), position: { x: 0, y: 0 }, label });
    return { ...node, parameters: { ...node.parameters, ...parameters } };
};
const linked = (...nodes) => ({
    nodes: Object.fromEntries(nodes.map((node, index) => [node.id, { ...node, position: { x: 0, y: index * 100 } }])),
    edges: nodes.slice(1).map((node, index) => createEdge({ source: nodes[index].id, target: node.id })),
});
function custom(graph, templates) {
    return { graph, templates, metrics: computeModelMetrics(graph, templates) };
}

test('a binding follows its own definition first, then the repeat index, then globals', () => {
    const body = createTemplate({ name: 'Body', graph: linked(layerNode('deepflow.template.Input'), layerNode('torch.nn.Linear', { in_features: '$width', out_features: '$width' }, 'proj'), layerNode('deepflow.template.Output')),
        freeParameters: [{ name: 'd_model', type: 'int', default: 64 }, { name: 'width', type: 'int', default: 64 }, { name: 'depth', type: 'int', default: 1 }] });
    const repeat = createRepeatNode({ templateId: body.id, position: { x: 0, y: 0 }, count: 2, label: 'Stack', bindings: { d_model: 512, width: '$d_model' } });
    const graph = { ...linked(layerNode('deepflow.io.Input', { shape: '(512,)' }), repeat, layerNode('deepflow.io.Output')), parameters: [{ name: 'd_model', type: 'int', default: 768 }] };
    const ctx = custom(graph, { [body.id]: body });
    const stack = find(rowsAt(ctx, 'blocks'), 'Stack');
    const width = target(ctx, stack, 'width', 'values');
    assert.deepEqual([width.kind, width.nodeId, width.source, width.field], ['rootNode', repeat.id, 'bindings', 'd_model'], 'width reads the repeat\'s own d_model, not the global');
    const inner = target(ctx, find(rowsAt(ctx, 'blocks'), 'proj'), 'in_features', 'values');
    assert.deepEqual([inner.kind, inner.field], ['rootNode', 'd_model'], 'A layer inside follows width, then d_model on the repeat');

    const indexed = createRepeatNode({ templateId: body.id, position: { x: 0, y: 0 }, count: 2, label: 'Indexed', bindings: { depth: '$i' } });
    const byIndex = custom({ ...linked(layerNode('deepflow.io.Input', { shape: '(64,)' }), indexed, layerNode('deepflow.io.Output')), parameters: [] }, { [body.id]: body });
    const index = target(byIndex, find(rowsAt(byIndex, 'blocks'), 'Indexed'), 'depth', 'values');
    assert.equal(index.kind, 'readonly');
    assert.match(index.reason, /instance index/);
});

test('a global computed from other globals is never overwritten with a value', () => {
    const linear = layerNode('torch.nn.Linear', { in_features: '$d_model', out_features: '$hidden' }, 'fc');
    const alias = layerNode('torch.nn.Linear', { in_features: '$d_model', out_features: '$width' }, 'alias');
    const graph = { ...linked(layerNode('deepflow.io.Input', { shape: '(64,)' }), linear, alias, layerNode('deepflow.io.Output')),
        parameters: [{ name: 'd_model', type: 'int', default: 64 }, { name: 'hidden', type: 'int', default: '4 * d_model' }, { name: 'width', type: 'int', default: '$d_model' }] };
    const ctx = custom(graph, {});
    const rows = rowsAt(ctx, 'root');
    const formula = target(ctx, find(rows, 'fc'), 'out_features', 'values');
    assert.equal(formula.kind, 'readonly');
    assert.match(formula.reason, /4 \* d_model/);
    const forwarded = target(ctx, find(rows, 'alias'), 'out_features', 'values');
    assert.deepEqual([forwarded.kind, forwarded.name], ['global', 'd_model'], 'A global that forwards another global edits the source');
});

test('an open container keeps its own problems; auto counts name their source', () => {
    const left = layerNode('deepflow.template.Input', {}, 'left');
    const right = layerNode('deepflow.template.Input', {}, 'right');
    const add = layerNode('deepflow.op.Add', {}, 'join');
    const out = layerNode('deepflow.template.Output');
    const pair = createTemplate({ name: 'Pair', graph: { nodes: { [left.id]: left, [right.id]: right, [add.id]: add, [out.id]: out },
        edges: [createEdge({ source: left.id, target: add.id }), createEdge({ source: right.id, target: add.id }), createEdge({ source: add.id, target: out.id })] } });
    const instance = createSuperNodeInstance({ templateId: pair.id, position: { x: 0, y: 100 }, label: 'Pair block' });
    const input = layerNode('deepflow.io.Input', { shape: '(8,)' });
    const graph = { nodes: { [input.id]: input, [instance.id]: instance }, edges: [createEdge({ source: input.id, target: instance.id, targetHandle: left.id })] };
    const ctx = custom(graph, { [pair.id]: pair });
    const open = find(rowsAt(ctx, 'blocks'), 'Pair block');
    assert.equal(open.expanded, true);
    assert.ok(open.errors.includes('Connect every template input port'), 'The open row still shows the unconnected port');

    const head = createTemplate({ name: 'Head', graph: linked(layerNode('deepflow.template.Input'), layerNode('torch.nn.ReLU'), layerNode('deepflow.template.Output')) });
    const broadcast = layerNode('deepflow.op.Broadcast', { count: 4 }, 'copies');
    const heads = createRepeatNode({ templateId: head.id, position: { x: 0, y: 0 }, count: 'auto', mode: 'parallel', label: 'Heads' });
    const parallel = custom(linked(layerNode('deepflow.io.Input', { shape: '(8,)' }), broadcast, heads), { [head.id]: head });
    const auto = target(parallel, find(rowsAt(parallel, 'blocks'), 'Heads'), 'count', 'symbols');
    assert.equal(auto.kind, 'readonly');
    assert.match(auto.reason, /Broadcast copies \(count 4\)/);
});

test('an eye alias bound on its instance is edited where the instance binds it', () => {
    load(resnetStoryboard);
    const block = Object.values(templatesStore.getState().templates).find((template) => template.name === 'ConvBlock');
    const conv = Object.values(block.graph.nodes).find((node) => node.libraryKey === 'torch.nn.Conv2d');
    templatesStore.getState().toggleFieldExposure(block.id, { nodeId: conv.id, field: 'stride', spec: { name: 'stride', type: 'tuple', default: 1 }, defaultValue: 1, label: 'Stride' });
    const unbound = model();
    const unboundRow = find(rowsAt(unbound, 'blocks'), 'Conv2d');
    const unboundTarget = target(unbound, unboundRow, 'stride', 'values');
    assert.deepEqual([unboundTarget.kind, unboundTarget.templateId, unboundTarget.name], ['templateDefault', block.id, 'stride'], 'Nothing binds it: the default is what every instance reads');
    assert.equal(targetTag(unboundTarget, { row: unboundRow, rootGraph: unbound.graph, templates: unbound.templates }), 'default of stride in ConvBlock', 'The editor says where the value goes');
    assert.equal(targetSummary(unboundTarget, 2, { row: unboundRow, rootGraph: unbound.graph, templates: unbound.templates, templateUses: unbound.metrics.templateUses }),
        'stride = 2 (default in ConvBlock, 3 uses)');

    const stages = Object.values(architectureStore.getState().graph.nodes).find((node) => node.label === 'ResNet stages');
    architectureStore.getState().patchRootNode(stages.id, { bindings: { ...stages.bindings, stride: 2 } });
    const bound = model();
    const row = find(rowsAt(bound, 'blocks'), 'Conv2d');
    assert.equal(fieldOf(row, 'stride').values.text, '2', 'The cell shows the value the instances read');
    const valuesTarget = target(bound, row, 'stride', 'values');
    assert.deepEqual([valuesTarget.kind, valuesTarget.nodeId, valuesTarget.source, valuesTarget.field, valuesTarget.literalOnly],
        ['rootNode', stages.id, 'bindings', 'stride', true], 'Values follows $stride to the stages binding');
    assert.equal(describeTarget(valuesTarget, { row, rootGraph: bound.graph, templates: bound.templates }), 'Sets stride on ResNet stages · all 3 instances');
    assert.equal(targetTag(valuesTarget, { row, rootGraph: bound.graph, templates: bound.templates }), 'stride on ResNet stages');
    assert.equal(targetSummary(valuesTarget, 3, { row, rootGraph: bound.graph, templates: bound.templates }), 'ResNet stages stride = 3 (all 3 instances)');
    const symbolsTarget = target(bound, row, 'stride', 'symbols');
    assert.deepEqual([symbolsTarget.kind, symbolsTarget.name], ['templateDefault', 'stride'], 'Symbols still edits the definition default');
});

test('values that differ per instance read as their range, whichever instance differs', () => {
    load(gpt2Storyboard);
    const stack = Object.values(architectureStore.getState().graph.nodes).find((node) => node.label === 'Transformer blocks');
    architectureStore.getState().setInspectedIteration([stack.id], 4);
    assert.equal(architectureStore.getState().setStructureScope('local', [stack.id]), true);
    const localId = Object.values(architectureStore.getState().graph.nodes[stack.id].structureOverrides)[0].templateId;
    const fc = Object.values(templatesStore.getState().templates[localId].graph.nodes).find((node) => node.label === 'mlp fc');
    templatesStore.getState().patchTemplateNode(localId, fc.id, { parameterValues: { ...fc.parameterValues, out_features: 1536 } });
    const proj = Object.values(templatesStore.getState().templates[localId].graph.nodes).find((node) => node.label === 'mlp proj');
    templatesStore.getState().patchTemplateNode(localId, proj.id, { parameterValues: { ...proj.parameterValues, in_features: 1536 } });
    const narrowed = model();
    assert.deepEqual(narrowed.metrics.issues.errors, [], 'Block 4 is a valid, narrower block');
    const rows = rowsAt(narrowed, 'blocks');
    const narrow = find(rows, 'mlp fc');
    assert.equal(cellText(narrow, 'values'), '768 → 1536…3072', 'A middle instance is never hidden behind first…last');
    assert.equal(fieldOf(narrow, 'out_features').values.title, 'i=0–3: 3072 · i=4: 1536 · i=5–11: 3072');
    assert.equal(narrow.shape.text, 'B × 1024 × 1536…3072', 'Consistent with the Output column');
    assert.equal(cellText(find(rows, 'mlp proj'), 'values'), '1536…3072 → 768');
    assert.deepEqual(narrow.params.perInstance.map((run) => [run.from, run.to, run.value]),
        [['i=0', 'i=3', 2362368], ['i=4', 'i=4', 1181184], ['i=5', 'i=11', 2362368]], 'Metric tooltips list which instances differ');
    assert.equal(fitShapeText(narrow.shape.text, 15), 'B × … × 1536…3072', 'A cut keeps the axis that varies');

    const resnet = load(resnetStoryboard);
    const stages = find(rowsAt(resnet, 'blocks'), 'ResNet stages');
    assert.deepEqual(stages.params.perInstance.map((run) => [run.from, run.value]), [['i=0', 1856], ['i=1', 73984], ['i=2', 295424]]);
    assert.deepEqual(stages.macs.perInstance.map((run) => run.value), [1769472, 75497472, 301989888]);
});

test('config cells name tanh GELU, switched-off weights and convolutions without bias, and read as sentences', () => {
    const gpt2 = load(gpt2Storyboard);
    const blocks = rowsAt(gpt2, 'blocks');
    const gelu = fieldOf(find(blocks, 'gelu'), 'approximate');
    assert.deepEqual([gelu.prefix, gelu.values.text], ['approximate', 'tanh'], 'Shown as approximate=tanh');
    const exact = layerRow('torch.nn.GELU', { approximate: 'none' });
    assert.deepEqual(exact.config.map((part) => [part.name, part.concealed]), [['approximate', true]], 'Exact GELU says nothing, yet stays editable');
    assert.equal(configAccessibleText(exact, 'values'), '');
    assert.equal(configAccessibleText(find(blocks, 'mlp fc'), 'values'), 'in_features 768 to out_features 3072');
    assert.equal(configAccessibleText(find(blocks, 'lm_head'), 'values'), 'in_features 768 to out_features 50257, no bias');
    assert.equal(configAccessibleText(find(blocks, 'Transformer blocks'), 'symbols'), 'count $n_layers, mlp_width 4 * d_model');
    const lmBias = find(rowsAt(gpt2, 'ops'), 'bias', 'lm_head');
    assert.ok(lmBias.config.some((part) => part.kind === 'note' && part.text === 'off'), 'lm_head bias is disabled: enabled = $bias = False');
    assert.equal(lmBias.params.total, 0);

    const resnet = load(resnetStoryboard);
    const conv = find(rowsAt(resnet, 'blocks'), 'Conv2d');
    assert.equal(cellText(conv, 'values'), '3…128 → 64…256 · k 3 s 1 p 1 no bias');
    assert.equal(configAccessibleText(conv, 'values'), 'in_channels 3 through 128 to out_channels 64 through 256, kernel_size 3, stride 1, padding 1, no bias');
});

test('rows after an unknown size or inside a stopped analysis are marked, never exact', () => {
    load(gpt2Storyboard);
    const stack = Object.values(architectureStore.getState().graph.nodes).find((node) => node.label === 'Transformer blocks');
    architectureStore.getState().patchRootNode(stack.id, { count: '$missing' });
    const missing = rowsAt(model(), 'blocks');
    const lmHead = find(missing, 'lm_head');
    assert.deepEqual([lmHead.macs.total, lmHead.macs.unresolved], [0, 1], 'lm_head MACs are unknown (?), not 38.6M from a batch-less placeholder');
    assert.equal(find(missing, 'ln_f').shape.unknown, true, 'Its output shape is a guess too');
    assert.equal(find(missing, 'Transformer blocks').multiplicity, null);

    load(gpt2Storyboard);
    architectureStore.getState().patchGlobalParameter('n_layers', { default: 96 });
    architectureStore.getState().patchGlobalParameter('n_heads', { default: 96 });
    const huge = model();
    const rows = rowsAt(huge, 'blocks');
    const ln1 = find(rows, 'ln_1');
    assert.ok(ln1.multiplicity < 96);
    assert.deepEqual(ln1.lowerBound, { label: 'Transformer blocks', reason: 'the analysis stopped at 20,000 expanded nodes' }, 'Counts inside are lower bounds (≥)');
    assert.equal(find(rows, 'Transformer blocks').multiplicity, 96, 'The declared count stays');
    assert.ok(rows.every((row) => row.errors.every((message) => !/Expanded graph exceeds/.test(message))), 'The budget is not a model error');
    assert.match(find(rows, 'ln_f').warnings[0], /analysis stopped at 20,000 expanded nodes/);
});

test('text cut to fit never reads as another value', () => {
    assert.equal(cutExpression('4 * d_model', 9), '4 * d_mo…');
    assert.equal(cutExpression('4 * d_model', 4), '4 *…', 'Never "4…", which would read as the value 4');
    assert.equal(cutExpression('64 * 2 ** i', 7), '64 *…');
    assert.equal(cutExpression('i === 0 ? 3 : 64', 12), 'i === 0 ?…');
    const field = (name, text, prefix = null) => ({ kind: 'field', name, text, prefix });
    const fit = (parts, budget) => {
        const { texts, hiddenFrom } = fitConfigParts(parts, { budget, display: (part) => part.text, expression: (part) => /[$*]/.test(part.text) });
        return [parts.slice(0, hiddenFrom ?? parts.length).map((part, index) => part.kind === 'field' ? texts[index] : part.text).join(' '), hiddenFrom];
    };
    const blocks = [field('count', '$n_layers', '×'), { kind: 'text', text: '·' }, field('mlp_width', '4 * d_model', 'mlp_width')];
    assert.deepEqual(fit(blocks, 40), ['$n_layers · 4 * d_model', null]);
    assert.deepEqual(fit(blocks, 28), ['$n_layers · 4 * d…', null], 'The formula is cut inside, where the ellipsis belongs to it');
    assert.deepEqual(fit(blocks, 23), ['$n_layers', 1], 'A whole group gives way before a $name is cut');
    const linear = [field('in_features', '$d_model'), { kind: 'text', text: '→' }, field('out_features', '$vocab_size'), { kind: 'note', text: 'no bias' }];
    assert.deepEqual(fit(linear, 23), ['$d_model → $vocab_size', 3], 'The note gives way first');
    assert.deepEqual(fit(linear, 20), ['$d_model → $vocab_…', 3], 'An in → out pair stays together');
    assert.equal(fitShapeText('B × 64…256 × 32 × 32', 12), 'B × 64…256 × …', 'Shapes keep whole axes');
    assert.equal(fitShapeText('12 tensors [B × 1024 × 64]', 14), '12 tensors [B × 1024 × 64]', 'A collection is measured as drawn: 12×[B×1024×64]');
    assert.equal(fitShapeText('12 tensors [B × 1024 × 64]', 12), '12 tensors [B × …]', 'and cut inside its brackets');
});

test('an Input shape is checked in the grammar validation reads before it is written', () => {
    const spec = { name: 'shape', type: 'string' };
    const check = (text) => precheckCellValue(parseCellValue(text, spec), { spec, libraryKey: 'deepflow.io.Input' });
    for (const valid of ['(1024,)', '(B, 3, 32, 32)', '(3, 32, 32)', '1024']) assert.equal(check(valid), null, valid);
    assert.match(check('(abc)'), /numbers, not names \(abc\)/);
    assert.match(check('(1024, $foo)'), /numbers, not names \(\$foo\)/);
    assert.match(check('()'), /sizes such as/);
    for (const size of ['(-3)', '(1.5,)', '(0,)']) assert.match(check(size), /positive whole numbers/, size);
    assert.match(check('(3, B)'), /Only the first size/);
    // Other tuples allow zero (padding) and resolve names.
    assert.equal(precheckCellValue('(0, 0)', { spec: { name: 'padding', type: 'tuple' } }), null);
    assert.equal(precheckCellValue('($k, $k)', { spec: { name: 'kernel_size', type: 'tuple' }, scope: { k: 3 } }), null);
    assert.equal(precheckCellValue('(3, $nope)', { spec: { name: 'kernel_size', type: 'tuple' }, scope: {} }), 'No parameter named $nope');
    assert.equal(precheckCellValue('()', { spec: { name: 'kernel_size', type: 'tuple' } }), 'Not a list of sizes');
});

test('partial states never claim a per-instance figure or × N for instances that are not known', () => {
    load(gpt2Storyboard);
    // d_model = 1024 with 12 heads: the head split fails, so every block after the first is unknown.
    architectureStore.getState().patchGlobalParameter('d_model', { default: 1024 });
    const partial = model();
    const ops = rowsAt(partial, 'ops');
    const q = find(ops, 'Q projection');
    assert.deepEqual([q.macs.total, q.macs.knownCount, q.macs.count], [1073741824, 1, 12], 'One of 12 instances is known exactly');
    assert.equal(eachText('macs', q.macs), null, 'No "1.07G ea" beside a total that covers one instance');
    const lines = metricLines('macs', q.macs, q);
    assert.equal(lines[0], 'At least 1,073,741,824 MACs: exact in 1 of 12 instances');
    assert.equal(lines[1], 'i=0: 1,073,741,824 · i=1–11: ?', 'Each instance says what is known');
    assert.ok(lines.every((line) => !/× 12 =/.test(line)));
    // Block 0 reads its width D from the embeddings; later blocks would read it after the failed split.
    assert.equal(eachText('params', q.params), null, 'Weights sized by a width inferred after the error are not counted');
    const paramLines = metricLines('params', q.params, q);
    assert.deepEqual(paramLines.slice(0, 2), ['At least 1,049,600 parameters: exact in 1 of 12 instances', 'i=0: 1,049,600 · i=1–11: ?']);
    assert.match(paramLines[2], /^Not counted: weight \(D is inferred from an input after causal self-attention › split Q: Split dimension 1024 must be divisible by 12 chunks\)/);
    assert.deepEqual(q.config.filter((part) => part.kind === 'field').map((part) => [part.values.text, part.values.title]),
        [['?', 'i=0: 1024 · i=1–11: ?'], ['?', 'i=0: 1024 · i=1–11: ?']], 'Values mode does not show the inferred width as known either');
    assert.match(q.config[0].unknownReason, /^D is inferred from an input after causal self-attention › split Q: /, 'and says why');

    const blocks = find(rowsAt(partial, 'blocks'), 'Transformer blocks');
    assert.equal(blocks.params.knownCount, 0);
    assert.deepEqual(blocks.params.perInstance.map((run) => [run.from, run.to, run.value]),
        [['i=0', 'i=0', { atLeast: 12596224 }], ['i=1', 'i=11', { atLeast: 8397824 }]], 'Lower bounds read ≥ per instance');
    assert.equal(metricLines('params', blocks.params, blocks)[1], 'i=0: ≥ 12,596,224 · i=1–11: ≥ 8,397,824');
    assert.match(cellStatus(q, 5, { lens: 'macs', modelTotal: partial.metrics.totals.macs, modelPartial: true }), /^At least 1,073,741,824 MACs: exact in 1 of 12 instances · i=0: 1,073,741,824 · i=1–11: \? · Not counted: /,
        'The status line carries the breakdown');
});

test('shares of a partial total are upper bounds, or unknown where the row is partial too', () => {
    load(gpt2Storyboard);
    const block = templatesStore.getState().templates[Object.values(architectureStore.getState().graph.nodes).find((node) => node.label === 'Transformer blocks').templateId];
    const proj = Object.values(block.graph.nodes).find((node) => node.label === 'mlp proj');
    templatesStore.getState().patchTemplateNode(block.id, proj.id, { parameterValues: { ...proj.parameterValues, out_features: 1024 } });
    const broken = model();
    const rows = rowsAt(broken, 'blocks');
    const total = broken.metrics.totals.params;
    assert.ok(broken.metrics.totals.unresolved.params.size > 0, 'The model total is a lower bound');
    const share = (label) => shareInfo(find(rows, label), 'params', total, true);
    assert.equal(share('token embedding').text, '≤26.3%', 'An exact row of a partial total: at most');
    assert.equal(share('ln_f').text, '<0.1%', '"<0.1%" is already a bound');
    assert.equal(share('Transformer blocks').text, '?', 'Partial row and partial total: either way');
    assert.match(share('Transformer blocks').title, /^Unknown share: /);
    assert.equal(shareInfo(find(rows, 'token embedding'), 'params', total, false).text, '26.3%');

    // Shared definitions name the instances that fail.
    const ln1 = find(rows, 'ln_1');
    assert.deepEqual([ln1.shape.text, ln1.shape.known, ln1.shape.count, knownShapeMark(ln1)], ['B × 1024 × 768', 1, 12, '1/12'],
        'A shape known in block 0 only says so in the cell');
    assert.equal(knownShapeMark(find(rows, 'token embedding')), null);
    assert.equal(failingText(ln1), 'Fails in i=1–11 (11 of 12)', 'Block 0 still reads 768 wide');
    assert.match(cellStatus(ln1, 0, { lens: 'params' }), /internal errors: Element-wise dim 2 mismatch: 1024 vs 768 · Fails in i=1–11 \(11 of 12\)$/);
    const lnF = find(rows, 'ln_f');
    assert.equal(lnF.shape.unknown, true);
    assert.ok(cellStatus(lnF, 2, { lens: 'params' }).endsWith(SHAPE_GUESS_NOTE), 'A ? shape says why the canvas still draws one');
});

test('edits are summarised by the value that really changes', () => {
    const ctx = load(gpt2Storyboard);
    const rows = rowsAt(ctx, 'blocks');
    const ln1 = find(rows, 'ln_1');
    const global = target(ctx, ln1, 'normalized_shape', 'values');
    const uses = ctx.metrics.templateUses;
    assert.equal(targetTag(global, { row: ln1, rootGraph: ctx.graph, templates: ctx.templates }), 'global d_model');
    assert.match(targetSummary(global, 1024, { row: ln1, rootGraph: ctx.graph, templates: ctx.templates, templateUses: uses }), /^d_model = 1024 \(global, \d+ fields\)$/);
    const dropout = find(rows, 'attention projection dropout');
    const shared = target(ctx, dropout, 'p', 'symbols');
    assert.equal(targetTag(shared, { row: dropout, rootGraph: ctx.graph, templates: ctx.templates }), null, 'The cell\'s own field needs no tag');
    assert.equal(targetSummary(shared, 0.2, { row: dropout, rootGraph: ctx.graph, templates: ctx.templates, templateUses: uses }),
        'attention projection dropout p = 0.2 (shared Transformer block, 12 uses)');
    const root = find(rows, 'Dropout');
    assert.equal(targetSummary(target(ctx, root, 'p', 'values'), null, { row: root, rootGraph: ctx.graph, templates: ctx.templates }), 'Dropout p = default');

});
