import assert from 'node:assert/strict';
import { test } from 'node:test';
import { computeModelMetrics, formatShare } from '../src/domain/metrics/index.js';
import { buildModelTableRows, readingOrder, rowKey } from '../src/domain/modelTable/rows.js';
import { describeTarget, followReference, parseCellValue, precheckCellValue, resolveWriteTarget } from '../src/domain/modelTable/writeTargets.js';
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
const cellText = (row, cells) => row.config.map((part) => part.kind === 'field' ? `${part.prefix && part.prefix !== '×' ? `${part.prefix} ` : part.prefix ?? ''}${part[cells].text}` : part.text).join(' ');
const target = (ctx, row, name, cells) => resolveWriteTarget({ row, field: fieldOf(row, name), cells, rootGraph: ctx.graph, templates: ctx.templates });

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
    assert.equal(precheckCellValue('$nope', { spec: { type: 'int' }, scope }), 'Unknown name $nope');
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
