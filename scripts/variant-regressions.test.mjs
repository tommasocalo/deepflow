import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLayerNode, patchNode } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { createVariantPatch, getNodeFamily, getPaletteEntries, NODE_FAMILIES } from '../src/domain/library/nodeFamilies.js';
import { generateArchitectureModule } from '../src/codegen/index.js';

function node(key, parameters, label) {
    return createLayerNode({ id: 'projection', libraryKey: key, entry: getEntry(key), position: { x: 100, y: 100 }, parameters, label });
}

test('palette exposes a single entry for each family and retains searchable variants', () => {
    const entries = getPaletteEntries();
    for (const family of NODE_FAMILIES) {
        const members = entries.filter((entry) => entry.family?.key === family.key);
        assert.equal(members.length, 1, `${family.name} appears once`);
        assert.equal(members[0].key, family.defaultKey);
        for (const variant of family.variants) {
            assert.ok(getEntry(variant.key), 'Every selectable variant is a supported operation');
            assert.ok(members[0].aliases.includes(variant.key));
            assert.ok(members[0].aliases.includes(variant.name));
        }
    }
    assert.deepEqual(NODE_FAMILIES.map((family) => family.key), ['linear', 'attention', 'normalization', 'activation', 'convolution', 'pooling', 'recurrent']);
    const linear = entries.find((entry) => entry.name === 'Linear');
    assert.ok(linear.aliases.includes('Embedding'));
    assert.ok(linear.aliases.includes('Positional embedding'));
    assert.equal(getNodeFamily('torch.nn.TransformerEncoderLayer'), null, 'An encoder block includes an FFN, so is not an attention variant');
    assert.equal(entries.filter((entry) => entry.key === 'torch.nn.Dropout').length, 1);
    assert.equal(entries.filter((entry) => entry.key === 'torch.nn.TransformerEncoderLayer').length, 1);
    assert.deepEqual(entries.filter((entry) => entry.category === 'Operations').map((entry) => entry.key).sort(), [
        'deepflow.op.Add', 'deepflow.op.AddScalar', 'deepflow.op.Arange', 'deepflow.op.Broadcast', 'deepflow.op.CausalMask', 'deepflow.op.Concat', 'deepflow.op.Cos', 'deepflow.op.Exp', 'deepflow.op.Gather', 'deepflow.op.MatMul', 'deepflow.op.Mean',
        'deepflow.op.Multiply', 'deepflow.op.Parameter', 'deepflow.op.PositionIndices', 'deepflow.op.Pow', 'deepflow.op.Rsqrt', 'deepflow.op.Scale', 'deepflow.op.Sin', 'deepflow.op.Slice', 'deepflow.op.Split', 'deepflow.op.Stack', 'deepflow.op.Subtract', 'deepflow.op.Transpose', 'deepflow.op.Unsqueeze', 'torch.nn.Flatten',
    ].sort());
    assert.ok(entries.every((entry) => !entry.key.startsWith('deepflow.template.')));
});

test('Linear variants preserve output width, remove stale parameters and retain graph identity', () => {
    const linear = node('torch.nn.Linear', { in_features: 768, out_features: '$width', bias: false }, 'Token projection');
    const edge = { id: 'edge', source: 'input', target: linear.id, gradientOp: 'reverse', gradientOpLambda: 0.5 };
    const graph = { nodes: { [linear.id]: linear }, edges: [edge] };
    const embedded = patchNode(graph, linear.id, createVariantPatch(linear, 'torch.nn.Embedding'));
    const embedding = embedded.nodes[linear.id];
    assert.equal(embedding.libraryKey, 'torch.nn.Embedding');
    assert.equal(embedding.label, 'Token projection');
    assert.equal(embedding.id, linear.id);
    assert.equal(embedding.position, linear.position);
    assert.equal(embedded.edges, graph.edges);
    assert.equal(embedded.edges[0], edge);
    assert.deepEqual(embedding.parameters, { num_embeddings: null, embedding_dim: '$width', padding_idx: null });
    assert.deepEqual(linear.parameters, { in_features: 768, out_features: '$width', bias: false });

    const position = { ...embedding, ...createVariantPatch(embedding, 'deepflow.nn.PositionalEmbedding') };
    assert.equal(position.libraryKey, 'deepflow.nn.PositionalEmbedding');
    assert.deepEqual(position.parameters, { max_len: null, embedding_dim: '$width' });
    const projected = { ...position, ...createVariantPatch(position, 'torch.nn.Linear') };
    assert.deepEqual(projected.parameters, { in_features: null, out_features: '$width', bias: true });
    assert.equal(projected.label, 'Token projection');
});

test('attention variants preserve heads, width and dropout while applying their actual parameter schemas', () => {
    const causal = node('deepflow.nn.CausalSelfAttention', { embed_dim: 768, num_heads: 12, dropout: 0.2, bias: false });
    const patch = createVariantPatch(causal, 'torch.nn.MultiheadAttention');
    assert.equal(patch.label, 'MultiheadAttention');
    assert.equal(patch.libraryKey, 'torch.nn.MultiheadAttention');
    assert.deepEqual(patch.parameters, { embed_dim: 768, num_heads: 12, dropout: 0.2, batch_first: true });
    const reversed = createVariantPatch({ ...causal, ...patch }, causal.libraryKey);
    assert.deepEqual(reversed.parameters, { embed_dim: 768, num_heads: 12, dropout: 0.2, bias: true });
    assert.equal(createVariantPatch(causal, 'torch.nn.Linear'), null);
    assert.equal(createVariantPatch(causal, causal.libraryKey), null);
    assert.equal(createVariantPatch(causal, 'missing'), null);
    assert.equal(createVariantPatch(causal, 'torch.nn.TransformerEncoderLayer'), null, 'An encoder is not an attention variant');
    assert.equal(createVariantPatch(node('torch.nn.TransformerEncoderLayer', {}), causal.libraryKey), null);
});

test('normalization variants preserve channels and shared settings without conflating normalized axes', () => {
    const batch = node('torch.nn.BatchNorm2d', { num_features: 64, eps: 0.001, momentum: 0.2, affine: false }, 'Feature normalization');
    const oneDimensional = createVariantPatch(batch, 'torch.nn.BatchNorm1d');
    assert.deepEqual(oneDimensional.parameters, batch.parameters);
    const grouped = createVariantPatch(batch, 'torch.nn.GroupNorm');
    assert.equal(grouped.label, 'Feature normalization');
    assert.deepEqual(grouped.parameters, { num_groups: null, num_channels: 64, eps: 0.001, affine: false });
    const reversed = createVariantPatch({ ...batch, ...grouped }, 'torch.nn.BatchNorm2d');
    assert.deepEqual(reversed.parameters, { num_features: 64, eps: 0.001, momentum: 0.1, affine: false });
    const layer = createVariantPatch(batch, 'torch.nn.LayerNorm');
    assert.deepEqual(layer.parameters, { normalized_shape: null, eps: 0.001, elementwise_affine: true });
    const fromLayer = createVariantPatch(node('torch.nn.LayerNorm', { normalized_shape: [64, 32, 32], eps: 0.002, elementwise_affine: false }), 'torch.nn.GroupNorm');
    assert.deepEqual(fromLayer.parameters, { num_groups: null, num_channels: null, eps: 0.002, affine: true });
});

test('activation variants carry compatible flags and remove parameters unsupported by the chosen function', () => {
    const relu = node('torch.nn.ReLU', { inplace: true });
    const leaky = { ...relu, ...createVariantPatch(relu, 'torch.nn.LeakyReLU') };
    assert.deepEqual(leaky.parameters, { negative_slope: 0.01, inplace: true });
    assert.equal(leaky.label, 'LeakyReLU');
    const gelu = { ...leaky, ...createVariantPatch(leaky, 'torch.nn.GELU') };
    assert.deepEqual(gelu.parameters, { approximate: 'none' });
    assert.deepEqual(createVariantPatch(gelu, 'torch.nn.Sigmoid').parameters, {});
    assert.deepEqual(createVariantPatch(gelu, 'torch.nn.Softmax').parameters, { dim: -1 });
    assert.equal(createVariantPatch(relu, 'torch.nn.Dropout'), null);
});

test('convolution variants preserve channel and scalar geometry settings while avoiding arbitrary tuple reduction', () => {
    const conv1 = node('torch.nn.Conv1d', { in_channels: '$in', out_channels: 64, kernel_size: 5, stride: 2, padding: 1, dilation: 2, bias: false });
    const conv2 = { ...conv1, ...createVariantPatch(conv1, 'torch.nn.Conv2d') };
    assert.deepEqual(conv2.parameters, { in_channels: '$in', out_channels: 64, kernel_size: 5, stride: 2, padding: 1, dilation: 2, groups: 1, bias: false });
    const transpose = createVariantPatch(conv2, 'torch.nn.ConvTranspose2d');
    assert.deepEqual(transpose.parameters, { in_channels: '$in', out_channels: 64, kernel_size: 5, stride: 2, padding: 1, output_padding: 0, bias: false });
    const anisotropic = { ...conv2, parameters: { ...conv2.parameters, kernel_size: [3, 5], padding: '(1, 2)', stride: '2' } };
    const backTo1d = createVariantPatch(anisotropic, 'torch.nn.Conv1d');
    assert.deepEqual(backTo1d.parameters, { in_channels: '$in', out_channels: 64, kernel_size: 3, stride: '2', padding: 0, dilation: 2, bias: false });
    assert.deepEqual(anisotropic.parameters.kernel_size, [3, 5], 'Switching does not mutate the prior node');
});

test('pooling variants distinguish window geometry from adaptive output size', () => {
    const max1 = node('torch.nn.MaxPool1d', { kernel_size: 3, stride: 2, padding: 1, ceil_mode: true });
    const max2 = { ...max1, ...createVariantPatch(max1, 'torch.nn.MaxPool2d') };
    assert.deepEqual(max2.parameters, max1.parameters);
    const average = { ...max2, ...createVariantPatch(max2, 'torch.nn.AvgPool2d') };
    assert.deepEqual(average.parameters, { kernel_size: 3, stride: 2, padding: 1, ceil_mode: true, count_include_pad: true });
    const adaptive = { ...average, ...createVariantPatch(average, 'torch.nn.AdaptiveAvgPool2d') };
    assert.deepEqual(adaptive.parameters, { output_size: 1 });
    const windowed = createVariantPatch(adaptive, 'torch.nn.MaxPool2d');
    assert.deepEqual(windowed.parameters, { kernel_size: null, stride: null, padding: 0, ceil_mode: false });
});

test('recurrent variant changes preserve graph connections and export the selected cell', () => {
    const lstm = node('torch.nn.LSTM', { input_size: 32, hidden_size: '$width', num_layers: 3, dropout: 0.25, bidirectional: true, batch_first: false }, 'Sequence encoder');
    const edge = { id: 'recurrent-edge', source: 'input', target: lstm.id, gradientOp: 'detach' };
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', entry: getEntry('deepflow.io.Input'), position: { x: 100, y: 0 }, parameters: { shape: '(24, 32)' } });
    const graph = { nodes: { input, projection: lstm }, edges: [edge] };
    const changed = patchNode(graph, lstm.id, createVariantPatch(lstm, 'torch.nn.GRU'));
    const gru = changed.nodes[lstm.id];
    assert.equal(gru.id, lstm.id);
    assert.equal(gru.position, lstm.position);
    assert.equal(gru.label, 'Sequence encoder');
    assert.equal(changed.edges, graph.edges);
    assert.equal(changed.edges[0], edge);
    assert.deepEqual(gru.parameters, { input_size: 32, hidden_size: '$width', num_layers: 3, dropout: 0.25, batch_first: false });
    const concrete = patchNode(changed, gru.id, { parameters: { ...gru.parameters, hidden_size: 64 } });
    const code = generateArchitectureModule(concrete);
    assert.match(code, /torch\.nn\.GRU\(input_size=32, hidden_size=64, num_layers=3, dropout=0\.25, batch_first=False\)/);
    assert.doesNotMatch(code, /torch\.nn\.LSTM|bidirectional=/);
});

test('switching an embedding variant changes exported operation semantics', () => {
    const input = createLayerNode({ id: 'input', libraryKey: 'deepflow.io.Input', entry: getEntry('deepflow.io.Input'), position: { x: 100, y: 0 }, parameters: { shape: '(1024,)', dtype: 'int64' } });
    const embedding = node('torch.nn.Embedding', { num_embeddings: 50257, embedding_dim: 768, padding_idx: null });
    const graph = { nodes: { input, projection: embedding }, edges: [{ id: 'input-edge', source: input.id, target: embedding.id }] };
    const before = generateArchitectureModule(graph);
    assert.match(before, /torch\.nn\.Embedding\(num_embeddings=50257, embedding_dim=768/);
    const patch = createVariantPatch(embedding, 'deepflow.nn.PositionalEmbedding');
    patch.parameters.max_len = 1024;
    const after = generateArchitectureModule(patchNode(graph, embedding.id, patch));
    assert.match(after, /torch\.nn\.Embedding\(1024, 768\)/);
    assert.doesNotMatch(after, /num_embeddings=50257/);
    assert.match(after, /arange/);
});
