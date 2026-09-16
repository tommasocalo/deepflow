import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createLayerNode, patchNode } from '../src/domain/architecture/index.js';
import { getEntry } from '../src/domain/library/index.js';
import { createVariantPatch, getNodeFamily, getPaletteEntries } from '../src/domain/library/nodeFamilies.js';
import { generateArchitectureModule } from '../src/codegen/index.js';

function node(key, parameters, label) {
    return createLayerNode({ id: 'projection', libraryKey: key, entry: getEntry(key), position: { x: 100, y: 100 }, parameters, label });
}

test('palette exposes a single entry for each family and retains searchable variants', () => {
    const entries = getPaletteEntries();
    assert.equal(entries.filter((entry) => entry.family?.key === 'linear').length, 1);
    assert.equal(entries.filter((entry) => entry.family?.key === 'attention').length, 1);
    const linear = entries.find((entry) => entry.name === 'Linear');
    assert.ok(linear.aliases.includes('Embedding'));
    assert.ok(linear.aliases.includes('Positional embedding'));
    assert.equal(getNodeFamily('torch.nn.TransformerEncoderLayer'), null, 'An encoder block includes an FFN, so is not an attention variant');
    assert.deepEqual(entries.filter((entry) => entry.category === 'Operations').map((entry) => entry.key).sort(), [
        'deepflow.op.Add', 'deepflow.op.Concat', 'deepflow.op.Multiply', 'deepflow.op.Split', 'torch.nn.Flatten',
    ]);
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
