import assert from 'node:assert/strict';
import { test } from 'node:test';
import { residualEdgeIds } from '../src/components/architecture/residualEdges.js';
import { buildTransformerBlockGraph } from '../src/storyboards/gpt2.js';

const edge = (source, target, id = `${source}-${target}`) => ({ id, source, target });

test('the GPT-2 transformer exposes only its two residual streams', () => {
    const graph = buildTransformerBlockGraph();
    const original = residualEdgeIds(graph.edges);
    const residuals = graph.edges.filter((connection) => original.has(connection.id));
    assert.equal(residuals.length, 2);
    assert.deepEqual(residuals.map(({ source, target }) => [graph.nodes[source].label, graph.nodes[target].label]), [
        ['block input', 'residual'],
        ['residual', 'residual'],
    ]);
    assert.equal(residuals[0].target, residuals[1].source, 'The second stream begins at the first residual merge');
    assert.ok(residuals.every(({ target }) => graph.nodes[target].libraryKey === 'deepflow.op.Add'));

    for (const node of Object.values(graph.nodes)) node.position = { x: 0, y: 0 };
    assert.deepEqual(residualEdgeIds(graph.edges), original, 'Overlapping or moving cards does not change flow topology');
});

test('the GPT-2 root embedding diamond is a branch and merge, without residual edges', () => {
    const edges = [
        edge('tokens', 'token embedding'), edge('tokens', 'positional embedding'),
        edge('token embedding', 'add'), edge('positional embedding', 'add'),
        edge('add', 'dropout'), edge('dropout', 'Transformer blocks'),
        edge('Transformer blocks', 'ln_f'), edge('ln_f', 'lm_head'), edge('lm_head', 'logits'),
    ];
    assert.deepEqual(residualEdgeIds(edges), new Set());
});

test('classification updates after deletion or reconnection of a bypass or its alternate path', () => {
    const chain = [edge('a', 'b'), edge('b', 'c')];
    const bypass = edge('a', 'c', 'bypass');
    assert.deepEqual(residualEdgeIds([...chain, bypass]), new Set(['bypass']));
    assert.deepEqual(residualEdgeIds([chain[0], bypass]), new Set(), 'Deleting the alternate path clears the residual');
    assert.deepEqual(residualEdgeIds(chain), new Set(), 'Deleting the bypass leaves an ordinary chain');
    assert.deepEqual(residualEdgeIds([...chain, { ...bypass, target: 'd' }]), new Set(), 'Reconnecting the bypass clears its former role');
    assert.deepEqual(residualEdgeIds([chain[0], { ...chain[1], target: 'd' }, bypass]), new Set(), 'Reconnecting the alternate path clears the residual');
    assert.deepEqual(residualEdgeIds([...chain, bypass]), new Set(['bypass']), 'Restoring connections restores their classification');
});

test('parallel direct connections alone are not residuals, even at a branch and merge', () => {
    const parallel = [edge('a', 'c', 'one'), edge('a', 'c', 'two')];
    assert.deepEqual(residualEdgeIds(parallel), new Set());
    assert.deepEqual(residualEdgeIds([...parallel, edge('a', 'b'), edge('d', 'c')]), new Set());
    assert.deepEqual(residualEdgeIds([...parallel, edge('a', 'b'), edge('b', 'c')]), new Set(['one', 'two']), 'Both direct wires bypass the same real computation');
});

test('cycles terminate without using the candidate edge or revisiting the source as an alternate path', () => {
    const cyclicBranch = [edge('a', 'c', 'candidate'), edge('a', 'b'), edge('b', 'a'), edge('d', 'c')];
    assert.deepEqual(residualEdgeIds(cyclicBranch), new Set(), 'A loop back through the candidate is not a bypass');
    const cyclicPath = [edge('a', 'c', 'candidate'), edge('a', 'b'), edge('b', 'd'), edge('d', 'b'), edge('d', 'c')];
    assert.deepEqual(residualEdgeIds(cyclicPath), new Set(['candidate']), 'A real alternate path remains detectable beside a cycle');
    assert.deepEqual(residualEdgeIds([edge('a', 'a'), edge('a', 'b'), edge('b', 'a')]), new Set(), 'Self loops do not create residual routes');
});

test('disconnected components and edge order cannot change residual classification', () => {
    const edges = [edge('a', 'b'), edge('b', 'c'), edge('a', 'c'), edge('d', 'e'), edge('e', 'd')];
    assert.deepEqual(residualEdgeIds(edges), new Set(['a-c']));
    assert.deepEqual(residualEdgeIds([...edges].reverse()), new Set(['a-c']));
    assert.deepEqual(residualEdgeIds([]), new Set());
});
