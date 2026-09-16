import assert from 'node:assert/strict';
import { test } from 'node:test';
import { buildTransformerBlockGraph } from '../src/storyboards/gpt2.js';
import { tensorFlowNodeIds } from '../src/components/architecture/tensorFlow.js';

function legacyBlock() {
    const graph = buildTransformerBlockGraph();
    for (const node of Object.values(graph.nodes)) {
        delete node.visualization;
        delete node.tensorRole;
    }
    return graph;
}

const labels = (graph, ids) => [...ids].map((id) => graph.nodes[id].label).sort();
const mlpLabels = ['ln_2', 'mlp fc', 'gelu', 'mlp proj', 'dropout', 'residual'].sort();

test('saved GPT-2 blocks without display metadata get the same MLP experiment as new storyboards', () => {
    const newGraph = buildTransformerBlockGraph();
    const saved = JSON.parse(JSON.stringify(legacyBlock()));
    const before = JSON.stringify(saved);
    assert.deepEqual(labels(newGraph, tensorFlowNodeIds(newGraph)), mlpLabels);
    assert.deepEqual(labels(saved, tensorFlowNodeIds(saved)), mlpLabels);
    assert.equal(JSON.stringify(saved), before, 'Recognition must not migrate or mutate a saved graph');
});

test('an ordinary MLP with the same labels does not opt into the GPT-2 experiment', () => {
    const graph = legacyBlock();
    const mlpIds = tensorFlowNodeIds(graph);
    const standalone = {
        nodes: Object.fromEntries(Object.entries(graph.nodes).filter(([id]) => mlpIds.has(id))),
        edges: graph.edges.filter((edge) => mlpIds.has(edge.source) && mlpIds.has(edge.target)),
    };
    assert.equal(tensorFlowNodeIds(standalone).size, 0);
    assert.equal(tensorFlowNodeIds({ nodes: {}, edges: [] }).size, 0);
});

test('broken GPT-2 residual topology stays unchanged, while explicit display metadata survives copying', () => {
    const graph = legacyBlock();
    const residuals = Object.values(graph.nodes).filter((node) => node.libraryKey === 'deepflow.op.Add');
    graph.edges = graph.edges.filter((edge) => !(edge.source === residuals[0].id && edge.target === residuals[1].id));
    assert.equal(tensorFlowNodeIds(graph).size, 0);
    const fc = Object.values(graph.nodes).find((node) => node.label === 'mlp fc');
    fc.visualization = 'tensor-flow';
    const copy = JSON.parse(JSON.stringify(graph));
    assert.deepEqual([...tensorFlowNodeIds(copy)], [fc.id]);
});
