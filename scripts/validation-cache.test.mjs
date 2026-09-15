import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createValidationCache } from '../src/validation/cache.js';
import { validate } from '../src/validation/validator.js';

test('many graph consumers share validation while scope and graph edits refresh shapes and errors', () => {
    let evaluations = 0;
    const readReport = createValidationCache((graph, options) => {
        evaluations += 1;
        return validate(graph, options);
    });
    const templates = {};
    const graph = {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.io.Input', parameters: { shape: '(B, 4)', dtype: 'float32' } },
            linear: { id: 'linear', kind: 'layer', libraryKey: 'torch.nn.Linear', parameters: { in_features: 4, out_features: '$width', bias: true } },
        },
        edges: [{ id: 'edge', source: 'input', target: 'linear' }],
    };
    const first = readReport(graph, templates, { width: 8, unused: 2 });
    for (let consumer = 0; consumer < 100; consumer += 1) {
        assert.equal(readReport(graph, templates, { unused: 2, width: 8 }), first);
    }
    assert.equal(evaluations, 1);
    assert.deepEqual(first.shapes.linear, [null, 8]);

    const resized = readReport(graph, templates, { width: 16, unused: 2 });
    assert.equal(evaluations, 2);
    assert.deepEqual(resized.shapes.linear, [null, 16]);

    const incompatible = {
        ...graph,
        nodes: { ...graph.nodes, linear: { ...graph.nodes.linear, parameters: { ...graph.nodes.linear.parameters, in_features: 7 } } },
    };
    const invalid = readReport(incompatible, templates, { width: 16, unused: 2 });
    assert.equal(evaluations, 3);
    assert.equal(invalid.ok, false);
    assert.ok(invalid.errors.linear.some((error) => error.includes('input has 4 features')));
});

test('template changes refresh a report even when the viewed graph is unchanged', () => {
    let evaluations = 0;
    const readReport = createValidationCache((graph, options) => {
        evaluations += 1;
        return validate(graph, options);
    });
    const graph = {
        nodes: { block: { id: 'block', kind: 'supernode', templateId: 'shared', parameterValues: {} } },
        edges: [],
    };
    const missing = readReport(graph, {}, {});
    assert.equal(missing.ok, false);
    const templates = {
        shared: {
            id: 'shared', name: 'Block', graph: {
                nodes: {
                    input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.template.Input', parameters: {} },
                    output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.template.Output', parameters: {} },
                },
                edges: [{ id: 'edge', source: 'input', target: 'output' }],
            },
        },
    };
    const available = readReport(graph, templates, {});
    assert.equal(evaluations, 2);
    assert.equal(available.ok, true);
    assert.equal(readReport(graph, templates, {}), available);
    assert.equal(evaluations, 2);
});
