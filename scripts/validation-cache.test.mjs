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

test('moving root and template nodes preserves reports while semantic edits invalidate them', () => {
    let evaluations = 0;
    const readReport = createValidationCache((graph, options) => {
        evaluations += 1;
        return validate(graph, options);
    });
    let inner = {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.template.Input', parameters: {}, position: { x: 0, y: 0 } },
            linear: { id: 'linear', kind: 'layer', libraryKey: 'torch.nn.Linear', parameters: { in_features: 4, out_features: '$width', bias: true }, position: { x: 0, y: 80 } },
            output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.template.Output', parameters: {}, position: { x: 0, y: 160 } },
        },
        edges: [{ id: 'a', source: 'input', target: 'linear' }, { id: 'b', source: 'linear', target: 'output' }],
    };
    let templates = { shared: { id: 'shared', name: 'Block', graph: inner, freeParameters: [{ name: 'width', default: 8 }] } };
    let graph = {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.io.Input', parameters: { shape: '(4,)', dtype: 'float32' }, position: { x: 0, y: 0 } },
            block: { id: 'block', kind: 'supernode', templateId: 'shared', parameterValues: {}, position: { x: 0, y: 80 } },
        },
        edges: [{ id: 'edge', source: 'input', target: 'block' }],
    };
    const initial = readReport(graph, templates);
    assert.deepEqual(initial.shapes.block, [null, 8]);
    for (let x = 1; x <= 60; x += 1) {
        graph = { ...graph, nodes: { ...graph.nodes, block: { ...graph.nodes.block, position: { x, y: 80 } } } };
        assert.equal(readReport(graph, templates), initial);
    }
    inner = { ...inner, nodes: { ...inner.nodes, linear: { ...inner.nodes.linear, position: { x: 50, y: 100 } } } };
    templates = { ...templates, shared: { ...templates.shared, graph: inner } };
    assert.equal(readReport(graph, templates), initial);
    assert.equal(evaluations, 1, 'Layout edits must not repeat shape inference');

    templates = { ...templates, shared: { ...templates.shared, freeParameters: [{ name: 'width', default: 16 }] } };
    assert.deepEqual(readReport(graph, templates).shapes.block, [null, 16]);
    assert.equal(evaluations, 2);
    inner = { ...inner, nodes: { ...inner.nodes, linear: { ...inner.nodes.linear, parameters: { ...inner.nodes.linear.parameters, in_features: 7 } } } };
    templates = { ...templates, shared: { ...templates.shared, graph: inner } };
    assert.ok(readReport(graph, templates).warnings.block.some((message) => message.includes('internal error')));
    assert.equal(evaluations, 3);
});

test('boundary shapes and edge edits invalidate cached reports', () => {
    const readReport = createValidationCache(validate);
    const graph = {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.template.Input', parameters: {} },
            output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.template.Output', parameters: {} },
        },
        edges: [{ id: 'edge', source: 'input', target: 'output' }],
    };
    const templates = {};
    assert.deepEqual(readReport(graph, templates, {}, { input: [null, 768] }).shapes.output, [null, 768]);
    assert.deepEqual(readReport(graph, templates, {}, { input: [null, 3072] }).shapes.output, [null, 3072]);
    const disconnected = readReport({ ...graph, edges: [] }, templates, {}, { input: [null, 3072] });
    assert.ok(disconnected.warnings.output.some((message) => message.includes('No incoming')));
});

test('scope keys distinguish null, undefined and non-finite expression values', () => {
    const readReport = createValidationCache(validate);
    const graph = {
        nodes: {
            input: { id: 'input', kind: 'layer', libraryKey: 'deepflow.template.Input', parameters: {} },
            linear: { id: 'linear', kind: 'layer', libraryKey: 'torch.nn.Linear', parameters: { in_features: 4, out_features: 'width + 1', bias: true } },
        },
        edges: [{ id: 'edge', source: 'input', target: 'linear' }],
    };
    const templates = {};
    const externalInputs = { input: [null, 4] };
    for (const width of [undefined, null, NaN, Infinity, -Infinity, 8]) {
        const scope = { width };
        assert.deepEqual(readReport(graph, templates, scope, externalInputs), validate(graph, { templates, scope, externalInputs }));
    }
});
