import assert from 'node:assert/strict';
import { test } from 'node:test';

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { friendlyIssues } = await import('../src/edu/lessons/engine/feedback.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { validate } = await import('../src/validation/index.js');
const { withGlobalParameters } = await import('../src/domain/expressions.js');
const { t } = await import('../src/edu/lessons/engine/i18n.js');

/** Build a chain of nodes with the fixture helpers and return the friendly issues of its real validation report. */
function issuesOf(build) {
    const h = createFixtureHelpers();
    const { nodes, edges = h.chain(...nodes), parameters } = build(h);
    const snapshot = h.snapshot({ nodes, edges, parameters });
    const report = validate(snapshot.graph, { templates: snapshot.templates, scope: withGlobalParameters(snapshot.graph) });
    return { issues: friendlyIssues(report, snapshot.graph, snapshot.templates), report };
}
const text = (issue) => t(issue.key, issue.params);

test('Linear layer in_features mismatch → fb.linear-in', () => {
    const { issues, report } = issuesOf((h) => ({ nodes: [h.input('in', [4]), h.layer('lin', 'torch.nn.Linear', { in_features: 5, out_features: 3 }, 'neuron'), h.output('out')] }));
    assert.match(report.errors.lin[0], /Linear in_features=5 but input has 4 features/);
    assert.equal(issues[0].key, 'fb.linear-in');
    assert.equal(issues[0].nodeId, 'lin');
    assert.equal(text(issues[0]), '**neuron** expects 5 input features but receives 4.');
});

test('standard Linear component: the inner MatMul mismatch is rewritten for the component', () => {
    const { issues, report } = issuesOf((h) => ({ nodes: [h.input('in', [784]), h.linear('lin1', 748, 128, { label: 'hidden' }), h.output('out')] }));
    assert.match(report.errors.lin1[0], /MatMul inner dimensions mismatch: 784 vs 748/);
    assert.equal(issues[0].key, 'fb.linear-in');
    assert.equal(text(issues[0]), '**hidden** expects 748 input features but receives 784.');
});

test('a Linear on image tensors with in_features = C·H·W → fb.missing-flatten (layer and component)', () => {
    const component = issuesOf((h) => ({ nodes: [h.input('in', [1, 28, 28]), h.linear('lin1', 784, 128, { label: 'hidden' }), h.output('out')] }));
    assert.equal(component.issues[0].key, 'fb.missing-flatten');
    assert.equal(text(component.issues[0]), '**hidden** works on the last axis only, so it receives 28 features instead of 784. Add **Flatten** before it.');
    const layer = issuesOf((h) => ({ nodes: [h.input('in', [1, 28, 28]), h.layer('lin', 'torch.nn.Linear', { in_features: 784, out_features: 10 }, 'classifier'), h.output('out')] }));
    assert.equal(layer.issues[0].key, 'fb.missing-flatten');
    const mismatch = issuesOf((h) => ({ nodes: [h.input('in', [1, 28, 28]), h.linear('lin1', 100, 10), h.output('out')] }));
    assert.equal(mismatch.issues[0].key, 'fb.linear-in', 'a width that is not C·H·W stays a plain mismatch');
});

test('Conv2d channel and rank errors', () => {
    const channels = issuesOf((h) => ({ nodes: [h.input('in', [1, 28, 28]), h.layer('conv', 'torch.nn.Conv2d', { in_channels: 3, out_channels: 8, kernel_size: 3 }, 'conv1'), h.output('out')] }));
    assert.equal(channels.issues[0].key, 'fb.conv-in');
    assert.equal(text(channels.issues[0]), '**conv1** expects 3 channels but receives 1.');
    const rank = issuesOf((h) => ({ nodes: [h.input('in', [1, 28, 28]), h.layer('flat', 'torch.nn.Flatten'), h.layer('conv', 'torch.nn.Conv2d', { in_channels: 1, out_channels: 8, kernel_size: 3 }, 'conv1'), h.output('out')] }));
    assert.equal(rank.issues[0].key, 'fb.conv-rank');
    assert.match(text(rank.issues[0]), /Did Flatten come too early\?/);
});

test('required parameters → fb.required with the inspector label', () => {
    const { issues } = issuesOf((h) => ({ nodes: [h.input('in', [4]), h.layer('lin', 'torch.nn.Linear', { in_features: 4, out_features: null }, 'neuron'), h.output('out')] }));
    const required = issues.find((issue) => issue.key === 'fb.required');
    assert.ok(required, 'out_features is required');
    assert.equal(text(required), 'Set **Output** on **neuron**.');
});

test('heads must divide the width → fb.heads-divide', () => {
    const { issues } = issuesOf((h) => ({ nodes: [h.input('in', [64]), h.layer('split', 'deepflow.op.Split', { chunks: 3, dim: -1 }, 'heads'), h.output('out')] }));
    assert.equal(issues[0].key, 'fb.heads-divide');
    assert.equal(text(issues[0]), 'The width 64 must be divisible by the number of heads (3).');
});

test('Add inputs with different shapes → fb.add-shapes', () => {
    const { issues } = issuesOf((h) => {
        const input = h.input('in', [32]);
        const narrow = h.linear('lin', 32, 16);
        const add = h.layer('add', 'deepflow.op.Add', {}, 'residual');
        const output = h.output('out');
        return { nodes: [input, narrow, add, output], edges: [...h.chain(input, narrow, add, output), h.edge(input, add)] };
    });
    assert.equal(issues[0].key, 'fb.add-shapes');
    assert.equal(text(issues[0]), 'Both inputs of **residual** must have the same shape.');
});

test('BatchNorm2d width → fb.norm-width; anything else → fb.generic', () => {
    const norm = issuesOf((h) => ({ nodes: [h.input('in', [3, 8, 8]), h.layer('bn', 'torch.nn.BatchNorm2d', { num_features: 16 }, 'bn0'), h.output('out')] }));
    assert.equal(norm.issues[0].key, 'fb.norm-width');
    assert.equal(text(norm.issues[0]), '**bn0** expects 16 features but receives 3.');
    const generic = issuesOf((h) => ({ nodes: [h.input('in', [2, 3]), h.layer('flat', 'torch.nn.Flatten', { start_dim: 2, end_dim: 1 }, 'flatten'), h.output('out')] }));
    assert.equal(generic.issues[0].key, 'fb.generic');
    assert.equal(generic.issues[0].params.message, 'Flatten dimensions must be ordered and within the input rank');
});

test('issues are sorted with the node nearest the Input first', () => {
    const { issues } = issuesOf((h) => ({ nodes: [
        h.input('in', [4]),
        h.layer('first', 'torch.nn.Linear', { in_features: 5, out_features: 3 }, 'first'),
        h.layer('second', 'torch.nn.Linear', { in_features: 7, out_features: 2 }, 'second'),
        h.output('out'),
    ] }));
    assert.deepEqual(issues.filter((issue) => issue.key === 'fb.linear-in').map((issue) => issue.nodeId), ['first']);
    assert.equal(issues[0].nodeId, 'first');
});
