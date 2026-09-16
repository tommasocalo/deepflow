import test from 'node:test';
import assert from 'node:assert/strict';
import { describePorts } from '../src/components/architecture/portShapes.js';

test('port labels distinguish upstream input from transformed output', () => {
    const node = { id: 'fc', libraryKey: 'torch.nn.Linear' };
    const graph = { edges: [{ source: 'input', target: 'fc' }] };
    assert.deepEqual(describePorts(graph, node, { input: [null, 1024, 768], fc: [null, 1024, 3072] }), {
        input: 'B × 1024 × 768', output: 'B × 1024 × 3072',
    });
});

test('multiple inputs keep distinct shapes and summarize equal shapes with a count', () => {
    const node = { id: 'add', libraryKey: 'deepflow.op.Add' };
    const graph = { edges: ['a', 'b', 'c'].map(source => ({ source, target: 'add' })) };
    assert.deepEqual(describePorts(graph, node, { a: [null, 128], b: [null, 128], c: [null, 64] }), {
        input: '2 × [B × 128] · B × 64', output: '—',
    });
});

test('boundary ports and uninferred Split outputs do not invent shapes', () => {
    const graph = { edges: [{ source: 'in', target: 'out' }] };
    const shapes = { in: [null, 768], out: [null, 768] };
    assert.deepEqual(describePorts(graph, { id: 'in', libraryKey: 'deepflow.template.Input' }, shapes), { input: null, output: 'B × 768' });
    assert.deepEqual(describePorts(graph, { id: 'out', libraryKey: 'deepflow.io.Output' }, shapes), { input: 'B × 768', output: null });
    assert.equal(describePorts(graph, { id: 'out', libraryKey: 'deepflow.op.Split' }, shapes).output, null);
});
