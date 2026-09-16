import assert from 'node:assert/strict';
import { test } from 'node:test';
import { compactConnectionWarnings, compactParameterError, partitionParameterErrors } from '../src/components/architecture/inspectorIssues.js';

test('parameter errors are attached once and unrelated node issues stay visible', () => {
    const messages = ['kernel_size is required', 'stride must be an integer', 'MaxPool2d expects 4D; got rank 2', 'This node is part of a cycle'];
    assert.deepEqual(partitionParameterErrors([{ name: 'kernel_size' }, { name: 'stride' }], messages), {
        fields: { kernel_size: ['kernel_size is required'], stride: ['stride must be an integer'] },
        remaining: messages.slice(2),
    });
    assert.equal(compactParameterError('kernel_size', messages[0]), 'Required');
    assert.equal(compactParameterError('stride', messages[1]), 'Must be an integer');
});

test('field matching uses exact parameter identifiers and keeps axis mismatch errors on the node', () => {
    const message = 'Causal Self-Attention embed_dim=768 but input has 256 features';
    const axisError = 'Concat dim 1 mismatch: 64 vs 128';
    assert.deepEqual(partitionParameterErrors([{ name: 'p' }, { name: 'dim' }, { name: 'embed_dim' }], [message, axisError]), {
        fields: { embed_dim: [message] },
        remaining: [axisError],
    });
    const jointError = 'embed_dim=768 is not divisible by num_heads=11';
    assert.deepEqual(partitionParameterErrors([{ name: 'num_heads' }, { name: 'embed_dim' }], [jointError]), {
        fields: { embed_dim: [jointError] },
        remaining: [],
    });
});

test('both disconnected ports share one warning without hiding other warnings', () => {
    assert.deepEqual(compactConnectionWarnings(['No incoming connections', 'Template has 2 internal errors', 'No outgoing connections']), [
        'No incoming or outgoing connections', 'Template has 2 internal errors',
    ]);
    assert.deepEqual(compactConnectionWarnings(['No outgoing connections']), ['No outgoing connections']);
});
