import assert from 'node:assert/strict';
import { test } from 'node:test';
import { groupGraphIssues } from '../src/components/architecture/graphIssues.js';
import { resolveGraphLevel } from '../src/domain/architecture/index.js';

test('graph checks group all severities under one node name and keep graph errors separate', () => {
    const groups = groupGraphIssues({
        globalErrors: ['Cycle detected'],
        errors: { pooling: ['kernel_size is required', 'Invalid stride'], unused: [] },
        warnings: { pooling: ['No incoming connection'], head: ['Unused output'] },
    }, { nodes: { pooling: { label: 'Spatial pool' }, head: { label: 'Domain head' } } });

    assert.deepEqual(groups.map(({ key, nodeId, label }) => ({ key, nodeId, label })), [
        { key: 'graph', nodeId: null, label: 'Graph' },
        { key: 'node:pooling', nodeId: 'pooling', label: 'Spatial pool' },
        { key: 'node:head', nodeId: 'head', label: 'Domain head' },
    ]);
    assert.deepEqual(groups[1].errors, ['kernel_size is required', 'Invalid stride']);
    assert.deepEqual(groups[1].warnings, ['No incoming connection']);
});

test('duplicate node names remain separate and retain their selection IDs', () => {
    const groups = groupGraphIssues({ errors: { first: ['Required'], second: ['Invalid'] } }, {
        nodes: { first: { label: 'Linear' }, second: { label: 'Linear' } },
    });
    assert.deepEqual(groups.map(({ key, nodeId, label }) => [key, nodeId, label]), [
        ['node:first', 'first', 'Linear'],
        ['node:second', 'second', 'Linear'],
    ]);
});

test('nested graph checks resolve labels in the open level, including template and operation fallbacks', () => {
    const templates = {
        inner: { id: 'inner', name: 'Residual block', graph: { nodes: {
            same: { id: 'same', kind: 'layer', label: 'Inner projection' },
            unnamed: { id: 'unnamed', kind: 'layer', libraryKey: 'torch.nn.MaxPool2d', label: '' },
            child: { id: 'child', kind: 'supernode', templateId: 'child-template' },
        }, edges: [] } },
        'child-template': { name: 'Feed forward' },
    };
    const root = { nodes: {
        same: { id: 'same', kind: 'layer', label: 'Root projection' },
        block: { id: 'block', kind: 'supernode', templateId: 'inner' },
    }, edges: [] };
    const { graph } = resolveGraphLevel(root, ['block'], templates);
    const groups = groupGraphIssues({ errors: { same: ['Invalid'], unnamed: ['Required'], child: ['Invalid'] } }, graph, templates);
    assert.deepEqual(groups.map(({ label }) => label), ['Inner projection', 'MaxPool2d', 'Feed forward']);
});
