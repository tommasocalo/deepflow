import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { buildParameterScope, resolveDimensionTuple } from '../src/domain/expressions.js';
import { instanceAsRepeat, repeatAsSingle } from '../src/domain/architecture/repeatStructure.js';
import { validate } from '../src/validation/validator.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';

const shapeParameter = { name: 'table_shape', type: 'tuple', editor: 'dimensions', default: [8, 4] };
const body = {
    id: 'table-body', name: 'Embedding table', freeParameters: [shapeParameter],
    graph: {
        nodes: {
            table: { id: 'table', kind: 'layer', libraryKey: 'deepflow.op.Parameter', label: 'table', parameters: { shape: '$table_shape' } },
            output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.template.Output', parameters: {} },
        },
        edges: [{ id: 'table-output', source: 'table', target: 'output' }],
    },
};
const repeatedTable = (bindings, mode = 'parallel') => ({ id: 'tables', kind: 'repeat', templateId: body.id, count: 2, mode, iterationVariable: 'i', bindings });
const fixture = (bindings, mode = 'parallel') => ({
    nodes: {
        tables: repeatedTable(bindings, mode),
        output: { id: 'output', kind: 'layer', libraryKey: 'deepflow.io.Output', parameters: {} },
    },
    edges: [{ id: 'tables-output', source: 'tables', target: 'output' }],
});

test('tuple editors keep every axis in every repeat while scalar arrays remain per-instance', () => {
    const parameters = [shapeParameter, { name: 'width', type: 'int' }];
    for (const index of [0, 1]) {
        const scope = buildParameterScope(parameters, { table_shape: ['$rows', '4 * (i + 1)'], width: [16, 32] }, { rows: 8, i: index }, index);
        assert.deepEqual(resolveDimensionTuple(scope.table_shape, scope), [8, 4 * (index + 1)]);
        assert.equal(scope.width, [16, 32][index]);
    }
    for (const binding of [[[8, 4], [12, 6]], ['(8, 4)', '(12, 6)']]) {
        assert.deepEqual(resolveDimensionTuple(buildParameterScope([shapeParameter], { table_shape: binding }, {}, 1).table_shape), [12, 6], 'Explicit per-instance tuple bindings remain supported');
    }
});

test('Repeat to Single preserves tuple axes and index formulas through its round-trip', () => {
    const repeat = repeatedTable({ table_shape: ['$rows', '4 * (i + 1)'] });
    const single = repeatAsSingle(repeat, body);
    assert.deepEqual(single.parameterValues.table_shape, ['$rows', 4]);
    assert.deepEqual(instanceAsRepeat(single), repeat);
    const defaultBody = { ...body, freeParameters: [{ ...shapeParameter, default: ['$rows', '4 * (i + 1)'] }] };
    assert.deepEqual(repeatAsSingle(repeatedTable({}), defaultBody).parameterValues.table_shape, ['$rows', 4]);
    assert.deepEqual(repeatAsSingle(repeatedTable({ table_shape: [[8, 4], [12, 6]] }), body).parameterValues.table_shape, [8, 4]);
});

test('shape inference keeps exposed weight shapes after repeating a component', () => {
    for (const mode of ['sequential', 'parallel']) {
        const report = validate(fixture({ table_shape: [8, 4] }, mode), { templates: { [body.id]: body } });
        assert.equal(report.ok, true, JSON.stringify(report.errors));
        assert.deepEqual(report.shapes.tables, mode === 'parallel' ? { kind: 'collection', items: [[8, 4], [8, 4]] } : [8, 4]);
    }
});

const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
test('exported repeats retain the complete weight shape and independent gradients', { skip: !torchAvailable }, () => {
    const source = generateArchitectureModule(fixture({ table_shape: [8, 4] }), { templates: { [body.id]: body }, className: 'Tables' });
    const result = spawnSync('python3', ['-c', `${source}
model = Tables()
tables = model(None)
assert len(tables) == 2
assert all(tuple(table.shape) == (8, 4) for table in tables)
assert tables[0].data_ptr() != tables[1].data_ptr()
(tables[0].sum() + 2 * tables[1].sum()).backward()
torch.testing.assert_close(tables[0].grad, torch.ones_like(tables[0]))
torch.testing.assert_close(tables[1].grad, 2 * torch.ones_like(tables[1]))
`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});
