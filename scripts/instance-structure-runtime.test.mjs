import assert from 'node:assert/strict';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { validate } from '../src/validation/validator.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';
import { analyzeGradientFlow } from '../src/domain/gradientFlow.js';
import { resolveLevelContext } from '../src/hooks/useValidation.js';
import { containsTemplate } from '../src/domain/architecture/templateReferences.js';
import { copyGraphSelection, pasteGraphSelection } from '../src/domain/architecture/clipboard.js';
import { useTemplatesStore } from '../src/stores/templatesStore.js';
import { createValidationCache } from '../src/validation/cache.js';

const layer = (id, libraryKey, parameters = {}) => ({ id, label: id, kind: 'layer', libraryKey, parameters, position: { x: 0, y: 0 } });
const edge = (source, target, targetHandle) => ({ id: `${source}_${target}`, source, target, targetHandle });
const graph = (nodes, edges) => ({ nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges });
const override = (templateId, enabled = true) => ({ templateId, enabled });
const repeat = (id, templateId, count = 3, mode = 'parallel', structureOverrides = {}) => ({ id, label: id, kind: 'repeat', templateId, count, mode, structureOverrides, position: { x: 0, y: 0 } });
const scale = (id, factor) => ({ id, name: id, graph: graph([
    layer('in', 'deepflow.template.Input'), layer('scale', 'deepflow.op.Scale', { factor }), layer('out', 'deepflow.template.Output'),
], [edge('in', 'scale'), edge('scale', 'out')]) });
const fixture = (node) => graph([
    layer('input', 'deepflow.io.Input', { shape: '(16,)', requires_grad: true }), node, layer('output', 'deepflow.io.Output'),
], [edge('input', node.id), edge(node.id, 'output')]);

test('parallel local structure affects only its occurrence in validation, preview, code and gradients', () => {
    const templates = { shared: scale('shared', 2), local: scale('local', 3) };
    templates.local.graph.edges[1].gradientOp = 'stop';
    const root = fixture(repeat('heads', 'shared', 3, 'parallel', { 1: override('local') }));
    const report = validate(root, { templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.heads, { kind: 'collection', items: [[null, 16], [null, 16], [null, 16]] });
    const source = generateArchitectureModule(root, { templates });
    assert.equal((source.match(/\* 2\)/g) ?? []).length, 2);
    assert.equal((source.match(/\* 3\)/g) ?? []).length, 1);
    assert.equal((source.match(/\.detach\(/g) ?? []).length, 1);
    const gradients = analyzeGradientFlow(root, { templates });
    assert.deepEqual(gradients.children.heads.map((child) => child.output.requires), ['yes', 'no', 'yes']);
    const context = resolveLevelContext(root, ['heads'], templates, { '["heads"]': 1 });
    assert.equal(context.graph, templates.local.graph);
    assert.equal(context.repeat.index, 1);
    assert.equal(resolveLevelContext(root, ['heads'], templates, { '["heads"]': 2 }).graph, templates.shared.graph);

    root.nodes.heads.structureOverrides[1].enabled = false;
    assert.equal((generateArchitectureModule(root, { templates }).match(/\* 2\)/g) ?? []).length, 3);
    assert.equal(resolveLevelContext(root, ['heads'], templates, { '["heads"]': 1 }).graph, templates.shared.graph);
    assert.equal(root.nodes.heads.structureOverrides[1].templateId, 'local', 'Disabling retains the saved local graph');
});

test('sequential input inference uses preceding local structures and each effective parameter schema', () => {
    const projection = (id, divisor) => ({ id, name: id,
        freeParameters: [{ name: 'width', type: 'int', derived: { kind: 'input-dimension', inputId: 'in', axis: -1 } }],
        graph: graph([layer('in', 'deepflow.template.Input'), layer('linear', 'torch.nn.Linear', { in_features: '$width', out_features: `width / ${divisor}`, bias: false }), layer('out', 'deepflow.template.Output')], [edge('in', 'linear'), edge('linear', 'out')]),
    });
    const templates = { half: projection('half', 2), quarter: projection('quarter', 4) };
    const root = fixture(repeat('stack', 'half', 3, 'sequential', { 0: override('quarter') }));
    const report = validate(root, { templates });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    assert.deepEqual(report.shapes.output, [null, 1]);
    const source = generateArchitectureModule(root, { templates });
    for (const pair of ['16, out_features=4', '4, out_features=2', '2, out_features=1']) assert.ok(source.includes(`in_features=${pair}`));
    const context = resolveLevelContext(root, ['stack'], templates, { '["stack"]': 1 });
    assert.equal(context.graph, templates.half.graph);
    assert.equal(context.scope.width, 4);
    assert.deepEqual(context.externalInputs.in, [null, 4]);
    assert.deepEqual(analyzeGradientFlow(root, { templates }).children.stack.map((child) => child.scope.width), [16, 4, 2]);
});

test('nested local repeat definitions resolve by outer occurrence and survive project JSON', () => {
    const nested = (id, structureOverrides = {}) => ({ id, name: id, graph: graph([
        layer('in', 'deepflow.template.Input'), repeat('heads', 'shared', 2, 'parallel', structureOverrides), layer('out', 'deepflow.template.Output'),
    ], [edge('in', 'heads'), edge('heads', 'out')]) });
    const templates = { shared: scale('shared', 2), local: scale('local', 3), body: nested('body'), localBody: nested('localBody', { 1: override('local') }) };
    const root = fixture(repeat('outer', 'body', 2, 'parallel', { 1: override('localBody') }));
    const saved = JSON.parse(JSON.stringify({ architecture: root, templates }));
    assert.equal(validate(saved.architecture, { templates: saved.templates }).ok, true);
    assert.equal(generateArchitectureModule(saved.architecture, { templates: saved.templates }), generateArchitectureModule(root, { templates }));
    const source = generateArchitectureModule(root, { templates });
    assert.equal((source.match(/\* 2\)/g) ?? []).length, 3);
    assert.equal((source.match(/\* 3\)/g) ?? []).length, 1);
    const indices = { '["outer"]': 1, '["outer","heads"]': 1 };
    assert.equal(resolveLevelContext(root, ['outer', 'heads'], templates, indices).graph, templates.local.graph);
    assert.equal(resolveLevelContext(root, ['outer', 'heads'], templates, { ...indices, '["outer"]': 0 }).graph, templates.shared.graph);
});

test('single components use their local graph and missing active occurrence definitions block export', () => {
    const templates = { shared: scale('shared', 2), local: scale('local', 3) };
    const node = { id: 'single', kind: 'supernode', templateId: 'shared', structureOverrides: { single: override('local') } };
    const root = fixture(node);
    assert.match(generateArchitectureModule(root, { templates }), /\* 3\)/);
    assert.equal(resolveLevelContext(root, ['single'], templates).graph, templates.local.graph);
    const repeated = fixture(repeat('heads', 'shared', 3, 'parallel', { 2: override('missing') }));
    assert.match(JSON.stringify(validate(repeated, { templates }).errors), /Instance 3 template is missing/);
    assert.throws(() => generateArchitectureModule(repeated, { templates }), /template is missing/);
    repeated.nodes.heads.structureOverrides[2].enabled = false;
    assert.equal(validate(repeated, { templates }).ok, true);
});

test('local boundary ports are checked against their actual interface', () => {
    const templates = { shared: scale('shared', 2), local: scale('local', 3) };
    templates.local.graph.nodes.in.id = 'local-input';
    templates.local.graph.nodes['local-input'] = templates.local.graph.nodes.in;
    delete templates.local.graph.nodes.in;
    templates.local.graph.edges[0].source = 'local-input';
    const root = fixture(repeat('heads', 'shared', 2, 'parallel', { 1: override('local') }));
    root.edges[0].targetHandle = 'in';
    assert.match(JSON.stringify(validate(root, { templates }).errors), /Unknown input port/);
    delete root.edges[0].targetHandle;
    assert.equal(validate(root, { templates }).ok, true, 'An unhandled boundary connection remaps to the local input');
});

test('reference checks and deep cloning include disabled saved local structures', () => {
    const child = repeat('child', 'shared', 2, 'parallel', { 1: override('local', false) });
    const templates = { shared: scale('shared', 2), local: scale('local', 3), outer: { id: 'outer', name: 'Outer', graph: graph([child], []) } };
    assert.equal(containsTemplate('outer', 'local', templates), true);
    const selection = copyGraphSelection(templates.outer.graph, ['child']);
    assert.match(pasteGraphSelection(graph([], []), selection, { templates, templateId: 'local' }).error, /cannot contain itself/);
    const { local: _local, ...missing } = templates;
    assert.match(pasteGraphSelection(graph([], []), selection, { templates: missing }).error, /no longer/);
    useTemplatesStore.setState({ templates });
    const copyId = useTemplatesStore.getState().cloneTemplate('outer');
    const copied = useTemplatesStore.getState().templates[copyId].graph.nodes.child;
    assert.notEqual(copied.templateId, 'shared');
    assert.notEqual(copied.structureOverrides[1].templateId, 'local');
    assert.equal(copied.structureOverrides[1].enabled, false);
    assert.deepEqual(useTemplatesStore.getState().templates[copied.structureOverrides[1].templateId].graph, templates.local.graph);
});

test('validation cache refreshes when an occurrence switches structure without changing edges', () => {
    const shared = scale('shared', 2);
    const local = { id: 'local', name: 'Local', graph: graph([
        layer('in', 'deepflow.template.Input'), layer('linear', 'torch.nn.Linear', { in_features: 16, out_features: 8, bias: false }), layer('out', 'deepflow.template.Output'),
    ], [edge('in', 'linear'), edge('linear', 'out')]) };
    const templates = { shared, local };
    const root = fixture(repeat('heads', 'shared', 2, 'parallel', { 1: override('local', false) }));
    const cached = createValidationCache(validate);
    const first = cached(root, templates);
    const changed = { ...root, nodes: { ...root.nodes, heads: { ...root.nodes.heads, structureOverrides: { 1: override('local') } } } };
    const second = cached(changed, templates);
    assert.notEqual(second, first);
    assert.deepEqual(first.shapes.heads.items, [[null, 16], [null, 16]]);
    assert.deepEqual(second.shapes.heads.items, [[null, 16], [null, 8]]);
});

const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
test('exported per-occurrence local structures execute distinct outputs and gradients', { skip: !torchAvailable }, () => {
    const templates = { shared: scale('shared', 2), local: scale('local', 3) };
    const root = fixture(repeat('heads', 'shared', 3, 'parallel', { 1: override('local') }));
    const source = generateArchitectureModule(root, { templates });
    const result = spawnSync('python3', ['-c', `${source}
x = torch.randn(2, 16, requires_grad=True)
outputs = Model()(x)
for actual, factor in zip(outputs, [2, 3, 2]):
    torch.testing.assert_close(actual, x * factor)
sum(output.sum() for output in outputs).backward()
torch.testing.assert_close(x.grad, torch.full_like(x, 7))
`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});
