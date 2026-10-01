import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import protobuf from 'protobufjs';
import { readOnnxModel, recognizeOnnxComponents, buildEditableOnnxArchitecture } from '../src/domain/onnx/index.js';
import { ONNX_SCHEMA } from '../src/domain/onnx/schema.js';
import { validate } from '../src/validation/validator.js';
import { generateArchitectureModule } from '../src/codegen/architecture.js';

const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url));
// These tiny binary fixtures were generated with Python's onnx.helper and
// passed onnx.checker.check_model before being committed.

test('decodes a real ONNX graph with named tensor edges and ordered ports', () => {
    const source = readOnnxModel(fixture('tiny-linear-relu.onnx'));
    assert.equal(source.name, 'Tiny linear + ReLU');
    assert.deepEqual(source.opsets, [{ domain: '', version: 17 }]);
    assert.deepEqual(source.inputs[0].shape, ['batch', 3]);
    assert.deepEqual(source.nodes.map((node) => node.opType), ['Gemm', 'Relu']);
    assert.equal(source.nodes[0].attributes.transB.value, 1);
    assert.deepEqual(source.initializers.map((item) => item.shape), [[2, 3], [2]]);
    assert.deepEqual(source.edges.filter((edge) => edge.to === 'onnx:0').map((edge) =>
        [edge.value, edge.inputIndex, edge.sourceKind]), [
        ['x', 0, 'input'], ['W', 1, 'initializer'], ['b', 2, 'initializer'],
    ]);
});

test('maps all Gemm and ReLU operators to editable nodes without loading weights', () => {
    const source = readOnnxModel(fixture('tiny-linear-relu.onnx'));
    const recognized = recognizeOnnxComponents(source);
    assert.equal(recognized.supported, true);
    assert.deepEqual(recognized.components.map((item) => [item.kind, item.source.nodeIds]), [
        ['Linear', ['onnx:0']], ['Relu', ['onnx:1']],
    ]);
    const { graph, source: importSource } = buildEditableOnnxArchitecture(source, { fileName: 'tiny-linear-relu.onnx' });
    assert.equal(importSource.provider, 'onnx');
    assert.equal(importSource.displayName, source.name);
    assert.equal(importSource.weightsLoaded, false);
    assert.equal(importSource.mappedOpCount, 2);
    assert.deepEqual(importSource.operatorGraph.nodes.map((node) => node.opType), ['Gemm', 'Relu']);
    const linear = Object.values(graph.nodes).find((node) => node.libraryKey === 'torch.nn.Linear');
    assert.deepEqual(linear.parameters, { in_features: 3, out_features: 2, bias: true });
    assert.deepEqual(linear.onnxSource.initializers, ['W', 'b']);
    assert.equal(graph.edges.length, 3);
    assert.equal(validate(graph, { templates: {} }).ok, true);
    const python = generateArchitectureModule(graph, { templates: {} });
    assert.match(python, /torch\.nn\.Linear\(in_features=3, out_features=2, bias=True\)/);
});

test('recognizes MatMul plus Add as one Linear while preserving both operator IDs', () => {
    const source = readOnnxModel(fixture('tiny-matmul-add.onnx'));
    const recognition = recognizeOnnxComponents(source);
    assert.equal(recognition.supported, true);
    assert.equal(recognition.components.length, 1);
    assert.deepEqual(recognition.components[0].source.nodeIds, ['onnx:0', 'onnx:1']);
    assert.deepEqual(recognition.components[0].parameters, { in_features: 3, out_features: 2, bias: true });
    assert.equal(validate(buildEditableOnnxArchitecture(source).graph, { templates: {} }).ok, true);
});

test('legacy ONNX initializer inputs remain identifiable defaults, not extra data inputs', () => {
    const source = readOnnxModel(fixture('tiny-linear-legacy-input.onnx'));
    assert.deepEqual(source.inputs.map((item) => [item.name, item.defaultInitializer]),
        [['x', false], ['W', true], ['b', true]]);
    assert.equal(source.edges.find((edge) => edge.value === 'W').sourceKind, 'input-with-default-initializer');
    assert.equal(recognizeOnnxComponents(source).supported, true);
});

test('maps Conv2d while retaining its initializer sources', () => {
    const source = readOnnxModel(fixture('tiny-conv.onnx'));
    const recognition = recognizeOnnxComponents(source);
    assert.equal(recognition.supported, true);
    assert.equal(recognition.components[0].libraryKey, 'torch.nn.Conv2d');
    assert.deepEqual(recognition.components[0].source.nodeIds, ['onnx:0']);
    assert.equal(validate(buildEditableOnnxArchitecture(source).graph, { templates: {} }).ok, true);
});

test('unsupported operators stay visible and cannot silently become an editable graph', () => {
    const source = readOnnxModel(fixture('tiny-unsupported.onnx'));
    const recognition = recognizeOnnxComponents(source);
    assert.equal(source.nodes[0].opType, 'Abs');
    assert.equal(recognition.supported, false);
    assert.deepEqual(recognition.issues.map((item) => item.nodeId), ['onnx:0']);
    assert.match(recognition.issues[0].message, /Abs/);
    assert.throws(() => buildEditableOnnxArchitecture(source), /Abs/);
});

test('a nonstandard Gemm coefficient is rejected rather than converted to Linear', () => {
    const codec = protobuf.parse(ONNX_SCHEMA).root.lookupType('onnx.ModelProto');
    const model = codec.decode(fixture('tiny-linear-relu.onnx'));
    model.graph.node[0].attribute.push({ name: 'alpha', type: 1, f: 2 });
    const altered = readOnnxModel(codec.encode(model).finish());
    assert.equal(recognizeOnnxComponents(altered).supported, false);
    assert.match(recognizeOnnxComponents(altered).issues[0].message, /standard Linear/);
});

test('external initializer metadata can define topology without loading weights', () => {
    const codec = protobuf.parse(ONNX_SCHEMA).root.lookupType('onnx.ModelProto');
    const model = codec.decode(fixture('tiny-linear-relu.onnx'));
    const weight = model.graph.initializer.find((item) => item.name === 'W');
    weight.floatData = [];
    weight.dataLocation = 1;
    weight.externalData = [{ key: 'location', value: 'weights.bin' }];
    const source = readOnnxModel(codec.encode(model).finish());
    assert.equal(source.initializers[0].storage, 'external');
    assert.match(source.limitations.join(' '), /External tensor files/);
    assert.equal(recognizeOnnxComponents(source).supported, true);
    assert.equal(buildEditableOnnxArchitecture(source).source.weightsLoaded, false);
});

test('float16 input is previewable but not claimed runnable through float32 Linear layers', () => {
    const codec = protobuf.parse(ONNX_SCHEMA).root.lookupType('onnx.ModelProto');
    const model = codec.decode(fixture('tiny-linear-relu.onnx'));
    model.graph.input[0].type.tensorType.elemType = 10;
    const source = readOnnxModel(codec.encode(model).finish());
    assert.equal(source.inputs[0].dtype, 'float16');
    assert.equal(recognizeOnnxComponents(source).supported, false);
    assert.match(recognizeOnnxComponents(source).issues[0].message, /floating-point tensor input/);
});

test('missing optional graph name uses producer name and malformed bytes fail clearly', () => {
    const codec = protobuf.parse(ONNX_SCHEMA).root.lookupType('onnx.ModelProto');
    const model = codec.decode(fixture('tiny-linear-relu.onnx'));
    model.graph.name = '';
    assert.equal(readOnnxModel(codec.encode(model).finish()).name, 'DeepFlow ONNX regression');
    assert.throws(() => readOnnxModel(new Uint8Array()), /1–64 MiB/);
    assert.throws(() => readOnnxModel(new Uint8Array([0x08])), /Cannot decode/);
});

const torchAvailable = spawnSync('python3', ['-c', 'import torch'], { encoding: 'utf8' }).status === 0;
test('the generated small imported model runs forward and backward in PyTorch', { skip: !torchAvailable }, () => {
    const graph = buildEditableOnnxArchitecture(readOnnxModel(fixture('tiny-linear-relu.onnx'))).graph;
    const python = generateArchitectureModule(graph, { templates: {} });
    const result = spawnSync('python3', ['-c', `${python}\nmodel = Model()\nx = torch.randn(4, 3)\ny = model(x)\nassert y.shape == (4, 2), y.shape\ny.sum().backward()\nassert model.dense.weight.grad is not None\n`],
        { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});

function alteredFixture(name, alter) {
    const codec = protobuf.parse(ONNX_SCHEMA).root.lookupType('onnx.ModelProto');
    const model = codec.decode(fixture(name));
    alter(model);
    return readOnnxModel(codec.encode(model).finish());
}

test('maps an exported residual CNN as a DAG with its actual skip connection', () => {
    const source = readOnnxModel(fixture('tiny-residual-cnn.onnx'));
    const recognition = recognizeOnnxComponents(source);
    assert.equal(recognition.supported, true, JSON.stringify(recognition.issues));
    assert.deepEqual(recognition.components.map((item) => item.libraryKey), [
        'torch.nn.Conv2d', 'torch.nn.ReLU', 'torch.nn.Conv2d', 'deepflow.op.Add',
        'torch.nn.AdaptiveAvgPool2d', 'torch.nn.Flatten', 'torch.nn.Linear',
    ]);
    const { graph } = buildEditableOnnxArchitecture(source);
    const input = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
    const merge = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.op.Add');
    assert.equal(graph.edges.filter((edge) => edge.source === input.id).length, 2);
    assert.equal(graph.edges.filter((edge) => edge.source === input.id && edge.target === merge.id).length, 1);
    assert.equal(graph.edges.filter((edge) => edge.target === merge.id).length, 2);
    const report = validate(graph, { templates: {} });
    assert.equal(report.ok, true, JSON.stringify(report.errors));
    const output = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Output');
    assert.deepEqual(report.shapes[output.id], [null, 2]);
    assert.equal(graph.importSource.operatorGraph.values.length, source.values.length);
});

test('multiple inputs and outputs retain ordered noncommutative ports and tensor names', () => {
    const source = readOnnxModel(fixture('tiny-ordered-dag.onnx'));
    const { graph } = buildEditableOnnxArchitecture(source);
    assert.equal(validate(graph, { templates: {} }).ok, true);
    const product = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.op.MatMul');
    const difference = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.op.Subtract');
    assert.deepEqual(graph.edges.filter((edge) => edge.target === product.id).map((edge) => [edge.targetHandle, edge.onnxValue]), [['a', 'a'], ['b', 'b']]);
    assert.deepEqual(graph.edges.filter((edge) => edge.target === difference.id).map((edge) => [edge.targetHandle, edge.onnxValue]), [['a', 'c'], ['b', 'product']]);
    assert.deepEqual(Object.values(graph.nodes).filter((node) => node.libraryKey === 'deepflow.io.Output').map((node) => node.label), ['joined', 'difference']);
});

test('shared initializer stays one frozen Parameter rather than duplicated Linear weights', () => {
    const source = readOnnxModel(fixture('tiny-shared-weight.onnx'));
    const { graph } = buildEditableOnnxArchitecture(source);
    const parameters = Object.values(graph.nodes).filter((node) => node.libraryKey === 'deepflow.op.Parameter');
    assert.equal(parameters.length, 1);
    assert.equal(parameters[0].parameters.requires_grad, false);
    assert.equal(parameters[0].parameters.shape, '(3, 2)');
    assert.equal(parameters[0].onnxSource.trainability, 'unknown-frozen-default');
    assert.equal(graph.edges.filter((edge) => edge.source === parameters[0].id).length, 2);
    assert.equal(Object.values(graph.nodes).filter((node) => node.libraryKey === 'torch.nn.Linear').length, 0);
    assert.equal(validate(graph, { templates: {} }).ok, true);
});

test('a fixed numeric batch does not introduce a second batch dimension', () => {
    const source = alteredFixture('tiny-conv.onnx', (model) => {
        model.graph.input[0].type.tensorType.shape.dim[0] = { dimValue: 2 };
    });
    const { graph } = buildEditableOnnxArchitecture(source);
    const input = Object.values(graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
    const report = validate(graph, { templates: {} });
    assert.equal(input.onnxSource.shape[0], 2);
    assert.equal(report.shapes[input.id].length, 4);
    assert.equal(report.ok, true, JSON.stringify(report.errors));
});

test('asymmetric Conv padding and non-float32 weights are blocked', () => {
    const asymmetric = alteredFixture('tiny-conv.onnx', (model) => {
        model.graph.node[0].attribute = model.graph.node[0].attribute.filter((item) => item.name !== 'pads');
        model.graph.node[0].attribute.push({ name: 'pads', type: 7, ints: [0, 1, 1, 1] });
    });
    assert.equal(recognizeOnnxComponents(asymmetric).supported, false);
    assert.match(recognizeOnnxComponents(asymmetric).issues[0].message, /symmetric/);
    const half = alteredFixture('tiny-conv.onnx', (model) => { model.graph.initializer[0].dataType = 10; });
    assert.equal(recognizeOnnxComponents(half).supported, false);
});

test('fusion leaves branched and graph-output MatMul tensors addressable', () => {
    const source = alteredFixture('tiny-matmul-add.onnx', (model) => {
        model.graph.output.push({ name: model.graph.node[0].output[0], type: model.graph.output[0].type });
    });
    const recognition = recognizeOnnxComponents(source);
    assert.equal(recognition.supported, true, JSON.stringify(recognition.issues));
    assert.deepEqual(recognition.components.filter((item) => item.source.nodeIds.length).map((item) => item.source.nodeIds), [['onnx:0'], ['onnx:1']]);
    const { graph } = buildEditableOnnxArchitecture(source);
    assert.equal(validate(graph, { templates: {} }).ok, true);
    const dense = Object.values(graph.nodes).find((node) => node.libraryKey === 'torch.nn.Linear');
    assert.equal(graph.edges.filter((edge) => edge.source === dense.id).length, 2);
});

test('legacy operator semantics and a shared module weight are not silently accepted', () => {
    const old = alteredFixture('tiny-linear-relu.onnx', (model) => { model.opsetImport[0].version = 6; });
    assert.equal(recognizeOnnxComponents(old).supported, false);
    assert.equal(recognizeOnnxComponents(old).issues[0].code, 'unsupported-opset');
    const sharedConv = alteredFixture('tiny-conv.onnx', (model) => {
        const clone = { ...model.graph.node[0], name: 'second_conv', output: ['second_output'] };
        model.graph.node.push(clone);
        model.graph.output.push({ name: 'second_output', type: model.graph.output[0].type });
    });
    assert.equal(recognizeOnnxComponents(sharedConv).supported, false);
});

test('an imported residual CNN executes forward/backward with fresh editable weights', { skip: !torchAvailable }, () => {
    const graph = buildEditableOnnxArchitecture(readOnnxModel(fixture('tiny-residual-cnn.onnx'))).graph;
    const python = generateArchitectureModule(graph, { templates: {} });
    const result = spawnSync('python3', ['-c', `${python}\nmodel = Model()\nx = torch.randn(2, 3, 8, 8, requires_grad=True)\ny = model(x)\nassert y.shape == (2, 2), y.shape\ny.sum().backward()\nassert x.grad is not None\nassert all(p.grad is not None for p in model.parameters())\n`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});

test('generated DAG matches MatMul/Sub/Concat values and declared output ordering', { skip: !torchAvailable }, () => {
    const graph = buildEditableOnnxArchitecture(readOnnxModel(fixture('tiny-ordered-dag.onnx'))).graph;
    const python = generateArchitectureModule(graph, { templates: {} });
    const result = spawnSync('python3', ['-c', `${python}\nmodel = Model()\na = torch.randn(2, 2, 3, requires_grad=True)\nb = torch.randn(2, 3, 4, requires_grad=True)\nc = torch.randn(2, 2, 4, requires_grad=True)\njoined, difference = model(a, b, c)\nproduct = a @ b\nexpected = c - product\ntorch.testing.assert_close(difference, expected)\ntorch.testing.assert_close(joined, torch.cat([expected, product], dim=1))\njoined.sum().backward()\nassert c.grad is not None\n`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});

test('a declared single output takes precedence over an unused later branch', { skip: !torchAvailable }, () => {
    const source = alteredFixture('tiny-ordered-dag.onnx', (model) => { model.graph.output = [model.graph.output[1]]; });
    const python = generateArchitectureModule(buildEditableOnnxArchitecture(source).graph, { templates: {} });
    const result = spawnSync('python3', ['-c', `${python}\na = torch.randn(2, 2, 3)\nb = torch.randn(2, 3, 4)\nc = torch.randn(2, 2, 4)\ntorch.testing.assert_close(Model()(a, b, c), c - a @ b)\n`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});

test('generated shared Parameter remains one constant operand used by both branches', { skip: !torchAvailable }, () => {
    const python = generateArchitectureModule(buildEditableOnnxArchitecture(readOnnxModel(fixture('tiny-shared-weight.onnx'))).graph, { templates: {} });
    const result = spawnSync('python3', ['-c', `${python}\nmodel = Model()\nparameters = list(model.parameters())\nassert len(parameters) == 1\nweight = parameters[0]\nassert not weight.requires_grad\nx = torch.randn(4, 3, requires_grad=True)\ny = model(x)\ntorch.testing.assert_close(y, 2 * (x @ weight))\ny.sum().backward()\nassert x.grad is not None\nassert weight.grad is None\n`], { encoding: 'utf8', timeout: 30000 });
    assert.equal(result.status, 0, result.stderr);
});
