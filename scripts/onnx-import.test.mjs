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

test('unmapped Conv stays in operator map and cannot silently become an editable graph', () => {
    const source = readOnnxModel(fixture('tiny-conv.onnx'));
    const recognition = recognizeOnnxComponents(source);
    assert.equal(source.nodes[0].opType, 'Conv');
    assert.equal(recognition.supported, false);
    assert.deepEqual(recognition.issues.map((item) => item.nodeId), ['onnx:0']);
    assert.match(recognition.issues[0].message, /Conv/);
    assert.throws(() => buildEditableOnnxArchitecture(source), /Conv/);
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
