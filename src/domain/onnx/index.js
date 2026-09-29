import protobuf from 'protobufjs';
import { ONNX_SCHEMA } from './schema.js';

const ModelProto = protobuf.parse(ONNX_SCHEMA).root.lookupType('onnx.ModelProto');
const decoder = new TextDecoder();
const MAX_MODEL_BYTES = 64 * 1024 * 1024;
const TENSOR_TYPES = [
    'undefined', 'float32', 'uint8', 'int8', 'uint16', 'int16', 'int32', 'int64', 'string', 'bool',
    'float16', 'float64', 'uint32', 'uint64', 'complex64', 'complex128', 'bfloat16',
    'float8e4m3fn', 'float8e4m3fnuz', 'float8e5m2', 'float8e5m2fnuz', 'uint4', 'int4',
    'float4e2m1', 'float8e8m0', 'uint2', 'int2', 'float6e2m3', 'float6e3m2',
];

const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const numberOrString = (value) => {
    const number = Number(value?.toString?.() ?? value);
    return Number.isSafeInteger(number) ? number : String(value);
};
const dimension = (value) => own(value, 'dimParam') ? value.dimParam
    : own(value, 'dimValue') ? numberOrString(value.dimValue) : null;
const tensorType = (number) => TENSOR_TYPES[number] ?? `onnx:${number}`;

function valueInfo(value) {
    const tensor = value.type?.tensorType;
    return {
        name: value.name,
        kind: tensor ? 'tensor' : 'unsupported-type',
        dtype: tensor ? tensorType(tensor.elemType) : null,
        shape: tensor?.shape ? tensor.shape.dim.map(dimension) : null,
    };
}

function tensorMetadata(tensor) {
    const typedValues = (tensor.floatData?.length ?? 0) + (tensor.int32Data?.length ?? 0)
        + (tensor.int64Data?.length ?? 0) + (tensor.doubleData?.length ?? 0)
        + (tensor.uint64Data?.length ?? 0) + (tensor.stringData?.length ?? 0);
    return {
        name: tensor.name,
        dtype: tensorType(tensor.dataType),
        shape: tensor.dims.map(numberOrString),
        storage: tensor.dataLocation === 1 || tensor.externalData?.length ? 'external'
            : tensor.rawData?.length || typedValues ? 'inline' : 'missing',
        rawBytes: tensor.rawData?.length ?? 0,
        externalData: Object.fromEntries((tensor.externalData ?? []).map(({ key, value }) => [key, value])),
    };
}

function attributeValue(attribute) {
    if (attribute.refAttrName) return { type: 'reference', value: attribute.refAttrName };
    // Attribute type is an ONNX enum; a decoded field's own presence also
    // covers old files written without the type discriminator.
    switch (attribute.type) {
        case 1: return { type: 'float', value: attribute.f };
        case 2: return { type: 'int', value: numberOrString(attribute.i) };
        case 3: return { type: 'string', value: decoder.decode(attribute.s) };
        case 4: return { type: 'tensor', value: tensorMetadata(attribute.t) };
        case 5: return { type: 'graph', value: { name: attribute.g?.name, nodeCount: attribute.g?.node?.length ?? 0 } };
        case 6: return { type: 'floats', value: [...attribute.floats] };
        case 7: return { type: 'ints', value: attribute.ints.map(numberOrString) };
        case 8: return { type: 'strings', value: attribute.strings.map((item) => decoder.decode(item)) };
        case 9: return { type: 'tensors', value: attribute.tensors.map(tensorMetadata) };
        case 10: return { type: 'graphs', value: attribute.graphs.map((graph) => ({ name: graph.name, nodeCount: graph.node.length })) };
        default:
            if (own(attribute, 'i')) return { type: 'int', value: numberOrString(attribute.i) };
            if (own(attribute, 'f')) return { type: 'float', value: attribute.f };
            if (own(attribute, 's')) return { type: 'string', value: decoder.decode(attribute.s) };
            return { type: 'unsupported', value: null };
    }
}

function bytesOf(input) {
    if (input instanceof Uint8Array) return input;
    if (input instanceof ArrayBuffer) return new Uint8Array(input);
    throw new TypeError('Expected ONNX ModelProto bytes (Uint8Array or ArrayBuffer).');
}

/**
 * Decode a local .onnx file into an operator-level, read-only map. This keeps
 * tensor names, ordered ports, attributes, shapes and initializer metadata;
 * weight values are intentionally not loaded into the editable graph.
 */
export function readOnnxModel(input) {
    const bytes = bytesOf(input);
    if (!bytes.length || bytes.length > MAX_MODEL_BYTES) {
        throw new Error(`ONNX file must contain 1–${MAX_MODEL_BYTES / (1024 * 1024)} MiB of data.`);
    }
    let model;
    try { model = ModelProto.decode(bytes); }
    catch (error) { throw new Error(`Cannot decode ONNX ModelProto: ${error.message}`); }
    if (!model.graph || !model.graph.input?.length || !model.graph.output?.length) {
        throw new Error('ONNX model needs a graph with inputs and outputs.');
    }
    const graph = model.graph;
    const initializers = graph.initializer.map(tensorMetadata);
    const initializerNames = new Set(initializers.map((item) => item.name));
    const inputs = graph.input.map((item) => ({ ...valueInfo(item), defaultInitializer: initializerNames.has(item.name) }));
    const outputs = graph.output.map(valueInfo);
    const nodes = graph.node.map((node, index) => ({
        id: `onnx:${index}`,
        index,
        name: node.name || `${node.opType}_${index + 1}`,
        domain: node.domain || '',
        opType: node.opType,
        inputs: [...node.input],
        outputs: [...node.output],
        attributes: Object.fromEntries(node.attribute.map((attribute) => [attribute.name, attributeValue(attribute)])),
    }));

    const producers = new Map();
    for (const [index, item] of inputs.entries()) producers.set(item.name, { id: `input:${index}`,
        kind: item.defaultInitializer ? 'input-with-default-initializer' : 'input', index: -1 });
    for (const [index, item] of initializers.entries()) {
        if (!producers.has(item.name)) producers.set(item.name, { id: `initializer:${index}`, kind: 'initializer', index: -1 });
    }
    for (const node of nodes) for (const [outputIndex, name] of node.outputs.entries()) {
        if (!name || producers.has(name)) throw new Error(`ONNX tensor “${name}” has multiple producers or an empty name.`);
        producers.set(name, { id: node.id, kind: 'node', index: node.index, outputIndex });
    }
    const edges = [];
    for (const node of nodes) for (const [inputIndex, name] of node.inputs.entries()) {
        if (!name) continue; // empty input is an optional ONNX port
        const producer = producers.get(name);
        if (!producer || producer.index >= node.index) throw new Error(`ONNX input “${name}” on ${node.name} is missing or out of order.`);
        edges.push({ from: producer.id, to: node.id, value: name, sourceKind: producer.kind,
            outputIndex: producer.outputIndex ?? 0, inputIndex });
    }
    for (const [index, output] of outputs.entries()) {
        const producer = producers.get(output.name);
        if (!producer) throw new Error(`ONNX output “${output.name}” has no producer.`);
        edges.push({ from: producer.id, to: `output:${index}`, value: output.name, sourceKind: producer.kind,
            outputIndex: producer.outputIndex ?? 0, inputIndex: 0 });
    }
    const values = [...inputs, ...graph.valueInfo.map(valueInfo), ...outputs];
    return {
        format: 'onnx-operator-graph',
        name: graph.name || model.producerName || 'ONNX model',
        irVersion: numberOrString(model.irVersion),
        opsets: model.opsetImport.map((item) => ({ domain: item.domain || '', version: numberOrString(item.version) })),
        producer: model.producerName || null,
        byteLength: bytes.length,
        inputs, outputs, values, initializers, nodes, edges,
        limitations: [
            'Operator graph only: tensor payloads are not loaded into DeepFlow.',
            ...initializers.some((item) => item.storage === 'external')
                ? ['External tensor files are not resolved.'] : [],
            ...nodes.some((node) => Object.values(node.attributes).some((item) => item.type === 'graph' || item.type === 'graphs'))
                ? ['Nested control-flow graphs are summarized, not expanded.'] : [],
        ],
    };
}

export { recognizeOnnxComponents, buildEditableOnnxArchitecture } from './editable.js';
