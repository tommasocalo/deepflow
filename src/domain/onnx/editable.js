import { createEdge, createLayerNode } from '../architecture/index.js';
import { getEntry } from '../library/index.js';
import { getOperationInputIds } from '../graphInputs.js';
import { inferShape } from '../../validation/shapes.js';

const ACTIVATIONS = {
    Relu: 'torch.nn.ReLU', Sigmoid: 'torch.nn.Sigmoid', Tanh: 'torch.nn.Tanh',
    Exp: 'deepflow.op.Exp', Sin: 'deepflow.op.Sin', Cos: 'deepflow.op.Cos',
};
const BINARY = { Add: 'deepflow.op.Add', Mul: 'deepflow.op.Multiply', Sub: 'deepflow.op.Subtract', MatMul: 'deepflow.op.MatMul' };
const positiveDimension = (value) => Number.isSafeInteger(value) && value > 0;
const positivePair = (value) => Array.isArray(value) && value.length === 2 && value.every(positiveDimension);
const onlyAttrs = (node, allowed = []) => Object.keys(node.attributes).every((name) => allowed.includes(name));
const attr = (node, name, fallback) => node.attributes[name]?.value ?? fallback;
const floatTensor = (value) => value?.dtype === 'float32';
const validInitializer = (value) => floatTensor(value) && value.shape.length > 0 && value.shape.every(positiveDimension);
const tuple = (shape) => `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;

function sourceFor(nodes, values, initializers = []) {
    return { nodeIds: nodes.map((node) => node.id), nodeNames: nodes.map((node) => node.name),
        values, initializers, fidelity: 'editable-topology-no-weights' };
}

function compactSourceGraph(operatorGraph) {
    const { name, irVersion, opsets, producer, byteLength, inputs, outputs, values, initializers, nodes, edges, limitations } = operatorGraph;
    return { name, irVersion, opsets, producer, byteLength, inputs, outputs, values, initializers, nodes, edges, limitations };
}

/**
 * Map a complete, supported float32 ONNX DAG into editable components. Named
 * tensors carry branch/merge routing, and ordered ports survive contraction of
 * recognized Linear patterns. Unknown operators/semantics block the import.
 * Initializer payloads and ONNX training state are never imported.
 */
export function recognizeOnnxComponents(operatorGraph) {
    if (operatorGraph?.format !== 'onnx-operator-graph') throw new TypeError('Expected an ONNX operator graph.');
    const issues = [], components = [];
    const visited = new Set(), parameterValues = new Set();
    const dynamicInputs = operatorGraph.inputs.filter((item) => !item.defaultInitializer);
    const initializers = new Map(operatorGraph.initializers.map((item) => [item.name, item]));
    const shapes = new Map((operatorGraph.values ?? []).map((item) => [item.name, item.shape?.map((dim) => positiveDimension(dim) ? dim : null) ?? null]));
    for (const item of operatorGraph.inputs) shapes.set(item.name, item.shape?.map((dim) => positiveDimension(dim) ? dim : null) ?? null);
    for (const item of operatorGraph.initializers) shapes.set(item.name, item.shape);
    const consumers = new Map();
    const graphOutputs = new Set(operatorGraph.outputs.map((item) => item.name));
    for (const node of operatorGraph.nodes) for (const value of node.inputs) {
        if (value) consumers.set(value, [...(consumers.get(value) ?? []), node]);
    }
    const issue = (message, node = null, code = 'unmapped-pattern') => {
        issues.push({ code, nodeId: node?.id ?? null, nodeName: node?.name ?? null, message });
    };
    const exclusiveInitializer = (value) => validInitializer(value)
        && (consumers.get(value.name)?.length ?? 0) === 1 && !graphOutputs.has(value.name);
    const opset = operatorGraph.opsets.find((item) => !item.domain || item.domain === 'ai.onnx')?.version;
    if (!Number.isInteger(opset) || opset < 13 || opset > 23) {
        issue('Editable ONNX mapping currently supports standard opsets 13–23.', null, 'unsupported-opset');
    }
    if (!dynamicInputs.length) issue('The editable graph needs a tensor input.', null, 'unsupported-input');
    for (const input of dynamicInputs) {
        if (input.kind !== 'tensor' || !floatTensor(input) || !input.shape || input.shape.length < 2) {
            issue(`The editable mapping needs a float32 floating-point tensor input with known rank of at least two: ${input.name}.`, null, 'unsupported-input');
        }
    }
    for (const output of operatorGraph.outputs) {
        if (output.kind !== 'tensor' || !floatTensor(output)) issue(`Output ${output.name} must be a float32 tensor.`, null, 'unsupported-output');
    }
    for (const value of operatorGraph.values ?? []) {
        if (value.kind !== 'tensor' || !floatTensor(value)) issue(`Tensor ${value.name} uses an unsupported type: ${value.dtype}.`, null, 'unsupported-dtype');
    }
    if (issues.length) return { supported: false, components, issues };

    const ensureParameter = (name, node) => {
        if (!initializers.has(name) || parameterValues.has(name)) return true;
        const value = initializers.get(name);
        if (!validInitializer(value)) {
            issue(`Initializer ${name} needs a non-scalar float32 tensor with positive dimensions.`, node, 'unsupported-initializer');
            return false;
        }
        // ONNX inference graphs do not encode requires_grad. Exposed constant
        // operands stay frozen unless the user explicitly opts into training.
        components.push({ kind: 'Parameter', status: 'mapped', libraryKey: 'deepflow.op.Parameter',
            label: name, parameters: { shape: tuple(value.shape), requires_grad: false },
            inputs: [], outputs: [name], source: { ...sourceFor([], [name], [name]), trainability: 'unknown-frozen-default' } });
        parameterValues.add(name);
        return true;
    };
    const add = (node, libraryKey, parameters = {}, inputs = node.inputs, sourceNodes = [node], output = node.outputs[0], weights = []) => {
        if (!inputs.every((name) => ensureParameter(name, node))) return;
        const result = inferShape(libraryKey, inputs.map((name) => shapes.get(name) ?? null), parameters);
        if (result.errors.length) {
            issue(`${node.name}: ${result.errors.join('; ')}`, node, 'incompatible-shape');
            return;
        }
        shapes.set(output, result.output ?? shapes.get(output) ?? null);
        components.push({ kind: ACTIVATIONS[node.opType] ? node.opType : libraryKey.split('.').at(-1), status: 'mapped', libraryKey, parameters,
            inputs: [...inputs], outputs: [output], source: sourceFor(sourceNodes, [...inputs, output], weights), label: node.name });
        for (const sourceNode of sourceNodes) visited.add(sourceNode.id);
    };
    const unsupported = (node, detail = '') => issue(
        `ONNX operator ${node.opType} on ${node.name} has no editable mapping for ${detail || 'these ports or attributes'}.`, node, 'unmapped-operator');
    const spatial = (node, kernel) => {
        const strides = attr(node, 'strides', [1, 1]), dilations = attr(node, 'dilations', [1, 1]);
        const pads = attr(node, 'pads', [0, 0, 0, 0]);
        if (!positivePair(kernel) || !positivePair(strides) || !positivePair(dilations)
            || !Array.isArray(pads) || pads.length !== 4 || !pads.every((n) => Number.isInteger(n) && n >= 0)
            || pads[0] !== pads[2] || pads[1] !== pads[3] || !['NOTSET', 'VALID'].includes(attr(node, 'auto_pad', 'NOTSET'))
            || (attr(node, 'auto_pad', 'NOTSET') === 'VALID' && pads.some(Boolean))) return null;
        return { kernel_size: tuple(kernel), stride: tuple(strides), padding: tuple(pads.slice(0, 2)), dilation: tuple(dilations) };
    };

    for (const node of operatorGraph.nodes) {
        if (visited.has(node.id)) continue;
        if (node.domain && node.domain !== 'ai.onnx') { issue(`Custom domain “${node.domain}” on ${node.name} is not mapped.`, node, 'custom-domain'); continue; }
        if (node.outputs.length !== 1 || !node.outputs[0] || !node.inputs.length || !node.inputs[0]) {
            unsupported(node, 'multiple outputs or optional data ports'); continue;
        }
        if (node.opType === 'Gemm') {
            const weight = initializers.get(node.inputs[1]), bias = node.inputs[2] ? initializers.get(node.inputs[2]) : null;
            const transB = attr(node, 'transB', 0);
            const outWidth = weight?.shape[transB ? 0 : 1], inWidth = weight?.shape[transB ? 1 : 0];
            if (!onlyAttrs(node, ['alpha', 'beta', 'transA', 'transB']) || attr(node, 'alpha', 1) !== 1 || attr(node, 'beta', 1) !== 1
                || attr(node, 'transA', 0) !== 0 || ![0, 1].includes(transB) || node.inputs.length < 2 || node.inputs.length > 3
                || !exclusiveInitializer(weight) || weight.shape.length !== 2 || shapes.get(node.inputs[0])?.length !== 2
                || (node.inputs[2] && (!exclusiveInitializer(bias) || bias.shape.length !== 1 || bias.shape[0] !== outWidth))) {
                issue(`${node.name} is not a standard Linear with an exclusive float32 W initializer and optional b[out].`, node); continue;
            }
            add(node, 'torch.nn.Linear', { in_features: inWidth, out_features: outWidth, bias: Boolean(bias) }, [node.inputs[0]], [node], node.outputs[0], [weight.name, ...(bias ? [bias.name] : [])]);
            continue;
        }
        if (node.opType === 'Conv') {
            const weight = initializers.get(node.inputs[1]), bias = node.inputs[2] ? initializers.get(node.inputs[2]) : null;
            const group = attr(node, 'group', 1), kernel = attr(node, 'kernel_shape', weight?.shape.slice(2));
            const params = spatial(node, kernel);
            if (!onlyAttrs(node, ['auto_pad', 'dilations', 'group', 'kernel_shape', 'pads', 'strides']) || !params
                || !exclusiveInitializer(weight) || weight.shape.length !== 4 || shapes.get(node.inputs[0])?.length !== 4
                || !positiveDimension(group) || weight.shape[0] % group !== 0 || kernel.some((dim, i) => dim !== weight.shape[i + 2])
                || node.inputs.length < 2 || node.inputs.length > 3
                || (node.inputs[2] && (!exclusiveInitializer(bias) || bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0]))) {
                unsupported(node, 'this convolution (requires Conv2d, exclusive float32 weights, and symmetric explicit padding)'); continue;
            }
            add(node, 'torch.nn.Conv2d', { ...params, in_channels: weight.shape[1] * group, out_channels: weight.shape[0], groups: group, bias: Boolean(bias) }, [node.inputs[0]], [node], node.outputs[0], [weight.name, ...(bias ? [bias.name] : [])]);
            continue;
        }
        if (node.opType === 'MatMul' && onlyAttrs(node) && node.inputs.length === 2) {
            const weight = initializers.get(node.inputs[1]);
            if (exclusiveInitializer(weight) && weight.shape.length === 2 && (shapes.get(node.inputs[0])?.length ?? 0) >= 2) {
                const candidates = consumers.get(node.outputs[0]) ?? [];
                const next = candidates.length === 1 && !graphOutputs.has(node.outputs[0]) ? candidates[0] : null;
                const biasName = next?.inputs.find((name) => name !== node.outputs[0]), bias = initializers.get(biasName);
                const fuse = next?.opType === 'Add' && !next.domain && onlyAttrs(next) && next.inputs.length === 2 && next.outputs.length === 1
                    && exclusiveInitializer(bias) && bias.shape.length === 1 && bias.shape[0] === weight.shape[1];
                add(node, 'torch.nn.Linear', { in_features: weight.shape[0], out_features: weight.shape[1], bias: Boolean(fuse) },
                    [node.inputs[0]], fuse ? [node, next] : [node], fuse ? next.outputs[0] : node.outputs[0], [weight.name, ...(fuse ? [bias.name] : [])]);
                continue;
            }
        }
        if (BINARY[node.opType] && node.inputs.length === 2 && node.inputs.every(Boolean) && onlyAttrs(node)) {
            add(node, BINARY[node.opType]); continue;
        }
        if (ACTIVATIONS[node.opType] && node.inputs.length === 1 && onlyAttrs(node)) {
            add(node, ACTIVATIONS[node.opType]); continue;
        }
        if (node.opType === 'LeakyRelu' && node.inputs.length === 1 && onlyAttrs(node, ['alpha']) && Number.isFinite(attr(node, 'alpha', 0.01))) {
            add(node, 'torch.nn.LeakyReLU', { negative_slope: attr(node, 'alpha', 0.01), inplace: false }); continue;
        }
        if (node.opType === 'Gelu' && opset >= 20 && node.inputs.length === 1 && onlyAttrs(node, ['approximate']) && ['none', 'tanh'].includes(attr(node, 'approximate', 'none'))) {
            add(node, 'torch.nn.GELU', { approximate: attr(node, 'approximate', 'none') }); continue;
        }
        if (node.opType === 'Identity' && node.inputs.length === 1 && onlyAttrs(node)) {
            add(node, 'deepflow.op.Scale', { factor: 1 }); continue;
        }
        if (node.opType === 'Softmax' && node.inputs.length === 1 && onlyAttrs(node, ['axis'])) {
            const axis = attr(node, 'axis', -1), rank = shapes.get(node.inputs[0])?.length;
            if (!Number.isInteger(axis) || !rank || axis < -rank || axis >= rank) { unsupported(node, 'this softmax axis'); continue; }
            add(node, 'torch.nn.Softmax', { dim: axis }); continue;
        }
        if (node.opType === 'Flatten' && node.inputs.length === 1 && onlyAttrs(node, ['axis']) && attr(node, 'axis', 1) === 1 && (shapes.get(node.inputs[0])?.length ?? 0) >= 2) {
            add(node, 'torch.nn.Flatten', { start_dim: 1, end_dim: -1 }); continue;
        }
        if (['MaxPool', 'AveragePool'].includes(node.opType) && node.inputs.length === 1 && shapes.get(node.inputs[0])?.length === 4) {
            const params = spatial(node, attr(node, 'kernel_shape', null)), isMax = node.opType === 'MaxPool';
            if (!params || !onlyAttrs(node, ['auto_pad', 'ceil_mode', 'dilations', 'kernel_shape', 'pads', 'strides', ...(isMax ? ['storage_order'] : ['count_include_pad'])])
                || attr(node, 'ceil_mode', 0) !== 0 || attr(node, 'storage_order', 0) !== 0
                || attr(node, 'dilations', [1, 1]).some((dim) => dim !== 1)
                || ![0, 1].includes(attr(node, 'count_include_pad', 0))
                || attr(node, 'pads', [0, 0, 0, 0]).slice(0, 2).some((pad, i) => pad > Math.floor(attr(node, 'kernel_shape')[i] / 2))) {
                unsupported(node, 'this pooling configuration'); continue;
            }
            const { dilation: _dilation, ...poolParams } = params;
            add(node, isMax ? 'torch.nn.MaxPool2d' : 'torch.nn.AvgPool2d', { ...poolParams, ceil_mode: false, ...(!isMax ? { count_include_pad: Boolean(attr(node, 'count_include_pad', 0)) } : {}) });
            continue;
        }
        if (node.opType === 'GlobalAveragePool' && node.inputs.length === 1 && onlyAttrs(node) && shapes.get(node.inputs[0])?.length === 4) {
            add(node, 'torch.nn.AdaptiveAvgPool2d', { output_size: '(1, 1)' }); continue;
        }
        if (node.opType === 'Concat' && node.inputs.length >= 2 && node.inputs.every(Boolean) && onlyAttrs(node, ['axis']) && Number.isInteger(attr(node, 'axis'))) {
            add(node, 'deepflow.op.Concat', { dim: attr(node, 'axis') }); continue;
        }
        if (node.opType === 'Transpose' && node.inputs.length === 1 && onlyAttrs(node, ['perm'])) {
            const rank = shapes.get(node.inputs[0])?.length;
            const perm = attr(node, 'perm', rank ? Array.from({ length: rank }, (_, i) => rank - i - 1) : []);
            if (!Array.isArray(perm)) { unsupported(node, 'this permutation'); continue; }
            const changed = perm.map((value, i) => value !== i ? i : -1).filter((i) => i >= 0);
            if (!rank || perm.length !== rank || new Set(perm).size !== rank || perm.some((value) => !Number.isInteger(value) || value < 0 || value >= rank)
                || (changed.length !== 2 && changed.length !== 0)) { unsupported(node, 'this permutation (only a single dimension swap is mapped)'); continue; }
            add(node, 'deepflow.op.Transpose', { dim0: changed[0] ?? 0, dim1: changed[1] ?? 0 }); continue;
        }
        unsupported(node);
    }
    for (const output of operatorGraph.outputs) ensureParameter(output.name, null);
    for (const node of operatorGraph.nodes) if (!visited.has(node.id) && !issues.some((item) => item.nodeId === node.id)) {
        issue(`ONNX operator ${node.opType} on ${node.name} was not mapped.`, node, 'unmapped-operator');
    }
    if (!components.length && !issues.length) issue('No editable component was recognized.', null, 'empty-graph');
    return { supported: issues.length === 0, components, issues };
}

/** Produce an editable graph only when every ONNX operator has a known mapping. */
export function buildEditableOnnxArchitecture(operatorGraph, { fileName = null } = {}) {
    const recognition = recognizeOnnxComponents(operatorGraph);
    if (!recognition.supported) throw new Error(recognition.issues.map((issue) => issue.message).join(' '));
    const all = [
        ...operatorGraph.inputs.filter((item) => !item.defaultInitializer).map((input) => ({
            key: 'deepflow.io.Input', label: input.name, parameters: {
                // DeepFlow Input treats the leading dimension as the batch. Keep
                // its exact source value in onnxSource; use B for shape inference
                // so a numeric ONNX batch is not accidentally prepended twice.
                shape: tuple(input.shape.map((dim, index) => index === 0 ? 'B' : dim ?? '?')), dtype: input.dtype,
            }, inputs: [], outputs: [input.name], source: { ...sourceFor([], [input.name]), shape: input.shape } })),
        ...recognition.components.map((component) => ({ key: component.libraryKey, label: component.label,
            parameters: component.parameters, source: component.source, inputs: component.inputs, outputs: component.outputs })),
        ...operatorGraph.outputs.map((output) => ({ key: 'deepflow.io.Output', label: output.name,
            parameters: { name: output.name }, inputs: [output.name], outputs: [], source: sourceFor([], [output.name]) })),
    ];
    const producers = new Map(), levels = new Map(), rows = new Map();
    const nodes = all.map((item, index) => {
        const entry = getEntry(item.key), id = `onnx-mapped:${index}`;
        const level = item.inputs.length ? Math.max(...item.inputs.map((value) => levels.get(value) ?? 0)) + 1 : 0;
        const column = rows.get(level) ?? 0;
        rows.set(level, column + 1);
        const node = createLayerNode({ id, libraryKey: item.key, entry, label: item.label,
            position: { x: 200 + column * 280, y: level * 155 },
            parameters: { ...Object.fromEntries(entry.parameters.map((param) => [param.name, param.default ?? null])), ...item.parameters } });
        for (const value of item.outputs) { producers.set(value, id); levels.set(value, level); }
        return { ...node, onnxSource: item.source };
    });
    const edges = [];
    all.forEach((item, index) => item.inputs.forEach((value, portIndex) => {
        const sourceId = producers.get(value);
        if (!sourceId) throw new Error(`Mapped tensor ${value} has no editable producer.`);
        edges.push({ ...createEdge({ id: `onnx-mapped-edge:${edges.length}`, source: sourceId, target: nodes[index].id,
            targetHandle: getOperationInputIds(nodes[index])?.[portIndex] }),
        onnxValue: value, onnxInputIndex: portIndex });
    }));
    const source = {
        provider: 'onnx', fileName, modelName: operatorGraph.name,
        displayName: operatorGraph.name || fileName || 'ONNX model', fidelity: 'editable-topology-no-weights', weightsLoaded: false,
        opCount: operatorGraph.nodes.length, mappedOpCount: operatorGraph.nodes.length,
        operatorGraph: compactSourceGraph(operatorGraph),
    };
    const graph = { nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges, importSource: source };
    return { graph, source, components: recognition.components };
}
