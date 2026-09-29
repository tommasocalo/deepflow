import { createEdge, createLayerNode } from '../architecture/index.js';
import { getEntry } from '../library/index.js';

const ACTIVATIONS = {
    Relu: 'torch.nn.ReLU',
    Sigmoid: 'torch.nn.Sigmoid',
    Tanh: 'torch.nn.Tanh',
};
const ALLOWED_DTYPES = new Set(['float32', 'float16', 'bfloat16', 'float64']);
const positiveDimension = (value) => Number.isSafeInteger(value) && value > 0;

function sourceFor(nodes, values, initializers = []) {
    return {
        nodeIds: nodes.map((node) => node.id),
        nodeNames: nodes.map((node) => node.name),
        values,
        initializers,
        fidelity: 'editable-topology-no-weights',
    };
}

function compactSourceGraph(operatorGraph) {
    const { name, irVersion, opsets, producer, byteLength, inputs, outputs, initializers, nodes, edges, limitations } = operatorGraph;
    return { name, irVersion, opsets, producer, byteLength, inputs, outputs, initializers, nodes, edges, limitations };
}

/**
 * Recognize a deliberately narrow, whole-graph conversion: one tensor input,
 * one output, and a linear chain of Gemm / MatMul(+bias Add) / activations.
 * Every ONNX op must be accounted for. The operator map remains available even
 * when this editable conversion is refused.
 */
export function recognizeOnnxComponents(operatorGraph) {
    if (operatorGraph?.format !== 'onnx-operator-graph') throw new TypeError('Expected an ONNX operator graph.');
    const issues = [];
    const components = [];
    const visited = new Set();
    const dynamicInputs = operatorGraph.inputs.filter((item) => !item.defaultInitializer);
    const input = dynamicInputs[0];
    const output = operatorGraph.outputs[0];
    const initializers = new Map(operatorGraph.initializers.map((item) => [item.name, item]));
    const consumers = new Map();
    const issue = (message, node = null, code = 'unmapped-pattern') => {
        issues.push({ code, nodeId: node?.id ?? null, nodeName: node?.name ?? null, message });
    };
    for (const node of operatorGraph.nodes) for (const value of node.inputs) {
        if (!value || initializers.has(value)) continue;
        consumers.set(value, [...(consumers.get(value) ?? []), node]);
    }

    if (dynamicInputs.length !== 1 || operatorGraph.outputs.length !== 1) {
        issue('This editable mapping currently needs exactly one tensor input and one output.', null, 'multiple-ports');
    }
    if (!input || input.kind !== 'tensor' || !ALLOWED_DTYPES.has(input.dtype)) {
        issue('The editable mapping needs a floating-point tensor input.', null, 'unsupported-input');
    }
    if (!output || output.kind !== 'tensor') issue('The editable mapping needs a tensor output.', null, 'unsupported-output');
    // Only dimensions are used to infer editable topology. No initializer
    // payload, inline or external, is copied into DeepFlow weights.
    if (issues.length) return { supported: false, components, issues };

    let currentValue = input.name;
    const maxSteps = operatorGraph.nodes.length + 1;
    for (let step = 0; currentValue !== output.name && step < maxSteps; step += 1) {
        const next = consumers.get(currentValue) ?? [];
        if (next.length !== 1) {
            issue(next.length ? `Tensor “${currentValue}” branches to ${next.length} operators; this mapping only handles a chain.`
                : `Tensor “${currentValue}” does not lead to the graph output.`, null, 'branch-or-gap');
            break;
        }
        const node = next[0];
        if (visited.has(node.id)) { issue(`Cycle near ${node.name}.`, node, 'cycle'); break; }
        if (node.domain) { issue(`Custom domain “${node.domain}” on ${node.name} is not mapped.`, node, 'custom-domain'); break; }
        if (node.inputs[0] !== currentValue || node.outputs.length !== 1 || !node.outputs[0]) {
            issue(`${node.name} has ports this editable mapping cannot reproduce.`, node, 'unsupported-ports'); break;
        }

        if (node.opType === 'Gemm') {
            const attrs = node.attributes;
            const get = (name, fallback) => attrs[name]?.value ?? fallback;
            const validAttrs = Object.keys(attrs).every((name) => ['alpha', 'beta', 'transA', 'transB'].includes(name));
            const weight = initializers.get(node.inputs[1]);
            const bias = node.inputs[2] ? initializers.get(node.inputs[2]) : null;
            if (!validAttrs || get('alpha', 1) !== 1 || get('beta', 1) !== 1
                || get('transA', 0) !== 0 || get('transB', 0) !== 1
                || node.inputs.length < 2 || node.inputs.length > 3
                || !weight || weight.shape.length !== 2 || !weight.shape.every(positiveDimension)
                || (node.inputs[2] && !bias) || (bias && (bias.shape.length !== 1 || bias.shape[0] !== weight.shape[0]))) {
                issue(`${node.name} is not a standard Linear: expected A · Wᵀ + b with W[out,in] and optional b[out].`, node);
                break;
            }
            components.push({ kind: 'Linear', status: 'mapped', libraryKey: 'torch.nn.Linear',
                parameters: { in_features: weight.shape[1], out_features: weight.shape[0], bias: Boolean(bias) },
                source: sourceFor([node], [currentValue, node.outputs[0]], [weight.name, ...(bias ? [bias.name] : [])]),
                label: node.name });
            visited.add(node.id);
            currentValue = node.outputs[0];
            continue;
        }

        if (node.opType === 'MatMul') {
            const weight = initializers.get(node.inputs[1]);
            if (node.inputs.length !== 2 || Object.keys(node.attributes).length || !weight
                || weight.shape.length !== 2 || !weight.shape.every(positiveDimension)) {
                issue(`${node.name} is not a Linear projection with a W[in,out] initializer.`, node);
                break;
            }
            const outWidth = weight.shape[1];
            const addCandidates = consumers.get(node.outputs[0]) ?? [];
            const add = addCandidates.length === 1 && addCandidates[0].opType === 'Add' ? addCandidates[0] : null;
            const biasName = add?.inputs.find((name) => name !== node.outputs[0]);
            const bias = biasName ? initializers.get(biasName) : null;
            if (add && (!bias || bias.shape.length !== 1 || bias.shape[0] !== outWidth
                || add.inputs.length !== 2 || add.outputs.length !== 1 || Object.keys(add.attributes).length)) {
                issue(`${add.name} is not a vector bias for ${node.name}.`, add);
                break;
            }
            const outputValue = add?.outputs[0] ?? node.outputs[0];
            components.push({ kind: 'Linear', status: 'mapped', libraryKey: 'torch.nn.Linear',
                parameters: { in_features: weight.shape[0], out_features: outWidth, bias: Boolean(add) },
                source: sourceFor(add ? [node, add] : [node], [currentValue, outputValue], [weight.name, ...(bias ? [bias.name] : [])]),
                label: add?.name || node.name });
            visited.add(node.id);
            if (add) visited.add(add.id);
            currentValue = outputValue;
            continue;
        }

        if (ACTIVATIONS[node.opType] && node.inputs.length === 1 && !Object.keys(node.attributes).length) {
            components.push({ kind: node.opType, status: 'mapped', libraryKey: ACTIVATIONS[node.opType], parameters: {},
                source: sourceFor([node], [currentValue, node.outputs[0]]), label: node.name });
            visited.add(node.id);
            currentValue = node.outputs[0];
            continue;
        }
        issue(`ONNX operator ${node.domain ? `${node.domain}::` : ''}${node.opType} on ${node.name} has no editable mapping yet.`, node, 'unmapped-operator');
        break;
    }
    if (currentValue !== output.name && !issues.length) issue('The graph output is not reached by the mapped components.', null, 'output-gap');
    for (const node of operatorGraph.nodes) if (!visited.has(node.id) && !issues.some((item) => item.nodeId === node.id)) {
        issue(`ONNX operator ${node.opType} on ${node.name} is outside the mapped chain.`, node, 'unmapped-operator');
    }
    if (!components.length && !issues.length) issue('No editable component was recognized.', null, 'empty-graph');
    return { supported: issues.length === 0, components, issues };
}

/** Produce a DeepFlow graph only when every ONNX operator has a known mapping. */
export function buildEditableOnnxArchitecture(operatorGraph, { fileName = null } = {}) {
    const recognition = recognizeOnnxComponents(operatorGraph);
    if (!recognition.supported) {
        throw new Error(recognition.issues.map((issue) => issue.message).join(' '));
    }
    const [input] = operatorGraph.inputs.filter((item) => !item.defaultInitializer);
    const [output] = operatorGraph.outputs;
    const all = [
        { key: 'deepflow.io.Input', label: input.name, parameters: {
            shape: input.shape && input.shape.every((dim) => dim != null) ? `(${input.shape.join(', ')}${input.shape.length === 1 ? ',' : ''})` : null,
            dtype: input.dtype,
        }, source: sourceFor([], [input.name]) },
        ...recognition.components.map((component) => ({ key: component.libraryKey, label: component.label,
            parameters: component.parameters, source: component.source })),
        { key: 'deepflow.io.Output', label: output.name, parameters: { name: output.name }, source: sourceFor([], [output.name]) },
    ];
    const nodes = all.map((item, index) => {
        const entry = getEntry(item.key);
        const node = createLayerNode({ id: `onnx-mapped:${index}`, libraryKey: item.key, entry,
            label: item.label, position: { x: 400, y: index * 150 },
            parameters: { ...Object.fromEntries(entry.parameters.map((param) => [param.name, param.default ?? null])), ...item.parameters } });
        return { ...node, onnxSource: item.source };
    });
    const source = {
        provider: 'onnx',
        fileName,
        modelName: operatorGraph.name,
        displayName: operatorGraph.name || fileName || 'ONNX model',
        fidelity: 'editable-topology-no-weights',
        weightsLoaded: false,
        opCount: operatorGraph.nodes.length,
        mappedOpCount: operatorGraph.nodes.length,
        operatorGraph: compactSourceGraph(operatorGraph),
    };
    const graph = {
        nodes: Object.fromEntries(nodes.map((node) => [node.id, node])),
        edges: nodes.slice(0, -1).map((node, index) => createEdge({ id: `onnx-mapped-edge:${index}`,
            source: node.id, target: nodes[index + 1].id })),
        importSource: source,
    };
    return { graph, source, components: recognition.components };
}
