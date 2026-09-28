import { topologicalSort } from '../validation/topology.js';
import { buildParameterScope, resolveLayerParameters, resolveRepeatCount, withGlobalParameters } from './expressions.js';
import { getIncomingEdges, getTemplateInputIds, mapInputEdgesToPorts, orderOperationEdges } from './graphInputs.js';
import { defaultParameterValues, getEntry } from './library/index.js';
import { isShapeCollection } from './shapeCollections.js';
import { validate } from '../validation/validator.js';
import { effectiveTemplateId } from './architecture/instanceStructure.js';
import { resolveGradientFactor } from './gradientFactor.js';

const tensor = (requires = 'unknown', dtype = 'float', dependencies = []) => ({ requires, dtype, dependencies: [...new Set(dependencies)] });
const items = (value) => value?.items ? value.items.flatMap(items) : [value ?? tensor()];
const merge = (values) => {
    const all = values.flatMap(items);
    const requires = all.some((v) => v.requires === 'yes') ? 'yes' : all.some((v) => v.requires === 'unknown') ? 'unknown' : 'no';
    return tensor(requires, all.every((v) => v.dtype === 'index') ? 'index' : 'float', all.flatMap((v) => v.dependencies));
};
const detach = (value) => value?.items ? { items: value.items.map(detach) } : tensor('no', value?.dtype ?? 'float');
const cut = (edge, scope) => ['stop', 'detach'].includes(edge.gradientOp) || (edge.gradientOp === 'reverse' && resolveGradientFactor(edge.gradientOpLambda ?? 1, scope).value === 0);
const invalidReverse = (edge, scope) => edge.gradientOp === 'reverse' && !!resolveGradientFactor(edge.gradientOpLambda ?? 1, scope).error;
const trainableModule = (key) => /^torch\.nn\.(Linear|Embedding|Bilinear|Conv|BatchNorm|LayerNorm|GroupNorm|InstanceNorm|RNN|GRU|LSTM|MultiheadAttention|Transformer)/.test(key)
    || key === 'deepflow.nn.PositionalEmbedding' || key === 'deepflow.nn.CausalSelfAttention';
const unknownModule = (key) => /^(deepflow\.op\.|deepflow\.(io|template)\.|torch\.nn\.|deepflow\.nn\.)/.test(key) === false;

/** Static autograd structure. This describes paths, never measured gradients. */
export function analyzeGradientFlow(graph, { templates = {}, scope: outerScope = {}, externalInputs = {}, externalShapes = {}, depth = 0, budget = { remaining: 20000 } } = {}) {
    const scope = withGlobalParameters(graph, outerScope);
    const report = { graph, scope, nodes: {}, edges: {}, children: {}, output: tensor(), inputDependencies: [] };
    const shapeReport = validate(graph, { templates, scope, externalInputs: externalShapes, depth });
    const allowed = new Set();
    const { order } = topologicalSort(graph);
    for (const id of order) {
        if (--budget.remaining < 0) break;
        const node = graph.nodes[id];
        const incoming = getIncomingEdges(graph, id);
        const valueFor = (edge) => !edge ? tensor() : cut(edge, scope) ? detach(report.nodes[edge.source])
            : invalidReverse(edge, scope) ? tensor() : report.nodes[edge.source] ?? tensor();
        let used = incoming;
        let output;
        if (node.kind === 'supernode' || node.kind === 'repeat') {
            const template = templates[node.templateId];
            const count = node.kind === 'repeat' ? resolveRepeatCount(node.count, scope, incoming.map((edge) => shapeReport.shapes[edge.source])).count : 1;
            if (!template || !count || depth >= 8) { report.nodes[id] = tensor(); continue; }
            const children = [];
            const outputs = [];
            let carry;
            let carryShape;
            const usedEdgeIds = new Set();
            for (let index = 0; index < count && budget.remaining > 0; index++) {
                const instanceTemplate = templates[effectiveTemplateId(node, index)];
                if (!instanceTemplate?.graph) {
                    children.push(null);
                    carry = tensor();
                    carryShape = null;
                    outputs.push(carry);
                    continue;
                }
                const ports = getTemplateInputIds(instanceTemplate.graph);
                const mapped = mapInputEdgesToPorts(incoming, ports);
                const initial = Object.fromEntries(ports.map((port) => [port, valueFor(mapped[port])]));
                const initialShapes = Object.fromEntries(ports.map((port) => [port, shapeReport.shapes[mapped[port]?.source] ?? null]));
                const repeated = node.kind === 'repeat';
                const childShapes = Object.fromEntries(ports.map((port) => [port, node.mode === 'parallel'
                    ? isShapeCollection(initialShapes[port]) ? initialShapes[port].items[index] ?? null : initialShapes[port]
                    : index === 0 ? initialShapes[port] : carryShape]));
                const childScope = buildParameterScope(instanceTemplate.freeParameters, repeated ? node.bindings : node.parameterValues,
                    repeated ? { ...scope, [node.iterationVariable || 'i']: index } : scope, repeated ? index : undefined, childShapes);
                const external = Object.fromEntries(ports.map((port) => [port, node.mode === 'parallel'
                    ? initial[port]?.items?.[index] ?? initial[port] : index === 0 ? initial[port] : carry]));
                const child = analyzeGradientFlow(instanceTemplate.graph, { templates, scope: childScope, externalInputs: external, externalShapes: childShapes, depth: depth + 1, budget });
                children.push(child);
                if (node.mode === 'parallel' || index === 0) {
                    for (const port of child.inputDependencies) if (mapped[port]) usedEdgeIds.add(mapped[port].id);
                } else if (!child.inputDependencies.length) usedEdgeIds.clear();
                // Dependency names are local to a template; map back through its boundary.
                const dependencies = child.inputDependencies.flatMap((port) => items(external[port]).flatMap((v) => v.dependencies));
                const remap = (value) => value?.items ? { items: value.items.map(remap) } : { ...value, dependencies };
                carry = remap(child.output);
                carryShape = child.outputShape;
                outputs.push(carry);
            }
            report.children[id] = children;
            used = incoming.filter((edge) => usedEdgeIds.has(edge.id));
            output = node.mode === 'parallel' ? { items: outputs } : outputs.at(-1) ?? tensor();
        } else {
            const key = node.libraryKey;
            const params = resolveLayerParameters({ ...node, parameters: { ...defaultParameterValues(getEntry(key)), ...node.parameters } }, scope).parameters;
            if (key === 'deepflow.template.Input') {
                const input = externalInputs[id] ?? tensor();
                const mark = (value) => value?.items ? { items: value.items.map(mark) } : { ...value, dependencies: [id] };
                output = mark(input);
                used = [];
            } else if (key === 'deepflow.io.Input') {
                const integer = ['int32', 'int64'].includes(params.dtype);
                output = tensor(integer ? 'no' : params.requires_grad === true ? 'yes' : params.requires_grad === false ? 'no' : 'unknown', integer ? 'index' : 'float');
                used = [];
            } else if (key === 'deepflow.op.Parameter') {
                output = tensor(params.requires_grad === false || params.enabled === false ? 'no' : 'yes');
                used = [];
            } else if (key === 'deepflow.op.PositionIndices' || key === 'deepflow.op.Arange') {
                output = tensor('no', key === 'deepflow.op.PositionIndices' ? 'index' : 'float');
                used = [];
            } else if (key === 'deepflow.op.Gather') {
                const ports = mapInputEdgesToPorts(incoming, ['indices', 'weight'], { broadcastSingle: false });
                used = ports.weight ? [ports.weight] : [];
                output = valueFor(ports.weight);
            } else if (key === 'torch.nn.Embedding' || key === 'deepflow.nn.PositionalEmbedding') {
                used = [];
                output = tensor('yes');
            } else if (key === 'deepflow.op.Split' || key === 'deepflow.op.Broadcast') {
                const count = Number(params.chunks ?? params.count);
                const value = merge(incoming.map(valueFor));
                output = Number.isInteger(count) && count > 0 && count <= 256 ? { items: Array.from({ length: count }, () => value) } : tensor();
            } else {
                used = orderOperationEdges(node, incoming).filter(Boolean);
                output = merge(used.map(valueFor));
                const affine = key === 'torch.nn.LayerNorm' ? params.elementwise_affine !== false : /Norm/.test(key) ? params.affine !== false : true;
                if (trainableModule(key) && affine) output = { ...output, requires: 'yes', dtype: 'float' };
                if (unknownModule(key)) output = tensor();
            }
        }
        report.nodes[id] = output;
        for (const edge of used) if (!cut(edge, scope) && !invalidReverse(edge, scope)) allowed.add(edge.id);
    }
    const outputs = Object.values(graph.nodes).filter((node) => ['deepflow.io.Output', 'deepflow.template.Output'].includes(node.libraryKey));
    report.output = outputs.length ? report.nodes[outputs.at(-1).id] ?? tensor() : tensor();
    report.outputShape = outputs.length ? shapeReport.shapes[outputs.at(-1).id] ?? null : null;
    report.inputDependencies = merge([report.output]).dependencies;
    const reachable = new Set(outputs.map((node) => node.id));
    for (const id of [...order].reverse()) {
        if (reachable.has(id)) for (const edge of getIncomingEdges(graph, id)) if (allowed.has(edge.id)) reachable.add(edge.source);
    }
    for (const edge of graph.edges) {
        const source = merge([report.nodes[edge.source]]);
        let status, reason;
        if (cut(edge, scope)) { status = 'stopped'; reason = 'Gradient is explicitly stopped on this connection'; }
        else if (invalidReverse(edge, scope)) { status = 'unknown'; reason = 'Gradient reversal factor is unresolved'; }
        else if (!report.nodes[edge.source] || !report.nodes[edge.target]) { status = 'unknown'; reason = 'Gradient path cannot be resolved for this graph'; }
        else if (!allowed.has(edge.id)) { status = 'constant'; reason = 'This input is used as indices or metadata, without a derivative'; }
        else if (source.dtype === 'index') { status = 'constant'; reason = 'Integer indices have no gradient'; }
        else if (!reachable.has(edge.target) && outputs.length) { status = 'constant'; reason = 'No differentiable route from this connection to a graph output'; }
        else if (source.requires === 'no') { status = 'constant'; reason = 'Constant or frozen tensor: no upstream gradient'; }
        else if (source.requires === 'unknown') { status = 'unknown'; reason = 'Input gradient requirement is not specified'; }
        else { status = edge.gradientOp === 'reverse' ? 'reverse' : 'flow'; reason = status === 'reverse' ? 'Gradient sign is reversed on this connection' : 'Differentiable path from trainable values'; }
        report.edges[edge.id] = { status, reason };
    }
    return report;
}

export function gradientReportAtPath(report, path, inspectedIterations = {}) {
    const traversed = [];
    for (const id of path) {
        traversed.push(id);
        const children = report?.children[id];
        if (!children?.length) return null;
        const index = Math.max(0, Math.min(children.length - 1, inspectedIterations[JSON.stringify(traversed)] ?? 0));
        report = children[index];
    }
    return report;
}
