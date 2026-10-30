/**
 * compileModel (docs/edu/SPEC.md §6.10.2): expand the architecture graph exactly as src/codegen/architecture.js does
 * (supernodes inlined with buildParameterScope, sequential repeats unrolled with their iteration scope and local
 * versions, edge gradient operators applied) into a flat program of ops the browser trainer can run.
 *
 * program = { inputShape, outputShape, inputValue, outputValue, ops: [{ id, kind, inputs, attrs, origin, shape }],
 *             params: [{ id, shape, init, scale, requiresGrad, enabled, disabledInit, paddingIdx, origin }],
 *             paramCount, trainableCount, activationsPerExample, constantActivations }
 */
import { topologicalSort } from '../../validation/topology.js';
import { buildParameterScope, resolveDimensionTuple, resolveLayerParameters, resolveRepeatCount, withGlobalParameters } from '../../domain/expressions.js';
import { getIncomingEdges, mapTemplateInputs, orderOperationEdges } from '../../domain/graphInputs.js';
import { cachedValidate } from '../../domain/architecture/levelContext.js';
import { effectiveTemplateId } from '../../domain/architecture/instanceStructure.js';
import { resolveGradientFactor } from '../../domain/gradientFactor.js';
import { isShapeCollection } from '../../domain/shapeCollections.js';
import { friendlyIssues } from '../lessons/engine/feedback.js';
import { t } from '../lessons/engine/i18n.js';

export const MAX_PARAMETERS = 2_000_000;
export const MAX_ACTIVATIONS_PER_BATCH = 4_000_000;
const MAX_DEPTH = 8;
const MAX_OPS = 20000;

/** Library keys the trainer runs, and their op kinds. Anything else is `unsupported-op`. */
export const SUPPORTED_OPS = Object.freeze({
    'torch.nn.Linear': 'linear',
    'torch.nn.ReLU': 'relu',
    'torch.nn.LeakyReLU': 'leakyrelu',
    'torch.nn.Tanh': 'tanh',
    'torch.nn.Sigmoid': 'sigmoid',
    'torch.nn.GELU': 'gelu',
    'torch.nn.Softmax': 'softmax',
    'torch.nn.LogSoftmax': 'logsoftmax',
    'torch.nn.Dropout': 'dropout',
    'torch.nn.Flatten': 'flatten',
    'deepflow.op.Parameter': 'param',
    'deepflow.op.Transpose': 'transpose',
    'deepflow.op.MatMul': 'matmul',
    'deepflow.op.Add': 'add',
    'deepflow.op.Multiply': 'multiply',
    'deepflow.op.Subtract': 'subtract',
    'deepflow.op.Scale': 'scale',
    'deepflow.op.AddScalar': 'addscalar',
    'deepflow.op.Pow': 'pow',
    'deepflow.op.Exp': 'exp',
    'deepflow.op.Rsqrt': 'rsqrt',
    'deepflow.op.Mean': 'mean',
});

const INPUT = 'deepflow.io.Input';
const OUTPUT = 'deepflow.io.Output';
const TEMPLATE_INPUT = 'deepflow.template.Input';
const TEMPLATE_OUTPUT = 'deepflow.template.Output';

const typeName = (node, templates) => node.kind === 'layer'
    ? node.libraryKey.split('.').at(-1)
    : templates[node.templateId]?.name ?? (node.kind === 'repeat' ? 'Repeat' : 'Component');
const formatShape = (shape) => (Array.isArray(shape) ? `(${shape.map((dim) => (dim === null ? 'B' : dim)).join(', ')})` : '(?)');
const normalAxis = (axis, rank) => (Number.isInteger(axis) ? (axis < 0 ? rank + axis : axis) : null);

/** Lesson markup (**bold**, `code`) without the marks, for the trainer's plain-text lines. */
const plainText = (markup) => String(markup).replace(/\*\*([^*]+)\*\*/g, '$1').replace(/`([^`]+)`/g, '$1');

export function unsupportedMessage(label) {
    return `${label} is not supported by the browser trainer. Use Code → train.py to train this model in PyTorch.`;
}

/**
 * @returns {{ ok: boolean, program: Object | null, issues: Object[], inputShape: Array|null, outputShape: Array|null,
 *             paramCount: number }}
 */
export function compileModel({ graph, templates = {} } = {}) {
    const issues = [];
    const error = (code, message, extra = {}) => issues.push({ severity: 'error', code, message, ...extra });
    if (!graph?.nodes) {
        error('invalid-graph', 'There is no model to train.');
        return { ok: false, program: null, issues, inputShape: null, outputShape: null, paramCount: 0 };
    }
    const rootScope = withGlobalParameters(graph);
    const report = cachedValidate(graph, templates, rootScope, {});
    const rootNodes = Object.values(graph.nodes);
    const inputs = rootNodes.filter((node) => node.kind === 'layer' && node.libraryKey === INPUT);
    const outputs = rootNodes.filter((node) => node.kind === 'layer' && node.libraryKey === OUTPUT);
    const inputShape = inputs.length === 1 ? report.shapes[inputs[0].id] ?? null : null;
    const outputShape = outputs.length === 1 ? report.shapes[outputs[0].id] ?? null : null;
    const result = (program = null) => ({ ok: !issues.some((issue) => issue.severity === 'error'), program, issues, inputShape, outputShape, paramCount: program?.paramCount ?? 0 });

    if (!report.ok) {
        // The lesson panel's friendly rewrite (docs/edu/SPEC.md §6.4.4), as plain text; the raw text stays alongside.
        for (const issue of friendlyIssues(report, graph, templates).slice(0, 6)) {
            error('invalid-graph', plainText(t(issue.key, issue.params)), { nodeId: issue.nodeId ?? undefined, validatorMessage: issue.message });
        }
        if (!issues.length) error('invalid-graph', 'The model has errors: open Graph checks on the canvas.');
        return result();
    }
    if (inputs.length !== 1) {
        error('multi-input', inputs.length ? `The model has ${inputs.length} Input blocks; the browser trainer needs exactly one.` : 'Add an Input block.');
    }
    if (outputs.length !== 1) {
        error('multi-output', outputs.length ? `The model has ${outputs.length} Output blocks; the browser trainer needs exactly one.` : 'Add an Output block.');
    }
    if (issues.length) return result();
    if (!Array.isArray(inputShape) || inputShape[0] !== null || inputShape.slice(1).some((dim) => !Number.isSafeInteger(dim) || dim < 1)) {
        error('input-mismatch', `Give the Input a shape, for example (2,): it is ${formatShape(inputShape)}.`, { nodeId: inputs[0].id });
        return result();
    }

    const ctx = { templates, ops: [], params: [], issues, valueShapes: new Map(), inputValue: null, outputValue: null };
    expandLevel(graph, rootScope, ctx, { prefix: '', externalInputs: {}, externalShapes: {}, depth: 0, report, rootNode: null, path: [] });
    if (issues.some((issue) => issue.severity === 'error')) return result();
    if (!ctx.outputValue) {
        error('invalid-graph', 'Connect the Output block to your model.', { nodeId: outputs[0].id });
        return result();
    }

    // Drop ops the output does not depend on (for example a disconnected branch).
    const needed = new Set([ctx.outputValue]);
    for (let i = ctx.ops.length - 1; i >= 0; i--) {
        if (needed.has(ctx.ops[i].id)) for (const input of ctx.ops[i].inputs) needed.add(input);
    }
    if (!needed.has(ctx.inputValue)) {
        error('invalid-graph', 'Connect the Input block to the Output block through your model.', { nodeId: inputs[0].id });
        return result();
    }
    const ops = ctx.ops.filter((op) => needed.has(op.id));
    const usedParams = new Set(ops.filter((op) => op.kind === 'param').map((op) => op.attrs.paramId));
    for (const op of ops) if (op.kind === 'linear') { usedParams.add(op.attrs.weight); if (op.attrs.bias) usedParams.add(op.attrs.bias); }
    const params = ctx.params.filter((param) => usedParams.has(param.id));
    const size = (shape) => shape.reduce((product, dim) => product * dim, 1);
    const paramCount = params.filter((param) => param.enabled !== false).reduce((sum, param) => sum + size(param.shape), 0);
    const trainableCount = params.filter((param) => param.enabled !== false && param.requiresGrad !== false).reduce((sum, param) => sum + size(param.shape), 0);
    let activationsPerExample = size(inputShape.slice(1));
    let constantActivations = 0;
    for (const op of ops) {
        const shape = op.shape;
        if (!Array.isArray(shape) || op.kind === 'param') continue;
        if (shape[0] === null) activationsPerExample += size(shape.slice(1).map((dim) => dim ?? 1));
        else constantActivations += size(shape.map((dim) => dim ?? 1));
    }
    if (paramCount > MAX_PARAMETERS) {
        error('too-large', `This model has ${paramCount.toLocaleString('en-US')} parameters; the browser trainer allows ${MAX_PARAMETERS.toLocaleString('en-US')}. Use Code → train.py to train it in PyTorch.`);
        return result();
    }
    return result({
        inputShape: [...inputShape],
        outputShape: Array.isArray(outputShape) ? [...outputShape] : null,
        inputValue: ctx.inputValue,
        outputValue: ctx.outputValue,
        ops,
        params,
        paramCount,
        trainableCount,
        activationsPerExample,
        constantActivations,
    });
}

function expandLevel(graph, scope, ctx, { prefix, externalInputs, externalShapes, depth, report: givenReport, rootNode, path }) {
    const { issues, templates } = ctx;
    const error = (code, message, extra = {}) => {
        if (!issues.some((issue) => issue.message === message)) issues.push({ severity: 'error', code, message, ...extra });
    };
    if (depth > MAX_DEPTH) {
        error('invalid-graph', `Components are nested more than ${MAX_DEPTH} levels deep.`);
        return { finalRef: null, finalShape: null };
    }
    const { order, adjacency } = topologicalSort(graph);
    const report = givenReport ?? cachedValidate(graph, templates, scope, externalShapes);
    const refs = {};
    const unique = (id) => `${prefix}${id}`;
    const emit = (node, kind, inputs, attrs, shape, idSuffix = '') => {
        if (ctx.ops.length >= MAX_OPS) {
            if (!issues.some((issue) => issue.code === 'too-large')) error('too-large', `The expanded model has more than ${MAX_OPS} operations.`);
            return null;
        }
        const id = unique(`${node.id}${idSuffix}`);
        ctx.ops.push({ id, kind, inputs, attrs, shape: shape ?? null, origin: originOf(node) });
        ctx.valueShapes.set(id, shape ?? null);
        return id;
    };
    const originOf = (node) => ({
        rootNodeId: rootNode ? rootNode.id : node.id,
        path: [...path, node.id],
        label: rootNode && rootNode.id !== node.id ? `${rootNode.label} › ${node.label}` : node.label,
        type: typeName(node, templates),
    });
    const unsupported = (node, reason) => error('unsupported-op', reason ?? unsupportedMessage(`${originOf(node).label} (${typeName(node, templates)})`), {
        nodeId: rootNode ? rootNode.id : node.id, path: [...path, node.id], type: typeName(node, templates),
    });

    for (const nodeId of order) {
        const node = graph.nodes[nodeId];
        if (!node) continue;
        const edges = getIncomingEdges(graph, nodeId);
        const edgeRefs = new Map();
        for (const edge of edges) {
            let ref = refs[edge.source] ?? null;
            const op = edge.gradientOp ?? 'none';
            if (ref && op !== 'none') {
                const sourceShape = ctx.valueShapes.get(ref) ?? report.shapes[edge.source] ?? null;
                if (op === 'reverse') {
                    const { value: lambda, error: problem } = resolveGradientFactor(edge.gradientOpLambda ?? 1, scope);
                    if (problem) error('invalid-graph', `Gradient reversal factor on a connection into ${node.label}: ${problem}`);
                    ref = emit(node, 'reversegrad', [ref], { lambda: lambda ?? 1 }, sourceShape, `<${edge.id}`);
                } else {
                    ref = emit(node, 'stopgrad', [ref], {}, sourceShape, `<${edge.id}`);
                }
            }
            edgeRefs.set(edge, ref);
        }
        const inputRefs = orderOperationEdges(node, edges).map((edge) => (edge ? edgeRefs.get(edge) ?? null : null));
        const shape = report.shapes[nodeId] ?? null;

        if (node.kind === 'layer') {
            const key = node.libraryKey;
            if (key === INPUT) {
                if (depth === 0) {
                    ctx.inputValue = unique(nodeId);
                    ctx.valueShapes.set(ctx.inputValue, shape);
                    refs[nodeId] = ctx.inputValue;
                }
                continue;
            }
            if (key === TEMPLATE_INPUT) {
                refs[nodeId] = Object.hasOwn(externalInputs, nodeId) ? externalInputs[nodeId] : inputRefs[0] ?? null;
                continue;
            }
            if (key === TEMPLATE_OUTPUT || key === OUTPUT) {
                refs[nodeId] = inputRefs[0] ?? null;
                if (key === OUTPUT && depth === 0) ctx.outputValue = refs[nodeId];
                continue;
            }
            const kind = SUPPORTED_OPS[key];
            if (!kind) {
                unsupported(node);
                refs[nodeId] = null;
                continue;
            }
            if (kind !== 'param' && (!inputRefs.length || inputRefs.some((ref) => ref === null))) {
                refs[nodeId] = null;
                continue;
            }
            const inputShapes = inputRefs.map((ref) => ctx.valueShapes.get(ref) ?? null);
            if (inputShapes.some(isShapeCollection)) {
                unsupported(node, unsupportedMessage(`${originOf(node).label} (a collection of tensors)`));
                refs[nodeId] = null;
                continue;
            }
            const p = resolveLayerParameters(node, scope).parameters;
            refs[nodeId] = layerOp(node, kind, p, inputRefs, inputShapes, shape, { ctx, emit, originOf, unsupported, error, unique });
            continue;
        }

        if (node.kind === 'supernode' || node.kind === 'repeat') {
            const template = templates[node.templateId];
            if (!template) {
                error('invalid-graph', `${node.label}: its component definition is missing.`, { nodeId: rootNode ? rootNode.id : node.id });
                refs[nodeId] = null;
                continue;
            }
            const isRepeat = node.kind === 'repeat';
            if (isRepeat && node.mode === 'parallel') {
                unsupported(node, unsupportedMessage(`${originOf(node).label} (parallel repeat)`));
                refs[nodeId] = null;
                continue;
            }
            const { count, error: countError } = isRepeat
                ? resolveRepeatCount(node.count, scope, edges.map((edge) => report.shapes[edge.source] ?? null))
                : { count: 1 };
            if (countError) {
                error('invalid-graph', `${node.label}: ${countError}`, { nodeId: rootNode ? rootNode.id : node.id });
                refs[nodeId] = null;
                continue;
            }
            let carry = inputRefs[0] ?? null;
            let carryShape = report.shapes[edges[0]?.source] ?? null;
            const childRoot = rootNode ?? node;
            for (let index = 0; index < count; index++) {
                const instanceTemplate = templates[effectiveTemplateId(node, index)];
                if (!instanceTemplate?.graph) {
                    error('invalid-graph', `${node.label}: the definition of instance ${index + 1} is missing.`);
                    carry = null;
                    break;
                }
                const initialShapes = mapTemplateInputs(instanceTemplate.graph, edges, (edge) => report.shapes[edge.source] ?? null);
                const initialInputs = mapTemplateInputs(instanceTemplate.graph, edges, (edge) => edgeRefs.get(edge) ?? null);
                const iterationScope = isRepeat ? { ...scope, [node.iterationVariable || 'i']: index } : scope;
                const childShapes = Object.fromEntries(Object.entries(initialShapes).map(([port, value]) => [port, isRepeat && index > 0 ? carryShape : value]));
                const subScope = buildParameterScope(instanceTemplate.freeParameters, isRepeat ? node.bindings : node.parameterValues, iterationScope, isRepeat ? index : undefined, childShapes);
                const subInputs = Object.fromEntries(Object.entries(initialInputs).map(([port, ref]) => [port, isRepeat && index > 0 ? carry : ref]));
                const sub = expandLevel(instanceTemplate.graph, subScope, ctx, {
                    prefix: `${unique(nodeId)}_${index}__`, externalInputs: subInputs, externalShapes: childShapes, depth: depth + 1,
                    report: null, rootNode: childRoot, path: [...path, nodeId],
                });
                carry = sub.finalRef;
                carryShape = sub.finalShape;
                if (!carry) break;
            }
            refs[nodeId] = carry;
            continue;
        }
        refs[nodeId] = inputRefs[0] ?? null;
    }

    let finalRef = null;
    for (const nodeId of order) {
        const node = graph.nodes[nodeId];
        if (!node) continue;
        if ((adjacency.out[nodeId]?.length ?? 0) === 0) finalRef = refs[nodeId] ?? null;
        if (node.kind === 'layer' && node.libraryKey === TEMPLATE_OUTPUT) finalRef = refs[nodeId] ?? null;
    }
    const output = Object.values(graph.nodes).filter((node) => node.libraryKey === TEMPLATE_OUTPUT).at(-1);
    return { finalRef, finalShape: output ? report.shapes[output.id] ?? null : null };
}

/** Emit the op for one supported layer node; returns its value id, or null after recording an issue. */
function layerOp(node, kind, p, inputRefs, inputShapes, shape, { ctx, emit, originOf, unsupported, error, unique }) {
    const label = originOf(node).label;
    const [first] = inputShapes;
    const rank = Array.isArray(first) ? first.length : null;
    const needRank2 = (name) => {
        if (inputShapes.every((s) => Array.isArray(s) && s.length === 2)) return true;
        const got = inputShapes.map(formatShape).join(' and ');
        unsupported(node, `${label} receives ${got}; the browser trainer runs ${name} only on 2D tensors such as (B, features). `
            + (name === 'Linear' ? 'Add a Flatten block before it, or use Code → train.py.' : 'Use Code → train.py to train this model in PyTorch.'));
        return false;
    };
    const param = (suffix, spec) => {
        const id = `${unique(node.id)}.${suffix}`;
        ctx.params.push({ id, origin: originOf(node), ...spec });
        return id;
    };

    switch (kind) {
        case 'linear': {
            if (!needRank2('Linear')) return null;
            const inFeatures = Number(p.in_features);
            const outFeatures = Number(p.out_features);
            if (!Number.isSafeInteger(inFeatures) || !Number.isSafeInteger(outFeatures) || inFeatures < 1 || outFeatures < 1) {
                error('invalid-graph', `${label}: set Input and Output features to positive whole numbers.`, { nodeId: originOf(node).rootNodeId });
                return null;
            }
            const bound = 1 / Math.sqrt(inFeatures);
            const weight = param('weight', { shape: [outFeatures, inFeatures], init: 'uniform', scale: bound, requiresGrad: true, enabled: true });
            const bias = isFalse(p.bias) ? null : param('bias', { shape: [outFeatures], init: 'uniform', scale: bound, requiresGrad: true, enabled: true });
            return emit(node, 'linear', inputRefs, { weight, bias }, shape);
        }
        case 'param': {
            const dims = Array.isArray(p.shape) ? p.shape : resolveDimensionTuple(p.shape);
            if (!dims || dims.some((dim) => !Number.isSafeInteger(dim) || dim < 1)) {
                error('invalid-graph', `${label}: the parameter shape must be positive whole numbers.`, { nodeId: originOf(node).rootNodeId });
                return null;
            }
            const scale = Number(p.scale ?? 1);
            const paramId = param('value', {
                shape: [...dims],
                init: p.init ?? 'normal',
                scale: Number.isFinite(scale) ? scale : 1,
                requiresGrad: !isFalse(p.requires_grad),
                enabled: !isFalse(p.enabled),
                disabledInit: p.disabled_init ?? 'zeros',
                paddingIdx: Number.isInteger(p.padding_idx) ? p.padding_idx : null,
            });
            return emit(node, 'param', [], { paramId }, shape ?? [...dims]);
        }
        case 'matmul':
            if (!needRank2('MatMul')) return null;
            return emit(node, 'matmul', inputRefs, {}, shape);
        case 'transpose': {
            if (!needRank2('Transpose')) return null;
            const dims = [normalAxis(Number(p.dim0 ?? -2), 2), normalAxis(Number(p.dim1 ?? -1), 2)].sort();
            if (dims[0] !== 0 || dims[1] !== 1) {
                unsupported(node, `${label}: the browser trainer swaps only the last two axes. Use Code → train.py for other transposes.`);
                return null;
            }
            return emit(node, 'transpose', inputRefs, {}, shape);
        }
        case 'flatten': {
            const start = normalAxis(Number(p.start_dim ?? 1), rank);
            const end = normalAxis(Number(p.end_dim ?? -1), rank);
            if (start !== 1 || end !== rank - 1) {
                unsupported(node, `${label}: the browser trainer flattens only from axis 1 to the last axis (start_dim 1, end_dim −1).`);
                return null;
            }
            return emit(node, 'flatten', inputRefs, {}, shape);
        }
        case 'softmax':
        case 'logsoftmax': {
            const axis = normalAxis(Number(p.dim ?? -1), rank);
            if (axis !== rank - 1) {
                unsupported(node, `${label}: the browser trainer applies ${kind === 'softmax' ? 'Softmax' : 'LogSoftmax'} only over the last axis (dim −1).`);
                return null;
            }
            return emit(node, kind, inputRefs, {}, shape);
        }
        case 'mean': {
            const dims = Array.isArray(p.dim) ? p.dim : resolveDimensionTuple(p.dim ?? -1);
            if (!dims || dims.length !== 1 || normalAxis(dims[0], rank) !== rank - 1) {
                unsupported(node, `${label}: the browser trainer averages only over the last axis (dim −1).`);
                return null;
            }
            return emit(node, 'mean', inputRefs, { keepdim: p.keepdim !== false }, shape);
        }
        case 'dropout': {
            const prob = Number(p.p ?? 0.5);
            if (!Number.isFinite(prob) || prob < 0 || prob >= 1) {
                error('invalid-graph', `${label}: the dropout probability p must be at least 0 and below 1.`, { nodeId: originOf(node).rootNodeId });
                return null;
            }
            return emit(node, 'dropout', inputRefs, { p: prob }, shape);
        }
        case 'leakyrelu':
            return emit(node, kind, inputRefs, { slope: Number.isFinite(Number(p.negative_slope)) ? Number(p.negative_slope) : 0.01 }, shape);
        case 'gelu':
            return emit(node, kind, inputRefs, { approximate: p.approximate === 'tanh' ? 'tanh' : 'none' }, shape);
        case 'scale':
            return emit(node, kind, inputRefs, { factor: Number(p.factor ?? 1) }, shape);
        case 'addscalar':
            return emit(node, kind, inputRefs, { value: Number(p.value ?? 0) }, shape);
        case 'pow':
            return emit(node, kind, inputRefs, { exponent: Number(p.exponent ?? 2) }, shape);
        case 'subtract':
            if (inputRefs.length !== 2) return null;
            return emit(node, kind, inputRefs, {}, shape);
        case 'add':
        case 'multiply':
            if (inputRefs.length < 1) return null;
            return inputRefs.length === 1 ? inputRefs[0] : emit(node, kind, inputRefs, {}, shape);
        default:
            return emit(node, kind, inputRefs, {}, shape);
    }
}

const isFalse = (value) => value === false || value === 'false' || value === 'False' || value === 0;
