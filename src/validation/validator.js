import { topologicalSort } from './topology.js';
import { validateLayerParameters } from './parameters.js';
import { inferShape } from './shapes.js';
import { buildParameterScope, resolveLayerParameters, resolveRepeatCount } from '../domain/expressions.js';
import { getIncomingEdges, getTemplateInputIds, getOperationInputIds, mapTemplateInputs, orderOperationEdges, validateInputPorts } from '../domain/graphInputs.js';
import { isShapeCollection, shapeCollection } from '../domain/shapeCollections.js';
import { effectiveTemplateId } from '../domain/architecture/instanceStructure.js';

const TEMPLATE_INPUT = 'deepflow.template.Input';
const TEMPLATE_OUTPUT = 'deepflow.template.Output';
const MAX_TEMPLATE_DEPTH = 8;
const MAX_VALIDATED_NODES = 20000;

/**
 * Validate tensors and explicit collections through a graph. externalInputs
 * maps template boundary node IDs to shapes. Repeats default to sequential;
 * parallel repeats select item i from each incoming collection.
 */
export function validate(graph, opts = {}) {
    const { templates = {}, scope = {}, externalInputs = {}, depth = 0 } = opts;
    const budget = opts.budget ?? { remaining: MAX_VALIDATED_NODES };
    const { order: topo, cyclic, adjacency: adj } = topologicalSort(graph);
    const errors = {}, warnings = {}, shapes = {}, globalErrors = [];
    const push = (bucket, id, message) => (bucket[id] ??= []).push(message);

    if (cyclic.length) {
        for (const id of cyclic) push(errors, id, 'This node is part of a cycle');
        globalErrors.push(`Graph has a cycle through ${cyclic.length} node(s)`);
    }

    for (const [id, node] of Object.entries(graph.nodes)) {
        const inDeg = adj.inn[id]?.length ?? 0;
        const outDeg = adj.out[id]?.length ?? 0;
        if (node.kind === 'layer') {
            const isSource = [TEMPLATE_INPUT, 'deepflow.io.Input', 'deepflow.op.Parameter', 'deepflow.op.Arange'].includes(node.libraryKey);
            const isSink = node.libraryKey === TEMPLATE_OUTPUT || node.libraryKey === 'deepflow.io.Output';
            if (isSource && inDeg > 0) push(errors, id, 'Source nodes cannot have inputs');
            if (!isSource && inDeg === 0) push(warnings, id, 'No incoming connections');
            if (isSink && outDeg > 0) push(errors, id, 'Output nodes cannot have outgoing connections');
            if (!isSink && outDeg === 0 && Object.keys(graph.nodes).length > 1) push(warnings, id, 'No outgoing connections');
            const resolved = resolveLayerParameters(node, scope);
            for (const issue of validateLayerParameters(resolved)) push(errors, id, issue.message);
            for (const [name, raw] of Object.entries(node.parameters ?? {})) {
                if (typeof raw === 'string' && raw.trim().startsWith('$') && resolved.parameters[name] === null && !Object.hasOwn(scope, raw.trim().slice(1))) {
                    push(errors, id, `${name} has an unresolved parameter reference: ${raw}`);
                }
            }
        } else if (node.kind === 'supernode' || node.kind === 'repeat') {
            if (inDeg === 0) push(warnings, id, 'No incoming connections');
            if (outDeg === 0 && Object.keys(graph.nodes).length > 1) push(warnings, id, 'No outgoing connections');
            if (!templates[node.templateId]) push(errors, id, 'Template missing — re-create or remove this node');
        }
    }

    for (const id of topo) {
        const node = graph.nodes[id];
        if (!node) continue;
        if (--budget.remaining < 0) {
            push(errors, id, `Expanded graph exceeds ${MAX_VALIDATED_NODES} nodes`);
            shapes[id] = null;
            break;
        }
        const edges = getIncomingEdges(graph, id);
        const inputShapes = orderOperationEdges(node, edges).map((edge) => edge ? shapes[edge.source] ?? null : null);
        if (node.kind === 'layer') {
            const operationPorts = getOperationInputIds(node);
            if (operationPorts) {
                for (const issue of validateInputPorts(edges, operationPorts)) push(errors, id, issue);
                if (edges.length !== operationPorts.length || orderOperationEdges(node, edges).some((edge) => !edge)) {
                    push(errors, id, `${node.libraryKey.split('.').at(-1)} needs ${operationPorts.length} connected inputs (${operationPorts.join(' and ')})`);
                }
            }
            if (node.libraryKey === TEMPLATE_INPUT && Object.hasOwn(externalInputs, id)) {
                shapes[id] = externalInputs[id];
                continue;
            }
            const result = inferShape(node.libraryKey, inputShapes, resolveLayerParameters(node, scope).parameters);
            shapes[id] = result.output;
            for (const error of result.errors) push(errors, id, error);
            continue;
        }
        if (node.kind !== 'supernode' && node.kind !== 'repeat') {
            shapes[id] = null;
            continue;
        }
        const template = templates[node.templateId];
        if (!template) { shapes[id] = null; continue; }
        if (depth >= MAX_TEMPLATE_DEPTH) {
            push(errors, id, `Template recursion exceeds depth ${MAX_TEMPLATE_DEPTH}`);
            shapes[id] = null;
            continue;
        }
        const isRepeat = node.kind === 'repeat';
        const parallel = isRepeat && node.mode === 'parallel';
        if (isRepeat && node.count === 'auto' && !parallel) {
            push(errors, id, 'Auto instances is available only for parallel repetition');
            shapes[id] = null;
            continue;
        }
        const countResult = isRepeat ? resolveRepeatCount(node.count, scope, inputShapes) : { count: 1 };
        if (countResult.error) {
            push(errors, id, countResult.error);
            shapes[id] = null;
            continue;
        }
        const { count } = countResult;
        if (parallel && inputShapes.some((shape) => isShapeCollection(shape) && shape.items.length !== count)) {
            push(errors, id, `Parallel repeat needs ${count} items in every input collection`);
            shapes[id] = null;
            continue;
        }
        let carry = inputShapes[0] ?? null;
        const outputs = [];
        const childErrors = [];
        for (let index = 0; index < count && budget.remaining >= 0; index++) {
            const instanceTemplateId = effectiveTemplateId(node, index);
            const instanceTemplate = templates[instanceTemplateId];
            if (!instanceTemplate?.graph) {
                childErrors.push(`Instance ${index + 1} template is missing`);
                carry = null;
                outputs.push(null);
                continue;
            }
            const inputIds = getTemplateInputIds(instanceTemplate.graph);
            const outputIds = Object.values(instanceTemplate.graph.nodes).filter((n) => n.libraryKey === TEMPLATE_OUTPUT).map((n) => n.id);
            const initialInputs = mapTemplateInputs(instanceTemplate.graph, edges, (edge) => shapes[edge.source] ?? null);
            if (parallel || index === 0) {
                for (const issue of validateInputPorts(edges, inputIds)) childErrors.push(issue);
                const connectedPorts = mapTemplateInputs(instanceTemplate.graph, edges, () => true);
                if (edges.length && Object.values(connectedPorts).some((connected) => !connected)) childErrors.push('Connect every template input port');
            }
            let childInputs;
            if (parallel) {
                childInputs = Object.fromEntries(Object.entries(initialInputs).map(([port, shape]) => [port, isShapeCollection(shape) ? shape.items[index] ?? null : shape]));
            } else if (!isRepeat || index === 0) {
                childInputs = initialInputs;
            } else {
                childInputs = Object.fromEntries(inputIds.map((port) => [port, carry]));
            }
            const iterationScope = isRepeat ? { ...scope, [node.iterationVariable || 'i']: index } : scope;
            const subScope = buildParameterScope(instanceTemplate.freeParameters, isRepeat ? node.bindings : node.parameterValues, iterationScope, isRepeat ? index : undefined, childInputs);
            for (const parameter of instanceTemplate.freeParameters ?? []) {
                const value = subScope[parameter.name];
                if (parameter.derived && value === null) {
                    const input = instanceTemplate.graph.nodes[parameter.derived.inputId];
                    childErrors.push(`${parameter.name} cannot be inferred from ${input?.label ?? parameter.derived.inputId}: connect an input with a known dimension`);
                    continue;
                }
                if (parameter.nullable && (value === null || value === undefined)) continue;
                if (parameter.type === 'bool' && typeof value !== 'boolean') childErrors.push(`${parameter.name} must resolve to true or false`);
                if (parameter.type === 'int' && (value === null || value === undefined || !Number.isSafeInteger(Number(value)))) {
                    childErrors.push(`${parameter.name} must resolve to an integer (got ${String(value)})`);
                }
                if (parameter.type === 'float' && (value === null || value === undefined || !Number.isFinite(Number(value)))) {
                    childErrors.push(`${parameter.name} must resolve to a finite number`);
                }
            }
            const child = validate(instanceTemplate.graph, { templates, scope: subScope, externalInputs: childInputs, depth: depth + 1, budget });
            carry = outputIds.reduce((output, port) => child.shapes[port] ?? output, null);
            outputs.push(carry);
            childErrors.push(...child.globalErrors, ...Object.values(child.errors).flat());
        }
        shapes[id] = parallel ? shapeCollection(outputs) : carry;
        if (childErrors.length) {
            const prefix = isRepeat ? 'Repeat' : `Template "${template.name}"`;
            push(errors, id, `${prefix} has ${childErrors.length} internal error${childErrors.length === 1 ? '' : 's'}: ${[...new Set(childErrors)].slice(0, 3).join('; ')}`);
        }
    }
    const ok = Object.keys(errors).length === 0 && globalErrors.length === 0;
    return { topo, cyclic, shapes, errors, warnings, globalErrors, ok };
}
