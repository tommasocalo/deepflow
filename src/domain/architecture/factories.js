import { defaultParameterValues } from '../library/index.js';

const newId = () =>
    typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `n_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

/**
 * @param {Object} args
 * @param {string} args.libraryKey
 * @param {import('./types.js').Position} args.position
 * @param {import('./types.js').LibraryEntry} args.entry
 * @param {string} [args.label]
 * @param {Object<string, *>} [args.parameters]
 * @param {string} [args.id]
 * @returns {import('./types.js').LayerNode}
 */
export function createLayerNode({ libraryKey, position, entry, label, parameters, id }) {
    return {
        id: id ?? newId(),
        kind: 'layer',
        libraryKey,
        label: label ?? entry?.name ?? libraryKey,
        position,
        parameters: parameters ?? defaultParameterValues(entry),
    };
}

/**
 * @param {Object} args
 * @param {string} args.templateId
 * @param {import('./types.js').Position} args.position
 * @param {string} [args.label]
 * @param {Object<string, *>} [args.parameterValues]
 * @param {string} [args.id]
 * @returns {import('./types.js').SuperNodeInstance}
 */
export function createSuperNodeInstance({ templateId, position, label = 'Supernode', parameterValues = {}, id }) {
    return {
        id: id ?? newId(),
        kind: 'supernode',
        templateId,
        label,
        position,
        parameterValues,
    };
}

/**
 * @param {Object} args
 * @param {string} args.templateId
 * @param {import('./types.js').Position} args.position
 * @param {string} [args.label]
 * @param {number | string} [args.count] Literal count or template expression
 * @param {'sequential' | 'parallel'} [args.mode]
 * @param {string} [args.iterationVariable]
 * @param {Object<string, *>} [args.bindings]
 * @param {string} [args.id]
 * @returns {import('./types.js').RepeatNode}
 */
export function createRepeatNode({
    templateId,
    position,
    label = 'Repeat',
    count = 2,
    mode = 'sequential',
    iterationVariable = 'i',
    bindings = {},
    id,
}) {
    return {
        id: id ?? newId(),
        kind: 'repeat',
        templateId,
        label,
        position,
        count,
        mode,
        iterationVariable,
        bindings,
    };
}

/**
 * @param {Object} args
 * @param {string} args.source
 * @param {string} args.target
 * @param {string} [args.sourceHandle]
 * @param {string} [args.targetHandle]
 * @param {import('./types.js').GradientOp} [args.gradientOp]
 * @param {number} [args.gradientOpLambda]
 * @param {string} [args.id]
 * @returns {import('./types.js').ArchEdge}
 */
export function createEdge({ source, target, sourceHandle, targetHandle, gradientOp, gradientOpLambda, id }) {
    return {
        id: id ?? `e_${source}_${target}_${Math.random().toString(36).slice(2, 6)}`,
        source,
        target,
        sourceHandle,
        targetHandle,
        gradientOp: gradientOp ?? 'none',
        ...(gradientOpLambda !== undefined ? { gradientOpLambda } : {}),
    };
}

/** @returns {import('./types.js').ArchitectureGraph} */
export function emptyGraph() {
    return { nodes: {}, edges: [] };
}

/**
 * @param {Object} args
 * @param {string} args.name
 * @param {string} [args.description]
 * @param {import('./types.js').ArchitectureGraph} [args.graph]
 * @param {import('./types.js').ParamSpec[]} [args.freeParameters]
 * @param {string} [args.id]
 * @returns {import('./types.js').ArchitectureTemplate}
 */
export function createTemplate({ name, description = '', graph, freeParameters = [], id }) {
    return {
        id: id ?? newId(),
        name,
        description,
        graph: graph ?? emptyGraph(),
        freeParameters,
    };
}

export const newDomainId = newId;
