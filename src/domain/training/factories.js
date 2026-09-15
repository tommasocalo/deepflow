const newId = () =>
    typeof crypto !== 'undefined' && crypto.randomUUID
        ? crypto.randomUUID()
        : `t_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

const OPTIMIZER_DEFAULTS = {
    Adam:    { learning_rate: 1e-3, weight_decay: 0,    betas: '(0.9, 0.999)',  eps: 1e-8 },
    AdamW:   { learning_rate: 1e-3, weight_decay: 1e-2, betas: '(0.9, 0.999)',  eps: 1e-8 },
    SGD:     { learning_rate: 1e-2, weight_decay: 0,    momentum: 0.9,          nesterov: false },
    RMSprop: { learning_rate: 1e-3, weight_decay: 0,    momentum: 0,            alpha: 0.99 },
};

const LOSS_DEFAULTS = {
    cross_entropy: { reduction: 'mean', label_smoothing: 0 },
    mse:           { reduction: 'mean' },
    bce:           { reduction: 'mean' },
    l1:            { reduction: 'mean' },
    custom:        { module: '', reduction: 'mean' },
};

/**
 * @param {Object} args
 * @param {import('../architecture/types.js').Position} args.position
 * @param {OptimizerNode['type']} [args.type]
 * @param {string} [args.label]
 * @param {string[]} [args.parameterGroup]
 * @param {Object<string, *>} [args.hyperparameters]
 * @param {string} [args.id]
 * @returns {import('./types.js').OptimizerNode}
 */
export function createOptimizerNode({
    position,
    type = 'Adam',
    label,
    parameterGroup = [],
    hyperparameters,
    id,
}) {
    return {
        id: id ?? newId(),
        kind: 'optimizer',
        position,
        label: label ?? `${type} optimizer`,
        type,
        hyperparameters: hyperparameters ?? { ...(OPTIMIZER_DEFAULTS[type] ?? {}) },
        parameterGroup,
    };
}

/**
 * @param {Object} args
 * @param {import('../architecture/types.js').Position} args.position
 * @param {LossNode['type']} [args.type]
 * @param {string} [args.label]
 * @param {string | null} [args.architectureOutputId]
 * @param {string} [args.labelKey]
 * @param {number} [args.weight]
 * @param {Object<string, *>} [args.hyperparameters]
 * @param {string} [args.id]
 * @returns {import('./types.js').LossNode}
 */
export function createLossNode({
    position,
    type = 'cross_entropy',
    label,
    architectureOutputId = null,
    labelKey = 'labels',
    weight = 1,
    hyperparameters,
    id,
}) {
    return {
        id: id ?? newId(),
        kind: 'loss',
        position,
        label: label ?? `${formatLossLabel(type)} loss`,
        type,
        hyperparameters: hyperparameters ?? { ...(LOSS_DEFAULTS[type] ?? {}) },
        architectureOutputId,
        labelKey,
        weight,
    };
}

/**
 * @param {Object} args
 * @param {import('../architecture/types.js').Position} args.position
 * @param {string} [args.label]
 * @param {string[]} [args.optimizerIds]
 * @param {string[]} [args.lossIds]
 * @param {number} [args.everyNSteps]
 * @param {number} [args.stepsPerCycle]
 * @param {string} [args.id]
 * @returns {import('./types.js').PhaseNode}
 */
export function createPhaseNode({
    position,
    label = 'Phase',
    optimizerIds = [],
    lossIds = [],
    everyNSteps = 1,
    stepsPerCycle = 1,
    id,
}) {
    return {
        id: id ?? newId(),
        kind: 'phase',
        position,
        label,
        optimizerIds,
        lossIds,
        everyNSteps,
        stepsPerCycle,
    };
}

/**
 * @param {Object} args
 * @param {string} args.source
 * @param {string} args.target
 * @param {string} [args.sourceHandle]
 * @param {string} [args.targetHandle]
 * @param {string} [args.id]
 * @returns {import('./types.js').TrainingEdge}
 */
export function createTrainingEdge({ source, target, sourceHandle, targetHandle, id }) {
    return {
        id: id ?? `te_${source}_${target}_${Math.random().toString(36).slice(2, 6)}`,
        source,
        target,
        sourceHandle,
        targetHandle,
    };
}

/** @returns {import('./types.js').TrainingGraph} */
export function emptyTrainingGraph() {
    return { nodes: {}, edges: [] };
}

function formatLossLabel(type) {
    return type === 'cross_entropy' ? 'Cross-Entropy'
        :  type === 'mse'           ? 'MSE'
        :  type === 'bce'           ? 'BCE'
        :  type === 'l1'            ? 'L1'
        :                              type;
}

export { OPTIMIZER_DEFAULTS, LOSS_DEFAULTS };
