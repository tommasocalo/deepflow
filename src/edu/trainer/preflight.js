/**
 * Preflight (docs/edu/SPEC.md §6.10.6): everything that must hold before Train. Errors block training, warnings do
 * not. Pure: the store passes the graph, templates, training graph, dataset spec and settings.
 */
import { defaultObjective, effectiveObjective } from '../objective.js';
import { parseInputShape } from '../data/registry.js';
import { compileModel, MAX_ACTIVATIONS_PER_BATCH } from './compile.js';

export const LOSS_NAMES = { cross_entropy: 'Cross-Entropy', bce: 'BCE', mse: 'MSE', l1: 'L1', custom: 'custom' };
const SUPPORTED_LOSSES = ['cross_entropy', 'bce', 'mse', 'l1'];
export const lossName = (type) => LOSS_NAMES[type] ?? String(type);

/** '(2,)', '(1, 8, 8)': the shape after the batch axis, written as in the Data tab. */
export function formatFeatureShape(dims) {
    if (!Array.isArray(dims)) return '(?)';
    return `(${dims.map((dim) => (dim === null ? '?' : dim)).join(', ')}${dims.length === 1 ? ',' : ''})`;
}

const sameDims = (a, b) => Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((dim, i) => dim === b[i]);
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;

/** The objective the trainer uses: the training graph's (§6.10.5) with defaults for whatever is missing. */
export function resolveObjective(training, graph, dataset, outputSize) {
    const effective = effectiveObjective(training, graph);
    const defaults = defaultObjective({
        targetKind: dataset?.target?.kind === 'regression' ? 'regression' : 'classes',
        numClasses: dataset?.target?.kind === 'classes' ? dataset.target.size : null,
        outputSize,
    });
    const optimizer = effective.optimizer
        ? { type: effective.optimizer.type, hyperparameters: { ...effective.optimizer.hyperparameters }, parameterGroup: [...(effective.optimizer.parameterGroup ?? [])], nodeId: effective.optimizer.id }
        : { ...defaults.optimizer, parameterGroup: [], nodeId: null, isDefault: true };
    const loss = effective.loss
        ? { type: effective.loss.type, hyperparameters: { ...effective.loss.hyperparameters }, nodeId: effective.loss.id }
        : { ...defaults.loss, nodeId: null, isDefault: true };
    return { optimizer, loss, lossBound: effective.lossBound, issues: effective.issues };
}

/**
 * @returns {{ ok, errors, warnings, program, objective, inputShape, outputShape, paramCount, trainableCount }}
 */
export function runPreflight({ graph, templates = {}, training, dataset = null, settings = {} } = {}) {
    const errors = [];
    const warnings = [];
    const add = (severity, code, message, extra = {}) => (severity === 'error' ? errors : warnings).push({ severity, code, message, ...extra });
    const done = (extra = {}) => ({ ok: errors.length === 0, errors, warnings, program: null, objective: null, inputShape: null, outputShape: null, paramCount: null, trainableCount: null, ...extra });

    if (!dataset) {
        add('error', 'no-dataset', 'No dataset yet. Open the Data tab, choose one and press Use this dataset.');
        return done();
    }
    if (dataset.training !== 'browser') {
        add('error', 'dataset-export-only', `${dataset.name} is too large to train in the browser. Open Code → train.py and run it in Colab.`);
        return done();
    }

    const compiled = compileModel({ graph, templates });
    for (const issue of compiled.issues) (issue.severity === 'error' ? errors : warnings).push(issue);
    const outputDims = Array.isArray(compiled.outputShape) ? compiled.outputShape.slice(1) : null;
    const outputSize = outputDims?.length === 1 ? outputDims[0] : null;
    const objective = resolveObjective(training, graph, dataset, outputSize);
    const summary = {
        objective: { optimizer: objective.optimizer, loss: objective.loss },
        inputShape: compiled.inputShape,
        outputShape: compiled.outputShape,
        paramCount: compiled.program?.paramCount ?? null,
        trainableCount: compiled.program?.trainableCount ?? null,
    };

    // The model's Input must match the dataset.
    const datasetDims = parseInputShape(dataset.input?.shape);
    const modelDims = Array.isArray(compiled.inputShape) ? compiled.inputShape.slice(1) : null;
    const shapeless = errors.findIndex((issue) => issue.code === 'input-mismatch');
    if (shapeless >= 0) {
        errors[shapeless] = { ...errors[shapeless], message: `Your Input has no usable shape, but ${dataset.name} needs ${formatFeatureShape(datasetDims)}: use Data → Use this dataset.` };
    } else if (datasetDims && modelDims && !sameDims(datasetDims, modelDims)) {
        const what = datasetDims.length === 1 ? plural(datasetDims[0], 'feature') : `images shaped ${formatFeatureShape(datasetDims)}`;
        add('error', 'input-mismatch', `Your Input is ${formatFeatureShape(modelDims)}, but ${dataset.name} has ${what}: use Data → Use this dataset.`);
    }

    // Training-graph objective.
    for (const issue of objective.issues) {
        if (issue.code === 'multi-phase') add('error', 'multi-phase', 'Phases train only in the exported code: remove the phase blocks in the Training tab, or use Code → train.py.');
        else if (issue.code === 'multi-loss') add('error', 'multi-loss', 'Several losses are bound to the Output; the browser trainer uses exactly one.');
        else if (issue.code === 'loss-unbound') add('warning', 'loss-unbound', 'The loss is not bound to the Output yet; the trainer uses it anyway.');
        else if (issue.code === 'optimizer-ambiguous') add('warning', 'optimizer-ambiguous', 'Several optimizers and none is connected to the loss: using Adam (lr 0.01).');
    }
    const missing = [objective.optimizer.isDefault ? 'optimizer' : null, objective.loss.isDefault ? 'loss' : null].filter(Boolean);
    if (missing.length) {
        const using = [objective.optimizer.isDefault ? 'Adam (lr 0.01)' : null, objective.loss.isDefault ? lossName(objective.loss.type) : null].filter(Boolean).join(' and ');
        add('warning', 'defaults-used', `No ${missing.join('/')} in the Training tab: using ${using}.`);
    }
    const lossType = objective.loss.type;
    if (!SUPPORTED_LOSSES.includes(lossType)) {
        add('error', 'custom-loss', 'A custom loss trains only in the exported code: choose cross_entropy, bce, mse or l1, or use Code → train.py.');
    }
    const weight = objective.loss.hyperparameters?.weight;
    if (weight !== null && weight !== undefined && weight !== '') {
        add('warning', 'loss-weight-ignored', 'The loss weight setting is ignored in the browser; train.py uses it.');
    }

    // Output size against the dataset's target.
    if (outputDims && SUPPORTED_LOSSES.includes(lossType)) {
        const target = dataset.target;
        const name = dataset.name;
        const fix = (k) => `set the last Linear's Output to ${k} or use the Data tab's Use this dataset`;
        let message = null;
        if (target.kind === 'classes') {
            const classes = target.size;
            if (lossType === 'cross_entropy') {
                if (outputSize !== classes) {
                    message = outputSize === null
                        ? `The model outputs ${formatFeatureShape(outputDims)} per example, but Cross-Entropy needs ${classes} class scores: add Flatten and a Linear with Output ${classes}.`
                        : `${name} has ${classes} classes: ${fix(classes)}.`;
                }
            } else if (lossType === 'bce') {
                if (classes !== 2) message = `BCE is for two classes, but ${name} has ${classes}: choose cross_entropy in the Training tab.`;
                else if (outputSize !== 1) message = `BCE needs 1 output score, but the model gives ${formatFeatureShape(outputDims)}: set the last Linear's Output to 1, or use cross_entropy with 2 outputs.`;
            } else if (classes !== 2 || outputSize !== 1) {
                message = classes !== 2
                    ? `${lossName(lossType)} predicts numbers, but ${name} has ${classes} classes: choose cross_entropy in the Training tab.`
                    : `${lossName(lossType)} on 0/1 labels needs 1 output: ${fix(1)}.`;
            }
        } else if (target.kind === 'regression') {
            if (lossType === 'cross_entropy' || lossType === 'bce') message = `${name} is a regression dataset (one number per point): choose mse or l1 in the Training tab.`;
            else if (outputSize !== 1) message = `${name} has one target value per point: ${fix(1)}.`;
        }
        if (message) add('error', 'loss-target-mismatch', message);
    }

    // Common mistakes that still train.
    const program = compiled.program;
    if (program) {
        const last = program.ops.find((op) => op.id === program.outputValue);
        if (last?.kind === 'softmax' && lossType === 'cross_entropy') {
            add('warning', 'softmax-before-ce', 'A Softmax comes right before the Output, but Cross-Entropy already applies softmax: remove it and output raw scores.', { nodeId: last.origin.rootNodeId });
        }
        if (last?.kind === 'sigmoid' && lossType === 'bce') {
            add('warning', 'sigmoid-before-bce', 'A Sigmoid comes right before the Output, but BCE applies the sigmoid itself: remove it and output the raw score.', { nodeId: last.origin.rootNodeId });
        }
        const batch = Math.max(1, Math.floor(Number(settings.batchSize) || 32));
        const activations = program.activationsPerExample * Math.max(batch, 256) + program.constantActivations;
        if (activations > MAX_ACTIVATIONS_PER_BATCH) {
            add('error', 'too-large', `One batch needs about ${Math.round(activations).toLocaleString('en-US')} values in memory; the browser trainer allows ${MAX_ACTIVATIONS_PER_BATCH.toLocaleString('en-US')}. Use Code → train.py to train it in PyTorch.`);
        }
    }
    return done({ ...summary, program: errors.length ? null : program });
}
