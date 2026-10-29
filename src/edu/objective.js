import { OPTIMIZER_DEFAULTS } from '../domain/training/index.js';

/**
 * Which optimizer and loss train the model (docs/edu/SPEC.md §6.10.5). Pure; shared by the trainer, lesson checks
 * and codegen. Do not edit; Stage E only.
 */
export function effectiveObjective(training, rootGraph) {
    const nodes = Object.values(training?.nodes ?? {});
    const edges = training?.edges ?? [];
    const output = Object.values(rootGraph?.nodes ?? {})
        .find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Output') ?? null;
    const issues = [];
    if (nodes.some((node) => node.kind === 'phase')) issues.push({ severity: 'error', code: 'multi-phase' });
    const lossNodes = nodes.filter((node) => node.kind === 'loss');
    let losses = output ? lossNodes.filter((node) => node.architectureOutputId === output.id) : [];
    if (losses.length > 1) issues.push({ severity: 'error', code: 'multi-loss' });
    if (!losses.length && lossNodes.length === 1) {
        losses = lossNodes;
        issues.push({ severity: 'warning', code: 'loss-unbound' });
    }
    const loss = losses[0] ?? null;
    const optimizers = nodes.filter((node) => node.kind === 'optimizer');
    const linked = loss ? optimizers.find((node) => edges.some((edge) => edge.source === loss.id && edge.target === node.id)) : null;
    const optimizer = linked ?? (optimizers.length === 1 ? optimizers[0] : null);
    if (!optimizer && optimizers.length > 1) issues.push({ severity: 'warning', code: 'optimizer-ambiguous' });
    return { optimizer, loss, lossBound: !!loss && !!output && loss.architectureOutputId === output.id, output, issues };
}

/** Defaults when the training graph has no optimizer or no loss (§6.10.5). */
export function defaultObjective({ targetKind, numClasses = null, outputSize = null }) {
    const lossType = targetKind === 'regression' ? 'mse'
        : numClasses === 2 && outputSize === 1 ? 'bce' : 'cross_entropy';
    return {
        optimizer: { type: 'Adam', hyperparameters: { ...OPTIMIZER_DEFAULTS.Adam, learning_rate: 0.01 } },
        loss: { type: lossType, hyperparameters: {} },
    };
}
