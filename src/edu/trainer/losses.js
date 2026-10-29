/**
 * Losses with PyTorch semantics (docs/edu/SPEC.md §6.10.4). Each returns the reduced loss and its gradient with
 * respect to the model output, which seeds Tape.backward. Reduction 'mean' (default) divides by the number of
 * elements, 'sum' does not.
 */
import { sigmoidValue } from './ops.js';

export const LOSS_TYPES = ['cross_entropy', 'bce', 'mse', 'l1'];

/** cross_entropy: logits (B, C), integer targets in [0, C); label smoothing ε as in torch.nn.CrossEntropyLoss. */
export function crossEntropy(logits, targets, { labelSmoothing = 0, reduction = 'mean' } = {}) {
    const [batch, classes] = logits.shape;
    const z = logits.data;
    const grad = new Float64Array(z.length);
    const eps = Number(labelSmoothing) || 0;
    let total = 0;
    for (let i = 0; i < batch; i++) {
        const base = i * classes;
        let max = -Infinity;
        for (let c = 0; c < classes; c++) max = Math.max(max, z[base + c]);
        let sum = 0;
        for (let c = 0; c < classes; c++) sum += Math.exp(z[base + c] - max);
        const lse = max + Math.log(sum);
        const y = targets[i];
        let smooth = 0;
        for (let c = 0; c < classes; c++) smooth -= z[base + c] - lse;
        total += (1 - eps) * (lse - z[base + y]) + eps * (smooth / classes);
        for (let c = 0; c < classes; c++) {
            grad[base + c] = Math.exp(z[base + c] - lse) - (c === y ? 1 - eps : 0) - eps / classes;
        }
    }
    return finish(total, grad, batch, reduction);
}

/** bce = BCEWithLogitsLoss: mean of max(x, 0) − x·y + log(1 + e^(−|x|)); targets in {0, 1}. */
export function bceWithLogits(logits, targets, { reduction = 'mean' } = {}) {
    const x = logits.data;
    const grad = new Float64Array(x.length);
    let total = 0;
    for (let i = 0; i < x.length; i++) {
        const y = targets[i];
        total += Math.max(x[i], 0) - x[i] * y + Math.log1p(Math.exp(-Math.abs(x[i])));
        grad[i] = sigmoidValue(x[i]) - y;
    }
    return finish(total, grad, x.length, reduction);
}

export function mse(outputs, targets, { reduction = 'mean' } = {}) {
    const x = outputs.data;
    const grad = new Float64Array(x.length);
    let total = 0;
    for (let i = 0; i < x.length; i++) {
        const d = x[i] - targets[i];
        total += d * d;
        grad[i] = 2 * d;
    }
    return finish(total, grad, x.length, reduction);
}

export function l1(outputs, targets, { reduction = 'mean' } = {}) {
    const x = outputs.data;
    const grad = new Float64Array(x.length);
    let total = 0;
    for (let i = 0; i < x.length; i++) {
        const d = x[i] - targets[i];
        total += Math.abs(d);
        grad[i] = Math.sign(d);
    }
    return finish(total, grad, x.length, reduction);
}

function finish(total, grad, count, reduction) {
    if (reduction === 'sum') return { loss: total, grad };
    const n = Math.max(1, count);
    for (let i = 0; i < grad.length; i++) grad[i] /= n;
    return { loss: total / n, grad };
}

/** Dispatch by DeepFlow loss type; `hyperparameters` are the loss node's (label_smoothing, reduction). */
export function computeLoss(type, outputs, targets, hyperparameters = {}) {
    const reduction = hyperparameters.reduction === 'sum' ? 'sum' : 'mean';
    if (type === 'cross_entropy') return crossEntropy(outputs, targets, { labelSmoothing: hyperparameters.label_smoothing, reduction });
    if (type === 'bce') return bceWithLogits(outputs, targets, { reduction });
    if (type === 'mse') return mse(outputs, targets, { reduction });
    if (type === 'l1') return l1(outputs, targets, { reduction });
    throw new Error(`Loss "${type}" is not supported by the browser trainer`);
}
