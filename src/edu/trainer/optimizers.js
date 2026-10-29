/**
 * Optimizers with PyTorch's update rules (docs/edu/SPEC.md §6.10.5). Hyperparameter names follow DeepFlow's training
 * graph: learning_rate, weight_decay, betas ("(0.9, 0.999)"), eps, momentum, nesterov, alpha, dampening.
 * Parameters whose gradient is null are skipped, as torch.optim does.
 */
import { OPTIMIZER_DEFAULTS } from '../../domain/training/index.js';

export const OPTIMIZER_TYPES = ['Adam', 'AdamW', 'SGD', 'RMSprop'];

const number = (value, fallback) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
};
const bool = (value) => value === true || value === 'true' || value === 'True' || value === 1;

/** "(0.9, 0.999)", "[0.9, 0.999]" or [0.9, 0.999] → [0.9, 0.999]. */
export function parseBetas(value, fallback = [0.9, 0.999]) {
    const parts = Array.isArray(value) ? value : String(value ?? '').replace(/[()[\]\s]/g, '').split(',').filter(Boolean);
    const betas = parts.map(Number);
    return betas.length === 2 && betas.every((beta) => Number.isFinite(beta) && beta >= 0 && beta < 1) ? betas : fallback;
}

/** Resolved numeric hyperparameters for a type, with DeepFlow's defaults filling the gaps. */
export function resolveHyperparameters(type, hyperparameters = {}) {
    const defaults = OPTIMIZER_DEFAULTS[type] ?? OPTIMIZER_DEFAULTS.Adam;
    const hp = { ...defaults, ...hyperparameters };
    return {
        lr: number(hp.learning_rate, defaults.learning_rate),
        weightDecay: number(hp.weight_decay, 0),
        betas: parseBetas(hp.betas),
        eps: number(hp.eps, 1e-8),
        momentum: number(hp.momentum, 0),
        nesterov: bool(hp.nesterov),
        alpha: number(hp.alpha, 0.99),
        dampening: number(hp.dampening, 0),
    };
}

/**
 * @param {string} type  Adam | AdamW | SGD | RMSprop
 * @param {Object} hyperparameters
 * @param {{ data: Float64Array, grad: Float64Array | null }[]} params  the trainable tensors
 */
export function createOptimizer(type, hyperparameters, params) {
    if (!OPTIMIZER_TYPES.includes(type)) throw new Error(`Optimizer "${type}" is not supported by the browser trainer`);
    const hp = resolveHyperparameters(type, hyperparameters);
    const state = params.map((p) => ({ step: 0, m: null, v: null, buffer: null, size: p.data.length }));

    function stepSGD(p, s) {
        const { lr, weightDecay, momentum, nesterov, dampening } = hp;
        const w = p.data, grad = p.grad;
        const g = new Float64Array(grad.length);
        for (let i = 0; i < g.length; i++) g[i] = weightDecay ? grad[i] + weightDecay * w[i] : grad[i];
        if (momentum !== 0) {
            if (!s.buffer) s.buffer = Float64Array.from(g);
            else for (let i = 0; i < g.length; i++) s.buffer[i] = momentum * s.buffer[i] + (1 - dampening) * g[i];
            if (nesterov) for (let i = 0; i < g.length; i++) g[i] = g[i] + momentum * s.buffer[i];
            else g.set(s.buffer);
        }
        for (let i = 0; i < w.length; i++) w[i] -= lr * g[i];
    }

    function stepAdam(p, s, decoupled) {
        const { lr, weightDecay, eps } = hp;
        const [beta1, beta2] = hp.betas;
        const w = p.data, grad = p.grad;
        s.m ??= new Float64Array(w.length);
        s.v ??= new Float64Array(w.length);
        s.step += 1;
        if (decoupled && weightDecay) for (let i = 0; i < w.length; i++) w[i] *= 1 - lr * weightDecay;
        const bias1 = 1 - Math.pow(beta1, s.step);
        const bias2 = 1 - Math.pow(beta2, s.step);
        for (let i = 0; i < w.length; i++) {
            const g = !decoupled && weightDecay ? grad[i] + weightDecay * w[i] : grad[i];
            s.m[i] = beta1 * s.m[i] + (1 - beta1) * g;
            s.v[i] = beta2 * s.v[i] + (1 - beta2) * g * g;
            w[i] -= lr * (s.m[i] / bias1) / (Math.sqrt(s.v[i] / bias2) + eps);
        }
    }

    function stepRMSprop(p, s) {
        const { lr, weightDecay, eps, alpha, momentum } = hp;
        const w = p.data, grad = p.grad;
        s.v ??= new Float64Array(w.length);
        if (momentum > 0) s.buffer ??= new Float64Array(w.length);
        for (let i = 0; i < w.length; i++) {
            const g = weightDecay ? grad[i] + weightDecay * w[i] : grad[i];
            s.v[i] = alpha * s.v[i] + (1 - alpha) * g * g;
            const avg = Math.sqrt(s.v[i]) + eps;
            if (momentum > 0) {
                s.buffer[i] = momentum * s.buffer[i] + g / avg;
                w[i] -= lr * s.buffer[i];
            } else {
                w[i] -= lr * g / avg;
            }
        }
    }

    return {
        type,
        hyperparameters: hp,
        state,
        step() {
            params.forEach((p, index) => {
                if (!p.grad) return;
                const s = state[index];
                if (type === 'SGD') stepSGD(p, s);
                else if (type === 'Adam') stepAdam(p, s, false);
                else if (type === 'AdamW') stepAdam(p, s, true);
                else stepRMSprop(p, s);
            });
        },
        zeroGrad() {
            for (const p of params) p.grad = null;
        },
    };
}
