/**
 * Parameter initialisation (docs/edu/SPEC.md §6.10.2, §6.10.7): every parameter is drawn from the 'init' stream of
 * the training seed, in program order and row-major within a tensor, with PyTorch's default distributions.
 */
import { normal, stream, uniform } from '../data/rng.js';
import { numel, Tensor } from './tensor.js';

/** fan_in as torch.nn.init computes it: shape[1] × the receptive field (product of shape[2:]). */
export function fanIn(shape) {
    if (shape.length < 2) return shape[0] ?? 1;
    return shape[1] * shape.slice(2).reduce((product, dim) => product * dim, 1);
}

/** Fill one parameter tensor. `init`: normal | uniform | kaiming_uniform | zeros | ones. */
export function fillParameter(param, next) {
    const size = numel(param.shape);
    const data = new Float64Array(size);
    const init = param.enabled === false ? param.disabledInit ?? 'zeros' : param.init;
    if (init === 'ones') data.fill(1);
    else if (init === 'normal') for (let i = 0; i < size; i++) data[i] = normal(next, 0, param.scale ?? 1);
    else if (init === 'uniform') {
        const bound = param.scale ?? 1;
        for (let i = 0; i < size; i++) data[i] = uniform(next, -bound, bound);
    } else if (init === 'kaiming_uniform') {
        // kaiming_uniform_(a=√5): gain √(2 / (1 + a²)) · √(3 / fan_in) = 1 / √fan_in.
        const bound = 1 / Math.sqrt(fanIn(param.shape));
        for (let i = 0; i < size; i++) data[i] = uniform(next, -bound, bound);
    }
    const row = param.paddingIdx;
    if (Number.isInteger(row) && param.shape.length >= 1) {
        const rows = param.shape[0];
        const index = row < 0 ? rows + row : row;
        const width = size / rows;
        if (index >= 0 && index < rows) data.fill(0, index * width, (index + 1) * width);
    }
    return data;
}

/** Tensors for every program parameter; trainable ones require gradients. */
export function initializeParameters(program, seed) {
    const next = stream(seed, 'init');
    const tensors = new Map();
    for (const param of program.params) {
        const trainable = param.enabled !== false && param.requiresGrad !== false;
        tensors.set(param.id, new Tensor([...param.shape], fillParameter(param, next), trainable));
    }
    return tensors;
}
