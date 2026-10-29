/**
 * Runs a compiled program (compile.js) on batches. Parameters come from init.js; the forward pass records on a Tape
 * in training mode and runs without one in evaluation mode (dropout off).
 *
 * The execution plan fuses MatMul(a, Transpose(b)) into one a·bᵀ kernel when the transposed value has no other use.
 * The kernels sum in the same order, so results are bit-identical to the unfused graph.
 */
import { initializeParameters } from './init.js';
import * as ops from './ops.js';

function buildPlan(program) {
    const consumers = new Map();
    for (const op of program.ops) for (const input of op.inputs) consumers.set(input, (consumers.get(input) ?? 0) + 1);
    const byId = new Map(program.ops.map((op) => [op.id, op]));
    const fusedTranspose = new Set();
    const steps = [];
    const slot = new Map([[program.inputValue, 0]]);
    for (const op of program.ops) {
        let kind = op.kind;
        let inputs = op.inputs;
        if (kind === 'matmul') {
            const right = byId.get(inputs[1]);
            if (right?.kind === 'transpose' && consumers.get(right.id) === 1 && right.id !== program.outputValue) {
                kind = 'matmul_t';
                inputs = [inputs[0], right.inputs[0]];
                fusedTranspose.add(right.id);
            }
        }
        steps.push({ id: op.id, kind, inputs, attrs: op.attrs });
    }
    // Drop transposes that were fused into their MatMul, then resolve value slots.
    const kept = steps.filter((step) => !fusedTranspose.has(step.id));
    kept.forEach((step, index) => slot.set(step.id, index + 1));
    for (const step of kept) step.slots = step.inputs.map((id) => slot.get(id));
    return { steps: kept, output: slot.get(program.outputValue), size: kept.length + 1 };
}

/** A model instance: the program, its parameter tensors (Map id → Tensor) and the execution plan. */
export function createModel(program, seed = 1) {
    return { program, params: initializeParameters(program, seed), plan: buildPlan(program) };
}

function runStep(step, inputs, model, training, tape, dropoutNext) {
    const [x, y] = inputs;
    const a = step.attrs ?? {};
    switch (step.kind) {
        case 'linear': return ops.linear(x, model.params.get(a.weight), a.bias ? model.params.get(a.bias) : null, tape);
        case 'param': return model.params.get(a.paramId);
        case 'relu': return ops.relu(x, tape);
        case 'leakyrelu': return ops.leakyRelu(x, a.slope, tape);
        case 'tanh': return ops.tanh(x, tape);
        case 'sigmoid': return ops.sigmoid(x, tape);
        case 'gelu': return ops.gelu(x, a.approximate, tape);
        case 'softmax': return ops.softmax(x, tape);
        case 'logsoftmax': return ops.logSoftmax(x, tape);
        case 'dropout': return ops.dropout(x, a.p, dropoutNext, training, tape);
        case 'flatten': return ops.flatten(x, tape);
        case 'transpose': return ops.transpose(x, tape);
        case 'matmul': return ops.matmul(x, y, tape);
        case 'matmul_t': return ops.matmulTransposed(x, y, tape);
        case 'add': return ops.add(inputs, tape);
        case 'multiply': return ops.multiply(inputs, tape);
        case 'subtract': return ops.subtract(x, y, tape);
        case 'scale': return ops.scale(x, a.factor, tape);
        case 'addscalar': return ops.addScalar(x, a.value, tape);
        case 'pow': return ops.pow(x, a.exponent, tape);
        case 'exp': return ops.exp(x, tape);
        case 'rsqrt': return ops.rsqrt(x, tape);
        case 'mean': return ops.meanLast(x, a.keepdim, tape);
        case 'stopgrad': return ops.stopGradient(x);
        case 'reversegrad': return ops.reverseGradient(x, a.lambda, tape);
        default: throw new Error(`Unknown op "${step.kind}"`);
    }
}

/**
 * Forward pass. `x` is a Tensor of shape (B, …input shape). Training mode needs `tape` (to backpropagate) and
 * `dropoutNext` (the run's 'dropout' stream).
 */
export function forward(model, x, { training = false, tape = null, dropoutNext = null } = {}) {
    const { steps, output, size } = model.plan;
    const values = new Array(size);
    values[0] = x;
    for (let s = 0; s < steps.length; s++) {
        const step = steps[s];
        values[s + 1] = runStep(step, step.slots.map((index) => values[index]), model, training, tape, dropoutNext);
    }
    return values[output];
}
