import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mulberry32 } from '../src/edu/data/rng.js';
import { Tape, Tensor } from '../src/edu/trainer/tensor.js';
import * as ops from '../src/edu/trainer/ops.js';
import { bceWithLogits, crossEntropy, l1, mse } from '../src/edu/trainer/losses.js';

// Finite-difference checks (docs/edu/SPEC.md §8.2): for an op f and fixed random weights w, the scalar
// L(x) = Σ w ⊙ f(x) has gradient backward(w). Central differences with h = 1e-6 in float64.
const H = 1e-6;
const TOLERANCE = 1e-6;
const rng = mulberry32(2024);
const random = (shape, { min = -1.5, max = 1.5, awayFromZero = 0 } = {}) => {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        let value = min + (max - min) * rng();
        if (awayFromZero && Math.abs(value) < awayFromZero) value = value < 0 ? value - awayFromZero : value + awayFromZero;
        data[i] = value;
    }
    return new Tensor([...shape], data, true);
};

function relativeError(analytic, numeric) {
    let worst = 0;
    for (let i = 0; i < analytic.length; i++) {
        const error = Math.abs(analytic[i] - numeric[i]) / Math.max(1e-6, Math.abs(analytic[i]) + Math.abs(numeric[i]));
        worst = Math.max(worst, error);
    }
    return worst;
}

/** f(inputs, tape) → Tensor. Checks the gradient of every input that requires one. */
function checkOp(name, inputs, f) {
    const probe = f(inputs.map((t) => new Tensor(t.shape, t.data, false)), null);
    const weights = Float64Array.from({ length: probe.size }, () => rng() * 2 - 1);
    const objective = () => {
        const out = f(inputs.map((t) => new Tensor(t.shape, t.data, false)), null);
        let sum = 0;
        for (let i = 0; i < out.size; i++) sum += weights[i] * out.data[i];
        return sum;
    };
    for (const input of inputs) input.grad = null;
    const tape = new Tape();
    const out = f(inputs, tape);
    tape.backward(out, weights);
    inputs.forEach((input, index) => {
        if (!input.requiresGrad) return;
        const numeric = new Float64Array(input.size);
        for (let i = 0; i < input.size; i++) {
            const saved = input.data[i];
            input.data[i] = saved + H;
            const plus = objective();
            input.data[i] = saved - H;
            const minus = objective();
            input.data[i] = saved;
            numeric[i] = (plus - minus) / (2 * H);
        }
        assert.ok(input.grad, `${name}: input ${index} received a gradient`);
        const error = relativeError(input.grad, numeric);
        assert.ok(error < TOLERANCE, `${name}: input ${index} relative error ${error}`);
    });
}

/** A loss l(outputs, targets) → { loss, grad }: compare grad with central differences of loss. */
function checkLoss(name, outputs, loss) {
    const { grad } = loss(outputs);
    const numeric = new Float64Array(outputs.size);
    for (let i = 0; i < outputs.size; i++) {
        const saved = outputs.data[i];
        outputs.data[i] = saved + H;
        const plus = loss(outputs).loss;
        outputs.data[i] = saved - H;
        const minus = loss(outputs).loss;
        outputs.data[i] = saved;
        numeric[i] = (plus - minus) / (2 * H);
    }
    const error = relativeError(grad, numeric);
    assert.ok(error < TOLERANCE, `${name}: relative error ${error}`);
}

test('linear layer, MatMul, fused a·bᵀ and Transpose', () => {
    checkOp('linear', [random([5, 4]), random([3, 4]), random([3])], ([x, w, b], tape) => ops.linear(x, w, b, tape));
    checkOp('linear without bias', [random([2, 3]), random([4, 3])], ([x, w], tape) => ops.linear(x, w, null, tape));
    checkOp('matmul', [random([4, 3]), random([3, 5])], ([a, b], tape) => ops.matmul(a, b, tape));
    checkOp('matmul (k = 5, unrolled)', [random([3, 5]), random([5, 2])], ([a, b], tape) => ops.matmul(a, b, tape));
    checkOp('matmulTransposed', [random([4, 6]), random([3, 6])], ([a, b], tape) => ops.matmulTransposed(a, b, tape));
    checkOp('transpose', [random([3, 5])], ([x], tape) => ops.transpose(x, tape));
    checkOp('transpose feeding matmul', [random([4, 3]), random([5, 3])], ([a, w], tape) => ops.matmul(a, ops.transpose(w, tape), tape));
});

test('broadcasting Add, Multiply and Subtract reduce gradients over the broadcast axes', () => {
    checkOp('add (B, n) + (n,)', [random([4, 3]), random([3])], (xs, tape) => ops.add(xs, tape));
    checkOp('add (B, n) + (B, 1)', [random([4, 3]), random([4, 1])], (xs, tape) => ops.add(xs, tape));
    checkOp('add three inputs', [random([2, 3]), random([2, 3]), random([1, 3])], (xs, tape) => ops.add(xs, tape));
    checkOp('add (2, 1, 3) + (4, 1)', [random([2, 1, 3]), random([4, 1])], (xs, tape) => ops.add(xs, tape));
    checkOp('multiply (B, n) · (n,)', [random([3, 4]), random([4])], (xs, tape) => ops.multiply(xs, tape));
    checkOp('multiply (B, n) · (B, 1) · (n,)', [random([3, 2]), random([3, 1]), random([2])], (xs, tape) => ops.multiply(xs, tape));
    checkOp('subtract (B, n) − (B, 1)', [random([3, 4]), random([3, 1])], ([a, b], tape) => ops.subtract(a, b, tape));
    checkOp('the same tensor used twice', [random([2, 3])], ([x], tape) => ops.multiply([x, ops.add([x, x], tape)], tape));
});

test('element-wise ops', () => {
    checkOp('scale', [random([3, 3])], ([x], tape) => ops.scale(x, -2.5, tape));
    checkOp('addScalar', [random([3, 3])], ([x], tape) => ops.addScalar(x, 0.7, tape));
    checkOp('pow 2', [random([3, 3])], ([x], tape) => ops.pow(x, 2, tape));
    checkOp('pow 3', [random([2, 3])], ([x], tape) => ops.pow(x, 3, tape));
    checkOp('exp', [random([3, 3])], ([x], tape) => ops.exp(x, tape));
    checkOp('rsqrt', [random([3, 3], { min: 0.2, max: 3 })], ([x], tape) => ops.rsqrt(x, tape));
    checkOp('relu', [random([4, 4], { awayFromZero: 0.05 })], ([x], tape) => ops.relu(x, tape));
    checkOp('leakyRelu', [random([4, 4], { awayFromZero: 0.05 })], ([x], tape) => ops.leakyRelu(x, 0.1, tape));
    checkOp('tanh', [random([4, 4])], ([x], tape) => ops.tanh(x, tape));
    checkOp('sigmoid', [random([4, 4], { min: -6, max: 6 })], ([x], tape) => ops.sigmoid(x, tape));
    checkOp('gelu none (A&S erf)', [random([4, 4], { min: -3, max: 3, awayFromZero: 0.01 })], ([x], tape) => ops.gelu(x, 'none', tape));
    checkOp('gelu tanh', [random([4, 4], { min: -3, max: 3 })], ([x], tape) => ops.gelu(x, 'tanh', tape));
});

test('GELU none stays within ½·|x|·1.5e-7 of x·Φ(x) (the A&S erf bound) and the tanh form within 1e-3', () => {
    // Φ from a high-precision series, for comparison only.
    const phi = (x) => {
        let term = x, sum = x;
        for (let n = 1; n < 80; n++) { term *= -x * x / 2 / n; sum += term / (2 * n + 1); }
        return 0.5 + sum / Math.sqrt(2 * Math.PI);
    };
    for (const x of [-3, -1.2, -0.3, 0.4, 1, 2.5]) {
        const exact = x * phi(x);
        const value = ops.gelu(new Tensor([1], Float64Array.of(x)), 'none', null).data[0];
        assert.ok(Math.abs(value - exact) <= 0.5 * Math.abs(x) * 1.5e-7 + 1e-12, `gelu(${x})`);
        const approx = ops.gelu(new Tensor([1], Float64Array.of(x)), 'tanh', null).data[0];
        assert.ok(Math.abs(approx - exact) < 1e-3, `gelu tanh(${x})`);
    }
});

test('reductions and normalisations over the last axis', () => {
    checkOp('mean keepdim', [random([3, 5])], ([x], tape) => ops.meanLast(x, true, tape));
    checkOp('mean no keepdim', [random([3, 5])], ([x], tape) => ops.meanLast(x, false, tape));
    checkOp('softmax', [random([3, 4], { min: -3, max: 3 })], ([x], tape) => ops.softmax(x, tape));
    checkOp('logSoftmax', [random([3, 4], { min: -3, max: 3 })], ([x], tape) => ops.logSoftmax(x, tape));
    const x = random([2, 3]);
    const sm = ops.softmax(x, null).data;
    assert.ok(Math.abs(sm[0] + sm[1] + sm[2] - 1) < 1e-15);
    const ls = ops.logSoftmax(x, null).data;
    for (let i = 0; i < 6; i++) assert.ok(Math.abs(Math.exp(ls[i]) - sm[i]) < 1e-15);
});

test('Flatten, Dropout with a fixed mask and gradient routing', () => {
    checkOp('flatten', [random([2, 1, 3, 2])], ([x], tape) => ops.linear(ops.flatten(x, tape), new Tensor([2, 6], Float64Array.from({ length: 12 }, (_, i) => (i % 5) / 4 - 0.5)), null, tape));
    const maskStream = () => { const next = mulberry32(5); return () => next(); };
    // Each evaluation recreates the same stream, so the mask is fixed while differences are taken.
    checkOp('dropout', [random([4, 5])], ([x], tape) => ops.dropout(x, 0.4, maskStream(), true, tape));

    const x = random([2, 2]);
    const tape = new Tape();
    const out = ops.scale(ops.reverseGradient(x, 0.5, tape), 3, tape);
    tape.backward(out);
    assert.deepEqual([...x.grad], [-1.5, -1.5, -1.5, -1.5], 'reverse multiplies the gradient by −λ');
    const y = random([2, 2]);
    const tape2 = new Tape();
    const stopped = ops.scale(ops.stopGradient(y), 2, tape2);
    tape2.backward(stopped);
    assert.equal(y.grad, null, 'stop / detach cut the gradient');
    const evalOut = ops.dropout(y, 0.5, mulberry32(1), false, null);
    assert.equal(evalOut, y, 'dropout is the identity in evaluation mode');
    const trainOut = ops.dropout(new Tensor([1, 1000], new Float64Array(1000).fill(1)), 0.5, mulberry32(1), true, null).data;
    const kept = trainOut.filter((value) => value > 0);
    assert.ok(kept.every((value) => value === 2), 'kept values are scaled by 1 / (1 − p)');
    assert.ok(kept.length > 430 && kept.length < 570, `about half are kept: ${kept.length}`);
});

test('losses: cross-entropy (with label smoothing), BCE with logits, MSE and L1', () => {
    const targets = Int32Array.of(2, 0, 1, 2);
    checkLoss('cross_entropy', random([4, 3], { min: -3, max: 3 }), (out) => crossEntropy(out, targets));
    checkLoss('cross_entropy smoothing', random([4, 3], { min: -3, max: 3 }), (out) => crossEntropy(out, targets, { labelSmoothing: 0.1 }));
    checkLoss('cross_entropy sum', random([4, 3], { min: -3, max: 3 }), (out) => crossEntropy(out, targets, { reduction: 'sum' }));
    const binary = Int32Array.of(1, 0, 0, 1, 1);
    checkLoss('bce', random([5, 1], { min: -8, max: 8 }), (out) => bceWithLogits(out, binary));
    const values = Float64Array.of(0.5, -1, 2, 0.25);
    checkLoss('mse', random([4, 1], { min: -3, max: 3 }), (out) => mse(out, values));
    checkLoss('l1', new Tensor([4, 1], Float64Array.of(1.2, -0.3, 0.1, 2.9), true), (out) => l1(out, values));
});

test('loss values match their definitions', () => {
    const logits = new Tensor([2, 3], Float64Array.of(1, 2, 3, 0, 0, 0));
    const ce = crossEntropy(logits, Int32Array.of(2, 1)).loss;
    const lse = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3));
    assert.ok(Math.abs(ce - ((lse - 3) + Math.log(3)) / 2) < 1e-15);
    assert.ok(Math.abs(crossEntropy(new Tensor([1, 3], new Float64Array(3)), Int32Array.of(0)).loss - Math.log(3)) < 1e-15, 'uniform scores give ln C');
    const smoothed = crossEntropy(logits, Int32Array.of(2, 1), { labelSmoothing: 0.2 }).loss;
    const logp = [1 - lse, 2 - lse, 3 - lse];
    const expected = (0.8 * -logp[2] + 0.2 * -(logp[0] + logp[1] + logp[2]) / 3 + 0.8 * Math.log(3) + 0.2 * Math.log(3)) / 2;
    assert.ok(Math.abs(smoothed - expected) < 1e-15);
    const bce = bceWithLogits(new Tensor([2, 1], Float64Array.of(0, 30)), Int32Array.of(1, 0)).loss;
    assert.ok(Math.abs(bce - (Math.log(2) + 30) / 2) < 1e-12, 'BCE with logits is stable for large logits');
    assert.equal(mse(new Tensor([2, 1], Float64Array.of(1, 3)), Float64Array.of(0, 1)).loss, 2.5);
    assert.equal(l1(new Tensor([2, 1], Float64Array.of(1, 3)), Float64Array.of(0, 1)).loss, 1.5);
});
