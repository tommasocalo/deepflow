import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createOptimizer, parseBetas, resolveHyperparameters } from '../src/edu/trainer/optimizers.js';
import { Tensor } from '../src/edu/trainer/tensor.js';
import { compileModel } from '../src/edu/trainer/compile.js';
import { createRun, drain, runSteps } from '../src/edu/trainer/run.js';
import { prepareDatasetSync } from '../src/edu/data/prepare.js';
import { getDataset } from '../src/edu/data/registry.js';
import { chainModel, objectiveOf, trainingGraph } from './fixtures/edu-trainer-graphs.mjs';

// f(p) = ½ (a·p₀² + b·p₁²) with ∇f = (a·p₀, b·p₁). Each optimizer runs 3 steps against an independent scalar
// re-statement of PyTorch's update rule (torch.optim docs), compared after every step.
const A = [1, 4];
const START = [1, -2];
const gradientOf = (p) => p.map((value, i) => A[i] * value);

function runOptimizer(type, hyperparameters) {
    const param = new Tensor([2], Float64Array.from(START), true);
    const optimizer = createOptimizer(type, hyperparameters, [param]);
    const trajectory = [];
    for (let step = 0; step < 3; step++) {
        param.grad = Float64Array.from(gradientOf([...param.data]));
        optimizer.step();
        trajectory.push([...param.data]);
    }
    return trajectory;
}

function assertTrajectory(actual, expected, name) {
    actual.forEach((point, step) => point.forEach((value, i) => {
        assert.ok(Math.abs(value - expected[step][i]) < 1e-12, `${name} step ${step + 1}, p${i}: ${value} vs ${expected[step][i]}`);
    }));
}

function referenceSGD({ lr, momentum = 0, weightDecay = 0, nesterov = false, dampening = 0 }) {
    let p = [...START];
    let buffer = null;
    const out = [];
    for (let t = 1; t <= 3; t++) {
        const g = gradientOf(p).map((value, i) => value + weightDecay * p[i]);
        let update = g;
        if (momentum !== 0) {
            buffer = buffer ? buffer.map((b, i) => momentum * b + (1 - dampening) * g[i]) : [...g];
            update = nesterov ? g.map((value, i) => value + momentum * buffer[i]) : buffer;
        }
        p = p.map((value, i) => value - lr * update[i]);
        out.push(p);
    }
    return out;
}

function referenceAdam({ lr, betas: [b1, b2], eps, weightDecay = 0, decoupled = false }) {
    let p = [...START];
    let m = [0, 0], v = [0, 0];
    const out = [];
    for (let t = 1; t <= 3; t++) {
        // The gradient comes from backward, before the step; AdamW then decays the weights it updates.
        const g = gradientOf(p).map((value, i) => (decoupled ? value : value + weightDecay * p[i]));
        if (decoupled) p = p.map((value) => value * (1 - lr * weightDecay));
        m = m.map((value, i) => b1 * value + (1 - b1) * g[i]);
        v = v.map((value, i) => b2 * value + (1 - b2) * g[i] * g[i]);
        const mHat = m.map((value) => value / (1 - b1 ** t));
        const vHat = v.map((value) => value / (1 - b2 ** t));
        p = p.map((value, i) => value - lr * mHat[i] / (Math.sqrt(vHat[i]) + eps));
        out.push(p);
    }
    return out;
}

function referenceRMSprop({ lr, alpha, eps, momentum = 0, weightDecay = 0 }) {
    let p = [...START];
    let v = [0, 0], buffer = [0, 0];
    const out = [];
    for (let t = 1; t <= 3; t++) {
        const g = gradientOf(p).map((value, i) => value + weightDecay * p[i]);
        v = v.map((value, i) => alpha * value + (1 - alpha) * g[i] * g[i]);
        const avg = v.map((value) => Math.sqrt(value) + eps);
        if (momentum > 0) {
            buffer = buffer.map((b, i) => momentum * b + g[i] / avg[i]);
            p = p.map((value, i) => value - lr * buffer[i]);
        } else {
            p = p.map((value, i) => value - lr * g[i] / avg[i]);
        }
        out.push(p);
    }
    return out;
}

test('SGD: plain, momentum (first step copies the gradient), Nesterov and weight decay', () => {
    assertTrajectory(runOptimizer('SGD', { learning_rate: 0.1, momentum: 0, weight_decay: 0 }), referenceSGD({ lr: 0.1 }), 'sgd');
    assertTrajectory(runOptimizer('SGD', { learning_rate: 0.1, momentum: 0.9, weight_decay: 0.01 }), referenceSGD({ lr: 0.1, momentum: 0.9, weightDecay: 0.01 }), 'momentum');
    assertTrajectory(runOptimizer('SGD', { learning_rate: 0.05, momentum: 0.9, nesterov: true }), referenceSGD({ lr: 0.05, momentum: 0.9, nesterov: true }), 'nesterov');
    assertTrajectory(runOptimizer('SGD', { learning_rate: 0.1, momentum: 0.5, dampening: 0.2 }), referenceSGD({ lr: 0.1, momentum: 0.5, dampening: 0.2 }), 'dampening');
    // By hand: first momentum step equals plain SGD: p = 1 − 0.1 · 1 = 0.9.
    assert.ok(Math.abs(runOptimizer('SGD', { learning_rate: 0.1, momentum: 0.9 })[0][0] - 0.9) < 1e-15);
});

test('Adam with bias correction and L2 weight decay', () => {
    const betas = [0.9, 0.999];
    assertTrajectory(runOptimizer('Adam', { learning_rate: 0.1, betas: '(0.9, 0.999)', eps: 1e-8, weight_decay: 0 }), referenceAdam({ lr: 0.1, betas, eps: 1e-8 }), 'adam');
    assertTrajectory(runOptimizer('Adam', { learning_rate: 0.1, betas: '(0.8, 0.99)', eps: 1e-8, weight_decay: 0.1 }), referenceAdam({ lr: 0.1, betas: [0.8, 0.99], eps: 1e-8, weightDecay: 0.1 }), 'adam l2');
    // Bias correction makes the first step lr · sign(g): 1 − 0.1 and −2 + 0.1.
    const [first] = runOptimizer('Adam', { learning_rate: 0.1 });
    assert.ok(Math.abs(first[0] - 0.9) < 1e-7 && Math.abs(first[1] + 1.9) < 1e-7);
});

test('AdamW decays the weights before the Adam step (decoupled)', () => {
    assertTrajectory(runOptimizer('AdamW', { learning_rate: 0.1, betas: '(0.9, 0.999)', eps: 1e-8, weight_decay: 0.1 }), referenceAdam({ lr: 0.1, betas: [0.9, 0.999], eps: 1e-8, weightDecay: 0.1, decoupled: true }), 'adamw');
    const adam = runOptimizer('Adam', { learning_rate: 0.1, weight_decay: 0.1 });
    const adamw = runOptimizer('AdamW', { learning_rate: 0.1, weight_decay: 0.1 });
    assert.notDeepEqual(adam, adamw, 'L2 inside the gradient differs from decoupled decay');
});

test('RMSprop with and without momentum', () => {
    assertTrajectory(runOptimizer('RMSprop', { learning_rate: 0.01, alpha: 0.99, momentum: 0 }), referenceRMSprop({ lr: 0.01, alpha: 0.99, eps: 1e-8 }), 'rmsprop');
    assertTrajectory(runOptimizer('RMSprop', { learning_rate: 0.01, alpha: 0.9, momentum: 0.9, weight_decay: 0.01 }), referenceRMSprop({ lr: 0.01, alpha: 0.9, eps: 1e-8, momentum: 0.9, weightDecay: 0.01 }), 'rmsprop momentum');
});

test('hyperparameters follow DeepFlow names and defaults; null gradients are skipped', () => {
    assert.deepEqual(parseBetas('(0.9, 0.999)'), [0.9, 0.999]);
    assert.deepEqual(parseBetas([0.5, 0.6]), [0.5, 0.6]);
    assert.deepEqual(parseBetas('nonsense'), [0.9, 0.999]);
    const sgd = resolveHyperparameters('SGD', {});
    assert.equal(sgd.lr, 0.01);
    assert.equal(sgd.momentum, 0.9, "DeepFlow's SGD default momentum");
    assert.equal(resolveHyperparameters('AdamW', {}).weightDecay, 1e-2);
    assert.equal(resolveHyperparameters('Adam', { learning_rate: '0.03' }).lr, 0.03, 'numeric strings are accepted');
    const untouched = new Tensor([2], Float64Array.of(1, 2), true);
    const optimizer = createOptimizer('AdamW', { learning_rate: 0.1, weight_decay: 0.5 }, [untouched]);
    optimizer.step();
    assert.deepEqual([...untouched.data], [1, 2], 'no gradient, no update (not even weight decay)');
    assert.throws(() => createOptimizer('LBFGS', {}, []), /not supported/);
});

test('a parameter group trains only the parameters of its root nodes', () => {
    const { graph, templates } = chainModel('(2,)', [['linear', 2, 4], ['relu'], ['linear', 4, 1]]);
    const { program } = compileModel({ graph, templates });
    const dataset = prepareDatasetSync(getDataset('blobs'), { n: 200, noise: 0.1, seed: 1 }, { val: 0.2, test: 0 });
    const training = trainingGraph({ loss: 'bce', lr: 0.05 });
    training.nodes.opt.parameterGroup = ['lin1'];
    const run = createRun({ program, dataset, objective: objectiveOf(training), settings: { epochs: 3, batchSize: 32, seed: 1 } });
    const before = new Map([...run.model.params].map(([id, tensor]) => [id, [...tensor.data]]));
    drain(runSteps(run));
    for (const param of program.params) {
        const changed = run.model.params.get(param.id).data.some((value, i) => value !== before.get(param.id)[i]);
        assert.equal(changed, param.origin.rootNodeId === 'lin1', `${param.origin.label} ${changed ? 'changed' : 'stayed'}`);
    }
    assert.equal(run.trainable.length, 2, 'weight and bias of lin1');
});
