import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createEdge, createRepeatNode, createSuperNodeInstance } from '../src/domain/architecture/index.js';
import { ensureStandardTemplate } from '../src/domain/components/index.js';
import { createPhaseNode } from '../src/domain/training/index.js';
import { compileModel, MAX_PARAMETERS } from '../src/edu/trainer/compile.js';
import { createModel, forward } from '../src/edu/trainer/model.js';
import { crossEntropy } from '../src/edu/trainer/losses.js';
import { runPreflight } from '../src/edu/trainer/preflight.js';
import { Tape, Tensor } from '../src/edu/trainer/tensor.js';
import { mulberry32 } from '../src/edu/data/rng.js';
import { getDataset } from '../src/edu/data/registry.js';
import { chainModel, layer, memoryTemplates, trainingGraph } from './fixtures/edu-trainer-graphs.mjs';

const rng = mulberry32(11);
const batch = (rows, width) => new Tensor([rows, width], Float64Array.from({ length: rows * width }, () => rng() * 4 - 2));
const graphOf = (nodes, wires) => ({
    nodes: Object.fromEntries(nodes.map((node, row) => [node.id, { ...node, position: { x: 400, y: row * 130 } }])),
    edges: wires.map(([source, target, extra = {}], index) => createEdge({ id: `w${index}`, source, target, ...extra })),
});
const errorsOf = (result, code) => result.issues.filter((issue) => issue.severity === 'error' && (!code || issue.code === code));

test('the standard Linear component and a torch.nn.Linear layer give bit-identical parameters, outputs and gradients', () => {
    const layerModel = chainModel('(3,)', [['linear', 3, 4, { component: false }], ['relu'], ['linear', 4, 2, { component: false }]]);
    const componentModel = chainModel('(3,)', [['linear', 3, 4], ['relu'], ['linear', 4, 2]]);
    const a = compileModel(layerModel);
    const b = compileModel(componentModel);
    assert.ok(a.ok && b.ok, JSON.stringify([a.issues, b.issues]));
    assert.equal(a.paramCount, 3 * 4 + 4 + 4 * 2 + 2);
    assert.equal(b.paramCount, a.paramCount);
    assert.ok(b.program.ops.some((op) => op.kind === 'matmul') && b.program.ops.some((op) => op.kind === 'transpose'), 'the component runs as its op graph');
    const ma = createModel(a.program, 7);
    const mb = createModel(b.program, 7);
    const values = (model) => a.program.params.map((_, i) => [...model.params.get((model === ma ? a : b).program.params[i].id).data]);
    assert.deepEqual(values(mb), values(ma), 'same init stream, same draws: weight then bias, U(±1/√in)');
    const x = batch(5, 3);
    const ta = new Tape(), tb = new Tape();
    const ya = forward(ma, x, { training: true, tape: ta });
    const yb = forward(mb, x, { training: true, tape: tb });
    assert.deepEqual([...yb.data], [...ya.data]);
    const seed = Float64Array.from({ length: ya.size }, () => rng() - 0.5);
    ta.backward(ya, seed);
    tb.backward(yb, seed);
    a.program.params.forEach((param, i) => {
        assert.deepEqual([...mb.params.get(b.program.params[i].id).grad], [...ma.params.get(param.id).grad], `${param.origin.label} gradient`);
    });
});

test('the LayerNorm component matches a direct computation', () => {
    const { templates, store } = memoryTemplates();
    const templateId = ensureStandardTemplate('layernorm', store);
    const norm = createSuperNodeInstance({ id: 'ln', templateId, label: 'norm', position: { x: 400, y: 130 }, parameterValues: { normalized_shape: 4, eps: 1e-5, elementwise_affine: true } });
    const graph = graphOf([layer('in', 'deepflow.io.Input', { shape: '(4,)' }), norm, layer('out', 'deepflow.io.Output')], [['in', 'ln'], ['ln', 'out']]);
    const { ok, program, issues } = compileModel({ graph, templates });
    assert.ok(ok, JSON.stringify(issues));
    assert.equal(program.paramCount, 8, 'scale and bias, 4 each');
    const model = createModel(program, 1);
    const [gamma, beta] = program.params.map((param) => model.params.get(param.id));
    assert.deepEqual([...gamma.data], [1, 1, 1, 1], 'scale starts at ones');
    assert.deepEqual([...beta.data], [0, 0, 0, 0], 'bias starts at zeros');
    gamma.data.set([0.5, 2, -1, 1.5]);
    beta.data.set([0.1, -0.2, 0.3, 0]);
    const x = batch(3, 4);
    const y = forward(model, x).data;
    for (let r = 0; r < 3; r++) {
        const row = [...x.data.slice(r * 4, r * 4 + 4)];
        const mean = row.reduce((s, v) => s + v, 0) / 4;
        const variance = row.reduce((s, v) => s + (v - mean) ** 2, 0) / 4;
        row.forEach((v, d) => {
            const expected = ((v - mean) / Math.sqrt(variance + 1e-5)) * gamma.data[d] + beta.data[d];
            assert.ok(Math.abs(y[r * 4 + d] - expected) < 1e-12);
        });
    }
});

test('unsupported layers give unsupported-op naming the node, also inside components', () => {
    const conv = layer('conv', 'torch.nn.Conv2d', { in_channels: 1, out_channels: 4, kernel_size: 3 }, 'conv1');
    const graph = graphOf([
        layer('in', 'deepflow.io.Input', { shape: '(1, 8, 8)' }), conv, layer('flat', 'torch.nn.Flatten'),
        layer('fc', 'torch.nn.Linear', { in_features: 144, out_features: 10 }), layer('out', 'deepflow.io.Output'),
    ], [['in', 'conv'], ['conv', 'flat'], ['flat', 'fc'], ['fc', 'out']]);
    const result = compileModel({ graph, templates: {} });
    assert.equal(result.ok, false);
    const [issue] = errorsOf(result, 'unsupported-op');
    assert.equal(issue.nodeId, 'conv');
    assert.equal(issue.message, 'conv1 (Conv2d) is not supported by the browser trainer. Use Code → train.py to train this model in PyTorch.');
    assert.deepEqual(result.inputShape, [null, 1, 8, 8], 'shapes are still reported');

    const { templates, store } = memoryTemplates();
    const embeddingId = ensureStandardTemplate('embedding', store);
    const embedding = createSuperNodeInstance({ id: 'emb', templateId: embeddingId, label: 'token embedding', position: { x: 400, y: 130 }, parameterValues: { num_embeddings: 10, embedding_dim: 4, padding_idx: null } });
    const tokens = graphOf([layer('in', 'deepflow.io.Input', { shape: '(5,)', dtype: 'int64' }), embedding, layer('out', 'deepflow.io.Output')], [['in', 'emb'], ['emb', 'out']]);
    const nested = compileModel({ graph: tokens, templates });
    const [inner] = errorsOf(nested, 'unsupported-op');
    assert.equal(inner.nodeId, 'emb', 'nested issues point at the root node to open');
    assert.match(inner.message, /^token embedding › lookup rows \(Gather\) is not supported/);
});

test('a Linear on images without Flatten, and other rank limits, explain the fix', () => {
    const graph = graphOf([
        layer('in', 'deepflow.io.Input', { shape: '(1, 8, 8)' }), layer('fc', 'torch.nn.Linear', { in_features: 8, out_features: 10 }, 'classifier'), layer('out', 'deepflow.io.Output'),
    ], [['in', 'fc'], ['fc', 'out']]);
    const [issue] = errorsOf(compileModel({ graph, templates: {} }), 'unsupported-op');
    assert.match(issue.message, /classifier receives \(B, 1, 8, 8\).*Add a Flatten block before it/);
    const partial = chainModel('(1, 8, 8)', [['layer', 'torch.nn.Flatten', { start_dim: 2, end_dim: -1 }]]);
    assert.match(errorsOf(compileModel(partial), 'unsupported-op')[0].message, /start_dim 1, end_dim −1/);
    const softmaxDim = chainModel('(4,)', [['layer', 'torch.nn.Softmax', { dim: 0 }]]);
    assert.match(errorsOf(compileModel(softmaxDim), 'unsupported-op')[0].message, /last axis/);
});

test('sequential repeats unroll with their own parameters; parallel repeats are rejected', () => {
    const { templates } = memoryTemplates();
    templates.block = {
        id: 'block', name: 'Block', description: '', freeParameters: [{ name: 'width', type: 'int', default: 4 }],
        graph: graphOf([
            layer('tin', 'deepflow.template.Input'), layer('fc', 'torch.nn.Linear', { in_features: '$width', out_features: '$width' }),
            layer('act', 'torch.nn.Tanh'), layer('tout', 'deepflow.template.Output'),
        ], [['tin', 'fc'], ['fc', 'act'], ['act', 'tout']]),
    };
    const repeat = createRepeatNode({ id: 'stack', templateId: 'block', label: 'blocks', position: { x: 400, y: 130 }, count: 3, mode: 'sequential', bindings: { width: 4 } });
    const graph = graphOf([layer('in', 'deepflow.io.Input', { shape: '(4,)' }), repeat, layer('out', 'deepflow.io.Output')], [['in', 'stack'], ['stack', 'out']]);
    const result = compileModel({ graph, templates });
    assert.ok(result.ok, JSON.stringify(result.issues));
    assert.equal(result.program.params.length, 6, 'three copies, each with its own weight and bias');
    assert.equal(new Set(result.program.params.map((param) => param.id)).size, 6);
    assert.equal(result.paramCount, 3 * (16 + 4));
    assert.ok(result.program.params.every((param) => param.origin.rootNodeId === 'stack'));

    const parallel = createRepeatNode({ id: 'heads', templateId: 'block', label: 'heads', position: { x: 400, y: 130 }, count: 2, mode: 'parallel', bindings: { width: 4 } });
    const parallelGraph = graphOf([layer('in', 'deepflow.io.Input', { shape: '(4,)' }), parallel, layer('out', 'deepflow.io.Output')], [['in', 'heads'], ['heads', 'out']]);
    const [issue] = errorsOf(compileModel({ graph: parallelGraph, templates }), 'unsupported-op');
    assert.match(issue.message, /heads \(parallel repeat\) is not supported by the browser trainer/);
});

test('edge gradient operators: detach stops the gradient, reverse multiplies it by −λ', () => {
    const build = (gradient) => graphOf([
        layer('in', 'deepflow.io.Input', { shape: '(2,)' }), layer('a', 'torch.nn.Linear', { in_features: 2, out_features: 3 }),
        layer('b', 'torch.nn.Linear', { in_features: 3, out_features: 1 }), layer('out', 'deepflow.io.Output'),
    ], [['in', 'a'], ['a', 'b', gradient], ['b', 'out']]);
    const gradients = (gradient) => {
        const { program } = compileModel({ graph: build(gradient), templates: {} });
        const model = createModel(program, 3);
        const tape = new Tape();
        const out = forward(model, new Tensor([2, 2], Float64Array.of(1, -1, 0.5, 2)), { training: true, tape });
        tape.backward(out);
        return program.params.map((param) => model.params.get(param.id).grad);
    };
    const plain = gradients({});
    const detached = gradients({ gradientOp: 'detach' });
    assert.equal(detached[0], null, 'nothing reaches the first Linear');
    assert.deepEqual([...detached[2]], [...plain[2]], 'the second Linear still trains');
    const reversed = gradients({ gradientOp: 'reverse', gradientOpLambda: 2 });
    plain[0].forEach((value, i) => assert.ok(Math.abs(reversed[0][i] + 2 * value) < 1e-15));
});

test('conversion errors: invalid graphs, inputs, outputs, disconnected models and size limits', () => {
    const mismatch = chainModel('(2,)', [['linear', 3, 1, { component: false }]]);
    const invalid = compileModel(mismatch);
    assert.equal(errorsOf(invalid, 'invalid-graph').length, 1);
    assert.match(errorsOf(invalid)[0].validatorMessage, /Linear in_features=3 but input has 2 features/);
    assert.match(errorsOf(invalid)[0].message, /^\S.* expects 3 input features but receives 2\.$/, 'the lesson panel\'s friendly rewrite, as plain text');
    assert.doesNotMatch(errorsOf(invalid)[0].message, /\*\*|in_features=/);
    const inComponent = compileModel(chainModel('(2,)', [['linear', 3, 1]]));
    assert.match(errorsOf(inComponent, 'invalid-graph')[0].message, /expects 3 input features but receives 2\./, 'not "Template … has 1 internal error"');

    const two = chainModel('(2,)', [['linear', 2, 1]]);
    two.graph.nodes.in2 = { ...two.graph.nodes.in, id: 'in2' };
    assert.equal(errorsOf(compileModel(two), 'multi-input').length, 1);
    const none = { nodes: { in: two.graph.nodes.in }, edges: [] };
    assert.match(errorsOf(compileModel({ graph: none, templates: {} }), 'multi-output')[0].message, /Add an Output block/);

    const loose = graphOf([layer('in', 'deepflow.io.Input', { shape: '(2,)' }), layer('out', 'deepflow.io.Output')], []);
    assert.match(errorsOf(compileModel({ graph: loose, templates: {} }), 'invalid-graph')[0].message, /Connect the Output block/);
    const blank = chainModel('', [['linear', 2, 1]]);
    assert.equal(errorsOf(compileModel(blank), 'input-mismatch').length, 1, 'an Input without a shape cannot train');

    const huge = chainModel('(1500,)', [['linear', 1500, 1500, { component: false }]]);
    const big = compileModel(huge);
    assert.match(errorsOf(big, 'too-large')[0].message, /2,251,500 parameters/);
    assert.ok(2_251_500 > MAX_PARAMETERS);
});

test('LogSoftmax followed by cross-entropy equals cross-entropy on the logits', () => {
    const { ok, program } = compileModel(chainModel('(3,)', [['linear', 3, 4], ['logsoftmax']]));
    assert.ok(ok);
    const plainProgram = compileModel(chainModel('(3,)', [['linear', 3, 4]])).program;
    const x = batch(6, 3);
    const targets = Int32Array.of(0, 1, 2, 3, 1, 0);
    const withLog = forward(createModel(program, 2), x);
    const logits = forward(createModel(plainProgram, 2), x);
    const a = crossEntropy(withLog, targets);
    const b = crossEntropy(logits, targets);
    assert.ok(Math.abs(a.loss - b.loss) < 1e-12);
    a.grad.forEach((value, i) => assert.ok(Math.abs(value - b.grad[i]) < 1e-12));
});

test('preflight explains dataset, input, output and objective problems', () => {
    const iris = getDataset('iris');
    const blobsModel = chainModel('(2,)', [['linear', 2, 1]]);
    const training = trainingGraph({ loss: 'cross_entropy', lr: 0.03 });
    const codes = (result) => result.errors.map((issue) => issue.code);

    assert.deepEqual(codes(runPreflight({ ...blobsModel, training, dataset: null })), ['no-dataset']);
    const mnistLike = { id: 'mnist', name: 'MNIST', training: 'export', input: { shape: '(1, 28, 28)' }, target: { kind: 'classes', size: 10, classes: [] } };
    assert.equal(runPreflight({ ...blobsModel, training, dataset: mnistLike }).errors[0].message,
        'MNIST is too large to train in the browser. Open Code → train.py and run it in Colab.');

    const wrongInput = runPreflight({ ...blobsModel, training, dataset: iris });
    assert.equal(wrongInput.errors.find((issue) => issue.code === 'input-mismatch').message,
        'Your Input is (2,), but Iris has 4 features: use Data → Use this dataset.');
    const oneOutput = runPreflight({ ...chainModel('(4,)', [['linear', 4, 1]]), training, dataset: iris });
    assert.equal(oneOutput.errors.find((issue) => issue.code === 'loss-target-mismatch').message,
        "Iris has 3 classes: set the last Linear's Output to 3 or use the Data tab's Use this dataset.");
    const bce = runPreflight({ ...chainModel('(4,)', [['linear', 4, 3]]), training: trainingGraph({ loss: 'bce' }), dataset: iris });
    assert.match(bce.errors[0].message, /BCE is for two classes, but Iris has 3/);

    const ok = runPreflight({ ...chainModel('(4,)', [['linear', 4, 3]]), training, dataset: iris });
    assert.equal(ok.ok, true);
    assert.equal(ok.paramCount, 15);
    assert.equal(ok.objective.loss.type, 'cross_entropy');

    const softmax = runPreflight({ ...chainModel('(4,)', [['linear', 4, 3], ['softmax']]), training, dataset: iris });
    assert.equal(softmax.ok, true, 'warnings do not block training');
    assert.deepEqual(softmax.warnings.map((issue) => issue.code), ['softmax-before-ce']);
    const defaults = runPreflight({ ...chainModel('(4,)', [['linear', 4, 3]]), training: { nodes: {}, edges: [] }, dataset: iris });
    assert.equal(defaults.warnings[0].message, 'No optimizer/loss in the Training tab: using Adam (lr 0.01) and Cross-Entropy.');
    assert.equal(defaults.objective.optimizer.hyperparameters.learning_rate, 0.01);

    const phases = trainingGraph({ loss: 'cross_entropy' });
    phases.nodes.phase = createPhaseNode({ id: 'phase', position: { x: 0, y: 0 } });
    assert.ok(codes(runPreflight({ ...chainModel('(4,)', [['linear', 4, 3]]), training: phases, dataset: iris })).includes('multi-phase'));
    const regression = runPreflight({ ...chainModel('(2,)', [['linear', 2, 1]]), training: trainingGraph({ loss: 'bce' }), dataset: getDataset('plane') });
    assert.match(regression.errors[0].message, /Plane \(regression\) is a regression dataset/);
});
