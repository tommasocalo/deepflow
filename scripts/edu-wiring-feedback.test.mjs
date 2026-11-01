import assert from 'node:assert/strict';
import { test } from 'node:test';

// Wiring feedback (review r1: in M2 s3 a novice who wired ReLU into Linear 2 before deleting the old wire saw
// 'Template "Linear" has 1 internal error: Too many unhandled incoming connections …'). Friendly rules name the extra
// wire and the fix; blocks without wires are named; the M2 s3 chip splices ReLU into the wire; Linear chips start at
// the open end's width; the run check says what to do next about the live model.

const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { friendlyIssues, wiringHints, extraWire } = await import('../src/edu/lessons/engine/feedback.js');
const { createFixtureHelpers } = await import('../src/edu/lessons/engine/fixtures.js');
const { createCheckContext, evaluateCheck, evaluateStep } = await import('../src/edu/lessons/engine/checks.js');
const { chipParams, openEndWidth, spliceTarget, besideWire } = await import('../src/edu/lessons/engine/palette.js');
const { validate } = await import('../src/validation/index.js');
const { withGlobalParameters } = await import('../src/domain/expressions.js');
const { t } = await import('../src/edu/lessons/engine/i18n.js');
const { MODULES } = await import('../src/edu/lessons/modules/index.js');

const m2 = MODULES.find((module) => module.id === 'm2');
const s3 = m2.steps.find((step) => step.id === 's3');
const text = (issue) => t(issue.key, issue.params);

/** Input → Linear 1 → [ReLU] → Linear 2 → Output, as in M2 s3, plus `extra` edges. */
function m2Model(h, { relu = true, wired = true, extra = [] } = {}) {
    const input = h.input('in', [2], { label: 'features' });
    const lin1 = h.linear('lin1', 2, 8, { label: 'Linear 1' });
    const act = h.layer('relu', 'torch.nn.ReLU', {}, 'ReLU');
    const lin2 = h.linear('lin2', 8, 1, { label: 'Linear 2' });
    const output = h.output('out', { label: 'logit' });
    const nodes = relu ? [input, lin1, act, lin2, output] : [input, lin1, lin2, output];
    const edges = wired ? h.chain(...nodes) : h.chain(input, lin1, lin2, output);
    return { nodes, edges: [...edges, ...extra.map(([from, to]) => ({ id: `extra-${from}-${to}`, source: from, target: to }))] };
}

function contextOf(build, extra = {}) {
    const h = createFixtureHelpers();
    const { nodes, edges } = build(h);
    const snapshot = h.snapshot({ nodes, edges });
    const report = validate(snapshot.graph, { templates: snapshot.templates, scope: withGlobalParameters(snapshot.graph) });
    const ctx = createCheckContext({ graph: snapshot.graph, templates: snapshot.templates, module: m2, ...extra });
    return { ctx, report, graph: snapshot.graph, templates: snapshot.templates, issues: friendlyIssues(report, snapshot.graph, snapshot.templates) };
}

test('M2 s3: ReLU wired into Linear 2 before the old wire was deleted names that wire and the fix', () => {
    const { ctx, report, issues, graph } = contextOf((h) => m2Model(h, { extra: [['lin1', 'lin2']] }));
    assert.match(report.errors.lin2[0], /Too many unhandled incoming connections/, 'the validator reports the raw error');
    assert.equal(issues[0].key, 'fb.extra-wire');
    assert.equal(issues[0].nodeId, 'lin2');
    assert.equal(issues[0].edgeId, 'extra-lin1-lin2', 'the wire to delete is named, so "Show the wire" can select it');
    assert.equal(text(issues[0]), '**Linear 2** has 2 incoming wires but takes one input. The old wire from **Linear 1** skips **ReLU**: click that wire and press Delete.');
    assert.deepEqual(extraWire(graph, 'lin2').edge.id, 'extra-lin1-lin2');
    const result = evaluateStep(s3, ctx);
    assert.equal(result.pass, false);
    assert.equal(result.issue.key, 'fb.extra-wire', 'the status line shows the friendly rewrite, not the validator text');
});

test('two wires into a Linear from parallel branches list both; a named port with two wires names the port', () => {
    const parallel = contextOf((h) => {
        const input = h.input('in', [2], { label: 'features' });
        const a = h.linear('a', 2, 8, { label: 'left' });
        const b = h.linear('b', 2, 8, { label: 'right' });
        const lin2 = h.linear('lin2', 8, 1, { label: 'Linear 2' });
        const output = h.output('out');
        return { nodes: [input, a, b, lin2, output], edges: [...h.chain(input, a, lin2, output), h.edge(input, b), h.edge(b, lin2)] };
    });
    assert.equal(parallel.issues[0].key, 'fb.too-many-inputs');
    assert.equal(text(parallel.issues[0]), '**Linear 2** has 2 incoming wires, from **left** and **right**, but takes one input. Click the wire that does not belong and press Delete.');

    const ports = contextOf((h) => {
        const input = h.input('in', [4], { label: 'x' });
        const w = h.layer('w', 'deepflow.op.Parameter', { shape: [4, 4] }, 'weight');
        const matmul = h.layer('mm', 'deepflow.op.MatMul', {}, 'product');
        const output = h.output('out');
        return { nodes: [input, w, matmul, output], edges: [
            { id: 'e1', source: 'in', target: 'mm', targetHandle: 'a' }, { id: 'e2', source: 'w', target: 'mm', targetHandle: 'a' }, h.edge(matmul, output),
        ] };
    });
    const twice = ports.issues.find((issue) => issue.key === 'fb.port-twice');
    assert.ok(twice, JSON.stringify(ports.issues.map((issue) => issue.message)));
    assert.equal(text(twice), 'Two wires, from **x** and **weight**, end at the **a** input of **product**. Keep one: click the other and press Delete.');
});

test('a second wire into a ReLU or the Output validates in DeepFlow but is a lesson error', () => {
    const relu = contextOf((h) => m2Model(h, { extra: [['in', 'relu']] }));
    assert.equal(relu.report.ok, true, 'DeepFlow quietly uses the first wire');
    const issue = relu.issues.find((item) => item.structural);
    assert.equal(issue?.key, 'fb.extra-wire');
    assert.equal(text(issue), '**ReLU** has 2 incoming wires but takes one input. The old wire from **features** skips **Linear 1**: click that wire and press Delete.');
    assert.equal(evaluateCheck({ valid: true }, relu.ctx).pass, false, 'the valid check fails on it');
    assert.equal(evaluateStep(s3, relu.ctx).issue?.key, 'fb.extra-wire');

    const output = contextOf((h) => m2Model(h, { extra: [['relu', 'out']] }));
    assert.equal(output.issues.find((item) => item.structural)?.nodeId, 'out');
    const add = contextOf((h) => {
        const input = h.input('in', [8], { label: 'x' });
        const lin = h.linear('lin', 8, 8, { label: 'F' });
        const sum = h.layer('add', 'deepflow.op.Add', {}, 'Add');
        const out = h.output('out');
        return { nodes: [input, lin, sum, out], edges: [...h.chain(input, lin, sum, out), h.edge(input, sum)] };
    });
    assert.deepEqual(add.issues, [], 'Add takes two inputs (a skip connection is not a mistake)');
});

test('a block on the canvas without wires is named after the check detail', () => {
    const loose = contextOf((h) => m2Model(h, { wired: false }));
    assert.deepEqual(wiringHints(loose.report, loose.graph, loose.templates).map((hint) => [hint.nodeId, hint.key]), [['relu', 'fb.unwired']]);
    const result = evaluateStep(s3, loose.ctx);
    assert.equal(result.issue, null);
    assert.equal(result.hint.key, 'fb.unwired');
    assert.equal(`${t(result.detail.key, result.detail.params)} ${t(result.hint.key, result.hint.params)}`, 'After **Linear 1**, connect an **activation (such as ReLU)**. **ReLU** has no wires yet.');
    const half = contextOf((h) => m2Model(h, { wired: false, extra: [['lin1', 'relu']] }));
    assert.equal(wiringHints(half.report, half.graph, half.templates)[0].key, 'fb.unwired-out');
});

test('the M2 s3 chip splices ReLU into the Linear 1 → Linear 2 wire while it exists, else goes beside it', async () => {
    const item = s3.palette[0];
    assert.deepEqual(item.into, { from: { id: 'lin1' }, to: { id: 'lin2' } });
    const start = contextOf((h) => m2Model(h, { relu: false }));
    const target = spliceTarget(item, start.graph, start.templates);
    assert.deepEqual([target.from.id, target.to.id], ['lin1', 'lin2']);
    const rewired = contextOf((h) => {
        const model = m2Model(h, { relu: false });
        return { nodes: model.nodes, edges: model.edges.filter((edge) => !(edge.source === 'lin1' && edge.target === 'lin2')) };
    });
    assert.equal(spliceTarget(item, rewired.graph, rewired.templates), null, 'no wire, no splice');
    const place = besideWire(item, rewired.graph, rewired.templates);
    const [lin1, lin2] = [rewired.graph.nodes.lin1.position, rewired.graph.nodes.lin2.position];
    assert.ok(place.x > Math.max(lin1.x, lin2.x) && place.y > lin1.y && place.y < lin2.y, 'beside the wire, between the two blocks');
});

test('a Linear chip starts at the width of the open end (2 in XOR, then 8), not 768', () => {
    const s7 = m2.steps.find((step) => step.id === 's7');
    const chip = s7.palette.find((item) => item.block === 'component:linear');
    const xor = contextOf((h) => ({ nodes: [h.input('in', [2], { label: 'features' }), h.output('out', { label: 'logit', row: 4 })], edges: [] }));
    assert.equal(openEndWidth(xor.graph, xor.templates), 2);
    assert.deepEqual(chipParams(chip, xor.graph, xor.templates), { out_features: 8, in_features: 2 });
    const first = contextOf((h) => {
        const input = h.input('in', [2], { label: 'features' });
        const lin = h.linear('lin1', 2, 8, { label: 'Linear' });
        return { nodes: [input, lin, h.output('out', { row: 4 })], edges: h.chain(input, lin) };
    });
    assert.deepEqual(chipParams(chip, first.graph, first.templates), { out_features: 8, in_features: 8 });
    assert.equal(chipParams({ block: 'torch.nn.ReLU' }, first.graph, first.templates), null, 'other chips keep their defaults');
    assert.deepEqual(chipParams({ block: 'component:linear' }, first.graph, first.templates), { in_features: 8, out_features: 8 });
});

test('the run check says what to do about the live model before "Press Train"', () => {
    const s5 = m2.steps.find((step) => step.id === 's5');
    const query = s5.check.all[1].run;
    const wide = contextOf((h) => m2Model(h), { trainer: { status: 'idle', runs: [], preflight: { ok: true, errors: [] } }, data: { datasetId: 'circle' } });
    const first = evaluateCheck({ run: query }, wide.ctx);
    assert.equal(first.detail.key, 'check.run.modelFirst', 'M2 s5 asks for the widths before a run');
    assert.match(t(first.detail.key, first.detail.params), /^Change the model first: Linear 1: Output is 8; this step needs at most 3\./);

    // After a run of the 8-unit model, the canvas now has 3 units: say that it changed, not "Output is 8".
    const trained = { id: 'r1', seq: 1, status: 'completed', datasetId: 'circle', settings: { epochs: 50 }, graph: wide.graph, templates: wide.templates, history: Array.from({ length: 51 }, (_, epoch) => ({ epoch, valAccuracy: 0.9, valLoss: 0.2 })) };
    const narrow = contextOf((h) => {
        const model = m2Model(h);
        model.nodes[1].parameterValues = { in_features: 2, out_features: 3 };
        model.nodes[3].parameterValues = { in_features: 3, out_features: 1 };
        return model;
    }, { trainer: { status: 'idle', runs: [trained], preflight: { ok: true, errors: [], datasetId: 'circle' } }, data: { datasetId: 'circle' } });
    assert.equal(evaluateCheck({ run: query }, narrow.ctx).detail.key, 'check.run.retrain');
    assert.equal(t('check.run.retrain'), 'You changed the model since the last run: press **Train** again.');

    // Nothing trained yet, and the trainer cannot start: its reason, not "Press Train".
    const blocked = contextOf((h) => m2Model(h), { trainer: { status: 'idle', runs: [], preflight: { ok: false, errors: [{ message: 'BCE needs 1 output score.' }], datasetId: 'xor' } }, data: { datasetId: 'xor' } });
    assert.equal(t(evaluateCheck({ run: { dataset: 'xor', minEpochs: 50 } }, blocked.ctx).detail.key, { message: 'BCE needs 1 output score.' }), 'Fix this before training: BCE needs 1 output score.');
});
