/**
 * Module 2 — Going non-linear (docs/edu/SPEC.md §4.2). Productive failure first (one neuron, then two stacked Linear
 * blocks on a circle), then a ReLU, the minimal hidden width, a look inside Linear and a "your turn" XOR build.
 * Text: text/en/m2.js.
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const TRAIN = { do: 'train' };
const linear = (inFeatures, outFeatures) => ({ block: 'component:linear', params: { in_features: inFeatures, out_features: outFeatures } });
const RELU = { block: 'torch.nn.ReLU' };
const CIRCLE = { n: 300, noise: 0, seed: 2 };
const TRAINER = { epochs: 50, batchSize: 32, seed: 2 };
const objective = (h, output) => [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss('bce', output)];

export default {
    id: 'm2',
    version: 1,
    order: 2,
    minutes: 20,
    datasets: ['circle', 'xor'],
    recommendedAfter: ['m1'],
    objectives: ['m2.lo1', 'm2.lo2', 'm2.lo3', 'm2.lo4'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const nodes = [h.input('in', [2], { label: 'features' }), h.linear('lin1', 2, 1, { label: 'neuron' }), h.output('out', { label: 'logit' })];
                return h.snapshot({ nodes, edges: h.chain(...nodes), training: objective(h, nodes[2]), data: h.data('circle', CIRCLE), trainer: h.trainer(TRAINER) });
            },
        },
        twoLinear: {
            kind: 'snapshot',
            build: (h) => {
                const nodes = [
                    h.input('in', [2], { label: 'features' }), h.linear('lin1', 2, 8, { label: 'Linear 1' }), h.linear('lin2', 8, 1, { label: 'Linear 2' }),
                    h.output('out', { label: 'logit' }),
                ];
                return h.snapshot({ nodes, edges: h.chain(...nodes), training: objective(h, nodes[3]), data: h.data('circle', CIRCLE), trainer: h.trainer(TRAINER) });
            },
        },
        xor: {
            kind: 'snapshot',
            build: (h) => {
                const input = h.input('in', [2], { label: 'features' });
                const output = h.output('out', { label: 'logit', row: 4 });
                return h.snapshot({ nodes: [input, output], edges: [], training: objective(h, output), data: h.data('xor', { n: 300, noise: 0, seed: 3 }), trainer: h.trainer(TRAINER) });
            },
        },
    },
    start: 'start',
    facts: {
        twoLinearParams: { value: 33, at: 's2', compute: (h) => h.total() },
    },
    warmup: [
        { id: 'm2.w1', objective: 'm1.lo2', options: ['five', 'one', 'two'], correct: 'five', misconceptions: { one: 'one-output-multiclass', two: 'one-output-multiclass' } },
        { id: 'm2.w2', objective: 'm1.lo3', options: ['leak', 'nothing', 'slower'], correct: 'leak', misconceptions: { nothing: 'scaling-leak', slower: 'validation-for-speed' } },
    ],
    steps: [
        {
            id: 's1', kind: 'predict', subgoal: 'train', tab: 'architecture', target: ['trainer.run'],
            locks: { gate: 'train' }, embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            // The reveal waits for the learner's run and reads whether its boundary bends (engine/observe.js).
            predict: { id: 'm2.oneNeuron', kind: 'choice', options: ['yes', 'no'], correct: 'no', observe: 'bends', diagnose: [{ equals: 'yes', misconception: 'neuron-bends' }] },
            check: { all: [{ answered: 'm2.oneNeuron' }, { run: { dataset: 'circle', minEpochs: 30 } }] },
            solution: [answer('m2.oneNeuron', 'no'), TRAIN],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'model', tab: 'architecture', target: ['trainer.run'],
            setup: [{ do: 'loadFixture', id: 'twoLinear' }],
            locks: { gate: 'train' }, embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            predict: { id: 'm2.stacked', kind: 'choice', options: ['yes', 'no'], correct: 'no', observe: 'bends', diagnose: [{ equals: 'yes', misconception: 'stacked-linear-bends' }] },
            check: { all: [
                { answered: 'm2.stacked' },
                { run: { dataset: 'circle', minEpochs: 30, model: { all: [{ has: { type: 'Linear' }, count: 2 }, { not: { has: { type: 'Activation' } } }] } } },
            ] },
            solution: [answer('m2.stacked', 'no'), TRAIN],
        },
        {
            id: 's3', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['lesson.palette', 'ribbon.components'],
            // The chip splices ReLU into the Linear 1 → Linear 2 wire (review r1: wiring it by hand was a dead end).
            palette: [{ block: 'torch.nn.ReLU', count: 1, into: { from: { id: 'lin1' }, to: { id: 'lin2' } } }],
            check: { all: [
                { chain: [{ type: 'Input' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }] },
                { valid: true },
            ] },
            diagnose: [{
                when: { chain: [{ type: 'Activation' }, { type: 'Output' }] }, misconception: 'activation-at-output',
                example: [{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin2' }, to: { id: 'out' } }],
            }],
            solution: [{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }],
        },
        {
            id: 's4', kind: 'task', subgoal: 'train', tab: 'architecture', target: ['trainer.run'],
            embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            check: { run: { dataset: 'circle', minEpochs: 50, final: { valAccuracy: { gte: 0.9 } }, model: { has: { type: 'Activation' } } } },
            solution: [TRAIN],
        },
        {
            id: 's5', kind: 'predict', subgoal: 'model', tab: 'architecture', target: [{ node: { id: 'lin1' } }, 'field:out_features'],
            locks: { gate: 'train' }, embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            predict: {
                id: 'm2.width', kind: 'choice', options: ['one', 'two', 'three', 'eight'], correct: 'three', observe: 'hiddenWidth',
                diagnose: [
                    { equals: 'one', misconception: 'too-few-edges' },
                    { equals: 'two', misconception: 'too-few-edges' },
                    { equals: 'eight', misconception: 'more-units-needed' },
                ],
            },
            check: { all: [
                { answered: 'm2.width' },
                { run: { dataset: 'circle', minEpochs: 50, model: { all: [{ param: { node: { type: 'Linear', nth: 0 }, name: 'out_features', max: 3 } }, { valid: true }] } } },
            ] },
            diagnose: [{
                when: { all: [{ param: { node: { type: 'Linear', nth: 0 }, name: 'out_features', max: 3 } }, { not: { valid: true } }] }, misconception: 'mismatched-widths',
                example: [{ do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 3 }],
            }],
            solution: [
                answer('m2.width', 'three'),
                { do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 3 },
                { do: 'setParam', node: { id: 'lin2' }, name: 'in_features', value: 3 },
                TRAIN,
            ],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: [{ nodeOpen: { id: 'lin1' } }, 'ribbon.dimensions'],
            setup: [{ do: 'setUi', showDimensions: false }],
            predict: {
                id: 'm2.matmul', kind: 'choice', options: ['B3', 'B2', '32'], correct: 'B3',
                diagnose: [{ equals: 'B2', misconception: 'matmul-keeps-width' }, { equals: '32', misconception: 'missing-batch-axis' }],
            },
            check: { all: [{ level: { open: { type: 'Linear' } } }, { answered: 'm2.matmul' }, { ui: { showDimensions: true } }] },
            solution: [{ do: 'openLevel', path: [{ id: 'lin1' }] }, answer('m2.matmul', 'B3'), { do: 'setUi', showDimensions: true }],
        },
        {
            id: 's7', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['lesson.palette'],
            setup: [{ do: 'loadFixture', id: 'xor' }],
            palette: [
                // A new Linear takes the open end's width (2, then 8) and outputs 8 hidden units, not 768.
                { block: 'component:linear', count: 2, params: { out_features: 8 } },
                { block: 'torch.nn.ReLU', count: 1 },
                { block: 'torch.nn.Sigmoid', count: 1, distractor: 'sigmoid-before-bce' },
            ],
            embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            check: { all: [
                { chain: [{ type: 'Input' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }], allow: ['Dropout'] },
                { valid: true },
                { run: { dataset: 'xor', minEpochs: 50, final: { valAccuracy: { gte: 0.9 } } } },
            ] },
            diagnose: [
                {
                    when: { chain: [{ type: 'Sigmoid' }, { type: 'Output' }] }, misconception: 'sigmoid-before-bce',
                    example: [{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [linear(2, 8), RELU, linear(8, 1), { block: 'torch.nn.Sigmoid' }] }],
                },
                {
                    when: { chain: [{ type: 'Activation' }, { type: 'Output' }] }, misconception: 'activation-at-output',
                    example: [{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [linear(2, 8), RELU, linear(8, 1), RELU] }],
                },
            ],
            solution: [{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [linear(2, 8), RELU, linear(8, 1)] }, TRAIN],
        },
        {
            id: 's8', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:model'],
            codeQuotes: { tab: 'model', text: ['def forward', 'torch.matmul(', 'torch.add(', 'self.ReLU('] },
            predict: { id: 'm2.forward', kind: 'choice', options: ['relu', 'sigmoid', 'matmul'], correct: 'relu', diagnose: [{ equals: 'matmul', misconception: 'stacked-linear-bends' }] },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'model' } }, { answered: 'm2.forward' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'model' }, answer('m2.forward', 'relu')],
        },
    ],
    quiz: [
        { id: 'm2.q1', objective: 'm2.lo1', options: ['compose', 'ignored', 'shared'], correct: 'compose', misconceptions: { ignored: 'second-linear-ignored', shared: 'stacked-share-weights' } },
        { id: 'm2.q2', objective: 'm2.lo2', options: ['relu', 'tanh', 'scale', 'sigmoid'], correct: 'scale', misconceptions: { relu: 'piecewise-linear-is-linear', tanh: 'nonlinear-squash', sigmoid: 'nonlinear-squash' } },
        { id: 'm2.q3', objective: 'm2.lo3', options: ['edge', 'circle', 'point'], correct: 'edge', misconceptions: { circle: 'unit-draws-circle', point: 'unit-memorises-point' } },
        { id: 'm2.q4', objective: 'm2.lo4', options: ['error', 'trains', 'adds'], correct: 'error', misconceptions: { trains: 'mismatched-widths', adds: 'tool-fixes-shapes' } },
    ],
};
