/**
 * Module 3 — Shapes and parameters (docs/edu/SPEC.md §4.3). MNIST shapes with a batch axis, Flatten, an MLP built
 * from chips, counting its parameters, a planted shape bug, a small version trained on Digits 8×8 and the export
 * for the full MNIST. Text: text/en/m3.js.
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const TRAIN = { do: 'train' };
const linear = (inFeatures, outFeatures) => ({ block: 'component:linear', params: { in_features: inFeatures, out_features: outFeatures } });
const RELU = { block: 'torch.nn.ReLU' };
const TRAINER = { epochs: 15, batchSize: 32, seed: 3 };
const objective = (h, output) => [h.optimizer('Adam', { learning_rate: 0.001 }), h.loss('cross_entropy', output)];
const firstLinear = { type: 'Linear', nth: 0 };

export default {
    id: 'm3',
    version: 1,
    order: 3,
    minutes: 25,
    datasets: ['mnist', 'digits'],
    recommendedAfter: ['m2'],
    objectives: ['m3.lo1', 'm3.lo2', 'm3.lo3', 'm3.lo4'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const input = h.input('in', [2], { label: 'images' });
                const output = h.output('out', { label: 'logits', row: 5 });
                return h.snapshot({ nodes: [input, output], edges: [], training: objective(h, output), data: h.data(null), trainer: h.trainer(TRAINER) });
            },
        },
        bug: {
            kind: 'snapshot',
            build: (h) => {
                const nodes = [
                    h.input('in', [1, 28, 28], { label: 'images' }),
                    h.layer('flat', 'torch.nn.Flatten', {}, 'Flatten'),
                    h.linear('lin1', 748, 128, { label: 'Linear 1' }),
                    h.layer('relu', 'torch.nn.ReLU', {}, 'ReLU'),
                    h.linear('lin2', 128, 10, { label: 'Linear 2' }),
                    h.output('out', { label: 'logits' }),
                ];
                return h.snapshot({ nodes, edges: h.chain(...nodes), training: objective(h, nodes[5]), data: h.data('mnist'), trainer: h.trainer(TRAINER) });
            },
        },
    },
    start: 'start',
    facts: {
        mnistImages: { value: 70000, at: 's1', compute: (h) => h.dataset().size.train + h.dataset().size.test },
        flattenMnist: { value: 784, at: 's2', compute: (h) => h.shape({ type: 'Flatten' })[1] },
        lin1Params: { value: 100480, at: 's4', compute: (h) => h.params({ type: 'Linear', nth: 0 }) },
        lin2Params: { value: 1290, at: 's4', compute: (h) => h.params({ type: 'Linear', nth: 1 }) },
        mlpParams: { value: 101770, at: 's4', compute: (h) => h.total() },
        digitsImages: { value: 1797, at: 's6', compute: async (h) => (await h.prepared()).train.n + (await h.prepared()).val.n },
        flattenDigits: { value: 64, at: 's6', compute: (h) => h.shape({ type: 'Flatten' })[1] },
        digitsMlpParams: { value: 9610, at: 's6', compute: (h) => h.total() },
        initLoss10: { value: '2.30', at: 'start', compute: (h) => h.lnText(10) },
        mnistMean: { value: '0.1307', at: 's7', compute: (h) => h.dataset().normalization.mean[0].toFixed(4) },
        mnistStd: { value: '0.3081', at: 's7', compute: (h) => h.dataset().normalization.std[0].toFixed(4) },
    },
    warmup: [
        { id: 'm3.w1', objective: 'm2.lo1', options: ['line', 'circle', 'curve'], correct: 'line', misconceptions: { circle: 'stacked-linear-bends', curve: 'stacked-linear-bends' } },
        { id: 'm3.w2', objective: 'm2.lo4', options: ['sixteen', 'three', 'two'], correct: 'sixteen', misconceptions: { three: 'mismatched-widths', two: 'mismatched-widths' } },
    ],
    steps: [
        {
            id: 's1', kind: 'predict', subgoal: 'data', tab: 'data', target: ['data.dataset:mnist', 'data.use'],
            setup: [{ do: 'setUi', showDimensions: false }],
            predict: {
                id: 'm3.batchShape', kind: 'shape', expectedShape: [null, 1, 28, 28],
                diagnose: [
                    { equals: [null, 28, 28], misconception: 'missing-channel-axis' },
                    { equals: [1, 28, 28], misconception: 'missing-batch-axis' },
                    { equals: [null, 784], misconception: 'flattened-too-early' },
                    { equals: [null, 28, 28, 1], misconception: 'channels-last' },
                ],
            },
            check: { all: [{ answered: 'm3.batchShape' }, { dataset: 'mnist' }, { inputShape: [null, 1, 28, 28] }] },
            solution: [answer('m3.batchShape', [null, 1, 28, 28]), { do: 'useDataset', id: 'mnist', setOutput: false }],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: ['lesson.palette', 'ribbon.operations', 'ribbon.dimensions'],
            palette: [{ block: 'torch.nn.Flatten', count: 1 }],
            predict: {
                id: 'm3.flatten', kind: 'number', expected: 784,
                diagnose: [{ equals: 56, misconception: 'added-not-multiplied' }, { equals: 28, misconception: 'one-axis-only' }],
            },
            check: { all: [
                { answered: 'm3.flatten' },
                { chain: [{ type: 'Input' }, { type: 'Flatten' }] },
                { shapeAt: { node: { type: 'Flatten' }, equals: [null, 784] } },
                { ui: { showDimensions: true } },
            ] },
            solution: [answer('m3.flatten', 784), { do: 'insertAfter', block: 'torch.nn.Flatten', after: { type: 'Input' } }, { do: 'setUi', showDimensions: true }],
        },
        {
            id: 's3', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['lesson.palette'],
            palette: [{ block: 'component:linear', count: 2 }, { block: 'torch.nn.ReLU', count: 1 }],
            check: { all: [
                { chain: [{ type: 'Input' }, { type: 'Flatten' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }] },
                { param: { node: firstLinear, name: 'out_features', equals: 128 } },
                { valid: true },
                { outputShape: [null, 10] },
            ] },
            diagnose: [{
                when: { chain: [{ type: 'Activation' }, { type: 'Output' }] }, misconception: 'activation-at-output',
                example: [{ do: 'buildChain', from: { type: 'Flatten' }, to: { type: 'Output' }, blocks: [linear(784, 128), RELU, linear(128, 10), RELU] }],
            }],
            solution: [{ do: 'buildChain', from: { type: 'Flatten' }, to: { type: 'Output' }, blocks: [linear(784, 128), RELU, linear(128, 10)] }],
        },
        {
            id: 's4', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: [{ node: firstLinear }],
            embeds: ['paramCount'],
            predict: {
                id: 'm3.params1', kind: 'number', expected: 100480,
                diagnose: [{ equals: 100352, misconception: 'forgot-bias' }, { equals: 912, misconception: 'added-not-multiplied' }],
            },
            check: { all: [{ answered: 'm3.params1' }, { paramCount: { equals: 101770 } }] },
            solution: [answer('m3.params1', 100480)],
        },
        {
            id: 's5', kind: 'task', subgoal: 'shapes', tab: 'architecture', target: ['graph.checks', { node: { id: 'lin1' } }],
            setup: [{ do: 'loadFixture', id: 'bug' }],
            check: { all: [{ valid: true }, { param: { node: firstLinear, name: 'in_features', equals: 784 } }] },
            solution: [{ do: 'setParam', node: { id: 'lin1' }, name: 'in_features', value: 784 }],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'train', tab: 'architecture', target: [{ node: { id: 'lin1' } }, 'trainer.run'],
            setup: [{ do: 'useDataset', id: 'digits', setOutput: false }, { do: 'setUi', showDimensions: false }],
            locks: { gate: 'train' }, embeds: ['trainer'], reveal: { run: 'initLoss' },
            predict: {
                id: 'm3.digitsFlatten', kind: 'number', expected: 64,
                diagnose: [{ equals: 16, misconception: 'added-not-multiplied' }, { equals: 8, misconception: 'one-axis-only' }],
            },
            check: { all: [
                { answered: 'm3.digitsFlatten' },
                { param: { node: firstLinear, name: 'in_features', equals: 64 } },
                { run: { dataset: 'digits', minEpochs: 10, final: { valAccuracy: { gte: 0.9 } } } },
            ] },
            solution: [answer('m3.digitsFlatten', 64), { do: 'setParam', node: { id: 'lin1' }, name: 'in_features', value: 64 }, TRAIN],
        },
        {
            id: 's7', kind: 'predict', subgoal: 'export', target: ['tab.data', 'toolbar.code', 'code.tab:train'],
            codeQuotes: { tab: 'train', text: ['transforms.Normalize((0.1307,), (0.3081,))', 'datasets.MNIST('] },
            predict: {
                id: 'm3.normalize', kind: 'choice', options: ['standardize', 'binarize', 'resize'], correct: 'standardize',
                diagnose: [{ equals: 'binarize', misconception: 'normalize-changes-pixels' }, { equals: 'resize', misconception: 'normalize-changes-pixels' }],
            },
            check: { all: [{ dataset: 'mnist' }, { valid: true }, { ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm3.normalize' }] },
            solution: [
                { do: 'useDataset', id: 'mnist', setOutput: false },
                { do: 'setParam', node: { id: 'lin1' }, name: 'in_features', value: 784 },
                { do: 'setUi', codeOpen: true, codeTab: 'train' },
                answer('m3.normalize', 'standardize'),
            ],
        },
    ],
    quiz: [
        { id: 'm3.q1', objective: 'm3.lo1', options: ['batch', 'brightness', 'bias'], correct: 'batch', misconceptions: { brightness: 'batch-axis-meaning', bias: 'batch-axis-meaning' } },
        { id: 'm3.q2', objective: 'm3.lo2', options: ['right', 'sum', 'map', 'nobatch'], correct: 'right', misconceptions: { sum: 'added-not-multiplied', map: 'one-map-only', nobatch: 'missing-batch-axis' } },
        { id: 'm3.q3', objective: 'm3.lo3', options: ['right', 'weights', 'sum'], correct: 'right', misconceptions: { weights: 'forgot-bias', sum: 'added-not-multiplied' } },
        { id: 'm3.q4', objective: 'm3.lo4', options: ['uniform', 'always', 'knows'], correct: 'uniform', misconceptions: { always: 'loss-starts-fixed', knows: 'loss-is-error-rate' } },
    ],
};
