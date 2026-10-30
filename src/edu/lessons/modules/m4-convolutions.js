/**
 * Module 4 — Convolutions (docs/edu/SPEC.md §4.4). Conv2d output shapes with and without padding, two convolution
 * blocks with pooling, Flatten into a classifier, weight sharing in the parameter count, and the Colab export for
 * Fashion-MNIST. The facts use Stage C's Conv2d scalar-kernel count (§6.11.4). Text: text/en/m4.js.
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const objective = (h, output) => [h.optimizer('Adam', { learning_rate: 0.001 }), h.loss('cross_entropy', output)];
const DATA = (h) => h.data('fashion-mnist');
const TRAINER = { epochs: 5, batchSize: 64, seed: 4 };
const conv = (inChannels, outChannels, padding = 1) => ({ block: 'torch.nn.Conv2d', params: { in_channels: inChannels, out_channels: outChannels, kernel_size: 3, padding } });
const POOL = { block: 'torch.nn.MaxPool2d', params: { kernel_size: 2 } };
const RELU = { block: 'torch.nn.ReLU' };
const valSize = (h) => Number(h.data().options?.valSize ?? h.dataset().options.valSize.default);

export default {
    id: 'm4',
    version: 1,
    order: 4,
    minutes: 20,
    datasets: ['fashion-mnist'],
    recommendedAfter: ['m3'],
    objectives: ['m4.lo1', 'm4.lo2', 'm4.lo3', 'm4.lo4'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const input = h.input('in', [1, 28, 28], { label: 'images' });
                const output = h.output('out', { label: 'logits', row: 2 });
                return h.snapshot({ nodes: [input, output], edges: [], training: objective(h, output), data: DATA(h), trainer: h.trainer(TRAINER), ui: { showDimensions: true } });
            },
        },
        conv1: {
            kind: 'snapshot',
            build: (h) => {
                const nodes = [
                    h.input('in', [1, 28, 28], { label: 'images' }),
                    h.layer('conv1', 'torch.nn.Conv2d', { in_channels: 1, out_channels: 8, kernel_size: 3, padding: 1, stride: 1 }, 'conv1'),
                    h.output('out', { label: 'logits' }),
                ];
                return h.snapshot({ nodes, edges: h.chain(...nodes), training: objective(h, nodes[2]), data: DATA(h), trainer: h.trainer(TRAINER), ui: { showDimensions: true } });
            },
        },
    },
    start: 'start',
    facts: {
        fmnistTrain: { value: 60000, at: 'start', compute: (h) => h.dataset().size.train },
        fmnistTest: { value: 10000, at: 'start', compute: (h) => h.dataset().size.test },
        mnistCnn: { value: '99.7%', source: 'literature', note: 'Fashion-MNIST README (Xiao, Rasul & Vollgraf 2017): "Convolutional nets can achieve 99.7% on MNIST".' },
        flattenCnn: { value: 784, at: 's5', compute: (h) => h.shape({ type: 'Flatten' })[1] },
        conv1Params: { value: 80, at: 's6', compute: (h) => h.params({ id: 'conv1' }) },
        conv2Params: { value: 1168, at: 's6', compute: (h) => h.params({ type: 'Conv2d', nth: 1 }) },
        fcParams: { value: 7850, at: 's6', compute: (h) => h.params({ type: 'Linear' }) },
        cnnParams: { value: 9098, at: 's6', compute: (h) => h.total() },
        mlpParams: { value: 101770, at: 's6', compute: (h) => h.fact('m3', 'mlpParams') },
        ratio: { value: 11, at: 's6', compute: (h) => Math.round(h.fact('m3', 'mlpParams') / h.fact('m4', 'cnnParams')) },
        fmnistTrainSplit: { value: 55000, at: 's7', compute: (h) => h.dataset().size.train - valSize(h) },
        fmnistVal: { value: 5000, at: 's7', compute: (h) => valSize(h) },
        benchmark: { value: '88–93%', source: 'literature', note: 'Fashion-MNIST README benchmark table: two-convolution networks report 0.876–0.925 test accuracy.' },
    },
    warmup: [
        { id: 'm4.w1', objective: 'm3.lo2', options: ['right', 'row', 'nobatch'], correct: 'right', misconceptions: { row: 'one-axis-only', nobatch: 'missing-batch-axis' } },
        { id: 'm4.w2', objective: 'm3.lo3', options: ['right', 'weights', 'sum'], correct: 'right', misconceptions: { weights: 'forgot-bias', sum: 'added-not-multiplied' } },
    ],
    steps: [
        {
            id: 's1', kind: 'read', subgoal: 'data', target: ['tab.data'],
            check: { all: [{ visited: 'data' }, { dataset: 'fashion-mnist' }, { inputShape: [null, 1, 28, 28] }] },
            solution: [{ do: 'setUi', tab: 'data' }],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: [{ node: { id: 'conv1' } }, 'ribbon.dimensions'],
            setup: [{ do: 'loadFixture', id: 'conv1' }, { do: 'setUi', showDimensions: false }],
            predict: {
                id: 'm4.conv1', kind: 'shape', expectedShape: [null, 8, 28, 28],
                diagnose: [
                    { equals: [null, 1, 28, 28], misconception: 'channels-unchanged' },
                    { equals: [null, 8, 26, 26], misconception: 'forgot-padding' },
                    { equals: [null, 8, 3, 3], misconception: 'kernel-is-output' },
                ],
            },
            check: { all: [{ answered: 'm4.conv1' }, { ui: { showDimensions: true } }] },
            solution: [answer('m4.conv1', [null, 8, 28, 28]), { do: 'setUi', showDimensions: true }],
        },
        {
            id: 's3', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: [{ node: { id: 'conv1' } }, 'field:padding'],
            setup: [{ do: 'setUi', showDimensions: false }],
            predict: {
                id: 'm4.noPad', kind: 'number', expected: 26,
                diagnose: [{ equals: 28, misconception: 'padding-no-effect' }, { equals: 25, misconception: 'minus-kernel' }],
            },
            check: { all: [{ answered: 'm4.noPad' }, { shapeAt: { node: { id: 'conv1' }, equals: [null, 8, 26, 26] } }, { ui: { showDimensions: true } }] },
            solution: [answer('m4.noPad', 26), { do: 'setParam', node: { id: 'conv1' }, name: 'padding', value: 0 }, { do: 'setUi', showDimensions: true }],
        },
        {
            id: 's4', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['lesson.palette', 'ribbon.components'],
            setup: [{ do: 'setParam', node: { id: 'conv1' }, name: 'padding', value: 1 }],
            palette: [{ block: 'torch.nn.ReLU', count: 2 }, { block: 'torch.nn.MaxPool2d', count: 2 }, { block: 'torch.nn.Conv2d', count: 1 }],
            check: { all: [
                { chain: [{ type: 'Input' }, { type: 'Conv2d' }, { type: 'Activation' }, { type: 'MaxPool2d' }, { type: 'Conv2d' }, { type: 'Activation' }, { type: 'MaxPool2d' }] },
                { shapeAt: { node: { type: 'MaxPool2d', nth: 1 }, equals: [null, 16, 7, 7] } },
            ] },
            diagnose: [{
                when: { all: [{ has: { type: 'Conv2d' }, count: { min: 2 } }, { not: { param: { node: { type: 'Conv2d', nth: 1 }, name: 'in_channels', equals: 8 } } }] },
                misconception: 'in-channels-mismatch',
                example: [{ do: 'buildChain', from: { id: 'conv1' }, to: { type: 'Output' }, blocks: [RELU, POOL, conv(1, 16), RELU, POOL] }],
            }],
            solution: [{ do: 'buildChain', from: { id: 'conv1' }, to: { type: 'Output' }, blocks: [RELU, POOL, conv(8, 16), RELU, POOL] }],
        },
        {
            id: 's5', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: ['lesson.palette', { node: { type: 'MaxPool2d', nth: 1 } }],
            setup: [{ do: 'setUi', showDimensions: false }],
            palette: [{ block: 'torch.nn.Flatten', count: 1 }, { block: 'component:linear', count: 1 }],
            predict: {
                id: 'm4.flatten', kind: 'number', expected: 784,
                diagnose: [
                    { equals: 49, misconception: 'one-map-only' },
                    { equals: 112, misconception: 'channels-times-height' },
                    { equals: 30, misconception: 'added-not-multiplied' },
                ],
            },
            check: { all: [
                { answered: 'm4.flatten' },
                { chain: [{ type: 'MaxPool2d' }, { type: 'Flatten' }, { type: 'Linear' }, { type: 'Output' }] },
                { valid: true },
                { outputShape: [null, 10] },
                { ui: { showDimensions: true } },
            ] },
            diagnose: [{
                when: { chain: [{ type: 'MaxPool2d' }, { type: 'Linear' }] }, misconception: 'missing-flatten',
                example: [{ do: 'buildChain', from: { type: 'MaxPool2d', nth: 1 }, to: { type: 'Output' }, blocks: [{ block: 'component:linear', params: { in_features: 784, out_features: 10 } }] }],
            }],
            solution: [
                answer('m4.flatten', 784),
                { do: 'buildChain', from: { type: 'MaxPool2d', nth: 1 }, to: { type: 'Output' }, blocks: [{ block: 'torch.nn.Flatten' }, { block: 'component:linear', params: { in_features: 784, out_features: 10 } }] },
                { do: 'setUi', showDimensions: true },
            ],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: [{ node: { id: 'conv1' } }],
            embeds: ['paramCount'],
            predict: {
                id: 'm4.conv1Params', kind: 'number', expected: 80,
                diagnose: [
                    { equals: 72, misconception: 'forgot-bias' },
                    { equals: 6272, misconception: 'weights-per-position' },
                    { equals: 10, misconception: 'one-filter-only' },
                ],
            },
            check: { all: [{ answered: 'm4.conv1Params' }, { paramCount: { equals: 9098 } }] },
            solution: [answer('m4.conv1Params', 80)],
        },
        {
            id: 's7', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:train'],
            codeQuotes: { tab: 'train', text: ['datasets.FashionMNIST(root, train=True, download=True, transform=transform)', 'def build_dataloaders'] },
            predict: { id: 'm4.loader', kind: 'choice', options: ['fashion', 'iris', 'random'], correct: 'fashion' },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm4.loader' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'train' }, answer('m4.loader', 'fashion')],
        },
    ],
    quiz: [
        { id: 'm4.q1', objective: 'm4.lo1', options: ['right', 'nopad', 'same'], correct: 'right', misconceptions: { nopad: 'forgot-padding', same: 'channels-unchanged' } },
        { id: 'm4.q2', objective: 'm4.lo2', options: ['right', 'channels', 'slide'], correct: 'right', misconceptions: { channels: 'pooling-halves-channels', slide: 'pool-stride-one' } },
        { id: 'm4.q3', objective: 'm4.lo3', options: ['reuse', 'noweights', 'fewer'], correct: 'reuse', misconceptions: { noweights: 'conv-no-weights', fewer: 'conv-sees-less' } },
        { id: 'm4.q4', objective: 'm4.lo4', options: ['right', 'map', 'sum'], correct: 'right', misconceptions: { map: 'one-map-only', sum: 'added-not-multiplied' } },
    ],
};
