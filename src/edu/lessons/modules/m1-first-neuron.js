/**
 * Module 1 — Your first neuron (docs/edu/SPEC.md §4.1). Whole game first: one neuron on two blobs, then a 3-class
 * Iris classifier, trained in the browser and exported to train.py. Text: text/en/m1.js.
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const TRAIN = { do: 'train' };

export default {
    id: 'm1',
    version: 1,
    order: 1,
    minutes: 20,
    datasets: ['blobs', 'iris'],
    objectives: ['m1.lo1', 'm1.lo2', 'm1.lo3', 'm1.lo4'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const input = h.input('in', [2], { label: 'features' });
                const neuron = h.linear('neuron', 2, 1, { label: 'neuron' });
                const output = h.output('out', { label: 'logit' });
                return h.snapshot({
                    nodes: [input, neuron, output],
                    edges: h.chain(input, neuron, output),
                    training: [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss('bce', output)],
                    data: h.data('blobs', { n: 200, noise: 0.1, seed: 1 }, { val: 0.2, test: 0 }),
                    trainer: h.trainer({ epochs: 30, batchSize: 32, seed: 1 }),
                });
            },
        },
    },
    start: 'start',
    facts: {
        points: { value: 200, at: 'start', compute: async (h) => (await h.prepared()).train.n + (await h.prepared()).val.n },
        trainPoints: { value: 160, at: 'start', compute: async (h) => (await h.prepared()).train.n },
        valPoints: { value: 40, at: 'start', compute: async (h) => (await h.prepared()).val.n },
        neuronParams: { value: 3, at: 'start', compute: (h) => h.params({ id: 'neuron' }) },
        irisTrain: { value: 120, at: 's4', compute: async (h) => (await h.prepared()).train.n },
        irisVal: { value: 30, at: 's4', compute: async (h) => (await h.prepared()).val.n },
        irisParams: { value: 15, at: 's5', compute: (h) => h.total() },
    },
    warmup: [
        { id: 'm1.w1', objective: 'm1.lo1', options: ['true', 'false'], correct: 'false' },
        { id: 'm1.w2', objective: 'm1.lo4', options: ['true', 'false'], correct: 'false' },
    ],
    steps: [
        {
            id: 's1', kind: 'read', subgoal: 'data', target: ['tab.data', 'data.preview'],
            check: { visited: 'data' },
            solution: [{ do: 'setUi', tab: 'data' }],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'model', tab: 'architecture', target: [{ node: { id: 'neuron' } }],
            embeds: ['paramCount'],
            predict: { id: 'm1.params', kind: 'number', expected: 3, diagnose: [{ equals: 2, misconception: 'forgot-bias' }] },
            check: { answered: 'm1.params' },
            solution: [answer('m1.params', 3)],
        },
        {
            id: 's3', kind: 'predict', subgoal: 'train', tab: 'architecture', target: ['trainer.run'],
            locks: { gate: 'train' }, embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            predict: {
                id: 'm1.boundary', kind: 'choice', options: ['line', 'circle', 'curve'], correct: 'line',
                diagnose: [{ equals: 'circle', misconception: 'neuron-bends' }, { equals: 'curve', misconception: 'neuron-bends' }],
            },
            check: { all: [{ answered: 'm1.boundary' }, { run: { dataset: 'blobs', minEpochs: 20, final: { valAccuracy: { gte: 0.9 } } } }] },
            solution: [answer('m1.boundary', 'line'), TRAIN],
        },
        {
            id: 's4', kind: 'predict', subgoal: 'data', tab: 'data', target: ['data.dataset:iris', 'data.use'],
            predict: {
                id: 'm1.features', kind: 'number', expected: 4,
                diagnose: [
                    { equals: 5, misconception: 'target-in-input' },
                    { equals: 3, misconception: 'classes-as-features' },
                    { equals: 150, misconception: 'examples-as-features' },
                ],
            },
            check: { all: [{ answered: 'm1.features' }, { dataset: 'iris' }, { inputShape: [null, 4] }] },
            solution: [answer('m1.features', 4), { do: 'useDataset', id: 'iris', setOutput: false }],
        },
        {
            id: 's5', kind: 'task', subgoal: 'model', tab: 'architecture', target: [{ node: { id: 'neuron' } }, 'field:in_features'],
            check: { all: [
                { param: { node: { type: 'Linear' }, name: 'in_features', equals: 4 } },
                { param: { node: { type: 'Linear' }, name: 'out_features', equals: 3 } },
                { valid: true },
                { outputShape: [null, 3] },
            ] },
            diagnose: [
                {
                    when: { param: { node: { type: 'Linear' }, name: 'in_features', equals: 5 } }, misconception: 'target-in-input',
                    example: [{ do: 'setParam', node: { id: 'neuron' }, name: 'in_features', value: 5 }],
                },
                {
                    when: { all: [
                        { param: { node: { type: 'Linear' }, name: 'in_features', equals: 4 } },
                        { param: { node: { type: 'Linear' }, name: 'out_features', equals: 1 } },
                    ] },
                    misconception: 'one-output-multiclass',
                    example: [{ do: 'setParam', node: { id: 'neuron' }, name: 'in_features', value: 4 }, { do: 'setParam', node: { id: 'neuron' }, name: 'out_features', value: 1 }],
                },
            ],
            solution: [
                { do: 'setParam', node: { id: 'neuron' }, name: 'in_features', value: 4 },
                { do: 'setParam', node: { id: 'neuron' }, name: 'out_features', value: 3 },
            ],
        },
        {
            id: 's6', kind: 'task', subgoal: 'objective', tab: 'training', target: ['training.loss'],
            check: { training: { loss: 'cross_entropy', lossBound: true } },
            solution: [{ do: 'setLoss', type: 'cross_entropy' }],
        },
        {
            id: 's7', kind: 'predict', subgoal: 'evaluate', tab: 'architecture', target: ['trainer.run'],
            setup: [{ do: 'setTrainer', epochs: 50, batchSize: 16 }],
            locks: { gate: 'train' }, embeds: ['trainer'], reveal: { run: 'finalValAccuracy' },
            predict: { id: 'm1.easiest', kind: 'choice', options: ['setosa', 'versicolor', 'virginica'], correct: 'setosa' },
            check: { all: [{ answered: 'm1.easiest' }, { run: { dataset: 'iris', minEpochs: 30, final: { valAccuracy: { gte: 0.85 } } } }] },
            solution: [answer('m1.easiest', 'setosa'), TRAIN],
        },
        {
            id: 's8', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:train'],
            codeQuotes: { tab: 'train', text: ['from sklearn.datasets import load_iris', 'mean = X[train_idx].mean(axis=0)', 'X = (X - mean) / std'] },
            predict: {
                id: 'm1.leak', kind: 'choice', options: ['train', 'own', 'none'], correct: 'train',
                diagnose: [{ equals: 'own', misconception: 'scaling-leak' }, { equals: 'none', misconception: 'unscaled-validation' }],
            },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm1.leak' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'train' }, answer('m1.leak', 'train')],
        },
    ],
    quiz: [
        { id: 'm1.q1', objective: 'm1.lo1', options: ['seven', 'twelve', 'fifteen'], correct: 'fifteen', misconceptions: { seven: 'added-not-multiplied', twelve: 'forgot-bias' } },
        { id: 'm1.q2', objective: 'm1.lo3', options: ['measure', 'more', 'faster'], correct: 'measure', misconceptions: { more: 'validation-trains', faster: 'validation-for-speed' } },
        { id: 'm1.q3', objective: 'm1.lo2', options: ['ce', 'bce', 'mse'], correct: 'ce', misconceptions: { bce: 'bce-for-multiclass', mse: 'mse-for-classes' } },
        { id: 'm1.q4', objective: 'm1.lo4', options: ['diagonal', 'row', 'column'], correct: 'diagonal', misconceptions: { row: 'confusion-axes', column: 'confusion-axes' } },
    ],
};
