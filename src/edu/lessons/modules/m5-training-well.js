/**
 * Module 5 — Training well (docs/edu/SPEC.md §4.5). Karpathy's recipe on the CS231n 3-class spiral: check the
 * initial loss, sweep the learning rate, overfit on purpose, read the curves (early stopping), add Dropout and
 * compare honestly, then find model.eval() in train.py. Text: text/en/m5.js.
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const TRAIN = { do: 'train' };
const lr = (value) => ({ do: 'setOptimizer', hyperparameters: { learning_rate: value } });
const objective = (h, output) => [h.optimizer('Adam', { learning_rate: 0.01 }), h.loss('cross_entropy', output)];

export default {
    id: 'm5',
    version: 1,
    order: 5,
    minutes: 25,
    datasets: ['spiral3'],
    recommendedAfter: ['m2'],
    objectives: ['m5.lo1', 'm5.lo2', 'm5.lo3', 'm5.lo4'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const nodes = [
                    h.input('in', [2], { label: 'features' }), h.linear('lin1', 2, 64, { label: 'Linear 1' }), h.layer('relu1', 'torch.nn.ReLU', {}, 'ReLU 1'),
                    h.linear('lin2', 64, 3, { label: 'Linear 2' }), h.output('out', { label: 'logits' }),
                ];
                return h.snapshot({
                    nodes, edges: h.chain(...nodes), training: objective(h, nodes[4]),
                    data: h.data('spiral3', { perClass: 100, noise: 0.2, seed: 5 }, { val: 0.2, test: 0 }),
                    trainer: h.trainer({ epochs: 60, batchSize: 32, seed: 5 }),
                });
            },
        },
        big: {
            kind: 'snapshot',
            build: (h) => {
                const nodes = [
                    h.input('in', [2], { label: 'features' }), h.linear('lin1', 2, 128, { label: 'Linear 1' }), h.layer('relu1', 'torch.nn.ReLU', {}, 'ReLU 1'),
                    h.linear('lin2', 128, 128, { label: 'Linear 2' }), h.layer('relu2', 'torch.nn.ReLU', {}, 'ReLU 2'),
                    h.linear('lin3', 128, 3, { label: 'Linear 3' }), h.output('out', { label: 'logits' }),
                ];
                return h.snapshot({
                    nodes, edges: h.chain(...nodes), training: objective(h, nodes[6]),
                    data: h.data('spiral3', { perClass: 50, noise: 0.3, seed: 5 }, { val: 0.5, test: 0 }),
                    trainer: h.trainer({ epochs: 300, batchSize: 16, seed: 5 }),
                });
            },
        },
    },
    start: 'start',
    facts: {
        initLoss3: { value: '1.10', at: 'start', compute: (h) => h.lnText(3) },
        trainPoints: { value: 75, at: 's3', compute: async (h) => (await h.prepared()).train.n },
        bigEpochs: { value: 300, at: 's3', compute: (h) => h.trainer().settings.epochs },
        dropoutSeeds: {
            value: '8 of 8', source: 'simulation',
            note: 'Measured with the in-browser trainer (scratch probe, Stage D): m5.big with Dropout p = 0.5 after each ReLU vs without, '
                + 'training seeds 1–8, data seed 5, 300 epochs; the final validation-minus-training loss gap shrank on 8 of 8 seeds '
                + '(the NumPy simulation of SPEC §11.2 gave 7 of 8).',
        },
    },
    warmup: [
        { id: 'm5.w1', objective: 'm4.lo1', options: ['right', 'same', 'minus'], correct: 'right', misconceptions: { same: 'padding-no-effect', minus: 'minus-kernel' } },
        { id: 'm5.w2', objective: 'm3.lo4', options: ['right', 'ten', 'tenth'], correct: 'right', misconceptions: { ten: 'loss-is-count', tenth: 'loss-is-probability' } },
    ],
    steps: [
        {
            id: 's1', kind: 'predict', subgoal: 'train', tab: 'architecture', target: ['trainer.initLoss'],
            embeds: ['trainer'], reveal: { run: 'initLoss' },
            predict: {
                id: 'm5.initLoss', kind: 'number', expected: 1.1, tolerance: 0.15,
                diagnose: [
                    { equals: 0, misconception: 'zero-initial-loss' },
                    { equals: 0.33, misconception: 'loss-is-probability' },
                    { equals: 0.67, misconception: 'loss-is-error-rate' },
                    { equals: 3, misconception: 'loss-is-count' },
                ],
            },
            check: { all: [{ answered: 'm5.initLoss' }, { run: { dataset: 'spiral3', initOnly: true } }] },
            solution: [answer('m5.initLoss', 1.1), { do: 'evaluateInit' }],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'train', tab: 'architecture', target: ['trainer.lr'],
            locks: { gate: 'train' }, embeds: ['trainer'],
            predict: {
                id: 'm5.bestLr', kind: 'choice', options: ['small', 'medium', 'large'], correct: 'medium',
                diagnose: [{ equals: 'small', misconception: 'smaller-lr-safer' }, { equals: 'large', misconception: 'bigger-lr-faster' }],
            },
            check: { all: [{ answered: 'm5.bestLr' }, { runs: { count: 3, lr: [0.0001, 0.01, 1], dataset: 'spiral3', minEpochs: 60 } }] },
            solution: [answer('m5.bestLr', 'medium'), lr(0.0001), TRAIN, lr(0.01), TRAIN, lr(1), TRAIN, lr(0.01)],
        },
        {
            id: 's3', kind: 'predict', subgoal: 'train', tab: 'architecture', target: ['trainer.run'],
            setup: [{ do: 'loadFixture', id: 'big' }],
            locks: { gate: 'train' }, embeds: ['trainer'], tagRun: 'm5.overfit',
            predict: {
                id: 'm5.whichRises', kind: 'choice', options: ['val', 'train', 'neither'], correct: 'val',
                diagnose: [{ equals: 'train', misconception: 'training-loss-rises' }, { equals: 'neither', misconception: 'more-epochs-always-better' }],
            },
            check: { all: [
                { answered: 'm5.whichRises' },
                { run: { dataset: 'spiral3', minEpochs: 300, overfit: { trainAccuracyGte: 0.97, valLossRiseGte: 1.25 } } },
            ] },
            solution: [answer('m5.whichRises', 'val'), TRAIN],
        },
        {
            id: 's4', kind: 'predict', subgoal: 'evaluate', tab: 'training', target: ['trainer.curves'],
            predict: {
                id: 'm5.diagnosis', kind: 'choice', options: ['overfitting', 'underfitting', 'fine'], correct: 'overfitting',
                diagnose: [{ equals: 'underfitting', misconception: 'underfitting-confusion' }, { equals: 'fine', misconception: 'training-loss-is-enough' }],
            },
            check: { answered: 'm5.diagnosis' },
            solution: [answer('m5.diagnosis', 'overfitting')],
        },
        {
            id: 's5', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['lesson.palette'],
            palette: [{ block: 'torch.nn.Dropout', count: 2 }],
            embeds: ['trainer'], reveal: { compare: { tag: 'm5.overfit', metric: 'gap' } },
            check: { all: [
                { chain: [{ type: 'Activation' }, { type: 'Dropout' }] },
                { param: { node: { type: 'Dropout' }, name: 'p', min: 0.2, max: 0.6, every: true } },
                { run: { dataset: 'spiral3', minEpochs: 300, model: { has: { type: 'Dropout' } } } },
            ] },
            diagnose: [{
                when: { param: { node: { type: 'Dropout' }, name: 'p', min: 0.61 } }, misconception: 'dropout-p-meaning',
                example: [{ do: 'insertAfter', block: 'torch.nn.Dropout', after: { id: 'relu1' }, params: { p: 0.9 } }],
            }],
            solution: [
                { do: 'insertAfter', block: 'torch.nn.Dropout', after: { id: 'relu1' }, params: { p: 0.5 } },
                { do: 'insertAfter', block: 'torch.nn.Dropout', after: { id: 'relu2' }, params: { p: 0.5 } },
                TRAIN,
            ],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:train'],
            codeQuotes: { tab: 'train', text: ['model.eval()', 'torch.no_grad()', 'model.train()'] },
            predict: {
                id: 'm5.eval', kind: 'choice', options: ['dropoutOff', 'freeze', 'smaller'], correct: 'dropoutOff',
                diagnose: [{ equals: 'freeze', misconception: 'eval-freezes-weights' }, { equals: 'smaller', misconception: 'eval-shrinks-model' }],
            },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm5.eval' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'train' }, answer('m5.eval', 'dropoutOff')],
        },
    ],
    quiz: [
        { id: 'm5.q1', objective: 'm5.lo1', options: ['bug', 'great', 'arbitrary'], correct: 'bug', misconceptions: { great: 'low-initial-loss-good', arbitrary: 'initial-loss-arbitrary' } },
        { id: 'm5.q2', objective: 'm5.lo2', options: ['lower', 'epochs', 'layers'], correct: 'lower', misconceptions: { epochs: 'more-epochs-always-better', layers: 'bigger-model-fixes-training' } },
        { id: 'm5.q3', objective: 'm5.lo3', options: ['lowestVal', 'last', 'lowestTrain'], correct: 'lowestVal', misconceptions: { last: 'more-epochs-always-better', lowestTrain: 'training-loss-is-enough' } },
        { id: 'm5.q4', objective: 'm5.lo4', options: ['nothing', 'same', 'inputs'], correct: 'nothing', misconceptions: { same: 'dropout-at-eval', inputs: 'dropout-at-eval' } },
        // A fifth item, beyond §4.0's 3–4: the test split, which none of the browser datasets uses (SPEC §12).
        { id: 'm5.q5', objective: 'm5.lo3', options: ['val', 'test', 'train'], correct: 'test', misconceptions: { val: 'validation-score-final', train: 'training-loss-is-enough' } },
    ],
};
