/**
 * Module 7 — Attention and a tiny Transformer (docs/edu/SPEC.md §4.7). DeepFlow's GPT-2 storyboard at a tiny size
 * for Tiny Shakespeare: output shapes, embeddings as tables, positions, causal attention, heads and width, the
 * Colab export with temperature sampling, then the same blocks at GPT-2 small's size. Text: text/en/m7.js.
 *
 * No step calls useDataset: the fixture binds Tiny Shakespeare already, and Use this dataset would replace
 * lm_head's $vocab_size link with a fixed 65 (Stage C report).
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const ROOT = { do: 'openLevel', path: [] };

export default {
    id: 'm7',
    version: 1,
    order: 7,
    minutes: 25,
    datasets: ['tiny-shakespeare'],
    recommendedAfter: ['m3', 'm6'],
    objectives: ['m7.lo1', 'm7.lo2', 'm7.lo3', 'm7.lo4'],
    fixtures: {
        start: {
            kind: 'storyboard',
            storyboard: 'gpt2',
            configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4, nLayers: 2, mlpWidth: 256 },
            includeTraining: true,
            after: [{ do: 'setOptimizer', type: 'AdamW', hyperparameters: { learning_rate: 0.001, weight_decay: 0.1 } }],
            slices: {
                data: { datasetId: 'tiny-shakespeare', options: { 'tiny-shakespeare': { blockSize: 64 } }, split: { val: 0.2, test: 0 } },
                trainer: { settings: { epochs: 10, batchSize: 32, seed: 7 } },
            },
            ui: { showDimensions: true },
        },
    },
    start: 'start',
    facts: {
        characters: { value: 1115394, at: 'start', compute: (h) => h.dataset().size.total },
        vocab: { value: 65, at: 'start', compute: (h) => h.global('vocab_size') },
        lastId: { value: 64, at: 'start', compute: (h) => h.dataset().target.size - 1 },
        context: { value: 64, at: 'start', compute: (h) => h.global('context') },
        lastPosition: { value: 63, at: 'start', compute: (h) => h.global('context') - 1 },
        width: { value: 64, at: 'start', compute: (h) => h.global('d_model') },
        tinyParams: { value: 112512, at: 'start', compute: (h) => h.total() },
        embParams: { value: 4160, at: 'start', compute: (h) => h.params({ label: 'token embedding' }) },
        dHead4: { value: 16, at: 'start', compute: (h) => h.global('d_model') / h.global('n_heads') },
        initLoss65: { value: '4.17', at: 'start', compute: (h) => h.lnText(65) },
        tinyParamsEightHeads: { value: 112512, at: 's5', compute: (h) => h.total() },
        gpt2Blocks: { value: 12, at: 's7', compute: (h) => h.global('n_layers') },
        gpt2Width: { value: 768, at: 's7', compute: (h) => h.global('d_model') },
        gpt2Vocab: { value: 50257, at: 's7', compute: (h) => h.global('vocab_size') },
        gpt2Context: { value: 1024, at: 's7', compute: (h) => h.global('context') },
        gpt2Heads: { value: 12, at: 's7', compute: (h) => h.global('n_heads') },
        gpt2Params: { value: 163037184, at: 's7', compute: (h) => h.total() },
        gpt2TiedParams: { value: 124439808, at: 's7', compute: (h) => h.total() - h.global('vocab_size') * h.global('d_model') },
    },
    warmup: [
        { id: 'm7.w1', objective: 'm6.lo2', options: ['yes', 'no'], correct: 'no', misconceptions: { yes: 'shared-structure-shared-weights' } },
        { id: 'm7.w2', objective: 'm3.lo4', options: ['right', 'count', 'zero'], correct: 'right', misconceptions: { count: 'loss-is-count', zero: 'zero-initial-loss' } },
    ],
    steps: [
        {
            id: 's1', kind: 'predict', subgoal: 'data', tab: 'data', target: ['data.dataset:tiny-shakespeare'],
            predict: {
                id: 'm7.outShape', kind: 'shape', expectedShape: [null, 64, 65],
                diagnose: [
                    { equals: [null, 65], misconception: 'one-prediction-per-sequence' },
                    { equals: [null, 64], misconception: 'ids-not-scores' },
                    { equals: [null, 65, 64], misconception: 'axes-swapped' },
                ],
            },
            check: { all: [{ answered: 'm7.outShape' }, { dataset: 'tiny-shakespeare' }, { outputShape: [null, 64, 65] }] },
            solution: [answer('m7.outShape', [null, 64, 65])],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'model', tab: 'architecture', target: [{ nodeOpen: { label: 'token embedding' } }],
            predict: {
                id: 'm7.embParams', kind: 'number', expected: 4160,
                diagnose: [
                    { equals: 129, misconception: 'added-not-multiplied' },
                    { equals: 65, misconception: 'one-number-per-token' },
                    { equals: 64, misconception: 'one-number-per-token' },
                ],
            },
            check: { all: [{ answered: 'm7.embParams' }, { level: { open: { type: 'Embedding' } } }] },
            solution: [{ do: 'openLevel', path: [{ label: 'token embedding' }] }, answer('m7.embParams', 4160)],
        },
        {
            id: 's3', kind: 'predict', subgoal: 'model', tab: 'architecture', target: [{ node: { label: 'positional embedding' } }],
            setup: [ROOT],
            predict: { id: 'm7.positions', kind: 'choice', options: ['yes', 'no'], correct: 'no', diagnose: [{ equals: 'yes', misconception: 'attention-knows-order' }] },
            check: { answered: 'm7.positions' },
            solution: [answer('m7.positions', 'no')],
        },
        {
            id: 's4', kind: 'predict', subgoal: 'model', tab: 'architecture', target: [{ nodeOpen: { label: 'Transformer blocks' } }],
            predict: {
                id: 'm7.causal', kind: 'choice', options: ['upto', 'all', 'only', 'next'], correct: 'upto',
                diagnose: [
                    { equals: 'all', misconception: 'attention-sees-future' },
                    { equals: 'next', misconception: 'attention-sees-future' },
                    { equals: 'only', misconception: 'attention-only-self' },
                ],
            },
            check: { all: [{ answered: 'm7.causal' }, { level: { open: { type: 'AttentionHead' } } }] },
            solution: [
                { do: 'openLevel', path: [{ label: 'Transformer blocks' }, { type: 'Attention' }, { type: 'AttentionHead' }] },
                answer('m7.causal', 'upto'),
            ],
        },
        {
            id: 's5', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: ['ribbon.parameters'],
            setup: [ROOT],
            predict: { id: 'm7.dHead', kind: 'number', expected: 16, diagnose: [{ equals: 256, misconception: 'heads-multiply-width' }] },
            check: { all: [{ answered: 'm7.dHead' }, { global: { name: 'n_heads', equals: 8 } }, { valid: true }] },
            diagnose: [{
                when: { not: { valid: true } }, misconception: 'heads-must-divide',
                example: [{ do: 'setGlobal', name: 'n_heads', value: 3 }],
            }],
            solution: [answer('m7.dHead', 16), { do: 'setGlobal', name: 'n_heads', value: 8 }],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:train'],
            embeds: ['softmaxTemperature'],
            codeQuotes: { tab: 'train', text: ['tinyshakespeare/input.txt', 'math.log(65)', 'torch.softmax(scores / temperature, dim=-1)', 'temperature=0.8'] },
            predict: {
                id: 'm7.initLoss', kind: 'number', expected: 4.17, tolerance: 0.2,
                diagnose: [{ equals: 65, misconception: 'loss-is-count' }, { equals: 0, misconception: 'zero-initial-loss' }],
            },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm7.initLoss' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'train' }, answer('m7.initLoss', 4.17)],
        },
        {
            id: 's7', kind: 'predict', subgoal: 'evaluate', tab: 'architecture', target: ['ribbon.parameters'],
            buttons: [{ id: 'load', actions: [{ do: 'loadStoryboard', id: 'gpt2' }] }],
            predict: {
                id: 'm7.gpt2Heads', kind: 'number', expected: 12,
                diagnose: [{ equals: 64, misconception: 'head-width-vs-count' }, { equals: 768, misconception: 'head-width-vs-count' }],
            },
            check: { all: [{ global: { name: 'd_model', equals: 768 } }, { global: { name: 'n_layers', equals: 12 } }, { answered: 'm7.gpt2Heads' }] },
            solution: [{ do: 'loadStoryboard', id: 'gpt2' }, answer('m7.gpt2Heads', 12)],
        },
    ],
    quiz: [
        { id: 'm7.q1', objective: 'm7.lo3', options: ['later', 'same', 'positions'], correct: 'later', misconceptions: { same: 'mask-hides-self', positions: 'mask-is-position' } },
        { id: 'm7.q2', objective: 'm7.lo4', options: ['sharper', 'creative', 'weights'], correct: 'sharper', misconceptions: { creative: 'temperature-creativity', weights: 'temperature-changes-weights' } },
        { id: 'm7.q3', objective: 'm7.lo1', options: ['right', 'sum', 'rows'], correct: 'right', misconceptions: { sum: 'added-not-multiplied', rows: 'one-number-per-token' } },
        { id: 'm7.q4', objective: 'm7.lo2', options: ['eight', 'five', 'seven'], correct: 'eight', misconceptions: { five: 'heads-must-divide', seven: 'heads-must-divide' } },
    ],
};
