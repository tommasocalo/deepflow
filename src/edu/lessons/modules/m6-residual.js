/**
 * Module 6 — Reusable blocks and residual connections (docs/edu/SPEC.md §4.6). A small CIFAR-10 CNN gets a skip
 * connection, the residual block becomes a component, the component is repeated three times, and the learner
 * reasons about shared structure, separate weights and the identity gradient path. Text: text/en/m6.js.
 */
const answer = (id, value) => ({ do: 'answer', id, value });
const BLOCK = ['relu0', 'c1', 'b1', 'r1', 'c2', 'b2'];

function adds(graph) {
    return Object.values(graph.nodes ?? {}).filter((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.op.Add');
}
const sources = (graph, id) => (graph.edges ?? []).filter((edge) => edge.target === id).map((edge) => edge.source);
const targets = (graph, id) => (graph.edges ?? []).filter((edge) => edge.source === id).map((edge) => edge.target);

/**
 * `residualAdd` (§4.6 s2): a root Add has exactly two inputs, relu0 and the BatchNorm2d at the end of the chain
 * c1 → b1 → r1 → c2 → b2, and it feeds r2. The details walk the learner through the three wires.
 */
function residualAdd(ctx) {
    const graph = ctx.graph;
    const candidates = adds(graph);
    if (!candidates.length) return { pass: false, detail: { key: 'm6.check.noAdd' } };
    const placed = candidates.find((add) => sources(graph, add.id).includes('b2') && targets(graph, add.id).includes('r2'));
    if (!placed) return { pass: false, detail: { key: 'm6.check.addPlace' } };
    const inputs = sources(graph, placed.id);
    if (!inputs.includes('relu0')) return { pass: false, detail: { key: 'm6.check.addSkip' } };
    if (inputs.length !== 2) return { pass: false, detail: { key: 'm6.check.addTwo' } };
    const intact = BLOCK.every((id, index) => index === 0 || targets(graph, BLOCK[index - 1]).includes(id));
    if (!intact) return { pass: false, detail: { key: 'm6.check.blockChain' } };
    return { pass: true };
}

/** An Add whose inputs have different inferred shapes (misconception add-shapes). */
function addShapesDiffer(ctx) {
    const shapes = ctx.report.shapes ?? {};
    return adds(ctx.graph).some((add) => {
        const inputs = sources(ctx.graph, add.id).map((id) => JSON.stringify(shapes[id] ?? null));
        return inputs.length >= 2 && new Set(inputs).size > 1;
    });
}

export default {
    id: 'm6',
    version: 1,
    order: 6,
    minutes: 20,
    datasets: ['cifar10'],
    recommendedAfter: ['m4'],
    objectives: ['m6.lo1', 'm6.lo2', 'm6.lo3', 'm6.lo4'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const conv = (id, inChannels, outChannels, label, bias = true) => h.layer(id, 'torch.nn.Conv2d', {
                    in_channels: inChannels, out_channels: outChannels, kernel_size: 3, padding: 1, ...(bias ? {} : { bias: false }),
                }, label);
                const bn = (id, label) => h.layer(id, 'torch.nn.BatchNorm2d', { num_features: 32 }, label);
                const relu = (id, label) => h.layer(id, 'torch.nn.ReLU', {}, label);
                const nodes = [
                    h.input('in', [3, 32, 32], { label: 'images' }),
                    conv('stem', 3, 32, 'stem'), bn('bn0', 'bn0'), relu('relu0', 'relu0'),
                    conv('c1', 32, 32, 'conv1', false), bn('b1', 'bn1'), relu('r1', 'relu1'),
                    conv('c2', 32, 32, 'conv2', false), bn('b2', 'bn2'), relu('r2', 'relu2'),
                    h.layer('pool', 'torch.nn.AdaptiveAvgPool2d', { output_size: 1 }, 'pool'),
                    h.layer('flat', 'torch.nn.Flatten', {}, 'flatten'),
                    h.linear('fc', 32, 10, { label: 'fc' }),
                    h.output('out', { label: 'logits' }),
                ];
                return h.snapshot({
                    nodes, edges: h.chain(...nodes),
                    training: [h.optimizer('Adam', { learning_rate: 0.001 }), h.loss('cross_entropy', nodes.at(-1))],
                    data: h.data('cifar10'), trainer: h.trainer({ epochs: 10, batchSize: 128, seed: 6 }), ui: { showDimensions: true },
                });
            },
        },
    },
    start: 'start',
    facts: {
        cifarImages: { value: 60000, at: 'start', compute: (h) => h.dataset().size.train + h.dataset().size.test },
        oneBlockModel: { value: 19850, at: 'start', compute: (h) => h.total() },
        blockParams: { value: 18560, at: 's3', compute: (h) => h.params({ type: 'Component' }) },
        threeBlockModel: { value: 56970, at: 's4', compute: (h) => h.total() },
        cifarDownload: { value: '170 MB', source: 'literature', note: 'torchvision\'s cifar-10-python.tar.gz is 170,498,071 bytes; train.py says "about 170 MB".' },
    },
    checks: { residualAdd, addShapesDiffer },
    warmup: [
        { id: 'm6.w1', objective: 'm5.lo3', options: ['over', 'under', 'normal'], correct: 'over', misconceptions: { under: 'underfitting-confusion', normal: 'training-loss-is-enough' } },
        { id: 'm6.w2', objective: 'm4.lo1', options: ['right', 'nopad', 'double'], correct: 'right', misconceptions: { nopad: 'forgot-padding', double: 'channels-unchanged' } },
    ],
    steps: [
        {
            id: 's1', kind: 'read', subgoal: 'data', target: ['tab.data'],
            check: { all: [{ visited: 'data' }, { dataset: 'cifar10' }, { inputShape: [null, 3, 32, 32] }] },
            solution: [{ do: 'setUi', tab: 'data' }],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'model', tab: 'architecture', target: ['lesson.palette', 'ribbon.operations'],
            palette: [{ block: 'deepflow.op.Add', count: 1 }],
            predict: { id: 'm6.addShapes', kind: 'choice', options: ['yes', 'no'], correct: 'no', diagnose: [{ equals: 'yes', misconception: 'add-shapes' }] },
            check: { all: [{ answered: 'm6.addShapes' }, { custom: 'residualAdd' }, { valid: true }] },
            diagnose: [{
                when: { custom: 'addShapesDiffer' }, misconception: 'add-shapes',
                example: [{ do: 'insertBetween', block: 'deepflow.op.Add', from: { id: 'b2' }, to: { id: 'r2' } }, { do: 'connect', from: { id: 'in' }, to: { type: 'Add' } }],
            }],
            solution: [
                answer('m6.addShapes', 'no'),
                { do: 'insertBetween', block: 'deepflow.op.Add', from: { id: 'b2' }, to: { id: 'r2' } },
                { do: 'connect', from: { id: 'relu0' }, to: { type: 'Add' } },
            ],
        },
        {
            id: 's3', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['ribbon.group'],
            check: { component: { contains: [{ type: 'Conv2d', count: 2 }, { type: 'BatchNorm2d', count: 2 }, { type: 'Add', count: 1 }] } },
            solution: [{ do: 'group', nodes: [{ id: 'c1' }, { id: 'b1' }, { id: 'r1' }, { id: 'c2' }, { id: 'b2' }, { type: 'Add' }, { id: 'r2' }], name: 'ResBlock' }],
        },
        {
            id: 's4', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: ['ribbon.sequential', 'field:count'],
            embeds: ['paramCount'],
            predict: {
                id: 'm6.blockParams', kind: 'number', expected: 18560,
                diagnose: [
                    { equals: 18432, misconception: 'forgot-batchnorm' },
                    { equals: 18496, misconception: 'forgot-batchnorm' },
                    { equals: 18624, misconception: 'conv-bias-when-disabled' },
                ],
            },
            check: { all: [
                { answered: 'm6.blockParams' },
                { repeat: { node: { type: 'Component' }, mode: 'sequential', count: 3 } },
                { valid: true },
                { paramCount: { equals: 56970 } },
            ] },
            solution: [answer('m6.blockParams', 18560), { do: 'repeat', node: { type: 'Component' }, count: 3, mode: 'sequential' }],
        },
        {
            id: 's5', kind: 'predict', subgoal: 'model', tab: 'architecture', target: [{ nodeOpen: { type: 'Component' } }, 'ribbon.instances'],
            predict: {
                id: 'm6.sharedWeights', kind: 'choice', options: ['yes', 'no'], correct: 'no',
                diagnose: [{ equals: 'yes', misconception: 'shared-structure-shared-weights' }],
            },
            check: { all: [{ answered: 'm6.sharedWeights' }, { level: { open: { type: 'Component' } } }] },
            solution: [{ do: 'openLevel', path: [{ type: 'Component' }] }, answer('m6.sharedWeights', 'no')],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'evaluate', tab: 'architecture', target: ['ribbon.gradients'],
            predict: {
                id: 'm6.residualGrad', kind: 'choice', options: ['identity', 'twice', 'nothing'], correct: 'identity',
                diagnose: [{ equals: 'twice', misconception: 'skip-doubles-gradient' }, { equals: 'nothing', misconception: 'skip-blocks-gradient' }],
            },
            check: { all: [{ answered: 'm6.residualGrad' }, { ui: { showGradientFlow: true } }] },
            solution: [{ do: 'setUi', showGradientFlow: true }, answer('m6.residualGrad', 'identity')],
        },
        {
            id: 's7', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:model'],
            codeQuotes: { tab: 'model', text: ['torch.add(', 'self.conv1_2(', 'self.relu1(', 'self.conv1_3 ='] },
            predict: { id: 'm6.codeAdd', kind: 'choice', options: ['add', 'conv', 'relu'], correct: 'add' },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'model' } }, { answered: 'm6.codeAdd' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'model' }, answer('m6.codeAdd', 'add')],
        },
    ],
    quiz: [
        { id: 'm6.q1', objective: 'm6.lo1', options: ['component', 'separate', 'merge'], correct: 'component', misconceptions: { separate: 'group-changes-model', merge: 'group-changes-model' } },
        { id: 'm6.q2', objective: 'm6.lo2', options: ['right', 'shared', 'split'], correct: 'right', misconceptions: { shared: 'shared-structure-shared-weights', split: 'repeat-splits-params' } },
        { id: 'm6.q3', objective: 'm6.lo3', options: ['sums', 'gradient', 'concat'], correct: 'sums', misconceptions: { gradient: 'add-shapes', concat: 'add-concatenates' } },
        { id: 'm6.q4', objective: 'm6.lo4', options: ['paths', 'size', 'lr'], correct: 'paths', misconceptions: { size: 'gradient-view-magnitudes', lr: 'gradient-view-magnitudes' } },
    ],
};
