/**
 * A small demo lesson for the Stage A engine and UI tests (docs/edu/SPEC.md §7.3). It exercises a warm-up, every
 * step kind (read, task, predict with number, shape and choice), hints, palette chips with a distractor, a train
 * gate, the paramCount embed, a quiz and a recap. It needs no Stage B code: no step trains.
 *
 * The browser check loads it through window.__deepflow.edu.engine.setModuleRegistry.
 */
export const TEST_MODULE = {
    id: 'demo',
    version: 1,
    order: 1,
    minutes: 5,
    datasets: ['blobs'],
    objectives: ['demo.lo1', 'demo.lo2'],
    fixtures: {
        start: {
            kind: 'snapshot',
            build: (h) => {
                const input = h.input('in', [2], { label: 'features' });
                const hidden = h.linear('lin1', 2, 8, { label: 'hidden' });
                const score = h.linear('lin2', 8, 1, { label: 'score' });
                const output = h.output('out', { label: 'logit' });
                return h.snapshot({
                    nodes: [input, hidden, score, output],
                    edges: h.chain(input, hidden, score, output),
                    training: [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss('bce', output)],
                    data: h.data('blobs', { n: 200, noise: 0.1, seed: 1 }),
                    trainer: h.trainer({ epochs: 30, batchSize: 32, seed: 1 }),
                });
            },
        },
        relu: {
            kind: 'snapshot',
            build: (h) => {
                const input = h.input('in', [2], { label: 'features' });
                const hidden = h.linear('lin1', 2, 8, { label: 'hidden' });
                const relu = h.layer('relu', 'torch.nn.ReLU', {}, 'ReLU');
                const score = h.linear('lin2', 8, 1, { label: 'score' });
                const output = h.output('out', { label: 'logit' });
                return h.snapshot({
                    nodes: [input, hidden, relu, score, output],
                    edges: h.chain(input, hidden, relu, score, output),
                    training: [h.optimizer('Adam', { learning_rate: 0.03 }), h.loss('bce', output)],
                    ui: { showDimensions: true },
                });
            },
        },
    },
    start: 'start',
    facts: {
        hiddenParams: { value: 24, compute: () => 2 * 8 + 8 },
    },
    warmup: [
        { id: 'demo.w1', objective: 'demo.lo1', options: ['true', 'false'], correct: 'false' },
        { id: 'demo.w2', objective: 'demo.lo2', options: ['small', 'huge'], correct: 'small' },
    ],
    steps: [
        {
            id: 's1', kind: 'read', subgoal: 'objective', target: ['tab.training'],
            check: { visited: 'training' },
            solution: [{ do: 'setUi', tab: 'training' }],
        },
        {
            id: 's2', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: [{ node: { id: 'lin1' } }],
            embeds: ['paramCount'],
            predict: { id: 'demo.params', kind: 'number', expected: 24, diagnose: [{ equals: 16, misconception: 'forgot-bias' }] },
            check: { answered: 'demo.params' },
            solution: [{ do: 'answer', id: 'demo.params', value: 24 }],
        },
        {
            id: 's3', kind: 'task', subgoal: 'model', tab: 'architecture', target: ['lesson.palette', 'ribbon.components'],
            palette: [{ block: 'torch.nn.ReLU', count: 1 }, { block: 'torch.nn.Sigmoid', distractor: 'sigmoid-before-bce' }],
            check: { all: [
                { chain: [{ type: 'Input' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }] },
                { valid: true },
            ] },
            diagnose: [{ when: { chain: [{ type: 'Activation' }, { type: 'Output' }] }, misconception: 'activation-at-output' }],
            solution: [{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }],
        },
        {
            id: 's4', kind: 'predict', subgoal: 'shapes', tab: 'architecture', target: ['ribbon.dimensions'],
            setup: [{ do: 'setUi', showDimensions: false }],
            predict: { id: 'demo.shape', kind: 'shape', expectedShape: [null, 8], diagnose: [{ equals: [8], misconception: 'missing-batch-axis' }] },
            check: { all: [{ answered: 'demo.shape' }, { ui: { showDimensions: true } }] },
            solution: [{ do: 'answer', id: 'demo.shape', value: [null, 8] }, { do: 'setUi', showDimensions: true }],
        },
        {
            id: 's5', kind: 'task', subgoal: 'model', tab: 'architecture', target: [{ node: { id: 'lin2' } }, 'field:out_features'],
            check: { all: [
                { param: { node: { id: 'lin2' }, name: 'out_features', equals: 3 } },
                { valid: true },
                { outputShape: [null, 3] },
            ] },
            diagnose: [{ when: { param: { node: { id: 'lin2' }, name: 'out_features', equals: 1 } }, misconception: 'one-output-multiclass' }],
            solution: [{ do: 'setParam', node: { id: 'lin2' }, name: 'out_features', value: 3 }],
        },
        {
            id: 's6', kind: 'predict', subgoal: 'objective', target: ['training.loss', 'tab.training'],
            locks: { gate: 'train' }, embeds: ['trainer'],
            predict: { id: 'demo.loss', kind: 'choice', options: ['cross_entropy', 'bce', 'mse'], correct: 'cross_entropy', diagnose: [{ equals: 'bce', misconception: 'one-output-multiclass' }] },
            check: { all: [{ answered: 'demo.loss' }, { training: { loss: 'cross_entropy', lossBound: true } }] },
            solution: [{ do: 'answer', id: 'demo.loss', value: 'cross_entropy' }, { do: 'setLoss', type: 'cross_entropy' }],
        },
        {
            id: 's7', kind: 'predict', subgoal: 'export', target: ['toolbar.code', 'code.tab:train'],
            predict: { id: 'demo.code', kind: 'choice', options: ['train', 'model'], correct: 'train' },
            check: { all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'demo.code' }] },
            solution: [{ do: 'setUi', codeOpen: true, codeTab: 'train' }, { do: 'answer', id: 'demo.code', value: 'train' }],
        },
    ],
    quiz: [
        { id: 'demo.q1', objective: 'demo.lo1', options: ['seven', 'twenty-four', 'sixteen'], correct: 'twenty-four', misconceptions: { sixteen: 'forgot-bias' } },
        { id: 'demo.q2', objective: 'demo.lo2', options: ['line', 'bend'], correct: 'bend' },
        { id: 'demo.q3', objective: 'demo.lo2', options: ['three', 'one'], correct: 'three', misconceptions: { one: 'one-output-multiclass' } },
    ],
};

export const TEST_TEXT = {
    en: {
        'demo.title': 'Tour of the lesson panel',
        'demo.goal': 'Try every kind of step once.',
        'demo.recap.1': 'A Linear block has one weight per input and output, plus one bias per output.',
        'demo.recap.2': 'A **ReLU** between Linear blocks lets the boundary bend.',
        'demo.recap.3': 'K classes need K outputs and cross-entropy.',
        'demo.lo1': 'Count the parameters of a Linear block.',
        'demo.lo2': 'Choose layers and a loss for a classifier.',

        'demo.w1.prompt': 'True or false: you need a PhD to build a neural network.',
        'demo.w1.opt.true': 'True',
        'demo.w1.opt.false': 'False',
        'demo.w1.why.true': 'Many people start with arithmetic and a little algebra.',
        'demo.w1.why.false': 'Right: arithmetic and a little algebra are enough to start.',
        'demo.w2.prompt': 'How much data do you need to learn the ideas?',
        'demo.w2.opt.small': 'A small dataset is enough',
        'demo.w2.opt.huge': 'A huge dataset and a GPU',
        'demo.w2.why.small': 'Right: 200 points train in about a second in your browser.',
        'demo.w2.why.huge': 'Big models need big data; learning the ideas does not.',

        'demo.s1.title': 'Find the loss and the optimizer',
        'demo.s1.body': 'The **Training** tab holds the **loss**, which measures how wrong the model is, and the **optimizer**, which changes the weights to lower it. Open it.',
        'demo.s1.why': 'Every network trains the same way: compute a loss, then let the optimizer reduce it.',
        'demo.s1.hint1': 'Look at the top bar, next to Data.',
        'demo.s1.hint2': 'Press **Training** in the top bar.',
        'demo.s1.done': 'This model trains with Adam and BCE.',

        'demo.s2.title': 'Count the parameters',
        'demo.s2.body': 'The **hidden** block is a Linear block from 2 inputs to 8 outputs. Each output has one **weight** per input and one **bias**. These {hiddenParams} numbers are what training changes.',
        'demo.s2.hint1': 'Count the weights first: inputs × outputs.',
        'demo.s2.hint2': '2 × 8 weights, plus 8 biases.',
        'demo.s2.predict.prompt': 'How many parameters does the hidden block have?',
        'demo.s2.predict.reveal': '2 × 8 = 16 weights plus 8 biases = {hiddenParams}.',

        'demo.s3.title': 'Add a non-linearity',
        'demo.s3.body': 'Insert a **ReLU** between **hidden** and **score**: relu(z) = max(0, z). Use the chip below or **Components ▾ → Activation**, then wire hidden → ReLU → score.',
        'demo.s3.hint1': 'Use the ReLU chip below, then connect it.',
        'demo.s3.hint2': 'Delete the wire from hidden to score, then wire hidden → ReLU → score.',
        'demo.s3.done': 'Now the network can bend its boundary.',

        'demo.s4.title': 'Read a shape',
        'demo.s4.body': 'Dimensions are off for now. Predict the shape, then turn **Dimensions** on in the ribbon to check.',
        'demo.s4.hint1': 'A batch of B points enters as (B, 2).',
        'demo.s4.hint2': 'The hidden block has 8 outputs per point.',
        'demo.s4.predict.prompt': 'What shape leaves the hidden block?',
        'demo.s4.predict.reveal': '(B, 8): one row per point in the batch, one column per hidden unit.',

        'demo.s5.title': 'One output per class',
        'demo.s5.body': 'A classifier for 3 classes needs 3 scores. Click **score** and set its **Output** to 3.',
        'demo.s5.hint1': 'Click the score block to open its settings.',
        'demo.s5.hint2': 'Set Output = 3.',

        'demo.s6.title': 'Pick the loss',
        'demo.s6.body': 'With 3 scores, the loss must compare a probability for each class. Answer first, then change the loss in the **Training** tab.',
        'demo.s6.hint1': 'BCE is for a single yes/no score.',
        'demo.s6.hint2': 'Open Training, click the loss block and set type to cross_entropy.',
        'demo.s6.predict.prompt': 'Which loss fits 3 scores?',
        'demo.s6.predict.opt.cross_entropy': 'Cross-entropy',
        'demo.s6.predict.opt.bce': 'BCE',
        'demo.s6.predict.opt.mse': 'MSE',
        'demo.s6.predict.reveal': 'Cross-entropy turns the 3 scores into probabilities with softmax and penalises a low probability for the true class.',

        'demo.s7.title': 'Read the code',
        'demo.s7.body': 'Press **Code** and open **train.py**. It rebuilds the model in PyTorch and trains it with the same loss and optimizer.',
        'demo.s7.hint1': 'Code is in the top bar, on the right.',
        'demo.s7.hint2': 'Press Code, then the train.py tab.',
        'demo.s7.predict.prompt': 'Which file holds the training loop?',
        'demo.s7.predict.opt.train': 'train.py',
        'demo.s7.predict.opt.model': 'model.py',
        'demo.s7.predict.reveal': 'train.py: model.py only defines the layers.',

        'demo.q1.prompt': 'How many parameters does a Linear block from 2 to 8 have?',
        'demo.q1.opt.seven': '7',
        'demo.q1.opt.twenty-four': '24',
        'demo.q1.opt.sixteen': '16',
        'demo.q1.why.seven': 'That counts nothing per weight: each output has one weight per input.',
        'demo.q1.why.twenty-four': 'Right: 2 × 8 weights + 8 biases.',
        'demo.q1.why.sixteen': 'Right number of weights, but add the 8 biases.',
        'demo.q2.prompt': 'What does a ReLU between two Linear blocks change?',
        'demo.q2.opt.line': 'Nothing: the boundary stays a line',
        'demo.q2.opt.bend': 'The boundary can bend',
        'demo.q2.why.line': 'Without the ReLU that is true; with it the map is no longer linear.',
        'demo.q2.why.bend': 'Right: each hidden unit adds a bend.',
        'demo.q3.prompt': 'How many outputs does a 3-class classifier need?',
        'demo.q3.opt.three': '3',
        'demo.q3.opt.one': '1',
        'demo.q3.why.three': 'Right: one score per class.',
        'demo.q3.why.one': 'One output only works for a yes/no question.',

        'mis.forgot-bias': 'Each output unit also has one bias, so add the number of outputs.',
        'mis.sigmoid-before-bce': 'DeepFlow\'s BCE applies the sigmoid itself. Output the raw score.',
        'mis.activation-at-output': 'A ReLU right before the output clips negative scores to 0. Put activations between Linear blocks.',
        'mis.missing-batch-axis': 'Models process batches: the first axis is the batch size B.',
        'mis.one-output-multiclass': 'With K classes the model needs K outputs, one score per class.',
    },
};
