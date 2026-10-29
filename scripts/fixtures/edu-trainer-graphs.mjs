// Stage B test helpers: build architecture and training graphs in memory, the way lesson fixtures do (Input at
// x = 400, rows 130 px apart), with the standard Linear component registered through ensureStandardTemplate.
import { createEdge, createLayerNode, createSuperNodeInstance } from '../../src/domain/architecture/index.js';
import { getEntry } from '../../src/domain/library/index.js';
import { ensureStandardTemplate } from '../../src/domain/components/index.js';
import { createLossNode, createOptimizerNode, createTrainingEdge, OPTIMIZER_DEFAULTS } from '../../src/domain/training/index.js';

/** A minimal stand-in for useTemplatesStore: getState().templates and createTemplate. */
export function memoryTemplates(initial = {}) {
    const templates = { ...initial };
    let counter = 0;
    const store = {
        getState: () => ({
            templates,
            createTemplate: ({ id, ...rest }) => {
                const templateId = id ?? `tpl_${++counter}`;
                templates[templateId] = { description: '', freeParameters: [], ...rest, id: templateId };
                return templateId;
            },
        }),
    };
    return { templates, store };
}

export function layer(id, libraryKey, parameters = {}, label) {
    const entry = getEntry(libraryKey);
    const node = createLayerNode({ id, libraryKey, entry, label: label ?? entry?.name ?? id, position: { x: 400, y: 0 } });
    return { ...node, parameters: { ...node.parameters, ...parameters } };
}

/**
 * Build a chain Input → blocks → Output. Each block is one of:
 *   ['linear', in, out, { bias, component = true }]   standard Linear component (or a torch.nn.Linear layer)
 *   ['relu'] ['tanh'] ['sigmoid'] ['gelu', approximate] ['leakyrelu', slope] ['softmax'] ['logsoftmax']
 *   ['dropout', p] ['flatten'] ['layer', libraryKey, parameters]
 */
export function chainModel(inputShape, blocks, { templates: given } = {}) {
    const { templates, store } = given ?? memoryTemplates();
    const nodes = [];
    nodes.push(layer('in', 'deepflow.io.Input', { shape: inputShape, dtype: 'float32' }, 'features'));
    let linears = 0;
    blocks.forEach((block, index) => {
        const [kind, ...args] = block;
        const id = kind === 'linear' ? `lin${++linears}` : `${kind}${index}`;
        if (kind === 'linear') {
            const [inFeatures, outFeatures, options = {}] = args;
            if (options.component === false) {
                nodes.push(layer(id, 'torch.nn.Linear', { in_features: inFeatures, out_features: outFeatures, bias: options.bias ?? true }, id));
            } else {
                const templateId = ensureStandardTemplate('linear', store);
                nodes.push(createSuperNodeInstance({ id, templateId, label: id, position: { x: 400, y: 0 },
                    parameterValues: { in_features: inFeatures, out_features: outFeatures, bias: options.bias ?? true } }));
            }
            return;
        }
        const keys = { relu: 'torch.nn.ReLU', tanh: 'torch.nn.Tanh', sigmoid: 'torch.nn.Sigmoid', gelu: 'torch.nn.GELU', leakyrelu: 'torch.nn.LeakyReLU',
            softmax: 'torch.nn.Softmax', logsoftmax: 'torch.nn.LogSoftmax', dropout: 'torch.nn.Dropout', flatten: 'torch.nn.Flatten' };
        if (kind === 'layer') nodes.push(layer(id, args[0], args[1] ?? {}, args[2] ?? id));
        else if (kind === 'gelu') nodes.push(layer(id, keys.gelu, { approximate: args[0] ?? 'none' }));
        else if (kind === 'leakyrelu') nodes.push(layer(id, keys.leakyrelu, { negative_slope: args[0] ?? 0.01 }));
        else if (kind === 'dropout') nodes.push(layer(id, keys.dropout, { p: args[0] ?? 0.5 }));
        else nodes.push(layer(id, keys[kind]));
    });
    nodes.push(layer('out', 'deepflow.io.Output', {}, 'logits'));
    nodes.forEach((node, row) => { node.position = { x: 400, y: row * 130 }; });
    const edges = nodes.slice(1).map((node, index) => createEdge({ id: `e${index}`, source: nodes[index].id, target: node.id }));
    return { graph: { nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges }, templates };
}

/** Optimizer + loss bound to the root Output, wired loss → optimizer (as DeepFlow's storyboards do). */
export function trainingGraph({ optimizer = 'Adam', lr = 0.01, loss = 'bce', outputId = 'out', hyperparameters = {}, lossHyperparameters } = {}) {
    const opt = createOptimizerNode({ id: 'opt', type: optimizer, position: { x: 60, y: 100 },
        hyperparameters: { ...OPTIMIZER_DEFAULTS[optimizer], learning_rate: lr, ...hyperparameters } });
    const lossNode = createLossNode({ id: 'loss', type: loss, position: { x: 360, y: 100 }, architectureOutputId: outputId,
        ...(lossHyperparameters ? { hyperparameters: lossHyperparameters } : {}) });
    return { nodes: { opt, loss: lossNode }, edges: [createTrainingEdge({ id: 'te', source: 'loss', target: 'opt' })] };
}

export const objectiveOf = (training) => ({
    optimizer: { type: training.nodes.opt.type, hyperparameters: training.nodes.opt.hyperparameters, parameterGroup: training.nodes.opt.parameterGroup ?? [] },
    loss: { type: training.nodes.loss.type, hyperparameters: training.nodes.loss.hyperparameters },
});
