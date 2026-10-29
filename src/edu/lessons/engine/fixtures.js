import { createEdge, createLayerNode, createSuperNodeInstance, createTemplate } from '../../../domain/architecture/index.js';
import { ensureStandardTemplate } from '../../../domain/components/index.js';
import { getEntry } from '../../../domain/library/index.js';
import { createLossNode, createOptimizerNode, createTrainingEdge, LOSS_DEFAULTS, OPTIMIZER_DEFAULTS } from '../../../domain/training/index.js';

/**
 * Fixture helpers (docs/edu/SPEC.md §6.8.3). They build a WorkspaceSnapshot without touching the live stores, so
 * module files stay pure and content tests can build every fixture in Node. Nodes sit in one column at x = 400,
 * rows 130 px apart; side inputs go to x = 820.
 */
export const COLUMN_X = 400;
export const SIDE_X = 820;
export const ROW_GAP = 130;

/** A minimal stand-in for useTemplatesStore, enough for ensureStandardTemplate. */
export function createTemplateRegistry(initial = {}) {
    const templates = { ...initial };
    const state = {
        templates,
        createTemplate: ({ name, description, graph, freeParameters, id, ...metadata }) => {
            const template = { ...metadata, ...createTemplate({ name, description, graph, freeParameters, id }) };
            templates[template.id] = template;
            return template.id;
        },
        getTemplate: (id) => templates[id],
    };
    return { getState: () => state, templates };
}

const shapeText = (shape) => {
    if (typeof shape === 'string') return shape;
    if (!Array.isArray(shape)) return null;
    return `(${shape.join(', ')}${shape.length === 1 ? ',' : ''})`;
};

export function createFixtureHelpers() {
    const registry = createTemplateRegistry();
    let row = 0;
    const position = (options = {}) => {
        const at = options.row ?? row;
        row = Math.max(row, at) + 1;
        return { x: options.x ?? COLUMN_X, y: options.y ?? at * ROW_GAP };
    };

    const layer = (id, libraryKey, params = {}, label, options = {}) => {
        const entry = getEntry(libraryKey);
        if (!entry) throw new Error(`Unknown library key in fixture: ${libraryKey}`);
        const node = createLayerNode({ id, libraryKey, entry, label: label ?? entry.name, position: position(options) });
        node.parameters = { ...node.parameters, ...params };
        return node;
    };

    const component = (id, key, params = {}, label, options = {}) => {
        const templateId = ensureStandardTemplate(key, registry);
        if (!templateId) throw new Error(`Unknown standard component in fixture: ${key}`);
        return createSuperNodeInstance({ id, templateId, label: label ?? registry.templates[templateId].name, position: position(options), parameterValues: { ...params } });
    };

    const helpers = {
        templates: registry.templates,
        registry,
        /** Input with shape '(2,)' or [2] (the batch axis is implicit). */
        input: (id, shape, options = {}) => layer(id, 'deepflow.io.Input', { shape: shapeText(shape), dtype: options.dtype ?? 'float32' }, options.label ?? 'Input', options),
        output: (id, options = {}) => layer(id, 'deepflow.io.Output', options.name ? { name: options.name } : {}, options.label ?? 'Output', options),
        layer,
        component,
        /** The standard Linear component (editable weight, transpose, MatMul and bias add). */
        linear: (id, inFeatures, outFeatures, options = {}) => component(id, 'linear', {
            in_features: inFeatures, out_features: outFeatures, ...(options.bias === false ? { bias: false } : {}),
        }, options.label ?? 'Linear', options),
        edge: (from, to, targetHandle) => createEdge({ source: typeof from === 'string' ? from : from.id, target: typeof to === 'string' ? to : to.id, targetHandle }),
        /** Edges between consecutive nodes. */
        chain: (...nodes) => nodes.slice(1).map((node, index) => helpers.edge(nodes[index], node)),
        optimizer: (type = 'Adam', hyperparameters = {}, options = {}) => createOptimizerNode({
            id: options.id ?? 'opt', type, label: options.label ?? type, position: options.position ?? { x: 60, y: 100 },
            hyperparameters: { ...(OPTIMIZER_DEFAULTS[type] ?? {}), ...hyperparameters },
        }),
        loss: (type = 'cross_entropy', outputId = null, options = {}) => createLossNode({
            id: options.id ?? 'loss', type, architectureOutputId: typeof outputId === 'string' ? outputId : outputId?.id ?? null,
            position: options.position ?? { x: 360, y: 100 }, hyperparameters: { ...(LOSS_DEFAULTS[type] ?? {}), ...(options.hyperparameters ?? {}) },
            ...(options.label ? { label: options.label } : {}),
        }),
        /** A training graph with the loss wired to the optimizer, as DeepFlow's storyboards do. */
        training: (optimizer, loss) => {
            const nodes = Object.fromEntries([optimizer, loss].filter(Boolean).map((node) => [node.id, node]));
            const edges = optimizer && loss ? [createTrainingEdge({ id: `te_${loss.id}_${optimizer.id}`, source: loss.id, target: optimizer.id })] : [];
            return { nodes, edges };
        },
        /** The data slice value (Stage B's capture shape). */
        data: (datasetId, options = {}, split = { val: 0.2, test: 0 }) => ({
            datasetId, options: datasetId ? { [datasetId]: { ...options } } : {}, split: { val: 0.2, test: 0, ...split },
        }),
        /** The trainer slice value. */
        trainer: (settings = {}) => ({ settings: { epochs: 50, batchSize: 32, seed: 1, ...settings } }),
        /** Assemble a WorkspaceSnapshot. `training` may be a graph or [optimizer, loss]. */
        snapshot: ({ nodes = [], edges = [], parameters, templates, training, data, trainer, ui, slices = {} } = {}) => {
            const graph = { nodes: Object.fromEntries(nodes.map((node) => [node.id, node])), edges };
            if (parameters?.length) graph.parameters = parameters.map((parameter) => ({ type: 'int', description: parameter.name, ...parameter }));
            const trainingGraph = Array.isArray(training) ? helpers.training(...training) : training ?? { nodes: {}, edges: [] };
            const allSlices = { ...slices };
            if (data !== undefined) allSlices.data = data;
            if (trainer !== undefined) allSlices.trainer = trainer;
            return {
                v: 1,
                savedAt: 0,
                graph,
                templates: { ...(templates ?? registry.templates) },
                training: trainingGraph,
                slices: allSlices,
                ...(ui ? { ui: { ...ui } } : {}),
            };
        },
    };
    return helpers;
}
