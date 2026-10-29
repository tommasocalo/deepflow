/**
 * Use this dataset (docs/edu/SPEC.md §6.9.4): bind a dataset and fit the model to it in one undoable edit.
 *
 * - setInput: the first root Input gets the dataset's shape and dtype.
 * - setOutput: walking back from the first root Output through activations, Dropout, Softmax and LogSoftmax, a
 *   Linear (layer or the editable component) gets out_features = the number of classes. Two classes give 1 output
 *   when the loss is single-output (BCE, MSE, L1) and 2 for cross-entropy; regression gives 1.
 *
 * It always edits the main pane's ROOT graph, whatever level is open. Exported as `useDataset` (the spec's name) and
 * `applyDataset` (it is not a React hook, so components call it by that name).
 */
import { patchLayerParameters, patchNode } from '../../domain/architecture/index.js';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useTrainingStore } from '../../stores/trainingStore.js';
import { effectiveObjective } from '../objective.js';
import { getDataset } from './registry.js';
import { useEduDataStore } from './store.js';

const PASS_THROUGH = new Set([
    'torch.nn.ReLU', 'torch.nn.LeakyReLU', 'torch.nn.GELU', 'torch.nn.Tanh', 'torch.nn.Sigmoid',
    'torch.nn.Dropout', 'torch.nn.Softmax', 'torch.nn.LogSoftmax',
]);
const isLinearComponent = (node, templates) => {
    const template = templates[node.templateId];
    return node.kind === 'supernode' && (template?.standardKey ?? template?.componentStyleKey) === 'linear';
};

/** The Linear that sets the model's output size, or null. */
export function findOutputLinear(graph, templates = {}) {
    const nodes = Object.values(graph?.nodes ?? {});
    let current = nodes.find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Output');
    const seen = new Set();
    while (current && !seen.has(current.id)) {
        seen.add(current.id);
        const incoming = (graph.edges ?? []).filter((edge) => edge.target === current.id && graph.nodes[edge.source]);
        if (incoming.length !== 1) return null;
        const previous = graph.nodes[incoming[0].source];
        if (previous.kind === 'layer' && previous.libraryKey === 'torch.nn.Linear') return { node: previous, component: false };
        if (isLinearComponent(previous, templates)) return { node: previous, component: true };
        if (previous.kind !== 'layer' || !PASS_THROUGH.has(previous.libraryKey)) return null;
        current = previous;
    }
    return null;
}

/** Output units for a dataset given the training graph's loss (§6.9.4). */
export function outputSizeFor(spec, { graph, training, currentSize = null } = {}) {
    if (spec.target.kind !== 'classes') return spec.target.size;
    if (spec.target.size !== 2) return spec.target.size;
    const loss = effectiveObjective(training, graph).loss;
    if (!loss) return Number(currentSize) === 1 ? 1 : 2;
    return loss.type === 'cross_entropy' ? 2 : 1;
}

const currentOutFeatures = ({ node, component }) => Number(component ? node.parameterValues?.out_features : node.parameters?.out_features);

/**
 * @returns {{ datasetId: string, changed: string[], warnings: string[], at: number }}
 */
export function applyDataset(id, { setInput = true, setOutput = true, options, split } = {}) {
    const spec = getDataset(id);
    if (!spec) throw new Error(`Unknown dataset "${id}"`);
    const data = useEduDataStore.getState();
    data.setDataset(id);
    if (options) data.setOptions(options, id);
    if (split) data.setSplit(split);
    const changed = [];
    const warnings = [];
    recordArchitectureEdit(`Use dataset ${spec.name}`, () => {
        const before = useArchitectureStore.getState().graph;
        const templates = useTemplatesStore.getState().templates;
        let graph = before;
        if (setInput) {
            const input = Object.values(graph.nodes).find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Input');
            if (!input) {
                warnings.push('There is no Input block to set.');
            } else if (input.parameters?.shape !== spec.input.shape || input.parameters?.dtype !== spec.input.dtype) {
                graph = patchLayerParameters(graph, input.id, { shape: spec.input.shape, dtype: spec.input.dtype });
                changed.push(`Input → ${spec.input.shape}`);
            } else {
                changed.push(`Input is already ${spec.input.shape}`);
            }
        }
        if (setOutput) {
            const found = findOutputLinear(graph, templates);
            if (!found) {
                warnings.push('Could not set the output size: the last block is not a Linear.');
            } else {
                const size = outputSizeFor(spec, { graph, training: useTrainingStore.getState().graph, currentSize: currentOutFeatures(found) });
                if (currentOutFeatures(found) !== size) {
                    graph = found.component
                        ? patchNode(graph, found.node.id, { parameterValues: { ...found.node.parameterValues, out_features: size } })
                        : patchLayerParameters(graph, found.node.id, { out_features: size });
                    changed.push(`last Linear (${found.node.label}) Output → ${size}`);
                } else {
                    changed.push(`last Linear (${found.node.label}) Output is already ${size}`);
                }
            }
        }
        if (graph !== before) useArchitectureStore.setState({ graph });
    });
    const result = { datasetId: id, changed, warnings, at: Date.now() };
    useEduDataStore.setState({ lastUse: result });
    return result;
}

export { applyDataset as useDataset };
