import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { recordArchitectureEdit, useArchitectureHistory } from '../../../stores/architectureHistory.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';
import { resolveGraphLevel } from '../../../domain/architecture/operations.js';
import { STANDARD_COMPONENTS } from '../../../domain/components/index.js';
import { withGlobalParameters } from '../../../domain/expressions.js';
import { getEntry } from '../../../domain/library/index.js';
import { createLossNode, createOptimizerNode, LOSS_DEFAULTS, OPTIMIZER_DEFAULTS } from '../../../domain/training/index.js';
import { callSource, readSource } from '../../bridge.js';
import { effectiveObjective } from '../../objective.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { ROW_GAP } from './fixtures.js';
import { describeMatch, matchNodes } from './nodes.js';
import { loadFixtureIntoWorkspace, runStoryboard } from './workspace.js';

/**
 * Lesson actions (docs/edu/SPEC.md §6.5). `runActions` runs a list in order and awaits each action. Consecutive
 * graph edits share one recordArchitectureEdit('Lesson: <label>'), so a whole Show me is a single Undo step;
 * workspace replacements (loadFixture, loadStoryboard) start a fresh history; training runs are awaited between
 * batches. Graph actions address the main pane's root graph.
 */
export const ACTIONS = Object.freeze([
    'loadFixture', 'loadStoryboard', 'useDataset', 'setDatasetOptions', 'setParam', 'setGlobal', 'insertBetween', 'insertAfter',
    'buildChain', 'connect', 'disconnect', 'remove', 'group', 'repeat', 'openLevel', 'setOptimizer', 'setLoss', 'setTrainer', 'train',
    'evaluateInit', 'answer', 'setUi',
]);
const ASYNC = new Set(['train', 'evaluateInit']);
const BARRIERS = new Set(['loadFixture', 'loadStoryboard']);

const architecture = () => useArchitectureStore.getState();
const rootGraph = () => architecture().graph;
const templates = () => useTemplatesStore.getState().templates;

class ActionError extends Error {}

function lookupOptions() {
    const graph = rootGraph();
    return { templates: templates(), scope: withGlobalParameters(graph), graph };
}

/** Resolve a NodeMatch on the root graph, or throw a readable error. */
export function resolveRootNode(match) {
    if (!match) throw new ActionError('A node match is required');
    const node = matchNodes(rootGraph(), match, lookupOptions())[0];
    if (!node) throw new ActionError(`No block matches ${describeMatch(match)}`);
    return node;
}

function goRoot() {
    if (architecture().currentPath.length) architecture().goToLevel(0);
}

/** BlockRef → a new node at `position`; returns its id. */
export function createBlock(block, { params, label, position }) {
    let id = null;
    if (typeof block === 'string' && block.startsWith('component:')) {
        const key = block.slice('component:'.length);
        if (!STANDARD_COMPONENTS.some((component) => component.key === key)) throw new ActionError(`Unknown component ${key}`);
        id = architecture().addStandardComponent(key, position);
        if (id && params) architecture().patchNode(id, { parameterValues: { ...rootGraph().nodes[id].parameterValues, ...params } });
    } else {
        if (!getEntry(block)) throw new ActionError(`Unknown block ${block}`);
        id = architecture().addLayerByLibraryKey(block, position);
        if (id && params) architecture().patchLayerParameters(id, params);
    }
    if (!id) throw new ActionError(`Could not add ${block}`);
    if (label) architecture().patchNode(id, { label });
    return id;
}

/** Make room below `node`: every root node lower on the canvas moves down by rows × 130 px. */
function pushDown(node, rows, except = new Set()) {
    const moves = Object.values(rootGraph().nodes)
        .filter((other) => !except.has(other.id) && other.position.y > node.position.y)
        .map((other) => ({ id: other.id, position: { x: other.position.x, y: other.position.y + rows * ROW_GAP } }));
    if (moves.length) architecture().moveNodes(moves);
}

function connectOrThrow(source, target, handles = {}) {
    const exists = rootGraph().edges.some((edge) => edge.source === source && edge.target === target
        && (edge.targetHandle ?? null) === (handles.targetHandle ?? null));
    if (exists) return;
    const id = architecture().connect({ source, target, sourceHandle: handles.sourceHandle, targetHandle: handles.targetHandle });
    if (!id) throw new ActionError('Could not connect the blocks');
}

function insertBlocks(fromNode, blocks) {
    pushDown(fromNode, blocks.length);
    return blocks.map((item, index) => createBlock(item.block, {
        params: item.params, label: item.label,
        position: { x: fromNode.position.x, y: fromNode.position.y + (index + 1) * ROW_GAP },
    }));
}

function defaultTrainingLabel(kind, type) {
    const position = { x: 0, y: 0 };
    return kind === 'optimizer' ? createOptimizerNode({ type, position }).label : createLossNode({ type, position }).label;
}

const relabel = (node, kind, nextType) => (
    [node.type, defaultTrainingLabel(kind, node.type)].includes(node.label) ? (kind === 'optimizer' ? nextType : defaultTrainingLabel(kind, nextType)) : node.label
);

function rootOutputId() {
    return Object.values(rootGraph().nodes).find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Output')?.id ?? null;
}

/** Synchronous actions; they run inside the batch's history transaction. */
const SYNC = {
    useDataset(action) {
        const { id, setInput = true, setOutput = true, options, split } = action;
        if (!readSource('data')) throw new ActionError('The Data tab is not available');
        return callSource('data', 'useDataset', id, { setInput, setOutput, options, split });
    },
    setDatasetOptions(action) {
        if (!readSource('data')) return console.warn('[edu actions] no data source: setDatasetOptions skipped');
        if (action.options) callSource('data', 'setOptions', action.options);
        if (action.split) callSource('data', 'setSplit', action.split);
        return undefined;
    },
    setParam({ node: match, name, value }) {
        goRoot();
        const node = resolveRootNode(match);
        if (node.kind === 'layer') architecture().patchLayerParameters(node.id, { [name]: value });
        else if (node.kind === 'supernode') architecture().patchNode(node.id, { parameterValues: { ...node.parameterValues, [name]: value } });
        else if (node.kind === 'repeat') {
            if (name === 'count') architecture().patchRepeatNode(node.id, { count: value });
            else architecture().patchNode(node.id, { bindings: { ...node.bindings, [name]: value } });
        }
    },
    setGlobal({ name, value }) {
        if (!(rootGraph().parameters ?? []).some((parameter) => parameter.name === name)) throw new ActionError(`No global parameter ${name}`);
        architecture().patchGlobalParameter(name, { default: value });
    },
    insertBetween({ block, from, to, params, label }) {
        goRoot();
        const source = resolveRootNode(from);
        const target = resolveRootNode(to);
        const edges = rootGraph().edges.filter((edge) => edge.source === source.id && edge.target === target.id);
        const [id] = insertBlocks(source, [{ block, params, label }]);
        for (const edge of edges) architecture().removeEdge(edge.id);
        connectOrThrow(source.id, id, { sourceHandle: edges[0]?.sourceHandle });
        connectOrThrow(id, target.id, { targetHandle: edges[0]?.targetHandle });
        return id;
    },
    insertAfter({ block, after, params, label }) {
        goRoot();
        const source = resolveRootNode(after);
        const outgoing = rootGraph().edges.filter((edge) => edge.source === source.id);
        const [id] = insertBlocks(source, [{ block, params, label }]);
        for (const edge of outgoing) {
            architecture().removeEdge(edge.id);
            connectOrThrow(id, edge.target, { targetHandle: edge.targetHandle });
        }
        connectOrThrow(source.id, id, { sourceHandle: outgoing[0]?.sourceHandle });
        return id;
    },
    buildChain({ from, to, blocks = [] }) {
        goRoot();
        const source = resolveRootNode(from);
        const target = resolveRootNode(to);
        for (const edge of rootGraph().edges.filter((item) => item.source === source.id && item.target === target.id)) architecture().removeEdge(edge.id);
        const ids = insertBlocks(source, blocks);
        const chain = [source.id, ...ids, target.id];
        for (let index = 1; index < chain.length; index++) connectOrThrow(chain[index - 1], chain[index]);
        return ids;
    },
    connect({ from, to, targetHandle }) {
        goRoot();
        connectOrThrow(resolveRootNode(from).id, resolveRootNode(to).id, { targetHandle });
    },
    disconnect({ from, to }) {
        goRoot();
        const source = resolveRootNode(from);
        const target = resolveRootNode(to);
        for (const edge of rootGraph().edges.filter((item) => item.source === source.id && item.target === target.id)) architecture().removeEdge(edge.id);
    },
    remove({ node }) {
        goRoot();
        architecture().removeNode(resolveRootNode(node).id);
    },
    group({ nodes = [], name = 'Component' }) {
        goRoot();
        const ids = nodes.map((match) => resolveRootNode(match).id);
        const templateId = architecture().groupSelection(ids, name);
        if (!templateId) throw new ActionError('These blocks cannot be grouped: they need one input and one output connection');
        return templateId;
    },
    repeat({ node: match, count, mode = 'sequential' }) {
        goRoot();
        const node = resolveRootNode(match);
        if (node.kind === 'supernode') {
            if (!architecture().convertInstanceToRepeat(node.id, count, mode)) throw new ActionError('Could not repeat this component');
        } else if (node.kind === 'repeat') {
            architecture().patchRepeatNode(node.id, { count, mode });
        } else {
            throw new ActionError('Only components can be repeated: group the blocks first');
        }
    },
    openLevel({ path = [] }) {
        const root = rootGraph();
        const ids = [];
        let graph = root;
        for (const match of path) {
            const node = matchNodes(graph, match, { templates: templates(), scope: withGlobalParameters(root), graph })[0];
            if (!node) throw new ActionError(`No block matches ${describeMatch(match)}`);
            ids.push(node.id);
            graph = resolveGraphLevel(root, ids, templates(), architecture().inspectedIterations).graph;
        }
        if (!architecture().openLevel(ids)) throw new ActionError('Could not open that level');
    },
    setOptimizer({ type, hyperparameters = {} }) {
        const training = useTrainingStore.getState();
        let { optimizer, loss } = effectiveObjective(training.graph, rootGraph());
        if (!optimizer) {
            const id = training.addOptimizer(type ?? 'Adam', { x: 60, y: 100 });
            if (loss) training.connect({ source: loss.id, target: id });
            optimizer = useTrainingStore.getState().graph.nodes[id];
        }
        if (type && type !== optimizer.type) {
            training.patchNode(optimizer.id, {
                type, label: relabel(optimizer, 'optimizer', type), hyperparameters: { ...(OPTIMIZER_DEFAULTS[type] ?? {}), ...hyperparameters },
            });
        } else if (Object.keys(hyperparameters).length) {
            training.patchHyperparameters(optimizer.id, hyperparameters);
        }
    },
    setLoss({ type }) {
        const training = useTrainingStore.getState();
        const { optimizer, loss } = effectiveObjective(training.graph, rootGraph());
        const outputId = rootOutputId();
        if (!loss) {
            const id = training.addLoss(type, { x: 360, y: 100 }, outputId);
            if (optimizer) training.connect({ source: id, target: optimizer.id });
            return;
        }
        training.patchNode(loss.id, {
            type,
            label: loss.type === type ? loss.label : relabel(loss, 'loss', type),
            hyperparameters: loss.type === type ? loss.hyperparameters : { ...(LOSS_DEFAULTS[type] ?? {}) },
            ...(!loss.architectureOutputId && outputId ? { architectureOutputId: outputId } : {}),
        });
    },
    setTrainer({ epochs, batchSize, seed }) {
        if (!readSource('trainer')) return console.warn('[edu actions] no trainer source: setTrainer skipped');
        const patch = Object.fromEntries(Object.entries({ epochs, batchSize, seed }).filter(([, value]) => value !== undefined));
        return callSource('trainer', 'setSettings', patch);
    },
    answer({ id, value }, ctx) {
        if (ctx.answer) return ctx.answer(id, value);
        if (ctx.answers) ctx.answers[id] = { value, correct: true, at: Date.now(), source: 'show-me' };
        return undefined;
    },
    setUi({ tab, showDimensions, showGradientFlow, codeOpen, codeTab }) {
        const ui = useEduUiStore.getState();
        if (tab) ui.setTab(tab);
        if (showDimensions !== undefined) architecture().setShowDimensions(showDimensions);
        if (showGradientFlow !== undefined) architecture().setShowGradientFlow(showGradientFlow);
        if (codeOpen === true) ui.openCode(codeTab);
        else if (codeOpen === false) ui.closeCode();
        else if (codeTab) ui.setCodeTab(codeTab);
    },
};

async function runTrainer(action, ctx) {
    if (!readSource('trainer')) throw new ActionError('The trainer is not available');
    const run = await callSource('trainer', action === 'train' ? 'start' : 'evaluateInit', { sync: !!ctx.headless });
    if (run?.status === 'failed') throw new ActionError(run.error?.message ?? run.error ?? 'Training failed');
    return run;
}

async function runBarrier(action, ctx) {
    if (action.do === 'loadFixture') {
        if (!ctx.module) throw new ActionError('loadFixture needs a module');
        await loadFixtureIntoWorkspace(ctx.module, action.id, {
            runAfter: async (after) => {
                const result = await runActions(after, { ...ctx, label: 'fixture' });
                if (!result.ok) throw new ActionError(result.error);
            },
        });
    } else {
        runStoryboard(action.id, { configuration: action.configuration, freshTemplates: true });
    }
}

/**
 * @param {object[]} actions ActionSpec list: { do: 'setParam', … }
 * @param {{ module?, headless?, label?, answer?: (id, value) => void, answers?: object }} ctx
 * @returns {Promise<{ ok: boolean, error?: string, action?: object }>}
 */
export async function runActions(actions = [], ctx = {}) {
    const label = `Lesson: ${ctx.label ?? 'show me'}`;
    let index = 0;
    while (index < actions.length) {
        const action = actions[index];
        try {
            if (!ACTIONS.includes(action?.do)) throw new ActionError(`Unknown action ${action?.do}`);
            if (BARRIERS.has(action.do)) {
                await runBarrier(action, ctx);
                index++;
                continue;
            }
            if (ASYNC.has(action.do)) {
                await runTrainer(action.do, ctx);
                index++;
                continue;
            }
            const batch = [];
            while (index < actions.length && ACTIONS.includes(actions[index]?.do) && !BARRIERS.has(actions[index].do) && !ASYNC.has(actions[index].do)) {
                batch.push(actions[index++]);
            }
            const before = useArchitectureHistory.getState().undoCount;
            const pending = [];
            try {
                recordArchitectureEdit(label, () => {
                    for (const item of batch) {
                        const result = SYNC[item.do](item, ctx);
                        if (result && typeof result.then === 'function') pending.push(result);
                    }
                });
            } catch (error) {
                // Leave the document as it was: undo the partial edit the failed batch recorded.
                if (useArchitectureHistory.getState().undoCount > before) useArchitectureHistory.getState().undo();
                throw error;
            }
            for (const promise of pending) await promise;
        } catch (error) {
            return { ok: false, error: error?.message ?? String(error), action };
        }
    }
    return { ok: true };
}
