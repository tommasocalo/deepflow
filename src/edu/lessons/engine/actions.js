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
import { describeMatch, matchNodes, orderedNodeIds } from './nodes.js';
import { loadFixtureIntoWorkspace, runStoryboard } from './workspace.js';

/**
 * Lesson actions (docs/edu/SPEC.md §6.5). `runActions` runs a list in order and awaits each action. Consecutive
 * graph edits share one recordArchitectureEdit('Lesson: <label>'), so a whole Show me is a single Undo step;
 * workspace replacements (loadFixture, loadStoryboard) start a fresh history; training runs are awaited between
 * batches. Graph actions address the main pane's root graph.
 *
 * Insert actions (insertBetween, insertAfter, buildChain) first reuse blocks of the same kind that the learner added
 * in this step and left unwired, or wired only to the blocks the action connects (a palette chip, Components ▾).
 * Show me then completes the learner's attempt instead of adding a duplicate beside it. The caller passes the ids of
 * the root nodes at step entry as `ctx.entryNodeIds`; without them nothing is reused.
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

/**
 * The shared definition (template) a path of components leads to from the root, e.g. [{ name: 'ResBlock' }]: editing
 * it changes every instance and every copy of a repeat at once (M6: the three ResBlock copies follow one definition).
 */
export function resolveDefinition(path = []) {
    if (!Array.isArray(path) || !path.length) throw new ActionError('A component path is required');
    let graph = rootGraph();
    let template = null;
    for (const match of path) {
        const options = { templates: templates(), scope: template ? {} : withGlobalParameters(graph), graph };
        const node = matchNodes(graph, match, options).find((item) => ['supernode', 'repeat'].includes(item.kind));
        if (!node) throw new ActionError(`No component matches ${describeMatch(match)}`);
        template = templates()[node.templateId];
        if (!template?.graph) throw new ActionError(`The definition of ${describeMatch(match)} is missing`);
        graph = template.graph;
    }
    return template;
}

/** setParam on a block inside a shared definition (`in`): the template changes, so all its instances do. */
function setDefinitionParam({ in: path, node: match, name, value }) {
    const template = resolveDefinition(path);
    const node = matchNodes(template.graph, match ?? {}, { templates: templates(), scope: {}, graph: template.graph })[0];
    if (!node) throw new ActionError(`No block matches ${describeMatch(match ?? {})} in ${template.name ?? 'the component'}`);
    const store = useTemplatesStore.getState();
    if (node.kind === 'layer') store.patchTemplateLayerParameters(template.id, node.id, { [name]: value });
    else store.patchTemplateNode(template.id, node.id, { parameterValues: { ...node.parameterValues, [name]: value } });
}

const isComponentRef = (block) => typeof block === 'string' && block.startsWith('component:');

function applyBlockSettings(id, block, { params, label } = {}) {
    if (params && isComponentRef(block)) architecture().patchNode(id, { parameterValues: { ...rootGraph().nodes[id].parameterValues, ...params } });
    else if (params) architecture().patchLayerParameters(id, params);
    if (label) architecture().patchNode(id, { label });
}

/** BlockRef → a new node at `position`; returns its id. */
export function createBlock(block, { params, label, position }) {
    let id = null;
    if (isComponentRef(block)) {
        const key = block.slice('component:'.length);
        if (!STANDARD_COMPONENTS.some((component) => component.key === key)) throw new ActionError(`Unknown component ${key}`);
        id = architecture().addStandardComponent(key, position);
    } else {
        if (!getEntry(block)) throw new ActionError(`Unknown block ${block}`);
        id = architecture().addLayerByLibraryKey(block, position);
    }
    if (!id) throw new ActionError(`Could not add ${block}`);
    applyBlockSettings(id, block, { params, label });
    return id;
}

/** The BlockRef a root node was made from: its library key, or `component:<key>` for a standard component. */
export function blockRefOf(node, templateMap = templates()) {
    if (node?.kind === 'layer') return node.libraryKey ?? null;
    if (node?.kind !== 'supernode') return null;
    const template = templateMap[node.templateId];
    const key = template?.standardKey ?? template?.componentStyleKey;
    return key ? `component:${key}` : null;
}

/**
 * Blocks added since the step began whose wires, if any, all end at `anchors` or at another such block: the
 * candidates an insert action may adopt, in canvas order. Empty without `ctx.entryNodeIds`.
 */
function adoptableBlocks(ctx, anchors, refs) {
    if (!Array.isArray(ctx?.entryNodeIds)) return [];
    const graph = rootGraph();
    const entry = new Set(ctx.entryNodeIds);
    const wanted = new Set(refs);
    const candidates = new Set(Object.values(graph.nodes)
        .filter((node) => !entry.has(node.id) && wanted.has(blockRefOf(node)))
        .map((node) => node.id));
    const allowed = (id) => anchors.has(id) || candidates.has(id);
    // Dropping one candidate can disqualify a neighbour wired to it: repeat until nothing changes.
    for (let changed = true; changed;) {
        changed = false;
        for (const edge of graph.edges) {
            for (const [end, other] of [[edge.source, edge.target], [edge.target, edge.source]]) {
                if (candidates.has(end) && !allowed(other)) { candidates.delete(end); changed = true; }
            }
        }
    }
    return orderedNodeIds(graph).filter((id) => candidates.has(id)).map((id) => graph.nodes[id]);
}

/** Pick one adoptable node per block (in order) and unwire the picked ones, so the action can wire them afresh. */
function adoptBlocks(blocks, ctx, anchors) {
    const spare = adoptableBlocks(ctx, anchors, blocks.map((item) => item.block));
    const picked = blocks.map((item) => {
        const index = spare.findIndex((node) => blockRefOf(node) === item.block);
        return index >= 0 ? spare.splice(index, 1)[0].id : null;
    });
    const ids = new Set(picked.filter(Boolean));
    for (const edge of rootGraph().edges.filter((item) => ids.has(item.source) || ids.has(item.target))) architecture().removeEdge(edge.id);
    return picked;
}

/**
 * Make room for `rows` new blocks below `node`: the root nodes lower on the canvas move down together, only as far
 * as the new rows need (a fixture's Output several rows below keeps its place), so the column stays compact.
 */
function pushDown(node, rows, except = new Set()) {
    const below = Object.values(rootGraph().nodes).filter((other) => !except.has(other.id) && other.position.y > node.position.y);
    if (!below.length) return;
    const shift = node.position.y + (rows + 1) * ROW_GAP - Math.min(...below.map((other) => other.position.y));
    if (shift > 0) architecture().moveNodes(below.map((other) => ({ id: other.id, position: { x: other.position.x, y: other.position.y + shift } })));
}

function connectOrThrow(source, target, handles = {}) {
    const exists = rootGraph().edges.some((edge) => edge.source === source && edge.target === target
        && (edge.targetHandle ?? null) === (handles.targetHandle ?? null));
    if (exists) return;
    const id = architecture().connect({ source, target, sourceHandle: handles.sourceHandle, targetHandle: handles.targetHandle });
    if (!id) throw new ActionError('Could not connect the blocks');
}

/** Place `blocks` in the column below `fromNode`: adopted nodes (`reuse[i]`) move into their slot, the rest are new. */
function insertBlocks(fromNode, blocks, reuse = []) {
    pushDown(fromNode, blocks.length, new Set(reuse.filter(Boolean)));
    return blocks.map((item, index) => {
        const position = { x: fromNode.position.x, y: fromNode.position.y + (index + 1) * ROW_GAP };
        const id = reuse[index];
        if (!id) return createBlock(item.block, { params: item.params, label: item.label, position });
        architecture().moveNodes([{ id, position }]);
        applyBlockSettings(id, item.block, item);
        return id;
    });
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
    setParam(action) {
        const { node: match, name, value } = action;
        if (action.in) return setDefinitionParam(action);
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
    insertBetween({ block, from, to, params, label }, ctx) {
        goRoot();
        const source = resolveRootNode(from);
        const target = resolveRootNode(to);
        const reuse = adoptBlocks([{ block }], ctx, new Set([source.id, target.id]));
        const edges = rootGraph().edges.filter((edge) => edge.source === source.id && edge.target === target.id);
        const [id] = insertBlocks(source, [{ block, params, label }], reuse);
        for (const edge of edges) architecture().removeEdge(edge.id);
        connectOrThrow(source.id, id, { sourceHandle: edges[0]?.sourceHandle });
        connectOrThrow(id, target.id, { targetHandle: edges[0]?.targetHandle });
        return id;
    },
    insertAfter({ block, after, params, label }, ctx) {
        goRoot();
        const source = resolveRootNode(after);
        const next = rootGraph().edges.filter((edge) => edge.source === source.id).map((edge) => edge.target);
        const reuse = adoptBlocks([{ block }], ctx, new Set([source.id, ...next]));
        const outgoing = rootGraph().edges.filter((edge) => edge.source === source.id);
        const [id] = insertBlocks(source, [{ block, params, label }], reuse);
        for (const edge of outgoing) {
            architecture().removeEdge(edge.id);
            connectOrThrow(id, edge.target, { targetHandle: edge.targetHandle });
        }
        connectOrThrow(source.id, id, { sourceHandle: outgoing[0]?.sourceHandle });
        return id;
    },
    buildChain({ from, to, blocks = [] }, ctx) {
        goRoot();
        const source = resolveRootNode(from);
        const target = resolveRootNode(to);
        const reuse = adoptBlocks(blocks, ctx, new Set([source.id, target.id]));
        for (const edge of rootGraph().edges.filter((item) => item.source === source.id && item.target === target.id)) architecture().removeEdge(edge.id);
        const ids = insertBlocks(source, blocks, reuse);
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
 * @param {{ module?, headless?, label?, answer?: (id, value) => void, answers?: object, entryNodeIds?: string[]|null }} ctx
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
