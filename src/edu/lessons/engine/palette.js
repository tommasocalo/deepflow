import { cachedValidate } from '../../../domain/architecture/levelContext.js';
import { withGlobalParameters } from '../../../domain/expressions.js';
import { recordArchitectureEdit } from '../../../stores/architectureHistory.js';
import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { blockRefOf, runActions } from './actions.js';
import { focusNode, requestFit, requestInsert } from './canvasBridge.js';
import { matchNodes, orderedNodeIds } from './nodes.js';

/**
 * Palette chips (docs/edu/SPEC.md §2 P3, §6.3.1 PaletteItem). A chip adds its block; lesson-only refinements keep a
 * novice out of dead ends (review r1): a chip with `into: { from, to }` splices its block into that wire while the wire
 * exists (M2 s3: ReLU into Linear 1 → Linear 2), and a Linear chip starts at the width that reaches the model's open
 * end instead of DeepFlow's 768. A chip never duplicates a block the learner already added in the step (review r2):
 * an `into` chip moves that block into its wire, any other chip points at it once the step has as many as it asks for.
 */

/** The root nodes a palette item's `into` names, when the wire between them exists; otherwise null. */
export function spliceTarget(item, graph, templates = {}) {
    if (!item?.into?.from || !item.into.to || !graph?.nodes) return null;
    const options = { templates, scope: withGlobalParameters(graph), graph };
    const from = matchNodes(graph, item.into.from, options)[0];
    const to = matchNodes(graph, item.into.to, options)[0];
    if (!from || !to || !(graph.edges ?? []).some((edge) => edge.source === from.id && edge.target === to.id)) return null;
    return { from, to };
}

/** Where a chip's block goes when it cannot splice: beside the middle of the wire it belongs in, clear of the column. */
export function besideWire(item, graph, templates = {}) {
    if (!item?.into?.from || !item.into.to || !graph?.nodes) return null;
    const options = { templates, scope: withGlobalParameters(graph), graph };
    const from = matchNodes(graph, item.into.from, options)[0];
    const to = matchNodes(graph, item.into.to, options)[0];
    if (!from || !to) return null;
    return { x: Math.max(from.position.x, to.position.x) + 260, y: Math.round((from.position.y + to.position.y) / 2) };
}

/**
 * The feature width at the model's open end: the block furthest from the Input that feeds nothing yet (the Output
 * aside), or null. M2 s7 starts with an Input of width 2; once a Linear 2 → 8 hangs off it, the open end is 8 wide.
 */
export function openEndWidth(graph, templates = {}) {
    if (!graph?.nodes) return null;
    const input = Object.values(graph.nodes).find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Input');
    if (!input) return null;
    const report = cachedValidate(graph, templates, withGlobalParameters(graph), {});
    const next = new Map();
    for (const edge of graph.edges ?? []) {
        if (!next.has(edge.source)) next.set(edge.source, []);
        next.get(edge.source).push(edge.target);
    }
    const depth = new Map([[input.id, 0]]);
    const queue = [input.id];
    while (queue.length) {
        const id = queue.shift();
        for (const target of next.get(id) ?? []) {
            if (depth.has(target)) continue;
            depth.set(target, depth.get(id) + 1);
            queue.push(target);
        }
    }
    const open = [...depth.keys()].filter((id) => !next.has(id) && graph.nodes[id]?.libraryKey !== 'deepflow.io.Output');
    open.sort((left, right) => depth.get(right) - depth.get(left));
    const shape = open.length ? report.shapes?.[open[0]] : null;
    const width = Array.isArray(shape) ? shape.at(-1) : null;
    return Number.isSafeInteger(width) && width > 0 ? width : null;
}

/** The parameters a chip gives its new block: the item's own, and for a Linear the open end's width. */
export function chipParams(item, graph, templates = {}) {
    const params = { ...(item?.params ?? {}) };
    if (item?.block === 'component:linear') {
        const width = openEndWidth(graph, templates);
        if (width && params.in_features === undefined) params.in_features = width;
        if (params.out_features === undefined && (params.in_features ?? width)) params.out_features = params.in_features ?? width;
    }
    return Object.keys(params).length ? params : null;
}

/** The blocks of a palette item's kind added since the step began (not in `entryNodeIds`), in canvas order. */
export function addedBlocks(block, graph, templates = {}, entryNodeIds = null) {
    if (!Array.isArray(entryNodeIds) || !graph?.nodes) return [];
    const entry = new Set(entryNodeIds);
    return orderedNodeIds(graph).map((id) => graph.nodes[id]).filter((node) => !entry.has(node.id) && blockRefOf(node, templates) === block);
}

/** How many more blocks of an item's kind the step asks for (its `count`, 1 by default). */
export function blocksLeft(item, graph, templates = {}, entryNodeIds = null) {
    return Math.max(0, (item?.count ?? 1) - addedBlocks(item?.block, graph, templates, entryNodeIds).length);
}

/**
 * The palette is used once every block it offers (the distractor aside) is on the canvas as often as the step asks:
 * the spotlight then moves off the chips (review r2), and off the ribbon buttons that add the same blocks.
 */
export function paletteUsed(items = [], graph, templates = {}, entryNodeIds = null) {
    const wanted = (items ?? []).filter((item) => !item.distractor);
    if (!wanted.length || !Array.isArray(entryNodeIds)) return false;
    return wanted.every((item) => blocksLeft(item, graph, templates, entryNodeIds) === 0);
}

function paletteEntry(block) {
    return block.startsWith('component:') ? { standardComponentKey: block.slice('component:'.length) } : { key: block };
}

/** Add a block at `position` with the store's own action (no mounted canvas, or a place the lesson chose). */
function addAt(block, position) {
    const state = useArchitectureStore.getState();
    const id = block.startsWith('component:') ? state.addStandardComponent(block.slice('component:'.length), position) : state.addLayerByLibraryKey(block, position);
    if (id) useArchitectureStore.getState().selectOnly(id);
    return id;
}

/** Without a mounted canvas (tests, a hidden tab), add the block beside the column. */
function besideColumn() {
    const nodes = Object.values(useArchitectureStore.getState().graph.nodes);
    const xs = nodes.map((node) => node.position.x);
    const ys = nodes.map((node) => node.position.y).sort((a, b) => a - b);
    return { x: (xs.length ? Math.max(...xs) : 400) + 260, y: ys.length ? ys[Math.floor(ys.length / 2)] : 200 };
}

function applyParams(id, block, params) {
    if (!id || !params) return;
    const state = useArchitectureStore.getState();
    const node = state.graph.nodes[id];
    if (block.startsWith('component:')) state.patchNode(id, { parameterValues: { ...node?.parameterValues, ...params } });
    else state.patchLayerParameters(id, params);
}

/** Wait until the architecture canvas is mounted, then insert through the canvas bridge (DeepFlow's own placement). */
async function insertBlock(item, label) {
    const ui = useEduUiStore.getState();
    if (ui.activeTab !== 'architecture') ui.setTab('architecture');
    const { graph } = useArchitectureStore.getState();
    const templates = useTemplatesStore.getState().templates;
    const params = chipParams(item, graph, templates);
    const place = besideWire(item, graph, templates);
    // One undo step adds the block with its starting sizes.
    const add = (insert) => recordArchitectureEdit(label, () => { const id = insert(); applyParams(id, item.block, params); return id; });
    if (place) return add(() => addAt(item.block, place));
    for (let attempt = 0; attempt < 20; attempt++) {
        const id = add(() => requestInsert(paletteEntry(item.block)));
        if (id) return id;
        if (typeof window === 'undefined' || !window.requestAnimationFrame) break;
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return add(() => addAt(item.block, besideColumn()));
}

/** The block of `item`'s kind that now sits between the two blocks its `into` names, if any. */
function blockBetween(item, graph, templates) {
    const anchors = item?.into ? resolveInto(item, graph, templates) : null;
    if (!anchors) return null;
    const after = new Set(graph.edges.filter((edge) => edge.source === anchors.from.id).map((edge) => edge.target));
    return graph.edges.map((edge) => edge.target === anchors.to.id && after.has(edge.source) ? graph.nodes[edge.source] : null)
        .find((node) => node && blockRefOf(node, templates) === item.block) ?? null;
}

function resolveInto(item, graph, templates) {
    const options = { templates, scope: withGlobalParameters(graph), graph };
    const from = matchNodes(graph, item.into.from, options)[0];
    const to = matchNodes(graph, item.into.to, options)[0];
    return from && to ? { from, to } : null;
}

const wiredBothWays = (node, graph) => graph.edges.some((edge) => edge.target === node.id) && graph.edges.some((edge) => edge.source === node.id);

/**
 * A chip click. An `into` chip splices its block into that wire with the lesson's insertBetween (one undo step), in
 * the column; a block of its kind the learner added in this step is moved there, out of any wire it sits in, rather
 * than a second one added (review r2: a ribbon ReLU plus the chip gave two ReLUs). Without that wire and such a block
 * it adds its block beside the wire. Any other chip adds its block until the step has as many as it asks for; after
 * that it selects one of them that still lacks a wire instead of adding another.
 * @returns {Promise<{ id: string|null, spliced: boolean, adopted: boolean }>}
 */
export async function insertPaletteItem(item, { entryNodeIds = null, name = item?.block } = {}) {
    const store = useArchitectureStore.getState();
    if (store.currentPath.length) store.goToLevel(0);
    const { graph } = useArchitectureStore.getState();
    const templates = useTemplatesStore.getState().templates;
    const added = addedBlocks(item.block, graph, templates, entryNodeIds);
    const label = `Add ${name}`;
    if (item.into && resolveInto(item, graph, templates) && (spliceTarget(item, graph, templates) || added.length)) {
        const before = new Set(Object.keys(graph.nodes));
        const result = await runActions([{ do: 'insertBetween', block: item.block, from: item.into.from, to: item.into.to, params: item.params }],
            { label: `insert ${name}`, entryNodeIds, lift: true });
        if (result.ok) {
            const after = useArchitectureStore.getState().graph;
            const id = Object.keys(after.nodes).find((key) => !before.has(key)) ?? blockBetween(item, after, templates)?.id ?? null;
            if (id) useArchitectureStore.getState().selectOnly(id);
            // The column grew by one row: keep the whole model in view.
            setTimeout(requestFit, 60);
            return { id, spliced: true, adopted: !!id && before.has(id) };
        }
    } else if (!item.into && added.length >= (item.count ?? 1)) {
        const loose = added.find((node) => !wiredBothWays(node, graph)) ?? added.at(-1);
        useArchitectureStore.getState().selectOnly(loose.id);
        focusNode(loose.id);
        return { id: loose.id, spliced: false, adopted: true };
    }
    return { id: await insertBlock(item, label), spliced: false, adopted: false };
}
