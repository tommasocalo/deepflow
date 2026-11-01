import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';
import { resolveGraphLevel } from '../../../domain/architecture/operations.js';
import { withGlobalParameters } from '../../../domain/expressions.js';
import { effectiveObjective } from '../../objective.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { matchNodes } from './nodes.js';
import { paletteUsed } from './palette.js';

/**
 * Spotlight targets (docs/edu/SPEC.md §6.6). A TargetRef resolves to an element at render time. Ids and labels are
 * compared in JavaScript and never interpolated into selectors. Unknown or hidden targets resolve to null.
 */
const RIBBON_LABELS = {
    'ribbon.components': 'Browse components',
    'ribbon.operations': 'Insert operations',
    'ribbon.group': 'Group selection into component',
    'ribbon.sequential': 'Repeat selection sequentially',
    'ribbon.parameters': 'Edit parameters',
    'ribbon.dimensions': 'Show dimensions',
    'ribbon.gradients': 'Show gradient flow',
};
const EDU_TARGETS = {
    'data.use': 'use-dataset',
    'data.preview': 'dataset-preview',
    'data.options': 'dataset-options',
    'trainer.run': 'trainer-run',
    'trainer.lr': 'trainer-lr',
    'trainer.curves': 'trainer-curves',
    'trainer.boundary': 'trainer-boundary',
    'trainer.confusion': 'trainer-confusion',
    'trainer.initLoss': 'trainer-init-loss',
    'code.tab:model': 'code-tab-model',
    'code.tab:train': 'code-tab-train',
    'code.notebook': 'code-notebook',
    'code.colabNote': 'code-colab-note',
    'lesson.palette': 'lesson-palette',
};
const TAB_TARGETS = ['tab.learn', 'tab.architecture', 'tab.data', 'tab.training'];

/** Static TargetRef names (node targets are objects). */
export const TARGET_NAMES = Object.freeze([
    ...TAB_TARGETS, 'toolbar.code', ...Object.keys(RIBBON_LABELS), 'ribbon.instances', 'graph.checks', ...Object.keys(EDU_TARGETS),
    'training.loss', 'training.optimizer',
]);

export function isValidTargetRef(ref) {
    if (typeof ref === 'string') {
        return TARGET_NAMES.includes(ref) || /^field:[A-Za-z_][\w]*$/.test(ref) || /^data\.dataset:[\w-]+$/.test(ref);
    }
    return !!ref && typeof ref === 'object' && (typeof ref.node === 'object' || typeof ref.nodeOpen === 'object');
}

const isJsdom = (view) => view?.navigator?.userAgent?.includes('jsdom') === true;

/** Rendered: connected and not display:none, visibility:hidden or [hidden]. */
export function isRendered(element) {
    if (!element?.isConnected) return false;
    if (element.closest?.('[hidden]')) return false;
    const view = element.ownerDocument?.defaultView;
    const style = view?.getComputedStyle?.(element);
    if (style && (style.display === 'none' || style.visibility === 'hidden')) return false;
    // jsdom has no layout: treat connected, styled-visible elements as rendered there.
    return isJsdom(view) || (element.getClientRects?.().length ?? 1) > 0;
}

/**
 * Rendered with its centre on screen: inside the window and inside every scrolling or clipping box around it. A
 * canvas node panned out of its canvas, or a panel row scrolled away, is not visible, so the panel offers
 * "Where is it?".
 */
export function isVisible(element) {
    if (!isRendered(element)) return false;
    const view = element.ownerDocument.defaultView;
    if (isJsdom(view) || !element.getBoundingClientRect) return true;
    const box = element.getBoundingClientRect();
    const x = box.left + box.width / 2;
    const y = box.top + box.height / 2;
    const inside = (area) => x >= area.left && x <= area.right && y >= area.top && y <= area.bottom;
    if (!inside({ left: 0, top: 0, right: view.innerWidth, bottom: view.innerHeight })) return false;
    for (let node = element.parentElement; node && node !== element.ownerDocument.body; node = node.parentElement) {
        const style = view.getComputedStyle(node);
        if (/auto|scroll|hidden|clip/.test(`${style.overflowX} ${style.overflowY}`) && !inside(node.getBoundingClientRect())) return false;
    }
    // Covered by something on top (an open menu, the code panel, the panel's sticky dock): not visible, so the ring
    // does not paint over it and "Where is it?" can bring the target out (review r1: the ring over the Examples menu).
    const top = element.ownerDocument.elementFromPoint?.(x, y);
    if (top && !element.contains(top) && !top.contains(element)) return false;
    return true;
}

const all = (root, selector) => [...(root?.querySelectorAll?.(selector) ?? [])];
const first = (elements, test) => elements.find((element) => test(element)) ?? null;

function mainPane(root) {
    return all(root, '[data-graph-pane]').find((element) => element.dataset.graphPane === 'main') ?? null;
}

/** The node id a NodeMatch names in the main pane's current level. */
export function resolveNodeTargetId(match) {
    const architecture = useArchitectureStore.getState();
    const templates = useTemplatesStore.getState().templates;
    const level = resolveGraphLevel(architecture.graph, architecture.currentPath, templates, architecture.inspectedIterations);
    const scope = level.isRoot ? withGlobalParameters(architecture.graph) : {};
    return matchNodes(level.graph, match ?? {}, { templates, scope, graph: level.graph })[0]?.id ?? null;
}

function flowNode(container, id) {
    if (!container || !id) return null;
    return all(container, '.react-flow__node').find((element) => element.dataset.id === id) ?? null;
}

function trainingNode(root, kind) {
    const objective = effectiveObjective(useTrainingStore.getState().graph, useArchitectureStore.getState().graph);
    const node = kind === 'loss' ? objective.loss : objective.optimizer;
    return node ? all(root, '.react-flow__node').filter((element) => element.dataset.id === node.id) : [];
}

/** Every element a TargetRef can name, in document order (the first one the test accepts wins). */
function candidates(ref, root) {
    if (typeof ref === 'object') {
        const element = flowNode(mainPane(root), resolveNodeTargetId(ref.node ?? ref.nodeOpen));
        if (!element || !ref.nodeOpen) return element ? [element] : [];
        const node = useArchitectureStore.getState().getViewGraph().nodes[element.dataset.id];
        const label = `Open ${node?.label ?? ''}`;
        return all(element, '[aria-label]').filter((button) => button.getAttribute('aria-label') === label);
    }
    const byAria = (label) => all(root, '[aria-label]').filter((element) => element.getAttribute('aria-label') === label);
    const byEdu = (value) => all(root, '[data-edu-target]').filter((element) => element.dataset.eduTarget === value);
    if (TAB_TARGETS.includes(ref)) return all(root, '.df-toolbar-tabs [data-tab]').filter((element) => element.dataset.tab === ref.slice('tab.'.length));
    if (ref === 'toolbar.code') return all(root, '.df-action-segment').filter((element) => element.getAttribute('aria-label') === 'Code');
    if (RIBBON_LABELS[ref]) return byAria(RIBBON_LABELS[ref]);
    if (ref === 'ribbon.instances') return all(root, '.df-ribbon-instance-button');
    if (ref === 'graph.checks') return all(mainPane(root), '[aria-label]').filter((element) => element.getAttribute('aria-label').startsWith('Graph checks:'));
    if (ref.startsWith('field:')) return all(root, '.df-inspector [data-parameter]').filter((element) => element.dataset.parameter === ref.slice('field:'.length));
    if (ref.startsWith('data.dataset:')) return byEdu(`dataset-${ref.slice('data.dataset:'.length)}`);
    if (EDU_TARGETS[ref]) return byEdu(EDU_TARGETS[ref]);
    if (ref === 'training.loss') return trainingNode(root, 'loss');
    if (ref === 'training.optimizer') return trainingNode(root, 'optimizer');
    return [];
}

/**
 * Resolve one TargetRef to a visible element, or null. With `{ rendered: true }` an element scrolled or panned off
 * screen counts too ("Where is it?" scrolls it into view).
 */
export function resolveTarget(ref, root = globalThis.document, { rendered = false } = {}) {
    if (!root || !ref) return null;
    return first(candidates(ref, root), rendered ? isRendered : isVisible);
}

/**
 * The block whose inside the main pane shows (at any depth) when a `nodeOpen` target names it: that target has been
 * reached, so a missing ring is not a lost target. Returns the node, or null.
 */
export function reachedOpenTarget(refs = []) {
    const opens = refs.filter((ref) => ref && typeof ref === 'object' && ref.nodeOpen).map((ref) => ref.nodeOpen);
    const architecture = useArchitectureStore.getState();
    const path = architecture.currentPath ?? [];
    if (!opens.length || !path.length) return null;
    const templates = useTemplatesStore.getState().templates;
    for (let depth = 0; depth < path.length; depth++) {
        const level = resolveGraphLevel(architecture.graph, path.slice(0, depth), templates, architecture.inspectedIterations);
        const node = level.graph?.nodes?.[path[depth]];
        if (!node) return null;
        const options = { templates, scope: level.isRoot ? withGlobalParameters(architecture.graph) : {}, graph: level.graph };
        if (opens.some((match) => matchNodes(level.graph, match, options).some((item) => item.id === node.id))) return node;
    }
    return null;
}

/** Ribbon buttons that add blocks: done, like the step's palette, once the blocks it asks for are on the canvas. */
const ADDING_TARGETS = new Set(['lesson.palette', 'ribbon.components', 'ribbon.operations']);

/**
 * A target the learner has already used, so the ring moves on to the next one: its tab was visited in this step
 * (`visitedTabs`; without it, the tab is open), the code panel is open (on that tab), the dataset row is the one
 * shown, the main pane is inside the block a nodeOpen target names, or the step's palette (`palette`, with the
 * root node ids at step entry, `entryNodeIds`) has all its blocks on the canvas (review r2: the ring stayed on the
 * ReLU chip after the learner added a ReLU from the ribbon).
 */
export function isTargetDone(ref, root = globalThis.document, { visitedTabs, palette, entryNodeIds } = {}) {
    if (ref && typeof ref === 'object') return !!ref.nodeOpen && reachedOpenTarget([ref]) !== null;
    if (typeof ref !== 'string') return false;
    if (ADDING_TARGETS.has(ref) && palette?.length) {
        return paletteUsed(palette, useArchitectureStore.getState().graph, useTemplatesStore.getState().templates, entryNodeIds);
    }
    const ui = useEduUiStore.getState();
    if (TAB_TARGETS.includes(ref)) {
        const tab = ref.slice('tab.'.length);
        return Array.isArray(visitedTabs) ? visitedTabs.includes(tab) : ui.activeTab === tab;
    }
    if (ref === 'toolbar.code') return !!ui.codeOpen;
    if (ref.startsWith('code.tab:')) return !!ui.codeOpen && ui.codeTab === ref.slice('code.tab:'.length);
    if (ref.startsWith('data.dataset:')) return candidates(ref, root)[0]?.getAttribute('aria-current') === 'true';
    return false;
}

/** The first visible target of a step's list that is not done yet, with the index of the ref that matched. */
export function resolveFirstTarget(refs = [], root = globalThis.document, options = {}) {
    for (let index = 0; index < refs.length; index++) {
        if (isTargetDone(refs[index], root, options)) continue;
        const element = resolveTarget(refs[index], root);
        if (element) return { element, index, ref: refs[index] };
    }
    return null;
}
