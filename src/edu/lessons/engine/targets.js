import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';
import { resolveGraphLevel } from '../../../domain/architecture/operations.js';
import { withGlobalParameters } from '../../../domain/expressions.js';
import { effectiveObjective } from '../../objective.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { matchNodes } from './nodes.js';

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

/** Rendered and not display:none or visibility:hidden. */
export function isVisible(element) {
    if (!element?.isConnected) return false;
    if (element.closest?.('[hidden]')) return false;
    const view = element.ownerDocument?.defaultView;
    const style = view?.getComputedStyle?.(element);
    if (style && (style.display === 'none' || style.visibility === 'hidden')) return false;
    const rects = element.getClientRects?.();
    // jsdom has no layout: treat connected, styled-visible elements as visible there.
    if (rects && rects.length === 0 && view?.navigator?.userAgent?.includes('jsdom') !== true) return false;
    return true;
}

const all = (root, selector) => [...(root?.querySelectorAll?.(selector) ?? [])];
const firstVisible = (elements) => elements.find(isVisible) ?? null;

function byAriaLabel(root, label) {
    return firstVisible(all(root, '[aria-label]').filter((element) => element.getAttribute('aria-label') === label));
}

function byEduTarget(root, value) {
    return firstVisible(all(root, '[data-edu-target]').filter((element) => element.dataset.eduTarget === value));
}

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

function trainingNodeElement(root, kind) {
    const objective = effectiveObjective(useTrainingStore.getState().graph, useArchitectureStore.getState().graph);
    const node = kind === 'loss' ? objective.loss : objective.optimizer;
    if (!node) return null;
    return firstVisible(all(root, '.react-flow__node').filter((element) => element.dataset.id === node.id));
}

/** Resolve one TargetRef to an element, or null. */
export function resolveTarget(ref, root = globalThis.document) {
    if (!root || !ref) return null;
    if (typeof ref === 'object') {
        const match = ref.node ?? ref.nodeOpen;
        const element = flowNode(mainPane(root), resolveNodeTargetId(match));
        if (!element) return null;
        if (!ref.nodeOpen) return isVisible(element) ? element : null;
        const node = useArchitectureStore.getState().getViewGraph().nodes[element.dataset.id];
        const label = `Open ${node?.label ?? ''}`;
        return firstVisible(all(element, '[aria-label]').filter((button) => button.getAttribute('aria-label') === label));
    }
    if (TAB_TARGETS.includes(ref)) {
        const tab = ref.slice('tab.'.length);
        return firstVisible(all(root, '.df-toolbar-tabs [data-tab]').filter((element) => element.dataset.tab === tab));
    }
    if (ref === 'toolbar.code') return firstVisible(all(root, '.df-action-segment').filter((element) => element.getAttribute('aria-label') === 'Code'));
    if (RIBBON_LABELS[ref]) return byAriaLabel(root, RIBBON_LABELS[ref]);
    if (ref === 'ribbon.instances') return firstVisible(all(root, '.df-ribbon-instance-button'));
    if (ref === 'graph.checks') {
        const pane = mainPane(root);
        return firstVisible(all(pane, '[aria-label]').filter((element) => element.getAttribute('aria-label').startsWith('Graph checks:')));
    }
    if (ref.startsWith('field:')) {
        const name = ref.slice('field:'.length);
        return firstVisible(all(root, '.df-inspector [data-parameter]').filter((element) => element.dataset.parameter === name));
    }
    if (ref.startsWith('data.dataset:')) return byEduTarget(root, `dataset-${ref.slice('data.dataset:'.length)}`);
    if (EDU_TARGETS[ref]) return byEduTarget(root, EDU_TARGETS[ref]);
    if (ref === 'training.loss') return trainingNodeElement(root, 'loss');
    if (ref === 'training.optimizer') return trainingNodeElement(root, 'optimizer');
    return null;
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

/**
 * A target the learner has already used, so the ring moves on to the next one: its tab is open, the code panel is
 * open (on that tab), the dataset row is the one shown, or the main pane is inside the block a nodeOpen target names.
 */
export function isTargetDone(ref, root = globalThis.document) {
    if (ref && typeof ref === 'object') return !!ref.nodeOpen && reachedOpenTarget([ref]) !== null;
    if (typeof ref !== 'string') return false;
    const ui = useEduUiStore.getState();
    if (TAB_TARGETS.includes(ref)) return ui.activeTab === ref.slice('tab.'.length);
    if (ref === 'toolbar.code') return !!ui.codeOpen;
    if (ref.startsWith('code.tab:')) return !!ui.codeOpen && ui.codeTab === ref.slice('code.tab:'.length);
    if (ref.startsWith('data.dataset:')) return byEduTarget(root, `dataset-${ref.slice('data.dataset:'.length)}`)?.getAttribute('aria-current') === 'true';
    return false;
}

/** The first visible target of a step's list that is not done yet, with the index of the ref that matched. */
export function resolveFirstTarget(refs = [], root = globalThis.document) {
    for (let index = 0; index < refs.length; index++) {
        if (isTargetDone(refs[index], root)) continue;
        const element = resolveTarget(refs[index], root);
        if (element) return { element, index, ref: refs[index] };
    }
    return null;
}
