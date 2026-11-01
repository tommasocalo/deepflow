import { Plus } from 'lucide-react';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { runActions } from '../lessons/engine/actions.js';
import { requestFit, requestInsert } from '../lessons/engine/canvasBridge.js';
import { logEvent } from '../lessons/engine/log.js';
import { t } from '../lessons/engine/i18n.js';
import { besideWire, chipParams, spliceTarget } from '../lessons/engine/palette.js';
import { useEduUiStore } from '../shell/uiStore.js';
import { blockName } from './format.js';

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
async function insertBlock(item) {
    const ui = useEduUiStore.getState();
    if (ui.activeTab !== 'architecture') ui.setTab('architecture');
    if (useArchitectureStore.getState().currentPath.length) useArchitectureStore.getState().goToLevel(0);
    const { graph } = useArchitectureStore.getState();
    const templates = useTemplatesStore.getState().templates;
    const params = chipParams(item, graph, templates);
    const place = besideWire(item, graph, templates);
    const label = `Add ${blockName(item.block)}`;
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

/** A chip with `into`: splice the block into that wire with the lesson's insertBetween (one undo step), in the column. */
async function spliceBlock(item) {
    if (useArchitectureStore.getState().currentPath.length) useArchitectureStore.getState().goToLevel(0);
    const { graph } = useArchitectureStore.getState();
    if (!spliceTarget(item, graph, useTemplatesStore.getState().templates)) return null;
    const before = new Set(Object.keys(graph.nodes));
    const result = await runActions([{ do: 'insertBetween', block: item.block, from: item.into.from, to: item.into.to, params: item.params }],
        { label: `insert ${blockName(item.block)}` });
    if (!result.ok) return null;
    const id = Object.keys(useArchitectureStore.getState().graph.nodes).find((key) => !before.has(key)) ?? null;
    if (id) useArchitectureStore.getState().selectOnly(id);
    // The column grew by one row: keep the whole model in view.
    setTimeout(requestFit, 60);
    return id;
}

/**
 * Parsons-style construction (docs/edu/SPEC.md §2 P3): chips for the blocks a step needs, plus at most one
 * distractor that looks exactly like the others. A chip inserts the block; the learner wires it. A chip whose item
 * names a wire (`into`) puts the block into that wire instead (engine/palette.js).
 */
export default function PaletteChips({ items = [], moduleId, stepId }) {
    if (!items.length) return null;
    const insert = async (item) => {
        const spliced = item.into ? await spliceBlock(item) : null;
        const id = spliced ?? await insertBlock(item);
        logEvent('palette-insert', { module: moduleId, step: stepId, data: { block: item.block, distractor: item.distractor ?? null, inserted: !!id, spliced: !!spliced } });
    };
    const splices = items.every((item) => item.into);
    return (
        <section className="edu-palette" data-edu-target="lesson-palette" aria-label={t('panel.palette')}>
            <h3 className="edu-section-label">{t('panel.palette')}</h3>
            <div className="edu-chip-row">
                {items.map((item, index) => (
                    <button key={`${item.block}-${index}`} type="button" className="edu-chip" onClick={() => insert(item)}
                        aria-label={t('panel.paletteInsert', { name: blockName(item.block) })}>
                        <Plus size={11} strokeWidth={3} aria-hidden="true" />
                        <span>{blockName(item.block)}</span>
                        {(item.count ?? 1) > 1 && <span className="edu-chip-count">{t('panel.paletteLeft', { count: item.count })}</span>}
                    </button>
                ))}
            </div>
            <p className="edu-muted">{t(splices ? 'panel.paletteHintSplice' : 'panel.paletteHint')}</p>
        </section>
    );
}
