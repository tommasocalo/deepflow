import { Plus } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { requestInsert } from '../lessons/engine/canvasBridge.js';
import { logEvent } from '../lessons/engine/log.js';
import { t } from '../lessons/engine/i18n.js';
import { useEduUiStore } from '../shell/uiStore.js';
import { blockName } from './format.js';

function paletteEntry(block) {
    return block.startsWith('component:') ? { standardComponentKey: block.slice('component:'.length) } : { key: block };
}

/** Without a mounted canvas (tests, a hidden tab), add the block beside the column with the store's own action. */
function insertWithoutCanvas(block) {
    const state = useArchitectureStore.getState();
    if (state.currentPath.length) state.goToLevel(0);
    const nodes = Object.values(useArchitectureStore.getState().graph.nodes);
    const xs = nodes.map((node) => node.position.x);
    const ys = nodes.map((node) => node.position.y).sort((a, b) => a - b);
    const position = { x: (xs.length ? Math.max(...xs) : 400) + 260, y: ys.length ? ys[Math.floor(ys.length / 2)] : 200 };
    const id = block.startsWith('component:')
        ? useArchitectureStore.getState().addStandardComponent(block.slice('component:'.length), position)
        : useArchitectureStore.getState().addLayerByLibraryKey(block, position);
    if (id) useArchitectureStore.getState().selectOnly(id);
    return id;
}

/** Wait until the architecture canvas is mounted, then insert through the canvas bridge. */
async function insertBlock(block) {
    const ui = useEduUiStore.getState();
    if (ui.activeTab !== 'architecture') ui.setTab('architecture');
    for (let attempt = 0; attempt < 20; attempt++) {
        const id = requestInsert(paletteEntry(block));
        if (id) return id;
        if (typeof window === 'undefined' || !window.requestAnimationFrame) break;
        await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return insertWithoutCanvas(block);
}

/**
 * Parsons-style construction (docs/edu/SPEC.md §2 P3): chips for the blocks a step needs, plus at most one
 * distractor that looks exactly like the others. A chip inserts the block; the learner wires it.
 */
export default function PaletteChips({ items = [], moduleId, stepId }) {
    if (!items.length) return null;
    const insert = async (item) => {
        const id = await insertBlock(item.block);
        logEvent('palette-insert', { module: moduleId, step: stepId, data: { block: item.block, distractor: item.distractor ?? null, inserted: !!id } });
    };
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
            <p className="edu-muted">{t('panel.paletteHint')}</p>
        </section>
    );
}
