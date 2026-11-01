import { Plus } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { logEvent } from '../lessons/engine/log.js';
import { t } from '../lessons/engine/i18n.js';
import { blocksLeft, insertPaletteItem } from '../lessons/engine/palette.js';
import { blockName } from './format.js';

/**
 * Parsons-style construction (docs/edu/SPEC.md §2 P3): chips for the blocks a step needs, plus at most one
 * distractor that looks exactly like the others. A chip inserts the block; the learner wires it. A chip whose item
 * names a wire (`into`) puts the block into that wire instead, and no chip adds a second copy of a block the learner
 * already added in the step (engine/palette.js). A chip's count says how many of its blocks the step still needs.
 */
export default function PaletteChips({ items = [], moduleId, stepId, entryNodeIds = null }) {
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    if (!items.length) return null;
    const insert = async (item) => {
        const { id, spliced, adopted } = await insertPaletteItem(item, { entryNodeIds, name: blockName(item.block) });
        logEvent('palette-insert', { module: moduleId, step: stepId, data: { block: item.block, distractor: item.distractor ?? null, inserted: !!id, spliced, adopted } });
    };
    const splices = items.every((item) => item.into);
    return (
        <section className="edu-palette" data-edu-target="lesson-palette" aria-label={t('panel.palette')}>
            <h3 className="edu-section-label">{t('panel.palette')}</h3>
            <div className="edu-chip-row">
                {items.map((item, index) => {
                    const left = blocksLeft(item, graph, templates, entryNodeIds);
                    return (
                        <button key={`${item.block}-${index}`} type="button" className="edu-chip" onClick={() => insert(item)}
                            aria-label={t('panel.paletteInsert', { name: blockName(item.block) })}>
                            <Plus size={11} strokeWidth={3} aria-hidden="true" />
                            <span>{blockName(item.block)}</span>
                            {left > 1 && <span className="edu-chip-count">{t('panel.paletteLeft', { count: left })}</span>}
                        </button>
                    );
                })}
            </div>
            <p className="edu-muted">{t(splices ? 'panel.paletteHintSplice' : 'panel.paletteHint')}</p>
        </section>
    );
}
