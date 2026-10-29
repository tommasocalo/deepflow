import { STANDARD_COMPONENTS } from '../../domain/components/index.js';
import { getEntry } from '../../domain/library/index.js';
import { hasText, t } from '../lessons/engine/i18n.js';

/** Display name of a BlockRef: the library or component name the editor shows. */
export function blockName(block) {
    if (block?.startsWith('component:')) return STANDARD_COMPONENTS.find((item) => item.key === block.slice('component:'.length))?.label ?? block;
    return getEntry(block)?.name ?? block;
}

export function datasetName(id) {
    return hasText(`dataset.${id}`) ? t(`dataset.${id}`) : id;
}

/** Five fixed scores for the characters e t a o z (docs/edu/SPEC.md §6.3.5). */
export const TEMPERATURE_SCORES = Object.freeze([
    { token: 'e', score: 2.0 }, { token: 't', score: 1.0 }, { token: 'a', score: 0.5 }, { token: 'o', score: 0.0 }, { token: 'z', score: -1.0 },
]);

export function softmaxWithTemperature(scores, temperature) {
    const scaled = scores.map((score) => score / temperature);
    const max = Math.max(...scaled);
    const exps = scaled.map((value) => Math.exp(value - max));
    const sum = exps.reduce((total, value) => total + value, 0);
    return exps.map((value) => value / sum);
}

/** "2 · Going non-linear" */
export function moduleTitle(module) {
    return t('panel.module', { number: module?.order ?? '', title: t(`${module?.id}.title`) });
}
