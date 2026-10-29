/**
 * All dataset presets in display order (docs/edu/SPEC.md §6.9.1): toy sets, Iris and digits (Stage B), then the
 * export-only image and text presets (Stage C, presets/images.js and presets/text.js).
 */
import toy from './toy.js';
import tabular from './tabular.js';
import digits from './digits.js';
import images from './images.js';
import text from './text.js';

export const PRESETS = [...toy, ...tabular, ...digits, ...images, ...text];

export default PRESETS;
