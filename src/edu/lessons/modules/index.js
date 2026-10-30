import m1 from './m1-first-neuron.js';
import m2 from './m2-non-linear.js';
import m3 from './m3-shapes.js';
import m4 from './m4-convolutions.js';
import m5 from './m5-training-well.js';
import m6 from './m6-residual.js';
import m7 from './m7-transformer.js';
import en from './text/en/index.js';

/**
 * The curriculum (docs/edu/SPEC.md §4, §6.3): seven modules in order, and their text bundles. The lesson engine
 * (engine/registry.js) loads this registry; scripts/edu-content.test.mjs validates and solves every module headless.
 */
export const MODULES = [m1, m2, m3, m4, m5, m6, m7];
export const TEXT = { en };
