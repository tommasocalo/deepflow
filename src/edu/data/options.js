/**
 * Dataset option schemas (docs/edu/SPEC.md §6.9.1): defaults, clamping and the data seed. Pure; shared by the
 * presets, the data store, the trainer and the Python loaders.
 */

const INTEGER_OPTIONS = new Set(['n', 'perClass', 'seed']);

/** Spec defaults, e.g. { n: 200, noise: 0.1, seed: 1 }. */
export function defaultOptions(spec) {
    return Object.fromEntries(Object.entries(spec?.options ?? {}).map(([name, option]) => [name, option.default]));
}

/** Clamp to the schema's range and step grid; unknown names are dropped, missing ones take the default. */
export function resolveOptions(spec, options = {}) {
    const resolved = {};
    for (const [name, option] of Object.entries(spec?.options ?? {})) {
        let value = Number(options?.[name]);
        if (!Number.isFinite(value)) value = option.default;
        if (Number.isFinite(option.min)) value = Math.max(option.min, value);
        if (Number.isFinite(option.max)) value = Math.min(option.max, value);
        if (INTEGER_OPTIONS.has(name)) value = Math.round(value);
        else if (option.step) value = Number((Math.round(value / option.step) * option.step).toFixed(6));
        resolved[name] = value;
    }
    return resolved;
}

/** The dataset seed that drives generation and the split: options.seed, else 1. */
export const dataSeedOf = (options) => (Number.isFinite(Number(options?.seed)) ? Number(options.seed) : 1);

/** Number of examples for the given options. */
export function datasetSize(spec, options = {}) {
    if (!spec) return 0;
    const resolved = resolveOptions(spec, options);
    if (spec.kind === 'toy2d') {
        const n = spec.id === 'spiral3' ? resolved.perClass * 3 : resolved.n;
        return spec.id === 'xor' || spec.id === 'plane' || spec.id === 'spiral3' ? n : 2 * Math.floor(n / 2);
    }
    return spec.size?.total ?? ((spec.size?.train ?? 0) + (spec.size?.test ?? 0));
}

/** Allowed split values in the Data tab (§3.4). */
export const SPLIT_LIMITS = { val: { min: 0.1, max: 0.5, step: 0.05 }, test: [0, 0.1, 0.2] };

/** Clamp a split patch to the Data tab's ranges; values outside them are moved to the nearest allowed value. */
export function resolveSplit(split = {}) {
    let val = Number(split.val);
    if (!Number.isFinite(val)) val = 0.2;
    val = Math.min(SPLIT_LIMITS.val.max, Math.max(SPLIT_LIMITS.val.min, val));
    val = Number((Math.round(val / SPLIT_LIMITS.val.step) * SPLIT_LIMITS.val.step).toFixed(6));
    let test = Number(split.test);
    if (!Number.isFinite(test)) test = 0;
    test = SPLIT_LIMITS.test.reduce((best, option) => (Math.abs(option - test) < Math.abs(best - test) ? option : best), 0);
    return { val, test };
}
