/** Display formatting for model metrics. Null is never rendered as NaN. */

const COUNT_UNITS = [[1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']];
const MAC_UNITS = [[1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'K']];
const BYTE_UNITS = [[1e12, ' TB'], [1e9, ' GB'], [1e6, ' MB'], [1e3, ' KB']];

const finite = (value) => typeof value === 'number' && Number.isFinite(value);

/**
 * Three significant figures below 10 (7.09M), one decimal above (38.6M,
 * 163.0M). Compact figures keep three significant figures (604M, 1.73K).
 */
function scaled(value, units, unitless = (number) => String(Math.round(number)), compact = false) {
    if (!finite(value)) return null;
    const abs = Math.abs(value);
    for (let index = 0; index < units.length; index++) {
        const [size, suffix] = units[index];
        if (abs < size) continue;
        const scaledValue = value / size;
        const magnitude = Math.abs(scaledValue);
        let text = scaledValue.toFixed(magnitude < 10 ? 2 : compact && magnitude >= 100 ? 0 : 1);
        if (Math.abs(Number(text)) >= 1000 && index > 0) {
            const [larger, largerSuffix] = units[index - 1];
            return `${(value / larger).toFixed(2)}${largerSuffix}`;
        }
        if (Math.abs(scaledValue) < 10 && Math.abs(Number(text)) >= 10) text = scaledValue.toFixed(1);
        return `${text}${suffix}`;
    }
    return unitless(value);
}

/** Parameter counts: 163.0M, 7.09M, 786.4K, 1.54K, 768. */
export function formatCount(value, { compact = false } = {}) {
    return scaled(value, COUNT_UNITS, undefined, compact);
}

/** Multiply-accumulates: 145.8G, 2.42G, 805.3M. */
export function formatMacs(value, { compact = false } = {}) {
    return scaled(value, MAC_UNITS, undefined, compact);
}

/** Decimal bytes: 205.9 MB, 4.03 GB, 652 B. */
export function formatBytes(value, { compact = false } = {}) {
    return scaled(value, BYTE_UNITS, (number) => `${Math.round(number)} B`, compact);
}

/** Share of the whole model: 52.2%, <0.1%; zero and unknown are blank. */
export function formatShare(fraction) {
    if (!finite(fraction) || fraction <= 0) return '';
    if (fraction < 0.001) return '<0.1%';
    if (fraction >= 0.9995) return '100%';
    return `${(fraction * 100).toFixed(1)}%`;
}

const exact = new Intl.NumberFormat('en-US', { maximumFractionDigits: 6 });

/** Thousands separators for tooltips: 163,037,184. */
export function formatExact(value) {
    return finite(value) ? exact.format(value) : '?';
}

/** The formatter for a metric lens. */
export function formatMetric(metric, value, options) {
    if (metric === 'macs') return formatMacs(value, options);
    if (metric === 'activation') return formatBytes(value, options);
    return formatCount(value, options);
}
