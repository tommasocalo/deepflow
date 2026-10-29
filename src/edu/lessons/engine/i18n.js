import { DEFAULT_LOCALE } from '../../config.js';
import UI_EN from './text/en/ui.js';

/**
 * Text bundles (docs/edu/SPEC.md §6.3.3). `t(key, params)` looks the key up in the active locale, then in
 * English, then returns the key itself. In development it warns once per missing key.
 */
const base = { en: { ...UI_EN } };
let bundles = { en: { ...UI_EN } };
let locale = DEFAULT_LOCALE;
const warned = new Set();

/** Merge lesson text bundles ({ en: {...}, it: {...} }) over the UI chrome strings. */
export function setTextBundles(text = {}) {
    const next = {};
    for (const [name, strings] of Object.entries(base)) next[name] = { ...strings };
    for (const [name, strings] of Object.entries(text ?? {})) next[name] = { ...(next[name] ?? {}), ...(strings ?? {}) };
    bundles = next;
}

export function setLocale(next) {
    locale = next || DEFAULT_LOCALE;
}

export function getLocale() {
    return locale;
}

export function hasText(key) {
    return typeof bundles[locale]?.[key] === 'string' || typeof bundles.en?.[key] === 'string';
}

/** The raw string without interpolation, or null. */
export function rawText(key) {
    const value = bundles[locale]?.[key] ?? bundles.en?.[key];
    return typeof value === 'string' ? value : null;
}

/** Numbers get locale grouping (101,770); non-integers keep up to four decimals. */
export function formatNumber(value, target = locale) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return String(value);
    const tag = target === 'en' ? 'en-US' : target;
    try {
        return value.toLocaleString(tag, { maximumFractionDigits: Number.isInteger(value) ? 0 : 4 });
    } catch {
        return String(value);
    }
}

export function formatValue(value) {
    if (typeof value === 'number') return formatNumber(value);
    if (value === null || value === undefined) return '';
    return String(value);
}

/** Replace {name} placeholders; unknown names stay as they are. */
export function interpolate(template, params = {}) {
    if (typeof template !== 'string') return '';
    return template.replace(/\{([A-Za-z_][\w.-]*)\}/g, (match, name) =>
        Object.hasOwn(params ?? {}, name) && params[name] !== undefined && params[name] !== null
            ? formatValue(params[name]) : match);
}

/** The inspector's label for a parameter name (in_features → Input), else the name with spaces. */
export function paramLabel(name) {
    const text = rawText(`param.${name}`);
    return text ?? String(name ?? '').replaceAll('_', ' ');
}

export function t(key, params) {
    const value = rawText(key);
    if (value === null) {
        if (import.meta.env?.DEV && !warned.has(key)) {
            warned.add(key);
            console.warn(`[edu i18n] missing text key: ${key}`);
        }
        return key;
    }
    return params ? interpolate(value, params) : value;
}
