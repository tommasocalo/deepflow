/**
 * Helpers for emitting Python source from JS-side data structures.
 */

/**
 * Make a string into a valid Python identifier; collapse non-alphanumerics
 * into underscores and prefix with `_` if it starts with a digit.
 */
export function toPyIdent(name, fallback = 'id') {
    let s = String(name ?? fallback).replace(/[^a-zA-Z0-9_]/g, '_');
    if (!s) s = fallback;
    if (/^\d/.test(s)) s = '_' + s;
    return s;
}

/**
 * Render a JS value as a Python literal for use in argument lists.
 * Strings already wrapped in parentheses (e.g. "(0.9, 0.999)") are passed
 * through unquoted so they render as tuples.
 */
export function toPyLiteral(value) {
    if (value === null || value === undefined) return 'None';
    if (typeof value === 'boolean') return value ? 'True' : 'False';
    if (typeof value === 'number') return Number.isFinite(value) ? String(value) : 'None';
    if (typeof value === 'string') {
        const trimmed = value.trim();
        if (trimmed === '') return 'None';
        // Tuple-like literals: leave them raw
        if (/^\(.*\)$/.test(trimmed) || /^\[.*\]$/.test(trimmed)) return trimmed;
        // Numeric strings: emit as numbers
        if (/^-?\d+(\.\d+)?$/.test(trimmed)) return trimmed;
        return JSON.stringify(value);
    }
    if (Array.isArray(value)) {
        return '[' + value.map(toPyLiteral).join(', ') + ']';
    }
    if (typeof value === 'object') {
        const pairs = Object.entries(value).map(
            ([k, v]) => `${JSON.stringify(k)}: ${toPyLiteral(v)}`,
        );
        return '{' + pairs.join(', ') + '}';
    }
    return 'None';
}

/**
 * Render keyword arguments from an object, skipping null / undefined / empty
 * values. Optionally restrict to a whitelist of names.
 */
export function kwArgs(params, allowedNames = null) {
    /** @type {string[]} */
    const out = [];
    for (const [k, v] of Object.entries(params ?? {})) {
        if (allowedNames && !allowedNames.includes(k)) continue;
        if (v === null || v === undefined || v === '') continue;
        out.push(`${k}=${toPyLiteral(v)}`);
    }
    return out.join(', ');
}

/** Indent a multi-line string by `level * 4` spaces. */
export function indent(text, level = 1) {
    const pad = ' '.repeat(level * 4);
    return text
        .split('\n')
        .map((l) => (l ? pad + l : l))
        .join('\n');
}
