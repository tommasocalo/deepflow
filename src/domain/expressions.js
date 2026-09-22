/**
 * Tiny safe-ish arithmetic evaluator for parameter bindings.
 *
 * Accepted forms:
 *   - a JavaScript number      → returned as-is
 *   - a numeric string         → parsed
 *   - an arithmetic expression in the variables of `scope`, restricted
 *     to digits, `+ - * / ** % ( ) , . _ $`, and identifier characters.
 *
 * Anything else returns `null` (unevaluable). The whitelist is
 * permissive enough for the patterns we care about — `64 * 2 ** i`,
 * `(num_heads * head_dim)`, `width // 2` (using `Math.floor` is too
 * heavy; we recommend integer-only formulas).
 *
 * @typedef {string | number} Expression
 *
 * @param {Expression} expr
 * @param {Object<string, number>} [scope]
 * @returns {number | null}
 */
// Whitelist for expression characters. Includes ternary (`?:`),
// comparison (`==`, `<=`, etc.), and short-circuit (`&&`, `||`) so users
// can write per-iteration formulas like `i === 0 ? 3 : 64 * 2 ** (i - 1)`.
// We substitute identifiers in scope with their literal numeric values
// before evaluating, so `=` cannot be abused for assignment to anything
// the host cares about.
const ALLOWED = /^[\s\d+\-*/().,a-zA-Z_$%?:=!<>&|]*$/;
export const MAX_REPEAT_COUNT = 256;

export function evaluateExpression(expr, scope = {}) {
    if (typeof expr === 'number') {
        return Number.isFinite(expr) ? expr : null;
    }
    if (typeof expr !== 'string') return null;
    const trimmed = expr.trim();
    if (trimmed === '') return null;

    // Pure numeric fast-path
    if (/^-?\d+(\.\d+)?$/.test(trimmed)) {
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : null;
    }

    if (!ALLOWED.test(trimmed)) return null;

    try {
        // Substitute identifiers from scope directly into the expression so
        // we don't pass them as Function parameters — JS reserved words
        // like `in` would otherwise cause a SyntaxError.
        const substituted = trimmed.replace(/\$?([a-zA-Z_][\w$]*)/g, (token, name) => {
            const value = scope[name] ?? scope[token];
            if (value === null || value === undefined || typeof value === 'boolean' || value === '') return token;
            const number = Number(value);
            return Number.isFinite(number) ? `(${number})` : token;
        });
        // Any identifier remaining means an unbound name → unevaluable.
        if (/[a-zA-Z_$]/.test(substituted)) return null;
        const fn = new Function(`"use strict"; return (${substituted});`);
        const result = fn();
        return Number.isFinite(result) ? result : null;
    } catch {
        return null;
    }
}

/**
 * If the value looks like a `$name` reference, return the name; otherwise
 * `null`.
 *
 * @param {*} value
 */
export function asFreeParamRef(value) {
    if (typeof value !== 'string') return null;
    const m = value.match(/^\$([a-zA-Z_$][\w$]*)$/);
    return m ? m[1] : null;
}

/**
 * Resolve a parameter value against a scope of free-parameter values.
 * Strings beginning with `$` are looked up in `scope`. Strings that look
 * like arithmetic expressions are evaluated. Everything else (booleans,
 * tuples, plain strings) is returned untouched.
 *
 * @param {*} value
 * @param {Object<string, *>} scope
 */
export function resolveParameterValue(value, scope = {}) {
    if (typeof value !== 'string') return value;

    const ref = asFreeParamRef(value.trim());
    if (ref !== null) {
        return scope[ref] ?? null;
    }

    if (Object.hasOwn(scope, value.trim())) return scope[value.trim()];
    // Tuple strings remain literal; arithmetic and numeric formulas may use
    // either bare identifiers or $references.
    if (!value.includes(',') && /[+\-*/%?:<>!&|=]/.test(value)) {
        const evaluated = evaluateExpression(value, scope);
        if (evaluated !== null) return evaluated;
    }

    // Numeric string → number
    if (/^-?\d+(\.\d+)?$/.test(value.trim())) {
        return Number(value.trim());
    }
    return value;
}

/**
 * Walk every parameter of a layer node and resolve `$ref` / expression
 * values against the given scope.
 *
 * @param {import('./architecture/types.js').LayerNode} node
 * @param {Object<string, *>} scope
 */
export function resolveLayerParameters(node, scope) {
    /** @type {Object<string, *>} */
    const out = {};
    for (const [key, value] of Object.entries(node.parameters ?? {})) {
        out[key] = resolveParameterValue(value, scope);
    }
    return { ...node, parameters: out };
}

/** Resolve ordered free parameters; later defaults can use earlier bindings. */
export function buildParameterScope(freeParameters, bindings = {}, parentScope = {}, iterationIndex) {
    const result = { ...parentScope };
    for (const parameter of freeParameters ?? []) {
        let raw = bindings[parameter.name];
        if (Array.isArray(raw) && iterationIndex !== undefined) raw = raw[iterationIndex];
        if (raw === undefined || raw === null || raw === '') raw = parameter.default;
        result[parameter.name] = resolveParameterValue(raw, result);
    }
    return result;
}

/** Guard all unrolling/allocation before entering a loop. */
export function resolveRepeatCount(raw, scope = {}) {
    const value = resolveParameterValue(raw, scope);
    const count = typeof value === 'number' ? value : NaN;
    if (!Number.isSafeInteger(count) || count < 1 || count > MAX_REPEAT_COUNT) {
        return { count: null, error: `Repeat count must resolve to a positive integer from 1 to ${MAX_REPEAT_COUNT}` };
    }
    return { count, error: null };
}
