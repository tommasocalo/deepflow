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
const ALLOWED_IDENT = /^[a-zA-Z_$][\w$]*$/;

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
        let substituted = trimmed;
        for (const [k, v] of Object.entries(scope)) {
            if (!ALLOWED_IDENT.test(k)) continue;
            const num = Number(v);
            if (!Number.isFinite(num)) continue;
            substituted = substituted.replace(new RegExp(`\\b${k}\\b`, 'g'), `(${num})`);
        }
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
export function resolveParameterValue(value, scope) {
    if (typeof value !== 'string') return value;

    const ref = asFreeParamRef(value);
    if (ref !== null) {
        return scope[ref] ?? null;
    }

    // Heuristic: if the string contains identifier-arithmetic, ternary, or
    // comparison operators, treat it as an expression. Includes `?:` and
    // `==`/`<`/`>` so per-iteration formulas with conditions parse.
    if (/[a-zA-Z_$]/.test(value) && /[+\-*/()%?:<>!&|=]/.test(value)) {
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
