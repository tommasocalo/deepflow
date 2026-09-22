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
const NUMERIC = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export function evaluateExpression(expr, scope = {}) {
    if (typeof expr === 'number') {
        return Number.isFinite(expr) ? expr : null;
    }
    if (typeof expr !== 'string') return null;
    const trimmed = expr.trim();
    if (trimmed === '') return null;

    // Pure numeric fast-path
    if (NUMERIC.test(trimmed)) {
        const n = Number(trimmed);
        return Number.isFinite(n) ? n : null;
    }

    if (!ALLOWED.test(trimmed)) return null;

    try {
        // Substitute identifiers from scope directly into the expression so
        // we don't pass them as Function parameters — JS reserved words
        // like `in` would otherwise cause a SyntaxError.
        const substituted = trimmed.replace(/((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|\$?([a-zA-Z_][\w$]*)/g, (token, numeric, name) => {
            if (numeric) return token;
            const value = scope[name] ?? scope[token];
            if (value === null || value === undefined || typeof value === 'boolean' || value === '') return token;
            const number = Number(value);
            return Number.isFinite(number) ? `(${number})` : token;
        });
        // Any identifier remaining means an unbound name → unevaluable.
        if (/[a-zA-Z_$]/.test(substituted.replace(/(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?/g, ''))) return null;
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
    if (NUMERIC.test(value.trim())) {
        return Number(value.trim());
    }
    return value;
}

/** Parse numeric tuples without evaluating arbitrary source text. */
export function resolveDimensionTuple(value, scope = {}) {
    let dimensions = resolveParameterValue(value, scope);
    if (typeof dimensions === 'number') return [dimensions];
    if (typeof dimensions === 'string') {
        dimensions = dimensions.trim();
        if (!dimensions) return null;
        if ((dimensions.startsWith('(') && dimensions.endsWith(')')) || (dimensions.startsWith('[') && dimensions.endsWith(']'))) {
            dimensions = dimensions.slice(1, -1);
        }
        dimensions = dimensions.split(',').map((part) => part.trim()).filter(Boolean);
    }
    if (!Array.isArray(dimensions)) return null;
    const resolved = dimensions.map((dimension) => resolveParameterValue(dimension, scope));
    return resolved.every((dimension) => typeof dimension === 'number' && Number.isFinite(dimension)) ? resolved : null;
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
        out[key] = node.libraryKey === 'deepflow.op.Parameter' && key === 'shape'
            ? resolveDimensionTuple(value, scope)
            : node.libraryKey === 'deepflow.op.Mean' && key === 'dim'
                ? resolveDimensionTuple(value, scope)
                : resolveParameterValue(value, scope);
    }
    return { ...node, parameters: out };
}

/** A derived dimension refers to a named template input, never to a guessed width. */
export function resolveDerivedParameter(derived, externalInputs = {}) {
    if (derived?.kind !== 'input-dimension') return null;
    const shape = externalInputs[derived.inputId];
    if (!Array.isArray(shape) || !Number.isInteger(derived.axis)) return null;
    const axis = derived.axis < 0 ? shape.length + derived.axis : derived.axis;
    const value = shape[axis];
    return Number.isSafeInteger(value) && value > 0 ? value : null;
}

/**
 * A tuple's flat array describes its axes, not separate repeat instances.
 * Explicit per-instance tuples remain supported as nested arrays or tuple
 * strings, while scalar parameters retain their existing array bindings.
 */
export function repeatBindingValue(value, parameter, index) {
    if (!Array.isArray(value) || index === undefined) return value;
    if (parameter?.type === 'tuple' && !value.some((part) => Array.isArray(part)
        || (typeof part === 'string' && part.includes(',')))) return value;
    return value[index];
}

/** Resolve ordered exposed controls and derived input dimensions in the local scope. */
export function buildParameterScope(freeParameters, bindings = {}, parentScope = {}, iterationIndex, externalInputs = {}) {
    const result = { ...parentScope };
    for (const parameter of freeParameters ?? []) {
        if (parameter.derived) {
            result[parameter.name] = resolveDerivedParameter(parameter.derived, externalInputs);
            continue;
        }
        let raw = repeatBindingValue(bindings[parameter.name], parameter, iterationIndex);
        if (raw === undefined || (raw === null && !parameter.nullable) || raw === '') raw = parameter.default;
        result[parameter.name] = resolveParameterValue(raw, result);
    }
    return result;
}

/** Guard all unrolling/allocation before entering a loop. */
export function resolveRepeatCount(raw, scope = {}, inputShapes = []) {
    let value;
    if (raw === 'auto') {
        const counts = inputShapes.filter((shape) => shape?.kind === 'collection' && Array.isArray(shape.items)).map((shape) => shape.items.length);
        if (!counts.length) return { count: null, error: 'Auto instances needs an incoming collection from Split or Broadcast' };
        if (counts.some((count) => count !== counts[0])) return { count: null, error: 'Auto instances needs equally sized input collections' };
        value = counts[0];
    } else value = resolveParameterValue(raw, scope);
    const count = typeof value === 'number' ? value : NaN;
    if (!Number.isSafeInteger(count) || count < 1 || count > MAX_REPEAT_COUNT) {
        return { count: null, error: `Repeat count must resolve to a positive integer from 1 to ${MAX_REPEAT_COUNT}` };
    }
    return { count, error: null };
}
