import { evaluateExpression } from './expressions.js';

/** A reverse edge accepts a number or an expression in the current parameter scope. */
export function resolveGradientFactor(raw = 1, scope = {}) {
    const value = evaluateExpression(raw, scope);
    if (value === null || value < 0) return {
        value: null,
        error: 'Use a non-negative number or a numeric parameter, such as $lambda.',
    };
    return { value, error: null };
}
