/** The editors' number grammar: 1, -2.5, .5, 1e-5 (but not "1." or "-"). */
export const NUMBER_LITERAL = /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$/;

/** Display formatting only: expression evaluation and saved values stay exact. */
export function formatParameterValue(value) {
    if (value === null || value === undefined) return 'Unresolved';
    if (Array.isArray(value)) return value.map(formatParameterValue).join(' × ');
    if (typeof value === 'number' && Number.isFinite(value) && !Number.isInteger(value)) {
        return String(Number(value.toPrecision(6)));
    }
    return String(value);
}

export function exactParameterValue(value) {
    return Array.isArray(value) ? value.map(exactParameterValue).join(' × ') : String(value);
}

export function isApproximateDisplay(value) {
    return typeof value === 'number' && Number.isFinite(value) && Number(formatParameterValue(value)) !== value;
}
