// A completion replaces only the reference containing the caret. Expressions
// on either side keep their original text, including whitespace and operators.
export function variableReferenceAtCaret(value, caret) {
    const text = String(value ?? '');
    const position = Math.max(0, Math.min(caret ?? text.length, text.length));
    const match = text.slice(0, position).match(/\$([A-Za-z_][A-Za-z0-9_]*|)$/);
    if (!match) return null;
    const tail = text.slice(position).match(/^[A-Za-z0-9_]*/)[0];
    return { start: position - match[0].length, end: position + tail.length, prefix: match[1] };
}

export function completeVariableReference(value, reference, name) {
    const text = String(value ?? '');
    const replacement = `$${name.replace(/^\$/, '')}`;
    return {
        value: `${text.slice(0, reference.start)}${replacement}${text.slice(reference.end)}`,
        caret: reference.start + replacement.length,
    };
}

export function formatVariableValue(value) {
    if (value === undefined || value === null) return 'unresolved';
    if (Array.isArray(value)) return value.map(formatVariableValue).join(' × ');
    return String(value);
}
