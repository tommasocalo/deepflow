/** "Attention heads" → head, "Transformer blocks" → block; otherwise instance/step. */
export function instanceNoun(label, mode) {
    const word = String(label ?? '').trim().split(/\s+/).at(-1)?.toLowerCase() ?? '';
    if (/^[a-z]{3,}ies$/.test(word)) return `${word.slice(0, -3)}y`;
    if (/^[a-z]{3,}s$/.test(word) && !/ss$/.test(word)) return word.slice(0, -1);
    return mode === 'parallel' ? 'instance' : 'step';
}

export const pluralOf = (noun) => (noun.endsWith('y') ? `${noun.slice(0, -1)}ies` : `${noun}s`);
