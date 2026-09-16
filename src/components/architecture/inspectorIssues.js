const escapePattern = (text) => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

/** Route messages from the resolved validation report without validating raw expressions again. */
export function partitionParameterErrors(parameters, errors) {
    const fields = {};
    const remaining = [];
    const matchers = parameters.map(({ name }) => ({
        name,
        // Parameter checks begin with the identifier. Shape constraints identify
        // a value with `name=…`; ordinary words such as "input" must not match `p`.
        pattern: new RegExp(`(?:^${escapePattern(name)}\\s+(?:is|must)\\b)|(?:^|[^\\w])${escapePattern(name)}(?=\\s*=)`, 'i'),
    }));
    for (const message of errors) {
        const matches = matchers
            .map(({ name, pattern }) => ({ name, index: message.search(pattern) }))
            .filter(({ index }) => index >= 0)
            .sort((a, b) => a.index - b.index);
        if (!matches.length) {
            remaining.push(message);
            continue;
        }
        const { name } = matches[0];
        (fields[name] ??= []).push(message);
    }
    return { fields, remaining };
}

export function compactParameterError(name, message) {
    if (message === `${name} is required`) return 'Required';
    const prefix = `${name} must `;
    if (message.startsWith(prefix)) return `Must ${message.slice(prefix.length)}`;
    return message;
}

export function compactConnectionWarnings(warnings) {
    const incoming = warnings.includes('No incoming connections');
    const outgoing = warnings.includes('No outgoing connections');
    if (!incoming || !outgoing) return warnings;
    let grouped = false;
    return warnings.flatMap((message) => {
        if (message !== 'No incoming connections' && message !== 'No outgoing connections') return [message];
        if (grouped) return [];
        grouped = true;
        return ['No incoming or outgoing connections'];
    });
}
