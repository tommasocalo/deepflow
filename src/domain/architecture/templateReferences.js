/** Whether a template references the target directly or through nested instances. */
export function containsTemplate(templateId, targetId, templates) {
    const pending = [templateId];
    const visited = new Set();
    while (pending.length) {
        const current = pending.pop();
        if (current === targetId) return true;
        if (visited.has(current)) continue;
        visited.add(current);
        for (const node of Object.values(templates[current]?.graph?.nodes ?? {})) {
            if (node.templateId) pending.push(node.templateId);
        }
    }
    return false;
}
