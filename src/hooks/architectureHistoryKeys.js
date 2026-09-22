/** Text editing keeps the browser's own undo stack, even inside a dialog. */
export function isEditingText(target) {
    const field = target?.closest?.('input, textarea, [contenteditable]:not([contenteditable="false"]), [role="textbox"]');
    if (!field) return false;
    if (field.tagName?.toLowerCase() !== 'input') return true;
    return !['button', 'checkbox', 'radio', 'range', 'submit', 'reset', 'file', 'color'].includes(field.type);
}

export function architectureHistoryCommand(event) {
    if (event.defaultPrevented || event.isComposing || event.altKey || !(event.metaKey || event.ctrlKey) || isEditingText(event.target)) return null;
    const key = event.key?.toLowerCase();
    if (key === 'z') return event.shiftKey ? 'redo' : 'undo';
    if (key === 'y' && event.ctrlKey && !event.metaKey && !event.shiftKey) return 'redo';
    return null;
}
