/** Only an intentional drop on empty canvas detaches a connection. */
export function isBlankCanvasDrop(event, connectionState, canvas) {
    if (!canvas || connectionState?.toHandle || connectionState?.toNode || connectionState?.isValid) return false;
    // Touch events retain their original target. Resolve the actual release point.
    const point = event.changedTouches?.[0] ?? event;
    const target = canvas.ownerDocument?.elementFromPoint?.(point.clientX, point.clientY) ?? event.target;
    return !!target && canvas.contains(target) && !!target.matches?.('.react-flow__pane');
}
