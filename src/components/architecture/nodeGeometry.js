/** Positions use top-centre origin so tensor matrices share a flow axis. */
export const OP_SIZE = 36;
export const CONTAINER_WIDTH = 260;

/** Half-width used to decide whether an edge passes "through" other nodes. */
export const COLUMN_HALF_WIDTH = 150;

export function isInteractive(target) {
    return !!target?.closest?.('input, button, select, textarea, label, a, summary, .react-flow__handle');
}

/** "B × 1024 × 768" — activation label for edges. */
export function activationLabel(shape) {
    if (!Array.isArray(shape) || !shape.length) return null;
    return shape.map((d, i) => (d === null ? (i === 0 ? 'B' : '?') : String(d))).join(' × ');
}

export const OP_KEYS = new Set(['deepflow.op.Add', 'deepflow.op.Multiply', 'deepflow.op.Concat', 'deepflow.op.Split']);
export function isOpKey(libraryKey) {
    return OP_KEYS.has(libraryKey);
}
