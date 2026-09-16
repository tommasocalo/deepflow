/**
 * Shared geometry for the compact "narrative" node language.
 *
 * Nodes are positioned by their top-centre (ReactFlow `nodeOrigin=[0.5, 0]`)
 * so blocks of different widths stack into a clean column. A block's width
 * encodes the size of the activation it produces (last dimension, log
 * scale) — the FFN bulge of a Transformer becomes visible at a glance.
 */
export const BLOCK_HEIGHT = 44;
export const OP_SIZE = 36;
export const CONTAINER_WIDTH = 260;

export function widthForShape(shape, fallback = 200) {
    if (!Array.isArray(shape) || shape.length < 2) return fallback;
    const last = shape[shape.length - 1];
    if (!Number.isFinite(last) || last <= 0) return fallback;
    const w = 130 + 30 * Math.log2(last / 64);
    return Math.round(Math.max(140, Math.min(300, w)));
}

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
