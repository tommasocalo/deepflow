/** Positions use top-centre origin so nodes share a flow axis. */
export const OP_SIZE = 36;

export function isInteractive(target) {
    return !!target?.closest?.('input, button, select, textarea, label, a, summary, .react-flow__handle');
}

/** "B × 1024 × 768" — output dimensions inside nodes. */
export function activationLabel(shape) {
    if (shape?.kind === 'collection') {
        const labels = shape.items.map(activationLabel);
        return labels.every((label) => label === labels[0])
            ? `${labels.length} tensors [${labels[0] ?? '?'}]`
            : `${labels.length} tensors`;
    }
    if (!Array.isArray(shape) || !shape.length) return null;
    return shape.map((d, i) => (d === null ? (i === 0 ? 'B' : '?') : String(d))).join(' × ');
}

export const OP_KEYS = new Set(['deepflow.op.Add', 'deepflow.op.Multiply', 'deepflow.op.Concat', 'deepflow.op.Split', 'deepflow.op.Broadcast', 'deepflow.op.Stack', 'deepflow.op.MatMul', 'deepflow.op.Transpose', 'deepflow.op.Scale', 'deepflow.op.CausalMask', 'torch.nn.Flatten']);
export function isOpKey(libraryKey) {
    return (libraryKey?.startsWith('deepflow.op.') && libraryKey !== 'deepflow.op.Parameter') || OP_KEYS.has(libraryKey);
}
