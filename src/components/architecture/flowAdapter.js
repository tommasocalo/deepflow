import { isOpKey } from './nodeGeometry.js';

const samePosition = (a, b) => a?.x === b?.x && a?.y === b?.y;

/**
 * Reconcile immutable domain snapshots with ReactFlow's local visual state.
 * ReactFlow owns in-progress positions and measured dimensions. Unchanged
 * domain nodes keep their data object, so dragging does not rerender cards.
 */
export function syncFlowNodes(previous, graph, selectedIds, tensorFlowIds, levelKey, resetPositions = false) {
    const byId = new Map(previous.map((node) => [node.id, node]));
    const selected = new Set(selectedIds);
    const next = Object.values(graph.nodes).map((node) => {
        const old = byId.get(node.id);
        const tensorFlow = tensorFlowIds.has(node.id);
        const unchanged = old?.data.node === node && old.data.tensorFlow === tensorFlow && old.data.levelKey === levelKey;
        const data = unchanged ? old.data : { node, tensorFlow, levelKey };
        const position = !resetPositions && unchanged && old.dragging ? old.position : node.position;
        const isSelected = selected.has(node.id);
        if (unchanged && old.selected === isSelected && samePosition(old.position, position) && (!resetPositions || !old.dragging)) return old;
        return {
            ...old,
            id: node.id,
            type: node.kind === 'supernode' ? 'archSupernode'
                : node.kind === 'repeat' ? 'archRepeat'
                    : isOpKey(node.libraryKey) ? 'archOp' : 'archLayer',
            position,
            data,
            selected: isSelected,
            dragging: !resetPositions && unchanged ? old?.dragging : false,
        };
    });
    return next.length === previous.length && next.every((node, index) => node === previous[index]) ? previous : next;
}

const sameContext = (a, b) => a.graph === b.graph && a.levelKey === b.levelKey;

/**
 * A drag produces many position batches, but only one domain transaction.
 * A context change or cancellation discards the unfinished transaction.
 * Position updates outside a drag (e.g. arrow keys) commit immediately.
 */
export function createDragBatcher() {
    let session = null;
    return {
        begin(context) {
            session = { context, positions: new Map(), draggingIds: new Set(), cancelled: false };
        },
        cancel() {
            if (session) session.cancelled = true;
        },
        reconcile(context) {
            if (session && !sameContext(session.context, context)) session.cancelled = true;
            return !!session?.cancelled;
        },
        consume(changes, context) {
            const positions = changes.filter((change) => change.type === 'position');
            if (!positions.length) return null;
            if (!session && positions.some((change) => change.dragging === true)) {
                session = { context, positions: new Map(), draggingIds: new Set(), cancelled: false };
            }
            if (!session) {
                const moves = positions.filter((change) => change.position).map(({ id, position }) => ({ id, position }));
                return moves.length ? { context, moves, cancelled: false, isDrag: false } : null;
            }
            if (!sameContext(session.context, context)) session.cancelled = true;
            for (const change of positions) {
                if (change.position) session.positions.set(change.id, change.position);
                if (change.dragging === true) session.draggingIds.add(change.id);
                if (change.dragging === false) session.draggingIds.delete(change.id);
            }
            if (session.draggingIds.size) return null;
            const finished = session;
            session = null;
            return {
                context: finished.context,
                moves: finished.cancelled ? [] : [...finished.positions].map(([id, position]) => ({ id, position })),
                cancelled: finished.cancelled,
                isDrag: true,
            };
        },
    };
}
