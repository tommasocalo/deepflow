import { useLayoutEffect, useRef } from 'react';
import { useStoreApi } from '@xyflow/react';
import { createInspectorPlacementScheduler, placeInspector } from './inspectorPlacement.js';

const relativeRect = (element, origin) => {
    const rect = element.getBoundingClientRect();
    return { x: rect.left - origin.left, y: rect.top - origin.top, width: rect.width, height: rect.height };
};


// Include visible port labels: they extend beyond the content-sized card.
const nodeRect = (element, origin) => {
    const boxes = [relativeRect(element, origin), ...[...element.querySelectorAll('.df-port-shape')]
        .filter((label) => getComputedStyle(label).visibility !== 'hidden')
        .map((label) => relativeRect(label, origin))];
    const x = Math.min(...boxes.map((box) => box.x));
    const y = Math.min(...boxes.map((box) => box.y));
    return {
        x, y,
        width: Math.max(...boxes.map((box) => box.x + box.width)) - x,
        height: Math.max(...boxes.map((box) => box.y + box.height)) - y,
    };
};

/**
 * Keep a floating panel beside its anchor. `anchorFor` returns the anchor
 * rectangle and the node elements to avoid, both relative to the canvas.
 */
function attachPlacement(panel, canvas, flowStore, anchorFor) {
    // Wait for valid geometry before showing a newly opened panel.
    if (!panel.dataset.placement) panel.style.visibility = 'hidden';
    const position = () => {
        const origin = canvas.getBoundingClientRect();
        if (!origin.width || !origin.height) return;
        panel.style.maxHeight = `${Math.min(460, Math.max(120, origin.height - 24))}px`;
        const target = anchorFor(origin, [...canvas.querySelectorAll('.react-flow__node')]);
        if (!target) return;
        const obstacles = [...canvas.querySelectorAll('.df-library, .react-flow__controls, [aria-label^="Graph checks:"], .df-inspector')]
            .filter((element) => element !== panel)
            .map((element) => relativeRect(element, origin));
        const placement = placeInspector({
            anchor: target.anchor,
            panel: { width: panel.offsetWidth || 260, height: panel.offsetHeight || 320 },
            bounds: { width: origin.width, height: origin.height },
            obstacles,
            nodes: target.nodes.map((element) => nodeRect(element, origin)),
        });
        panel.style.left = `${Math.round(placement.x)}px`;
        panel.style.top = `${Math.round(placement.y)}px`;
        panel.dataset.placement = placement.side;
        panel.style.visibility = '';
    };
    const scheduler = createInspectorPlacementScheduler({
        getNodes: () => flowStore.getState().nodes,
        position,
        requestFrame: requestAnimationFrame,
        cancelFrame: cancelAnimationFrame,
    });
    const schedule = scheduler.schedule;
    const resizeObserver = new ResizeObserver(schedule);
    resizeObserver.observe(canvas);
    resizeObserver.observe(panel);
    const palette = canvas.querySelector('.df-library');
    if (palette) resizeObserver.observe(palette);
    // Palette collapse and parameter disclosures change available space.
    const mutationObserver = new window.MutationObserver(schedule);
    mutationObserver.observe(canvas, { childList: true, subtree: true });
    const unsubscribe = flowStore.subscribe(schedule);
    scheduler.flush();
    return () => {
        unsubscribe();
        resizeObserver.disconnect();
        mutationObserver.disconnect();
        scheduler.dispose();
    };
}

/** Read geometry after React Flow commits and move only the overlay, not graph state. */
export function useInspectorPlacement(nodeId) {
    const panelRef = useRef(null);
    const flowStore = useStoreApi();
    useLayoutEffect(() => {
        const panel = panelRef.current;
        const canvas = panel?.closest('[data-architecture-canvas]');
        if (!panel || !canvas || !nodeId) return;
        return attachPlacement(panel, canvas, flowStore, (origin, nodeElements) => {
            const selected = nodeElements.find((element) => element.dataset.id === nodeId);
            return selected ? { anchor: nodeRect(selected, origin), nodes: nodeElements.filter((element) => element !== selected) } : null;
        });
    }, [nodeId, flowStore]);
    return panelRef;
}

/**
 * Place a connection's panel beside the point where the connection was
 * clicked (flow coordinates, so it follows pan and zoom), or beside the whole
 * connection when it was selected another way.
 */
export function useConnectionPlacement(edgeId, point) {
    const panelRef = useRef(null);
    const flowStore = useStoreApi();
    const x = point?.x;
    const y = point?.y;
    useLayoutEffect(() => {
        const panel = panelRef.current;
        const canvas = panel?.closest('[data-architecture-canvas]');
        if (!panel || !canvas || !edgeId) return;
        return attachPlacement(panel, canvas, flowStore, (origin, nodeElements) => {
            if (Number.isFinite(x) && Number.isFinite(y)) {
                const [tx, ty, zoom] = flowStore.getState().transform;
                return { anchor: { x: x * zoom + tx - 12, y: y * zoom + ty - 12, width: 24, height: 24 }, nodes: nodeElements };
            }
            const edge = [...canvas.querySelectorAll('.react-flow__edge')].find((element) => element.dataset.id === edgeId);
            return edge ? { anchor: relativeRect(edge, origin), nodes: nodeElements } : null;
        });
    }, [edgeId, x, y, flowStore]);
    return panelRef;
}
