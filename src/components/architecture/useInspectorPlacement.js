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

/** Read geometry after React Flow commits and move only the overlay, not graph state. */
export function useInspectorPlacement(nodeId) {
    const panelRef = useRef(null);
    const flowStore = useStoreApi();
    useLayoutEffect(() => {
        const panel = panelRef.current;
        const canvas = panel?.closest('[data-architecture-canvas]');
        if (!panel || !canvas || !nodeId) return;
        // Wait for valid geometry before showing a newly opened inspector.
        if (!panel.dataset.placement) panel.style.visibility = 'hidden';
        const position = () => {
            const origin = canvas.getBoundingClientRect();
            if (!origin.width || !origin.height) return;
            panel.style.maxHeight = `${Math.min(460, Math.max(120, origin.height - 24))}px`;
            const nodeElements = [...canvas.querySelectorAll('.react-flow__node')];
            const selected = nodeElements.find((element) => element.dataset.id === nodeId);
            if (!selected) return;
            const bounds = { width: origin.width, height: origin.height };
            const obstacles = [...canvas.querySelectorAll('.df-library, .react-flow__controls, [aria-label^="Graph checks:"]')]
                .map((element) => relativeRect(element, origin));
            const nodes = nodeElements.filter((element) => element !== selected).map((element) => nodeRect(element, origin));
            const placement = placeInspector({
                anchor: nodeRect(selected, origin),
                panel: { width: panel.offsetWidth || 260, height: panel.offsetHeight || 320 },
                bounds, obstacles, nodes,
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
    }, [nodeId, flowStore]);
    return panelRef;
}
