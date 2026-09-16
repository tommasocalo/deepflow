import { useLayoutEffect, useRef } from 'react';
import { useStoreApi } from '@xyflow/react';
import { placeInspector } from './inspectorPlacement.js';

const relativeRect = (element, origin) => {
    const rect = element.getBoundingClientRect();
    return { x: rect.left - origin.left, y: rect.top - origin.top, width: rect.width, height: rect.height };
};

/** Read geometry after React Flow commits and move only the overlay, not graph state. */
export function useInspectorPlacement(nodeId) {
    const panelRef = useRef(null);
    const flowStore = useStoreApi();
    useLayoutEffect(() => {
        const panel = panelRef.current;
        const canvas = panel?.closest('[data-architecture-canvas]');
        if (!panel || !canvas || !nodeId) return;
        let frame = null;
        const position = () => {
            frame = null;
            const origin = canvas.getBoundingClientRect();
            if (!origin.width || !origin.height) return;
            panel.style.maxHeight = `${Math.min(460, Math.max(120, origin.height - 24))}px`;
            const nodeElements = [...canvas.querySelectorAll('.react-flow__node')];
            const selected = nodeElements.find((element) => element.dataset.id === nodeId);
            if (!selected) return;
            const bounds = { width: origin.width, height: origin.height };
            const obstacles = [...canvas.querySelectorAll('.df-library, .react-flow__controls, [aria-label^="Graph checks:"]')]
                .map((element) => relativeRect(element, origin));
            const nodes = nodeElements.filter((element) => element !== selected).map((element) => relativeRect(element, origin));
            const placement = placeInspector({
                anchor: relativeRect(selected, origin),
                panel: { width: panel.offsetWidth || 272, height: panel.offsetHeight || 320 },
                bounds, obstacles, nodes,
            });
            panel.style.left = `${Math.round(placement.x)}px`;
            panel.style.top = `${Math.round(placement.y)}px`;
            panel.dataset.placement = placement.side;
        };
        const schedule = () => { if (frame === null) frame = requestAnimationFrame(position); };
        const resizeObserver = new ResizeObserver(schedule);
        resizeObserver.observe(canvas);
        resizeObserver.observe(panel);
        const palette = canvas.querySelector('.df-library');
        if (palette) resizeObserver.observe(palette);
        // Palette collapse and parameter disclosures change available space.
        const mutationObserver = new window.MutationObserver(schedule);
        mutationObserver.observe(canvas, { childList: true, subtree: true });
        const unsubscribe = flowStore.subscribe(schedule);
        position();
        return () => {
            unsubscribe();
            resizeObserver.disconnect();
            mutationObserver.disconnect();
            if (frame !== null) cancelAnimationFrame(frame);
        };
    }, [nodeId, flowStore]);
    return panelRef;
}
