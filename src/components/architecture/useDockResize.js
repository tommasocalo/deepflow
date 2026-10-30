import { useEffect, useRef, useState } from 'react';

const WIDTH_KEY = 'deepflow:model-table-width';
export const MIN_WIDTH = 380;
export const DEFAULT_WIDTH = 760;
const MIN_HEIGHT = 160;
/** A bottom sheet the user drags keeps this much canvas above it... */
const MIN_CANVAS = 120;
/** ...except in a pane too short for that and three rows: there the sheet may
 * take all but 90px, as its default does (see model-table.css). */
const SHEET_ROWS = 190;
const CANVAS_SLIVER = 90;
const MAX_WIDTH_SHARE = 0.72;
const KEY_STEP = 16;

/** The heights a bottom sheet can take in a pane: [lowest, highest]. */
function sheetRange(paneHeight) {
    const highest = Math.max(0, paneHeight - MIN_CANVAS, Math.min(SHEET_ROWS, paneHeight - CANVAS_SLIVER));
    return [Math.min(MIN_HEIGHT, highest), highest];
}

/** A width the user chose earlier, or null for the default. */
function readWidth() {
    try {
        const saved = Number(localStorage.getItem(WIDTH_KEY));
        return Number.isFinite(saved) && saved >= MIN_WIDTH ? saved : null;
    } catch {
        return null;
    }
}
function saveWidth(width) {
    try {
        if (width === null) localStorage.removeItem(WIDTH_KEY);
        else localStorage.setItem(WIDTH_KEY, String(Math.round(width)));
    } catch {
        // The width is a convenience; storage may be unavailable.
    }
}

/**
 * The size of the docked table: a width beside the canvas, or a height when
 * narrow panes turn it into a bottom sheet. The separator resizes it by drag,
 * arrow keys, Home and End; a double-click restores the default. Widths are
 * remembered, sheet heights last for the session.
 */
export function useDockResize(asideRef) {
    const [width, setWidth] = useState(readWidth);
    const [height, setHeight] = useState(null);
    const [layout, setLayout] = useState({ sheet: false, width: null, height: null, floor: null, room: null });
    const dragRef = useRef(null);
    const { sheet } = layout;

    // The workspace stacks the panel under the canvas in narrow panes (CSS decides).
    useEffect(() => {
        const aside = asideRef.current;
        const workspace = aside?.parentElement;
        if (!workspace || typeof ResizeObserver === 'undefined') return undefined;
        const observer = new ResizeObserver(() => {
            const box = aside.getBoundingClientRect();
            const sheet = getComputedStyle(workspace).flexDirection === 'column';
            const parent = workspace.getBoundingClientRect();
            // The sizes a drag can reach, for the separator's value range.
            const [floor, room] = sheet ? sheetRange(parent.height).map(Math.round) : [MIN_WIDTH, Math.round(parent.width * MAX_WIDTH_SHARE)];
            const next = { sheet, width: Math.round(box.width), height: Math.round(box.height), floor, room };
            setLayout((current) => current.sheet === next.sheet && current.width === next.width && current.height === next.height
                && current.floor === next.floor && current.room === next.room ? current : next);
        });
        observer.observe(workspace);
        observer.observe(aside);
        return () => observer.disconnect();
    }, [asideRef]);

    const parentBox = () => asideRef.current?.parentElement?.getBoundingClientRect();
    const clampWidth = (value) => Math.max(MIN_WIDTH, Math.min((parentBox()?.width ?? 0) * MAX_WIDTH_SHARE || Infinity, value));
    const clampHeight = (value) => {
        const paneHeight = parentBox()?.height;
        if (!paneHeight) return Math.max(MIN_HEIGHT, value);
        const [lowest, highest] = sheetRange(paneHeight);
        return Math.max(lowest, Math.min(highest, value));
    };
    const currentWidth = () => width ?? asideRef.current?.getBoundingClientRect().width ?? DEFAULT_WIDTH;
    const currentHeight = () => height ?? asideRef.current?.getBoundingClientRect().height ?? 320;
    const resizeBy = (delta) => {
        if (sheet) setHeight(clampHeight(currentHeight() + delta));
        else {
            const next = clampWidth(currentWidth() + delta);
            setWidth(next);
            saveWidth(next);
        }
    };

    const size = sheet ? layout.height : layout.width;
    const minimum = sheet ? layout.floor ?? MIN_HEIGHT : MIN_WIDTH;
    const separator = {
        role: 'separator',
        tabIndex: 0,
        'aria-label': 'Resize model table',
        'aria-orientation': sheet ? 'horizontal' : 'vertical',
        'aria-valuenow': size ?? undefined,
        'aria-valuemin': minimum,
        'aria-valuemax': layout.room ? Math.max(minimum, layout.room) : undefined,
        'aria-valuetext': size ? `${size} pixels ${sheet ? 'tall' : 'wide'}` : undefined,
        title: `Drag, or use the arrow keys, to resize the table; double-click to restore its size`,
        onPointerDown: (event) => {
            dragRef.current = { x: event.clientX, y: event.clientY, width: currentWidth(), height: currentHeight() };
            event.currentTarget.setPointerCapture?.(event.pointerId);
            event.preventDefault();
        },
        onPointerMove: (event) => {
            const drag = dragRef.current;
            if (!drag) return;
            if (sheet) setHeight(clampHeight(drag.height + drag.y - event.clientY));
            else setWidth(clampWidth(drag.width + drag.x - event.clientX));
        },
        onPointerUp: (event) => {
            dragRef.current = null;
            event.currentTarget.releasePointerCapture?.(event.pointerId);
            if (!sheet && width) saveWidth(width);
        },
        onPointerCancel: () => { dragRef.current = null; },
        onDoubleClick: () => {
            if (sheet) setHeight(null);
            else {
                setWidth(null);
                saveWidth(null);
            }
        },
        onKeyDown: (event) => {
            const keys = sheet ? { ArrowUp: KEY_STEP, ArrowDown: -KEY_STEP } : { ArrowLeft: KEY_STEP, ArrowRight: -KEY_STEP };
            if (keys[event.key]) resizeBy(keys[event.key]);
            else if (event.key === 'Home' || event.key === 'End') resizeBy(event.key === 'Home' ? -100000 : 100000);
            else return;
            event.preventDefault();
        },
    };
    const style = { ...(width ? { '--model-table-width': `${width}px` } : {}), ...(height ? { '--model-table-height': `${height}px` } : {}) };
    return { sheet, width: layout.width, separator, style };
}
