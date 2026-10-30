import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { resolveFirstTarget } from '../lessons/engine/targets.js';

const RING_OFFSET = 4;
const RING_WIDTH = 3;

const sameRect = (a, b) => a && b && a.top === b.top && a.left === b.left && a.width === b.width && a.height === b.height;

/**
 * The lesson spotlight (docs/edu/SPEC.md §3.8, §3.9): a 3 px accent ring 4 px outside the first visible target.
 * It never adds text, never takes pointer events and is hidden from assistive technology; the step text names
 * the control in words. It follows the target while the canvas pans and re-resolves when the UI changes.
 */
export default function Spotlight({ targets = [], active = true, onResolve, options }) {
    const [rect, setRect] = useState(null);
    const [pulse, setPulse] = useState(0);
    const elementRef = useRef(null);
    const onResolveRef = useRef(onResolve);
    const optionsRef = useRef(options);
    const targetKey = JSON.stringify(targets ?? []);

    useEffect(() => { onResolveRef.current = onResolve; }, [onResolve]);
    useEffect(() => { optionsRef.current = options; }, [options]);

    useEffect(() => {
        const refs = JSON.parse(targetKey);
        if (!active || !refs.length) {
            elementRef.current = null;
            onResolveRef.current?.(undefined);
            return undefined;
        }
        let frame = 0;
        let last;
        const resolve = () => {
            const found = resolveFirstTarget(refs, globalThis.document, optionsRef.current);
            const element = found?.element ?? null;
            if (element !== last) {
                last = element;
                elementRef.current = element;
                onResolveRef.current?.(found);
                if (element) setPulse((value) => value + 1);
            }
        };
        const track = () => {
            const element = elementRef.current;
            if (element?.isConnected) {
                const bounds = element.getBoundingClientRect();
                const next = bounds.width > 0 || bounds.height > 0
                    ? { top: Math.round(bounds.top), left: Math.round(bounds.left), width: Math.round(bounds.width), height: Math.round(bounds.height) } : null;
                setRect((previous) => (sameRect(previous, next) ? previous : next));
            } else {
                setRect((previous) => (previous ? null : previous));
            }
            frame = requestAnimationFrame(track);
        };
        resolve();
        const timer = setInterval(resolve, 400);
        frame = requestAnimationFrame(track);
        return () => {
            clearInterval(timer);
            cancelAnimationFrame(frame);
        };
    }, [active, targetKey]);

    if (!active || !targets?.length || !rect || typeof document === 'undefined') return null;
    const outset = RING_OFFSET + RING_WIDTH;
    return createPortal(
        <div
            key={pulse}
            className="edu-spotlight"
            aria-hidden="true"
            data-edu-spotlight=""
            style={{ top: rect.top - outset, left: rect.left - outset, width: rect.width + outset * 2, height: rect.height + outset * 2 }}
        />,
        document.body,
    );
}
