import React, { useEffect, useRef, useState } from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from '@xyflow/react';
import { Scissors, Square, Minus, RotateCcw } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';

/**
 * Custom edge that surfaces an edge-level gradient operator. Click the
 * mid-edge badge (or the small "+" hint) to pick none/detach/stop/reverse.
 * The popover persists until you click outside it or pick a value.
 */
const STYLES = {
    none:    { stroke: '#000000', dasharray: undefined,  badgeClass: 'bg-white text-black',           label: 'grad' },
    detach:  { stroke: '#71717a', dasharray: '4 4',      badgeClass: 'bg-stone-200 text-stone-800',   label: 'detach' },
    stop:    { stroke: '#dc2626', dasharray: '2 6',      badgeClass: 'bg-red-200 text-red-900',       label: 'stop' },
    reverse: { stroke: '#9333ea', dasharray: undefined,  badgeClass: 'bg-purple-200 text-purple-900', label: 'rev λ' },
};

const ICONS = {
    none:    Minus,
    detach:  Scissors,
    stop:    Square,
    reverse: RotateCcw,
};

const GradientEdge = ({
    id,
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
    data,
    markerEnd,
    selected,
    interactionWidth,
}) => {
    const [open, setOpen] = useState(false);
    const ref = useRef(null);
    const setEdgeGradientOp = useArchitectureStore((s) => s.setEdgeGradientOp);

    const op = data?.gradientOp ?? 'none';
    const lambda = data?.gradientOpLambda ?? 1;
    const style = STYLES[op] ?? STYLES.none;
    const Icon = ICONS[op] ?? Minus;

    // When the edge is selected, surface the menu automatically — that's
    // the natural "now what?" moment. Closing again on deselect.
    useEffect(() => {
        if (selected) setOpen(true);
        else setOpen(false);
    }, [selected, data?.menuVersion]);

    // Click-outside dismissal as a backup (the open menu must persist
    // while the cursor moves over it).
    useEffect(() => {
        if (!open) return;
        const onDoc = (e) => {
            if (!ref.current?.contains(e.target)) setOpen(false);
        };
        document.addEventListener('mousedown', onDoc);
        return () => document.removeEventListener('mousedown', onDoc);
    }, [open]);

    const [path, labelX, labelY] = getBezierPath({
        sourceX, sourceY, targetX, targetY,
        sourcePosition, targetPosition,
    });
    const anchorX = data?.menuPosition?.x ?? labelX;
    const anchorY = data?.menuPosition?.y ?? labelY;

    return (
        <>
            <BaseEdge
                id={id}
                path={path}
                markerEnd={markerEnd}
                interactionWidth={interactionWidth ?? 20}
                style={{
                    stroke: style.stroke,
                    strokeWidth: selected ? 3 : 2,
                    strokeDasharray: style.dasharray,
                }}
            />
            <EdgeLabelRenderer>
                <div
                    ref={ref}
                    style={{
                        position: 'absolute',
                        transform: `translate(-50%, -50%) translate(${anchorX}px, ${anchorY}px)`,
                        pointerEvents: 'all',
                        // High z-index so the dropdown menu sits above
                        // architecture nodes when the edge is selected.
                        zIndex: 9999,
                    }}
                    className="nodrag nopan"
                >
                    {/* No on-canvas badge ever — the operator is shown
                        purely through the edge path's stroke colour and
                        dasharray. The picker menu below appears only when
                        the edge is selected. */}
                    {open && (
                        <div
                            className="absolute top-full mt-1 left-1/2 -translate-x-1/2 bg-white border-2 border-black rounded-md shadow-hard p-1 flex flex-col gap-0.5 min-w-[150px]"
                            style={{ zIndex: 9999 }}
                            onMouseDown={(e) => e.stopPropagation()}
                        >
                            {(['none', 'detach', 'stop', 'reverse']).map((k) => {
                                const I = ICONS[k];
                                return (
                                    <button
                                        key={k}
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            setEdgeGradientOp(id, k, k === 'reverse' ? lambda : undefined);
                                            if (k !== 'reverse') setOpen(false);
                                        }}
                                        className={`flex items-center gap-1.5 px-2 py-1 rounded text-[11px] font-bold hover:bg-stone-100 ${k === op ? 'bg-stone-200' : ''}`}
                                    >
                                        <I size={11} strokeWidth={3} />
                                        {STYLES[k].label}
                                    </button>
                                );
                            })}
                            {op === 'reverse' && (
                                <label className="flex items-center gap-1.5 px-2 py-1 text-[11px] border-t border-black/10">
                                    λ:
                                    <input
                                        type="number"
                                        step="any"
                                        value={lambda}
                                        onChange={(e) =>
                                            setEdgeGradientOp(id, 'reverse', Number.parseFloat(e.target.value))
                                        }
                                        onClick={(e) => e.stopPropagation()}
                                        className="border-2 border-black rounded px-1 py-0.5 text-[11px] w-20 font-mono"
                                    />
                                </label>
                            )}
                        </div>
                    )}
                </div>
            </EdgeLabelRenderer>
        </>
    );
};

export default GradientEdge;
