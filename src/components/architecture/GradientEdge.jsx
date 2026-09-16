import { useEffect, useRef, useState } from 'react';
import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath } from '@xyflow/react';
import { ArrowRight, Check, RotateCcw, Scissors, Square } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';

const STYLES = {
    none: { stroke: '#000', label: 'Normal', icon: ArrowRight },
    detach: { stroke: '#94a3b8', dasharray: '4 4', label: 'Detach', icon: Scissors },
    stop: { stroke: '#dc7373', dasharray: '2 6', label: 'Stop', icon: Square },
    reverse: { stroke: '#8b5cf6', label: 'Reverse', icon: RotateCcw },
};

const DETOUR_OFFSET = 190;
const CORNER = 10;

/**
 * Orthogonal path that leaves the column on the right, for edges that
 * would otherwise run straight through the nodes between source and
 * target (residual / skip connections).
 */
function detourPath(x1, y1, x2, y2) {
    const xr = Math.max(x1, x2) + DETOUR_OFFSET;
    const yTop = y1 + 14;
    const yBottom = y2 - 14;
    const r = CORNER;
    const d = [
        `M ${x1} ${y1}`,
        `L ${x1} ${yTop - r}`,
        `Q ${x1} ${yTop} ${x1 + r} ${yTop}`,
        `L ${xr - r} ${yTop}`,
        `Q ${xr} ${yTop} ${xr} ${yTop + r}`,
        `L ${xr} ${yBottom - r}`,
        `Q ${xr} ${yBottom} ${xr - r} ${yBottom}`,
        `L ${x2 + r} ${yBottom}`,
        `Q ${x2} ${yBottom} ${x2} ${yBottom + r}`,
        `L ${x2} ${y2}`,
    ].join(' ');
    return [d, xr, (yTop + yBottom) / 2];
}

function GradientPicker({ id, op, lambda, x, y }) {
    const [open, setOpen] = useState(true);
    const [draftLambda, setDraftLambda] = useState(String(lambda));
    const ref = useRef(null);
    const setEdgeGradientOp = useArchitectureStore((state) => state.setEdgeGradientOp);
    useEffect(() => {
        if (!open) return;
        const outside = (event) => { if (!ref.current?.contains(event.target)) setOpen(false); };
        const escape = (event) => { if (event.key === 'Escape') setOpen(false); };
        document.addEventListener('mousedown', outside);
        document.addEventListener('keydown', escape);
        return () => { document.removeEventListener('mousedown', outside); document.removeEventListener('keydown', escape); };
    }, [open]);
    if (!open) return null;
    return (
        <div ref={ref} className="nodrag nopan df-gradient-picker" style={{ transform: `translate(-50%, 10px) translate(${x}px, ${y}px)` }} onMouseDown={(event) => event.stopPropagation()} onClick={(event) => event.stopPropagation()}>
            <h3>Gradient flow</h3>
            {Object.entries(STYLES).map(([key, style]) => {
                const Icon = style.icon;
                return <button type="button" key={key} aria-pressed={key === op} onClick={() => { setEdgeGradientOp(id, key, key === 'reverse' ? lambda : undefined); if (key !== 'reverse') setOpen(false); }}><Icon size={13} /><span>{style.label}</span>{key === op && <Check size={12} />}</button>;
            })}
            {op === 'reverse' && <label className="df-gradient-lambda">Scale λ<input type="text" inputMode="decimal" value={draftLambda} onChange={(event) => setDraftLambda(event.target.value)} onBlur={() => { const value = Number(draftLambda); if (draftLambda.trim() && Number.isFinite(value)) setEdgeGradientOp(id, 'reverse', value); else setDraftLambda(String(lambda)); }} /></label>}
        </div>
    );
}

export default function GradientEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, selected, interactionWidth }) {
    const op = data?.gradientOp ?? 'none';
    const style = STYLES[op] ?? STYLES.none;
    let path, labelX, labelY;
    if (data?.detour) {
        [path, labelX, labelY] = detourPath(sourceX, sourceY, targetX, targetY);
    } else {
        [path, labelX, labelY] = getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: CORNER, offset: 12 });
    }
    const activation = data?.activation;
    const showLabel = !!activation && Math.abs(targetY - sourceY) > 40;
    return <>
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} style={{ stroke: style.stroke, strokeWidth: selected ? 3 : 2, strokeDasharray: style.dasharray }} />
        {(showLabel || op !== 'none') && (
            <EdgeLabelRenderer>
                <div
                    className={`df-activation${op !== 'none' ? ' df-activation--gradop' : ''}`}
                    style={{ transform: `translate(${data?.detour ? 8 : 10}px, -50%) translate(${labelX}px, ${labelY}px)`, '--edge-color': style.stroke }}
                >
                    {activation}
                    {op !== 'none' && <span className="df-activation-op">{style.label.toLowerCase()}{op === 'reverse' ? ` λ=${data?.gradientOpLambda ?? 1}` : ''}</span>}
                </div>
            </EdgeLabelRenderer>
        )}
        {selected && <EdgeLabelRenderer><GradientPicker key={data?.menuVersion ?? 0} id={id} op={op} lambda={data?.gradientOpLambda ?? 1} x={data?.menuPosition?.x ?? labelX} y={data?.menuPosition?.y ?? labelY} /></EdgeLabelRenderer>}
    </>;
}
