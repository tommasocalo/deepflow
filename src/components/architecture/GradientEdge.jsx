import { useEffect, useRef, useState } from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from '@xyflow/react';
import { ArrowRight, Check, RotateCcw, Scissors, Square } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';

const STYLES = {
    none: { stroke: '#94a3b8', label: 'Normal', icon: ArrowRight },
    detach: { stroke: '#94a3b8', dasharray: '4 4', label: 'Detach', icon: Scissors },
    stop: { stroke: '#dc7373', dasharray: '2 6', label: 'Stop', icon: Square },
    reverse: { stroke: '#8b5cf6', label: 'Reverse', icon: RotateCcw },
};

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
    const [path, labelX, labelY] = getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition });
    return <>
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} style={{ stroke: selected ? '#6366f1' : style.stroke, strokeWidth: selected ? 2 : 1.5, strokeDasharray: style.dasharray }} />
        {selected && <EdgeLabelRenderer><GradientPicker key={data?.menuVersion ?? 0} id={id} op={op} lambda={data?.gradientOpLambda ?? 1} x={data?.menuPosition?.x ?? labelX} y={data?.menuPosition?.y ?? labelY} /></EdgeLabelRenderer>}
    </>;
}
