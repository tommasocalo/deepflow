import { memo } from 'react';
import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, getBezierPath } from '@xyflow/react';


const DETOUR_OFFSET = 190;
const CORNER = 10;

/** A stable lane for residual connections; endpoints still follow the nodes. */
function detourPath(sourceX, sourceY, targetX, targetY) {
    const laneX = Math.max(sourceX, targetX) + DETOUR_OFFSET;
    const top = sourceY + 16;
    const bottom = targetY - 16;
    const direction = bottom >= top ? 1 : -1;
    const radius = Math.min(CORNER, Math.abs(bottom - top) / 2);
    const path = [
        `M ${sourceX} ${sourceY}`,
        `L ${sourceX} ${top - CORNER}`,
        `Q ${sourceX} ${top} ${sourceX + CORNER} ${top}`,
        `L ${laneX - radius} ${top}`,
        `Q ${laneX} ${top} ${laneX} ${top + direction * radius}`,
        `L ${laneX} ${bottom - direction * radius}`,
        `Q ${laneX} ${bottom} ${laneX - radius} ${bottom}`,
        `L ${targetX + CORNER} ${bottom}`,
        `Q ${targetX} ${bottom} ${targetX} ${bottom + CORNER}`,
        `L ${targetX} ${targetY}`,
    ].join(' ');
    return [path, laneX, (top + bottom) / 2];
}

/** Architecture shows forward tensor flow. Gradient controls belong to Training. */
function ArchitectureEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, selected, interactionWidth }) {
    const [path, labelX, labelY] = data?.detour
        ? detourPath(sourceX, sourceY, targetX, targetY)
        : getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: CORNER, offset: 12 });
    const showLabel = selected && !!data?.activation;
    const stream = data?.featureStream && !data.detour && targetY > sourceY + 8;
    return <>
        {stream && <g className="df-feature-stream" pointerEvents="none" aria-hidden="true">
            {[-.48, -.32, -.16, .16, .32, .48].map((offset) => {
                const [strand] = getBezierPath({
                    sourceX: sourceX + offset * (data.sourceSpan ?? 0), sourceY,
                    targetX: targetX + offset * (data.targetSpan ?? 0), targetY,
                    sourcePosition, targetPosition, curvature: .35,
                });
                return <path key={offset} d={strand} fill="none" stroke={data.accent ?? '#666'} strokeWidth={selected ? 1.5 : 1} opacity={selected ? .7 : .4} />;
            })}
        </g>}
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} style={{ stroke: '#111', strokeWidth: selected ? 2.5 : stream ? 1 : 2 }} />
        {showLabel && <EdgeLabelRenderer>
            <div className="df-activation" style={{ transform: `translate(12px, -50%) translate(${labelX}px, ${labelY}px)` }}>
                {data.activation}
            </div>
        </EdgeLabelRenderer>}
    </>;
}

export default memo(ArchitectureEdge);
