import { memo } from 'react';
import { BaseEdge, getSmoothStepPath } from '@xyflow/react';


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

/** Offset the visible grip into the native reconnect hit area, clear of node ports. */
function gripPosition(x, y, position) {
    return {
        cx: x + (position === 'left' ? -12 : position === 'right' ? 12 : 0),
        cy: y + (position === 'top' ? -12 : position === 'bottom' ? 12 : 0),
    };
}

/** A single connection carries tensor flow. Shapes belong to the node. */
function ArchitectureEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, selected, interactionWidth }) {
    const [path] = data?.detour
        ? detourPath(sourceX, sourceY, targetX, targetY)
        : getSmoothStepPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, borderRadius: CORNER, offset: 12 });
    return <>
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} style={{ stroke: selected ? '#9d174d' : '#111', strokeWidth: 2 }} />
        {selected && <g className="df-connection-grips" pointerEvents="none" aria-hidden="true" fill="white" stroke="#9d174d" strokeWidth="1.5">
            <circle {...gripPosition(sourceX, sourceY, sourcePosition)} r="3.5" />
            <circle {...gripPosition(targetX, targetY, targetPosition)} r="3.5" />
        </g>}
    </>;
}

export default memo(ArchitectureEdge);
