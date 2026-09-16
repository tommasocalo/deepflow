import { memo } from 'react';
import { BaseEdge } from '@xyflow/react';
import { connectionPath } from './connectionPaths.js';

/** Offset the visible grip into the native reconnect hit area, clear of node ports. */
function gripPosition(x, y, position) {
    return {
        cx: x + (position === 'left' ? -12 : position === 'right' ? 12 : 0),
        cy: y + (position === 'top' ? -12 : position === 'bottom' ? 12 : 0),
    };
}

/** A single connection carries tensor flow. Shapes belong to the node. */
function ArchitectureEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, selected, interactionWidth }) {
    const path = connectionPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, detour: data?.detour });
    return <>
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} style={{ stroke: selected ? '#9d174d' : '#111', strokeWidth: 2 }} />
        {selected && <g className="df-connection-grips" pointerEvents="none" aria-hidden="true" fill="white" stroke="#9d174d" strokeWidth="1.5">
            <circle {...gripPosition(sourceX, sourceY, sourcePosition)} r="3.5" />
            <circle {...gripPosition(targetX, targetY, targetPosition)} r="3.5" />
        </g>}
    </>;
}

export default memo(ArchitectureEdge);
