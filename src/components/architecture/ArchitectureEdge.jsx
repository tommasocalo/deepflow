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
    const gradient = data?.gradient;
    const color = gradient?.status === 'stopped' ? '#b91c1c' : gradient?.status === 'reverse' ? '#6d28d9'
        : ['constant', 'unknown'].includes(gradient?.status) ? '#94949b' : '#111';
    const dash = gradient?.status === 'unknown' ? '2 5' : ['constant', 'stopped'].includes(gradient?.status) ? '6 4' : undefined;
    return <>
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} className={gradient ? `df-gradient-edge--${gradient.status}` : undefined} style={{ stroke: selected ? '#9d174d' : color, strokeWidth: 2, ...(dash ? { strokeDasharray: dash } : {}) }} />
        {selected && <g className="df-connection-grips" pointerEvents="none" aria-hidden="true" fill="white" stroke="#9d174d" strokeWidth="1.5">
            <circle {...gripPosition(sourceX, sourceY, sourcePosition)} r="3.5" />
            <circle {...gripPosition(targetX, targetY, targetPosition)} r="3.5" />
        </g>}
    </>;
}

export default memo(ArchitectureEdge);
