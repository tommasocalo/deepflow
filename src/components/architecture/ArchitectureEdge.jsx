import { memo } from 'react';
import { BaseEdge } from '@xyflow/react';
import { connectionPath } from './connectionPaths.js';

/** A single connection carries tensor flow. Shapes belong to the node.
 * Selection is a highlighter under the wire: colour stays reserved for gradients,
 * and reconnection uses the native endpoint areas without extra markers. */
function ArchitectureEdge({ id, sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, markerEnd, selected, interactionWidth }) {
    const path = connectionPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, detour: data?.detour });
    const gradient = data?.gradient;
    const color = gradient?.status === 'stopped' ? '#b91c1c' : gradient?.status === 'reverse' ? '#6d28d9'
        : ['constant', 'unknown'].includes(gradient?.status) ? '#94949b' : '#111';
    const dash = gradient?.status === 'unknown' ? '2 5' : ['constant', 'stopped'].includes(gradient?.status) ? '6 4' : undefined;
    return <>
        {selected && <path d={path} className="df-connection-selection" fill="none" pointerEvents="none" aria-hidden="true" />}
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} className={gradient ? `df-gradient-edge--${gradient.status}` : undefined} style={{ stroke: color, strokeWidth: selected ? 2.5 : 2, ...(dash ? { strokeDasharray: dash } : {}) }} />
    </>;
}

export default memo(ArchitectureEdge);
