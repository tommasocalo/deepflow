import { memo } from 'react';
import { BaseEdge, EdgeLabelRenderer, getSmoothStepPath, useStore } from '@xyflow/react';

import TensorPreview from './TensorPreview.jsx';

const showTensorDetail = (state) => state.transform[2] >= 0.45;

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
    const detailVisible = useStore(showTensorDetail);
    const showTensor = data?.tensorFlow && data?.shape && !data?.detour && detailVisible && targetY - sourceY >= 40;
    const showLabel = !!data?.activation && Math.abs(targetY - sourceY) > 40;
    return <>
        <BaseEdge id={id} path={path} markerEnd={markerEnd} interactionWidth={interactionWidth ?? 20} style={{ stroke: '#111', strokeWidth: selected ? 3 : 2 }} />
        {(showTensor || showLabel) && <EdgeLabelRenderer>
            {showTensor ? (
                <div className="df-tensor-edge nodrag nopan" tabIndex={0} style={{ transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)` }}>
                    <TensorPreview shape={data.shape} compact />
                </div>
            ) : (
                <div className="df-activation" style={{ transform: `translate(12px, -50%) translate(${labelX}px, ${labelY}px)` }}>
                    {data.activation}
                </div>
            )}
        </EdgeLabelRenderer>}
    </>;
}

export default memo(ArchitectureEdge);
