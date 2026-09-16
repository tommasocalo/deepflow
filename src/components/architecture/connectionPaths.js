import { getBezierPath } from '@xyflow/react';

// Cards are at most 240px wide. A 144px lane leaves 24px of clearance
// beside a centred column without the former 190px excursion.
const LANE_OFFSET = 144;
const CORNER = 24;

function bypassPath(sourceX, sourceY, targetX, targetY) {
    const laneX = Math.max(sourceX, targetX) + LANE_OFFSET;
    const top = sourceY + 32;
    const bottom = targetY - 32;
    const direction = bottom >= top ? 1 : -1;
    const radius = Math.min(CORNER, Math.abs(bottom - top) / 2);
    return [
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
}

/** A wire depends only on its endpoints and whether it bypasses a graph branch. */
export function connectionPath({ sourceX, sourceY, targetX, targetY, sourcePosition = 'bottom', targetPosition = 'top', detour = false }) {
    const vertical = sourcePosition === 'bottom' && targetPosition === 'top';
    if (vertical && detour) return bypassPath(sourceX, sourceY, targetX, targetY);
    if (vertical && targetY < sourceY) {
        // Fold a backwards wire around its endpoints gradually, rather than
        // jumping to a lateral lane as soon as the nodes cross vertically.
        const depth = sourceY - targetY;
        const fold = Math.min(1, depth / 80);
        const lead = Math.min(24, depth * 0.35);
        const turn = Math.min(48, depth / 2);
        const middleX = (sourceX + targetX) / 2;
        const middleY = (sourceY + targetY) / 2;
        const laneX = middleX + (Math.max(sourceX, targetX) + LANE_OFFSET - middleX) * fold;
        return `M ${sourceX} ${sourceY} L ${sourceX} ${sourceY + lead} C ${sourceX} ${sourceY + lead + turn} ${laneX} ${sourceY + lead + turn} ${laneX} ${middleY} C ${laneX} ${targetY - lead - turn} ${targetX} ${targetY - lead - turn} ${targetX} ${targetY - lead} L ${targetX} ${targetY}`;
    }
    if (vertical) {
        // Short straight leads keep port labels centred on the wire. Between
        // them a continuous curve follows the endpoints, without elbow flips.
        const lead = Math.min(24, (targetY - sourceY) * 0.35);
        const middleY = (sourceY + targetY) / 2;
        return `M ${sourceX} ${sourceY} L ${sourceX} ${sourceY + lead} C ${sourceX} ${middleY} ${targetX} ${middleY} ${targetX} ${targetY - lead} L ${targetX} ${targetY}`;
    }
    return getBezierPath({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, curvature: 0.2 })[0];
}
