import { useId } from 'react';
import { epochTicks, linearScale, linePath, spreadLabels } from '../charts.js';
import { tr } from '../text.js';

const FULL = { left: 36, right: 52, top: 12, bottom: 22 };
const COMPACT = { left: 2, right: 2, top: 3, bottom: 3 };

/**
 * A dependency-free SVG line chart over epochs (docs/edu/SPEC.md §3.5): the current run's series, dimmed earlier
 * runs ("ghosts") labelled at their end, a dashed reference line (chance level) and an optional ▲ marker. Lines are
 * labelled directly, and the caller passes a text alternative (`desc`) plus a visually hidden table.
 *
 * series: [{ id, points: [[epoch, value]], label, variant: 'train' | 'val' }]
 * ghosts: [{ id, points, label, short? }]   `short` is drawn at the line end, `label` goes to the legend
 */
export default function CurveChart({
    width, height, xMax, yMin = 0, yMax, yFormat, series = [], ghosts = [], reference = null, marker = null,
    title, desc, compact = false, table = null,
}) {
    const titleId = useId();
    const descId = useId();
    const pad = compact ? COMPACT : FULL;
    const w = Math.max(80, width);
    const sx = linearScale(0, Math.max(1, xMax), pad.left, w - pad.right);
    const sy = linearScale(yMin, yMax, height - pad.bottom, pad.top);
    const clamp = [yMin, yMax];
    const yTicks = [yMin, (yMin + yMax) / 2, yMax];
    const endLabels = compact ? [] : spreadLabels([
        ...series.filter((item) => item.points.length > 1).map((item) => ({ id: item.id, text: item.label, y: sy(Math.min(yMax, Math.max(yMin, item.points.at(-1)[1]))), x: sx(item.points.at(-1)[0]), kind: item.variant })),
        ...ghosts.filter((item) => item.points.length > 1).map((item) => ({ id: item.id, text: item.short ?? item.label, y: sy(Math.min(yMax, Math.max(yMin, item.points.at(-1)[1]))), x: sx(item.points.at(-1)[0]), kind: 'ghost' })),
    ], 10, [pad.top + 4, height - pad.bottom]);

    return (
        <div className={`edu-tr-chart-box${compact ? ' is-compact' : ''}`}>
            <svg className="edu-tr-svg" width={w} height={height} viewBox={`0 0 ${w} ${height}`} role="img" aria-labelledby={`${titleId} ${descId}`}>
                <title id={titleId}>{title}</title>
                <desc id={descId}>{desc}</desc>
                {!compact && (
                    <g className="edu-tr-grid" aria-hidden="true">
                        {yTicks.map((value, index) => (
                            <g key={`y${index}`}>
                                <line x1={pad.left} x2={w - pad.right} y1={sy(value)} y2={sy(value)} />
                                <text x={pad.left - 4} y={sy(value) + 3} textAnchor="end">{yFormat(value)}</text>
                            </g>
                        ))}
                        {epochTicks(Math.max(1, xMax)).map((value) => (
                            <text key={`x${value}`} x={sx(value)} y={height - 7} textAnchor={value === 0 ? 'start' : 'middle'}>{value}</text>
                        ))}
                        <text className="edu-tr-axis-title" x={w - 2} y={height - 7} textAnchor="end">{tr('chart.epoch')}</text>
                    </g>
                )}
                {reference && Number.isFinite(reference.y) && reference.y <= yMax && reference.y >= yMin && (
                    <g className="edu-tr-reference" aria-hidden="true">
                        <line x1={pad.left} x2={w - pad.right} y1={sy(reference.y)} y2={sy(reference.y)} />
                        {!compact && <text x={w - pad.right - 2} y={sy(reference.y) - 3} textAnchor="end">{reference.label}</text>}
                    </g>
                )}
                <g aria-hidden="true">
                    {ghosts.map((item) => <path key={item.id} className="edu-tr-line is-ghost" d={linePath(item.points, sx, sy, clamp)} />)}
                    {series.map((item) => (
                        <g key={item.id}>
                            <path className={`edu-tr-line is-${item.variant}`} d={linePath(item.points, sx, sy, clamp)} />
                            {item.points.length === 1 && Number.isFinite(item.points[0][1]) && (
                                <circle className={`edu-tr-dot is-${item.variant}`} cx={sx(item.points[0][0])} cy={sy(Math.min(yMax, item.points[0][1]))} r={compact ? 2 : 3} />
                            )}
                        </g>
                    ))}
                    {marker && Number.isFinite(marker.y) && (
                        <path className="edu-tr-marker" d={`M${sx(marker.x)},${Math.min(height - pad.bottom, sy(Math.min(yMax, marker.y)) + 3)}l4,7h-8z`}>
                            <title>{marker.title}</title>
                        </path>
                    )}
                    {endLabels.map((label) => (
                        <text key={label.id} className={`edu-tr-end-label is-${label.kind}`} x={Math.min(label.x + 4, w - pad.right + 4)} y={label.y + 3}>{label.text}</text>
                    ))}
                </g>
            </svg>
            {table}
        </div>
    );
}
