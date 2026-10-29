import { useId } from 'react';
import { ClassMarker } from './ClassMarker.jsx';
import { divergingColor } from './classStyle.js';
import { dtx } from './text.js';
import { formatTick, niceTicks } from '../trainer/charts.js';

const W = 320;
const H = 292;
const PAD = { left: 38, right: 10, top: 10, bottom: 34 };

function extent(values, margin, square) {
    let min = Infinity, max = -Infinity;
    for (const value of values) {
        if (value < min) min = value;
        if (value > max) max = value;
    }
    if (!Number.isFinite(min)) return [-1, 1];
    const span = Math.max(max - min, 1e-6);
    return square ? [min, max] : [min - span * margin, max + span * margin];
}

/**
 * 2D scatter of a prepared dataset in its raw units (before scaling): class colour + shape, training points filled,
 * validation hollow, test faint. `xFeature` / `yFeature` choose the axes for datasets with more than 2 features.
 */
export default function ScatterPreview({ prepared, xFeature = 0, yFeature = 1, name }) {
    const titleId = useId();
    const descId = useId();
    const { raw, featureSize } = prepared;
    const n = raw.n;
    const regression = prepared.numClasses === null;
    const role = new Uint8Array(n);
    for (const index of prepared.val.indices) role[index] = 1;
    for (const index of prepared.test.indices) role[index] = 2;

    const xs = new Float64Array(n);
    const ys = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        xs[i] = raw.x[i * featureSize + xFeature];
        ys[i] = raw.x[i * featureSize + yFeature];
    }
    const square = featureSize === 2 && prepared.bounds;
    const [x0, x1] = square ? prepared.bounds[0] : extent(xs, 0.08, false);
    const [y0, y1] = square ? prepared.bounds[1] : extent(ys, 0.08, false);
    const sx = (value) => PAD.left + (value - x0) / (x1 - x0) * (W - PAD.left - PAD.right);
    const sy = (value) => H - PAD.bottom - (value - y0) / (y1 - y0) * (H - PAD.top - PAD.bottom);
    const size = n <= 200 ? 3.6 : n <= 500 ? 3 : 2.4;

    const order = [0, 2, 1];
    const points = [];
    for (const pass of order) {
        for (let i = 0; i < n; i++) {
            if (role[i] !== pass) continue;
            const x = sx(xs[i]);
            const y = sy(ys[i]);
            if (regression) {
                points.push(<circle key={i} cx={x} cy={y} r={size * 0.9} fill={pass === 1 ? '#fff' : divergingColor(raw.y[i])}
                    stroke={pass === 1 ? divergingColor(raw.y[i] < 0 ? -1 : 1) : '#00000055'} strokeWidth={pass === 1 ? 1.4 : 0.5} opacity={pass === 2 ? 0.5 : 1} />);
            } else {
                points.push(<ClassMarker key={i} x={x} y={y} index={raw.y[i]} size={size} filled={pass === 0} faint={pass === 2} />);
            }
        }
    }

    const names = prepared.featureNames ?? ['x₁', 'x₂'];
    const counts = regression ? null : prepared.classNames.map((label, index) => {
        let count = 0;
        for (let i = 0; i < n; i++) if (raw.y[i] === index) count += 1;
        return `${count} ${label}`;
    });
    const xTicks = niceTicks(x0, x1, 4);
    const yTicks = niceTicks(y0, y1, 4);
    const description = dtx('preview.scatterDesc', {
        counts: counts ? counts.join(', ') : dtx('classes.regression'),
        train: prepared.train.n,
        val: prepared.val.n,
        test: prepared.test.n ? dtx('preview.testPart', { test: prepared.test.n }) : '',
    });

    return (
        <svg className="edu-data-scatter" viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={`${titleId} ${descId}`}>
            <title id={titleId}>{dtx('preview.scatterTitle', { name, n })}</title>
            <desc id={descId}>{description}</desc>
            <rect x={PAD.left} y={PAD.top} width={W - PAD.left - PAD.right} height={H - PAD.top - PAD.bottom} fill="#fff" stroke="#0000002e" />
            {square && x0 < 0 && x1 > 0 && <line x1={sx(0)} x2={sx(0)} y1={PAD.top} y2={H - PAD.bottom} stroke="#00000014" />}
            {square && y0 < 0 && y1 > 0 && <line x1={PAD.left} x2={W - PAD.right} y1={sy(0)} y2={sy(0)} stroke="#00000014" />}
            <g>{points}</g>
            <g className="edu-data-axis" aria-hidden="true">
                {xTicks.map((value) => (
                    <g key={`x${value}`}>
                        <line x1={sx(value)} x2={sx(value)} y1={H - PAD.bottom} y2={H - PAD.bottom + 3} stroke="#0000004d" />
                        <text x={sx(value)} y={H - PAD.bottom + 13} textAnchor="middle">{formatTick(value)}</text>
                    </g>
                ))}
                {yTicks.map((value) => (
                    <g key={`y${value}`}>
                        <line x1={PAD.left - 3} x2={PAD.left} y1={sy(value)} y2={sy(value)} stroke="#0000004d" />
                        <text x={PAD.left - 5} y={sy(value) + 3} textAnchor="end">{formatTick(value)}</text>
                    </g>
                ))}
                <text className="edu-data-axis-title" x={(PAD.left + W - PAD.right) / 2} y={H - 4} textAnchor="middle">{names[xFeature]}</text>
                <text className="edu-data-axis-title" x={11} y={(PAD.top + H - PAD.bottom) / 2} textAnchor="middle" transform={`rotate(-90 11 ${(PAD.top + H - PAD.bottom) / 2})`}>{names[yFeature]}</text>
            </g>
        </svg>
    );
}
