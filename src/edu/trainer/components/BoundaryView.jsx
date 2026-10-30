import { useEffect, useRef } from 'react';
import { ClassMarker } from '../../data/ClassMarker.jsx';
import { boundaryRgb, divergingRgb } from '../../data/classStyle.js';
import { getDataset } from '../../data/registry.js';
import { usePrepared } from '../../data/usePrepared.js';
import { tr } from '../text.js';

/** "blue in 52% and orange in 48%" from the predicted class of every grid cell. */
function areaSummary(boundary, classNames) {
    const counts = new Array(boundary.numClasses ?? 2).fill(0);
    for (const value of boundary.classes) counts[value] = (counts[value] ?? 0) + 1;
    const total = boundary.classes.length || 1;
    return tr('boundary.summary', {
        parts: counts.map((count, index) => tr('boundary.part', { name: classNames?.[index] ?? index, share: `${Math.round(count / total * 100)}%` })).join(', '),
    });
}

/**
 * The decision boundary of a run (docs/edu/SPEC.md §3.5): a 64 × 64 grid over the data bounds, coloured by the
 * predicted class with confidence as colour strength, and the run's data points on top (training filled, validation
 * hollow). Regression shows the predicted value. Drawn on a canvas, scaled smoothly; the points are SVG.
 */
export default function BoundaryView({ record, size = 220, thumbnail = false, target = true }) {
    const canvasRef = useRef(null);
    const boundary = record?.boundary ?? null;
    const spec = boundary ? getDataset(record.datasetId) : null;
    const { prepared } = usePrepared(spec, record?.dataOptions, record?.split, record?.dataSeed);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || !boundary) return;
        let context = null;
        try { context = canvas.getContext('2d'); } catch { context = null; }
        if (!context) return;
        const image = context.createImageData(boundary.size, boundary.size);
        const cells = boundary.size * boundary.size;
        for (let i = 0; i < cells; i++) {
            const [r, g, b] = boundary.values ? divergingRgb(boundary.values[i]) : boundaryRgb(boundary.classes[i], boundary.confidence[i], boundary.numClasses ?? 2);
            image.data[i * 4] = r;
            image.data[i * 4 + 1] = g;
            image.data[i * 4 + 2] = b;
            image.data[i * 4 + 3] = 255;
        }
        context.putImageData(image, 0, 0);
    }, [boundary]);

    if (!boundary) return null;
    const [[x0, x1], [y0, y1]] = boundary.bounds;
    const px = (value) => (value - x0) / (x1 - x0) * size;
    const py = (value) => (y1 - value) / (y1 - y0) * size;
    const markerSize = thumbnail ? 1.9 : prepared && prepared.raw.n > 400 ? 2.4 : 3.1;
    const points = [];
    if (prepared && prepared.featureSize === 2) {
        const { raw } = prepared;
        const draw = (split, filled) => {
            for (const index of split.indices) {
                const x = px(raw.x[index * 2]);
                const y = py(raw.x[index * 2 + 1]);
                if (boundary.values) {
                    points.push(<circle key={`${filled ? 't' : 'v'}${index}`} cx={x} cy={y} r={markerSize * 0.8} fill={filled ? '#111' : '#fff'} stroke="#111" strokeWidth={filled ? 0 : 1} />);
                } else {
                    points.push(<ClassMarker key={`${filled ? 't' : 'v'}${index}`} x={x} y={y} index={raw.y[index]} size={markerSize} filled={filled} strokeWidth={thumbnail ? 1 : 1.4} />);
                }
            }
        };
        draw(prepared.train, true);
        draw(prepared.val, false);
    }
    const summary = boundary.values ? tr('boundary.regression') : areaSummary(boundary, record.classNames);
    return (
        <figure className={`edu-tr-boundary${thumbnail ? ' is-thumbnail' : ''}`} data-edu-target={target ? 'trainer-boundary' : undefined}>
            <div className="edu-tr-boundary-plot" style={{ width: size, height: size }}>
                <canvas ref={canvasRef} width={boundary.size} height={boundary.size} aria-hidden="true" />
                <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden="true" focusable="false">{points}</svg>
            </div>
            <figcaption className={thumbnail ? 'edu-sr-only' : 'edu-tr-caption'}>
                <span className="edu-sr-only">{tr('boundary.title')}. </span>{summary}
            </figcaption>
        </figure>
    );
}
