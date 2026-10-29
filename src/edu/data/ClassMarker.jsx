import { classColor, classShape } from './classStyle.js';

/**
 * One data point with its class colour and shape (classStyle.js). `filled` points train the model, hollow ones are
 * validation, and `faint` marks test points. Sizes are in SVG user units around (x, y).
 */
export function ClassMarker({ x, y, index, size = 4, filled = true, faint = false, strokeWidth = 1.4 }) {
    const color = classColor(index);
    const common = {
        fill: filled ? color : '#fff',
        stroke: filled ? '#ffffff' : color,
        strokeWidth: filled ? 0.6 : strokeWidth,
        opacity: faint ? 0.5 : 1,
    };
    const shape = classShape(index);
    if (shape === 'triangle') {
        const r = size * 1.18;
        return <path d={`M${x},${y - r}L${x + r * 0.9},${y + r * 0.6}L${x - r * 0.9},${y + r * 0.6}Z`} {...common} />;
    }
    if (shape === 'square') {
        const half = size * 0.82;
        return <rect x={x - half} y={y - half} width={half * 2} height={half * 2} {...common} />;
    }
    if (shape === 'diamond') {
        const r = size * 1.1;
        return <path d={`M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`} {...common} />;
    }
    return <circle cx={x} cy={y} r={size * 0.9} {...common} />;
}

/** A small legend swatch: the class marker in a 14 × 14 box. */
export function ClassSwatch({ index, filled = true }) {
    return (
        <svg className="edu-class-swatch" width="14" height="14" viewBox="0 0 14 14" aria-hidden="true" focusable="false">
            <ClassMarker x={7} y={7.6} index={index} size={4.2} filled={filled} />
        </svg>
    );
}
