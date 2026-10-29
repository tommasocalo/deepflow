/**
 * Class encodings shared by the Data preview and the trainer's boundary view (docs/edu/SPEC.md §3.4, §3.9): every
 * class has an Okabe-Ito colour AND a shape, so colour is never the only cue. Class 0 circle #0072B2, class 1
 * triangle #E69F00, class 2 square #009E73.
 */
export const CLASS_COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000', '#7f7f7f', '#8c564b'];
export const CLASS_SHAPES = ['circle', 'triangle', 'square', 'diamond', 'circle', 'triangle', 'square', 'diamond', 'circle', 'triangle'];

export const classColor = (index) => CLASS_COLORS[index % CLASS_COLORS.length];
export const classShape = (index) => CLASS_SHAPES[index % CLASS_SHAPES.length];

/** '#0072B2' → [0, 114, 178]. */
export function hexToRgb(hex) {
    const value = Number.parseInt(hex.slice(1), 16);
    return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/** Diverging blue → white → orange for regression values in [−1, 1], as [r, g, b]. */
export function divergingRgb(value) {
    const t = Math.max(-1, Math.min(1, Number(value) || 0));
    const [r0, g0, b0] = t < 0 ? [0, 114, 178] : [230, 159, 0];
    const k = Math.abs(t);
    const mix = (c) => Math.round(255 + (c - 255) * k);
    return [mix(r0), mix(g0), mix(b0)];
}

export function divergingColor(value) {
    const [r, g, b] = divergingRgb(value);
    return `rgb(${r}, ${g}, ${b})`;
}

/**
 * The boundary colour of one grid cell: the class colour mixed into white by confidence, from a light tint at chance
 * level to a clear colour when the model is sure (points stay visible on top).
 */
export function boundaryRgb(classIndex, confidence, numClasses) {
    const [r, g, b] = hexToRgb(classColor(classIndex));
    const chance = 1 / Math.max(2, numClasses);
    const sure = Math.max(0, Math.min(1, (confidence - chance) / (1 - chance)));
    const alpha = 0.1 + 0.5 * sure;
    return [Math.round(255 + (r - 255) * alpha), Math.round(255 + (g - 255) * alpha), Math.round(255 + (b - 255) * alpha)];
}
