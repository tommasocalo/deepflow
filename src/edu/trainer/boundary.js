/**
 * Decision boundary on a 64 × 64 grid over the dataset bounds (docs/edu/SPEC.md §3.5, §6.10.7): the predicted class
 * and its probability per cell (for regression, the predicted value), computed in evaluation mode.
 */
import { sigmoidValue } from './ops.js';
import { forward } from './model.js';
import { Tensor } from './tensor.js';

export const BOUNDARY_SIZE = 64;
const CHUNK = 1024;

/** Cell centres, row 0 at the top (largest x₂). */
export function gridPoints(bounds, size = BOUNDARY_SIZE) {
    const [[x0, x1], [y0, y1]] = bounds;
    const data = new Float64Array(size * size * 2);
    for (let row = 0; row < size; row++) {
        const y = y1 - (row + 0.5) * (y1 - y0) / size;
        for (let col = 0; col < size; col++) {
            const i = row * size + col;
            data[i * 2] = x0 + (col + 0.5) * (x1 - x0) / size;
            data[i * 2 + 1] = y;
        }
    }
    return data;
}

/**
 * @returns {{ size, bounds, numClasses, classes: Uint8Array, confidence: Float32Array, values: Float32Array | null } | null}
 */
export function computeBoundary(model, dataset, lossType, size = BOUNDARY_SIZE) {
    if (dataset.featureSize !== 2 || !dataset.bounds) return null;
    const cells = size * size;
    const points = gridPoints(dataset.bounds, size);
    let width = 1;
    const outputs = [];
    for (let start = 0; start < cells; start += CHUNK) {
        const end = Math.min(cells, start + CHUNK);
        const out = forward(model, new Tensor([end - start, 2], points.slice(start * 2, end * 2)), { training: false });
        width = out.shape[out.shape.length - 1] ?? 1;
        outputs.push(out.data);
    }
    const data = new Float64Array(cells * width);
    outputs.reduce((offset, chunk) => { data.set(chunk, offset); return offset + chunk.length; }, 0);
    const out = { data };
    const classes = new Uint8Array(cells);
    const confidence = new Float32Array(cells);
    const regression = dataset.numClasses === null;
    const values = regression ? new Float32Array(cells) : null;
    for (let i = 0; i < cells; i++) {
        const base = i * width;
        if (regression) {
            values[i] = out.data[base];
            continue;
        }
        if (lossType === 'cross_entropy' && width > 1) {
            let best = 0, max = -Infinity;
            for (let c = 0; c < width; c++) if (out.data[base + c] > max) { max = out.data[base + c]; best = c; }
            let sum = 0;
            for (let c = 0; c < width; c++) sum += Math.exp(out.data[base + c] - max);
            classes[i] = best;
            confidence[i] = 1 / sum;
        } else {
            const p = lossType === 'bce' ? sigmoidValue(out.data[base]) : Math.min(1, Math.max(0, out.data[base]));
            classes[i] = p > 0.5 ? 1 : 0;
            confidence[i] = Math.max(p, 1 - p);
        }
    }
    return { size, bounds: dataset.bounds, numClasses: dataset.numClasses, classes, confidence, values };
}
