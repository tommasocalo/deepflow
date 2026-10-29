/**
 * Dense float64 tensors and a gradient tape (docs/edu/SPEC.md §6.10.3).
 *
 * A tensor is { shape: number[], data: Float64Array, grad: Float64Array | null, requiresGrad: boolean }, row-major.
 * Ops record { output, backward } on a Tape when any input requires a gradient; Tape.backward walks the records in
 * reverse, and each backward closure accumulates into its inputs' `.grad`. Passing `tape = null` runs without
 * recording (evaluation mode).
 */

export const numel = (shape) => shape.reduce((product, dim) => product * dim, 1);

export class Tensor {
    constructor(shape, data = null, requiresGrad = false) {
        this.shape = shape;
        this.data = data ?? new Float64Array(numel(shape));
        if (this.data.length !== numel(shape)) throw new Error(`Tensor data has ${this.data.length} values for shape (${shape.join(', ')})`);
        this.grad = null;
        this.requiresGrad = requiresGrad;
    }

    get size() {
        return this.data.length;
    }
}

export const tensor = (shape, values, requiresGrad = false) => new Tensor([...shape], Float64Array.from(values), requiresGrad);
export const zeros = (shape) => new Tensor([...shape]);

/** Add g into t.grad (copying g on first use, so callers may pass shared arrays). */
export function accumulate(t, g) {
    if (!t.requiresGrad) return;
    if (!t.grad) {
        t.grad = Float64Array.from(g);
        return;
    }
    const grad = t.grad;
    for (let i = 0; i < grad.length; i++) grad[i] += g[i];
}

/** Like accumulate, but takes ownership of a freshly allocated g. */
export function adopt(t, g) {
    if (!t.requiresGrad) return;
    if (!t.grad) t.grad = g;
    else for (let i = 0; i < g.length; i++) t.grad[i] += g[i];
}

export class Tape {
    constructor() {
        this.records = [];
    }

    /** Record an op when any input needs a gradient; marks the output as requiring one too. */
    record(output, inputs, backward) {
        if (inputs.some((input) => input?.requiresGrad)) {
            output.requiresGrad = true;
            this.records.push({ output, backward });
        }
        return output;
    }

    /** Backpropagate from `output`, seeded with `seed` (a Float64Array of the output's size, default ones). */
    backward(output, seed = null) {
        if (!output.requiresGrad) return;
        output.grad = seed ? Float64Array.from(seed) : new Float64Array(output.size).fill(1);
        for (let i = this.records.length - 1; i >= 0; i--) {
            const { output: value, backward } = this.records[i];
            if (value.grad) backward(value.grad);
        }
    }
}

/** NumPy broadcasting of several shapes; throws on incompatible dimensions. */
export function broadcastShapes(shapes) {
    const rank = Math.max(...shapes.map((shape) => shape.length));
    const out = [];
    for (let offset = 1; offset <= rank; offset++) {
        let dim = 1;
        for (const shape of shapes) {
            const size = offset <= shape.length ? shape[shape.length - offset] : 1;
            if (size === 1) continue;
            if (dim !== 1 && dim !== size) throw new Error(`Cannot broadcast shapes ${shapes.map((s) => `(${s.join(', ')})`).join(' and ')}`);
            dim = size;
        }
        out.unshift(dim);
    }
    return out;
}

const broadcastCache = new Map();

/**
 * For each flat index of `outShape`, the flat index of the broadcast input of shape `shape`. Null when the shapes
 * are equal, so callers can skip the indirection. Maps are cached by shape pair (batches repeat the same shapes).
 */
export function broadcastIndex(shape, outShape) {
    if (shape.length === outShape.length && shape.every((dim, i) => dim === outShape[i])) return null;
    const key = `${shape.join(',')}>${outShape.join(',')}`;
    const cached = broadcastCache.get(key);
    if (cached) return cached;
    const map = buildBroadcastIndex(shape, outShape);
    if (broadcastCache.size > 256) broadcastCache.clear();
    broadcastCache.set(key, map);
    return map;
}

function buildBroadcastIndex(shape, outShape) {
    const rank = outShape.length;
    const strides = new Array(rank).fill(0);
    let stride = 1;
    for (let i = shape.length - 1, j = rank - 1; i >= 0; i--, j--) {
        strides[j] = shape[i] === 1 ? 0 : stride;
        stride *= shape[i];
    }
    const total = numel(outShape);
    const map = new Int32Array(total);
    const counter = new Array(rank).fill(0);
    let index = 0;
    for (let flat = 0; flat < total; flat++) {
        map[flat] = index;
        for (let axis = rank - 1; axis >= 0; axis--) {
            counter[axis] += 1;
            index += strides[axis];
            if (counter[axis] < outShape[axis]) break;
            index -= strides[axis] * counter[axis];
            counter[axis] = 0;
        }
    }
    return map;
}
