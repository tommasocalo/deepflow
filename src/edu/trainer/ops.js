/**
 * Forward ops with hand-written gradients (docs/edu/SPEC.md §6.10.2–6.10.3). Every op takes its inputs, its
 * attributes and a Tape (or null for evaluation without recording). Gradients are checked by finite differences in
 * scripts/edu-trainer-autograd.test.mjs.
 *
 * A Linear layer and the editable Linear component (Transpose + MatMul + Add, fused by model.js into one a·bᵀ
 * product) run the same kernels in the same order, so they produce bit-identical outputs and gradients. Kernels never
 * skip zeros, so NaN and Infinity propagate as in PyTorch.
 */
import { accumulate, adopt, broadcastIndex, broadcastShapes, numel, Tensor } from './tensor.js';

const record = (tape, output, inputs, backward) => (tape ? tape.record(output, inputs, backward) : output);
const needs = (t) => t?.requiresGrad;

// ---- matrix kernels -----------------------------------------------------------------------------------------

/** out (n×m) += a (n×k) · b (k×m). Four rows of b per pass; partial sums are added to out in p order. */
export function matmulNN(a, b, n, k, m, out = new Float64Array(n * m)) {
    for (let i = 0; i < n; i++) {
        const ai = i * k;
        const oi = i * m;
        let p = 0;
        for (; p + 3 < k; p += 4) {
            const a0 = a[ai + p], a1 = a[ai + p + 1], a2 = a[ai + p + 2], a3 = a[ai + p + 3];
            const b0 = p * m, b1 = b0 + m, b2 = b1 + m, b3 = b2 + m;
            for (let j = 0; j < m; j++) out[oi + j] += (a0 * b[b0 + j] + a1 * b[b1 + j]) + (a2 * b[b2 + j] + a3 * b[b3 + j]);
        }
        for (; p < k; p++) {
            const av = a[ai + p];
            const bp = p * m;
            for (let j = 0; j < m; j++) out[oi + j] += av * b[bp + j];
        }
    }
    return out;
}

/** out (n×m) = a (n×k) · bᵀ, with b stored as (m×k): one transposed copy of b, then the NN kernel. */
export function matmulNT(a, b, n, k, m, out = new Float64Array(n * m)) {
    return matmulNN(a, transposeData(b, m, k), n, k, m, out);
}

/** out (n×m) += aᵀ · b, with a stored as (k×n) and b as (k×m). Four rows of a and b per pass. */
export function matmulTN(a, b, k, n, m, out = new Float64Array(n * m)) {
    let p = 0;
    for (; p + 3 < k; p += 4) {
        const a0 = p * n, a1 = a0 + n, a2 = a1 + n, a3 = a2 + n;
        const b0 = p * m, b1 = b0 + m, b2 = b1 + m, b3 = b2 + m;
        for (let i = 0; i < n; i++) {
            const x0 = a[a0 + i], x1 = a[a1 + i], x2 = a[a2 + i], x3 = a[a3 + i];
            const oi = i * m;
            for (let j = 0; j < m; j++) out[oi + j] += (x0 * b[b0 + j] + x1 * b[b1 + j]) + (x2 * b[b2 + j] + x3 * b[b3 + j]);
        }
    }
    for (; p < k; p++) {
        const ap = p * n;
        const bp = p * m;
        for (let i = 0; i < n; i++) {
            const av = a[ap + i];
            const oi = i * m;
            for (let j = 0; j < m; j++) out[oi + j] += av * b[bp + j];
        }
    }
    return out;
}

function transposeData(data, rows, cols) {
    const out = new Float64Array(rows * cols);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[j * rows + i] = data[i * cols + j];
    return out;
}

const rank2 = (t, name) => {
    if (t.shape.length !== 2) throw new Error(`${name} expects a rank-2 tensor, got (${t.shape.join(', ')})`);
};

// ---- linear algebra -----------------------------------------------------------------------------------------

/** torch.nn.Linear: x (B×in) · Wᵀ + b, with W (out×in) and b (out) or null. */
export function linear(x, weight, bias, tape) {
    rank2(x, 'Linear');
    const [n, k] = x.shape;
    const [m, kw] = weight.shape;
    if (k !== kw) throw new Error(`Linear expects ${kw} input features, got ${k}`);
    const out = new Tensor([n, m], matmulNT(x.data, weight.data, n, k, m));
    if (bias) {
        const o = out.data, b = bias.data;
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) o[i * m + j] = o[i * m + j] + b[j];
    }
    return record(tape, out, [x, weight, bias], (g) => {
        if (needs(x)) adopt(x, matmulNN(g, weight.data, n, m, k));
        if (needs(weight)) adopt(weight, matmulTN(g, x.data, n, m, k));
        if (bias && needs(bias)) {
            const gb = new Float64Array(m);
            for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) gb[j] += g[i * m + j];
            adopt(bias, gb);
        }
    });
}

/** deepflow.op.MatMul on rank-2 tensors: a (n×k) · b (k×m). */
export function matmul(a, b, tape) {
    rank2(a, 'MatMul');
    rank2(b, 'MatMul');
    const [n, k] = a.shape;
    const [kb, m] = b.shape;
    if (k !== kb) throw new Error(`MatMul inner dimensions mismatch: ${k} vs ${kb}`);
    const out = new Tensor([n, m], matmulNN(a.data, b.data, n, k, m));
    return record(tape, out, [a, b], (g) => {
        if (needs(a)) adopt(a, matmulNT(g, b.data, n, m, k));
        if (needs(b)) adopt(b, matmulTN(a.data, g, n, k, m));
    });
}

/** a (n×k) · bᵀ where b is stored (m×k): the fused form of MatMul(a, Transpose(b)). */
export function matmulTransposed(a, b, tape) {
    rank2(a, 'MatMul');
    rank2(b, 'MatMul');
    const [n, k] = a.shape;
    const [m, kb] = b.shape;
    if (k !== kb) throw new Error(`MatMul inner dimensions mismatch: ${k} vs ${kb}`);
    const out = new Tensor([n, m], matmulNT(a.data, b.data, n, k, m));
    return record(tape, out, [a, b], (g) => {
        if (needs(a)) adopt(a, matmulNN(g, b.data, n, m, k));
        if (needs(b)) adopt(b, matmulTN(g, a.data, n, m, k));
    });
}

/** Swap the two axes of a rank-2 tensor. */
export function transpose(x, tape) {
    rank2(x, 'Transpose');
    const [rows, cols] = x.shape;
    const out = new Tensor([cols, rows], transposeData(x.data, rows, cols));
    return record(tape, out, [x], (g) => adopt(x, transposeData(g, cols, rows)));
}

// ---- element-wise with broadcasting -------------------------------------------------------------------------

function broadcastPlan(inputs) {
    const shape = broadcastShapes(inputs.map((t) => t.shape));
    return { shape, size: numel(shape), maps: inputs.map((t) => broadcastIndex(t.shape, shape)) };
}

/** Reduce a gradient over the broadcast axes back to the input's shape. */
function reduceTo(g, map, size) {
    if (!map) return Float64Array.from(g);
    const out = new Float64Array(size);
    for (let i = 0; i < g.length; i++) out[map[i]] += g[i];
    return out;
}

/** deepflow.op.Add: n inputs, NumPy broadcasting, summed in input order. */
export function add(inputs, tape) {
    const { shape, size, maps } = broadcastPlan(inputs);
    const out = new Float64Array(size);
    inputs.forEach((t, k) => {
        const data = t.data, map = maps[k];
        for (let i = 0; i < size; i++) out[i] += data[map ? map[i] : i];
    });
    return record(tape, new Tensor(shape, out), inputs, (g) => {
        inputs.forEach((t, k) => { if (needs(t)) adopt(t, reduceTo(g, maps[k], t.size)); });
    });
}

/** deepflow.op.Multiply: n inputs, NumPy broadcasting, multiplied in input order. */
export function multiply(inputs, tape) {
    const { shape, size, maps } = broadcastPlan(inputs);
    const value = (k, i) => inputs[k].data[maps[k] ? maps[k][i] : i];
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) {
        let product = value(0, i);
        for (let k = 1; k < inputs.length; k++) product *= value(k, i);
        out[i] = product;
    }
    return record(tape, new Tensor(shape, out), inputs, (g) => {
        inputs.forEach((t, k) => {
            if (!needs(t)) return;
            const local = new Float64Array(size);
            for (let i = 0; i < size; i++) {
                let others = g[i];
                for (let j = 0; j < inputs.length; j++) if (j !== k) others *= value(j, i);
                local[i] = others;
            }
            adopt(t, reduceTo(local, maps[k], t.size));
        });
    });
}

/** deepflow.op.Subtract: a − b with broadcasting. */
export function subtract(a, b, tape) {
    const { shape, size, maps } = broadcastPlan([a, b]);
    const out = new Float64Array(size);
    for (let i = 0; i < size; i++) out[i] = a.data[maps[0] ? maps[0][i] : i] - b.data[maps[1] ? maps[1][i] : i];
    return record(tape, new Tensor(shape, out), [a, b], (g) => {
        if (needs(a)) adopt(a, reduceTo(g, maps[0], a.size));
        if (needs(b)) {
            const negative = reduceTo(g, maps[1], b.size);
            for (let i = 0; i < negative.length; i++) negative[i] = -negative[i];
            adopt(b, negative);
        }
    });
}

/** An element-wise unary op: f gives the value, df(x, y) the local derivative. */
function unary(x, tape, f, df) {
    const n = x.size;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = f(x.data[i]);
    return record(tape, new Tensor([...x.shape], out), [x], (g) => {
        const gx = new Float64Array(n);
        for (let i = 0; i < n; i++) gx[i] = g[i] * df(x.data[i], out[i]);
        adopt(x, gx);
    });
}

export const scale = (x, factor, tape) => unary(x, tape, (v) => v * factor, () => factor);
export const addScalar = (x, value, tape) => unary(x, tape, (v) => v + value, () => 1);
export const pow = (x, exponent, tape) => unary(x, tape, (v) => Math.pow(v, exponent), (v) => exponent * Math.pow(v, exponent - 1));
export const exp = (x, tape) => unary(x, tape, Math.exp, (_v, y) => y);
export const rsqrt = (x, tape) => unary(x, tape, (v) => 1 / Math.sqrt(v), (_v, y) => -0.5 * y * y * y);
export const relu = (x, tape) => unary(x, tape, (v) => (v > 0 ? v : 0), (v) => (v > 0 ? 1 : 0));
export const leakyRelu = (x, slope, tape) => unary(x, tape, (v) => (v > 0 ? v : slope * v), (v) => (v > 0 ? 1 : slope));
export const tanh = (x, tape) => unary(x, tape, Math.tanh, (_v, y) => 1 - y * y);

export function sigmoidValue(v) {
    if (v >= 0) return 1 / (1 + Math.exp(-v));
    const e = Math.exp(v);
    return e / (1 + e);
}
export const sigmoid = (x, tape) => unary(x, tape, sigmoidValue, (_v, y) => y * (1 - y));

// Abramowitz & Stegun 7.1.26 (absolute error ≤ 1.5e-7). The gradient is the exact derivative of this approximation.
const P = 0.3275911;
const A = [0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429];
export function erf(z) {
    const s = Math.abs(z);
    const t = 1 / (1 + P * s);
    const poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    const value = 1 - poly * Math.exp(-s * s);
    return z >= 0 ? value : -value;
}
function erfDerivative(z) {
    const s = Math.abs(z);
    const t = 1 / (1 + P * s);
    const poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    const dPoly = A[0] + t * (2 * A[1] + t * (3 * A[2] + t * (4 * A[3] + t * 5 * A[4])));
    return Math.exp(-s * s) * (dPoly * P * t * t + 2 * s * poly);
}
const SQRT1_2 = Math.SQRT1_2;
const TANH_C = Math.sqrt(2 / Math.PI);

/** GELU: 'none' = x·Φ(x) with the erf above; 'tanh' = PyTorch's tanh approximation. */
export function gelu(x, approximate, tape) {
    if (approximate === 'tanh') {
        return unary(x, tape,
            (v) => 0.5 * v * (1 + Math.tanh(TANH_C * (v + 0.044715 * v * v * v))),
            (v) => {
                const th = Math.tanh(TANH_C * (v + 0.044715 * v * v * v));
                return 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * TANH_C * (1 + 3 * 0.044715 * v * v);
            });
    }
    return unary(x, tape,
        (v) => 0.5 * v * (1 + erf(v * SQRT1_2)),
        (v) => 0.5 * (1 + erf(v * SQRT1_2)) + 0.5 * v * erfDerivative(v * SQRT1_2) * SQRT1_2);
}

// ---- reductions over the last axis --------------------------------------------------------------------------

const lastAxis = (x) => {
    const width = x.shape[x.shape.length - 1];
    return { width, rows: x.size / width };
};

/** deepflow.op.Mean over the last axis. */
export function meanLast(x, keepdim, tape) {
    const { width, rows } = lastAxis(x);
    const out = new Float64Array(rows);
    for (let r = 0; r < rows; r++) {
        let sum = 0;
        for (let d = 0; d < width; d++) sum += x.data[r * width + d];
        out[r] = sum / width;
    }
    const shape = keepdim ? [...x.shape.slice(0, -1), 1] : x.shape.slice(0, -1);
    return record(tape, new Tensor(shape, out), [x], (g) => {
        const gx = new Float64Array(x.size);
        for (let r = 0; r < rows; r++) for (let d = 0; d < width; d++) gx[r * width + d] = g[r] / width;
        adopt(x, gx);
    });
}

/** Softmax over the last axis. */
export function softmax(x, tape) {
    const { width, rows } = lastAxis(x);
    const out = new Float64Array(x.size);
    for (let r = 0; r < rows; r++) {
        const base = r * width;
        let max = -Infinity;
        for (let d = 0; d < width; d++) max = Math.max(max, x.data[base + d]);
        let sum = 0;
        for (let d = 0; d < width; d++) { out[base + d] = Math.exp(x.data[base + d] - max); sum += out[base + d]; }
        for (let d = 0; d < width; d++) out[base + d] /= sum;
    }
    return record(tape, new Tensor([...x.shape], out), [x], (g) => {
        const gx = new Float64Array(x.size);
        for (let r = 0; r < rows; r++) {
            const base = r * width;
            let dot = 0;
            for (let d = 0; d < width; d++) dot += g[base + d] * out[base + d];
            for (let d = 0; d < width; d++) gx[base + d] = out[base + d] * (g[base + d] - dot);
        }
        adopt(x, gx);
    });
}

/** LogSoftmax over the last axis. */
export function logSoftmax(x, tape) {
    const { width, rows } = lastAxis(x);
    const out = new Float64Array(x.size);
    for (let r = 0; r < rows; r++) {
        const base = r * width;
        let max = -Infinity;
        for (let d = 0; d < width; d++) max = Math.max(max, x.data[base + d]);
        let sum = 0;
        for (let d = 0; d < width; d++) sum += Math.exp(x.data[base + d] - max);
        const lse = max + Math.log(sum);
        for (let d = 0; d < width; d++) out[base + d] = x.data[base + d] - lse;
    }
    return record(tape, new Tensor([...x.shape], out), [x], (g) => {
        const gx = new Float64Array(x.size);
        for (let r = 0; r < rows; r++) {
            const base = r * width;
            let total = 0;
            for (let d = 0; d < width; d++) total += g[base + d];
            for (let d = 0; d < width; d++) gx[base + d] = g[base + d] - Math.exp(out[base + d]) * total;
        }
        adopt(x, gx);
    });
}

// ---- shape, regularisation and gradient routing -------------------------------------------------------------

/** torch.nn.Flatten(1, −1): (B, …) → (B, product); shares the data. */
export function flatten(x, tape) {
    const batch = x.shape[0];
    const out = new Tensor([batch, x.size / batch], x.data);
    return record(tape, out, [x], (g) => accumulate(x, g));
}

/** Inverted dropout in training mode (keep with probability 1 − p, scale by 1/(1 − p)); identity otherwise. */
export function dropout(x, p, next, training, tape) {
    if (!training || !(p > 0)) return x;
    const n = x.size;
    const mask = new Float64Array(n);
    if (p < 1) {
        const keep = 1 / (1 - p);
        for (let i = 0; i < n; i++) mask[i] = next() >= p ? keep : 0;
    }
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = x.data[i] * mask[i];
    return record(tape, new Tensor([...x.shape], out), [x], (g) => {
        const gx = new Float64Array(n);
        for (let i = 0; i < n; i++) gx[i] = g[i] * mask[i];
        adopt(x, gx);
    });
}

/** Edge gradient operators: 'detach' / 'stop' cut the gradient; 'reverse' multiplies it by −λ. */
export function stopGradient(x) {
    return new Tensor(x.shape, x.data, false);
}

export function reverseGradient(x, lambda, tape) {
    const out = new Tensor(x.shape, x.data);
    return record(tape, out, [x], (g) => {
        const gx = new Float64Array(g.length);
        for (let i = 0; i < g.length; i++) gx[i] = -lambda * g[i];
        adopt(x, gx);
    });
}
