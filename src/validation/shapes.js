import { isShapeCollection, shapeCollection, flattenShapeInputs } from '../domain/shapeCollections.js';
import { resolveRepeatCount, resolveDimensionTuple } from '../domain/expressions.js';

/**
 * Static shape inference. A shape is an array of dimensions, where each
 * dim is either a positive integer or `null` (unknown). The leading batch
 * dim is conventionally null.
 *
 * Rules are written per library key. Layers without a rule are treated as
 * shape-preserving with no constraints (we propagate the input shape and
 * raise no errors). This is the safe default for early development.
 *
 * @typedef {Array<number | null>} Shape
 *
 * @typedef {Object} ShapeResult
 * @property {Shape | null} output   Inferred output shape (null if unknown)
 * @property {string[]} errors       Human-readable shape errors
 */

const numOrNull = (v) => {
    if (v === null || v === undefined || v === '') return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
};

/**
 * Parse an Input node's `shape` parameter — accepted forms:
 *   "(3, 224, 224)"     → [null, 3, 224, 224]   (batch prepended)
 *   "(None, 3, 224, 224)" → [null, 3, 224, 224]
 *   "[3, 224, 224]"     → [null, 3, 224, 224]
 *   "B, 3, 224, 224"    → [null, 3, 224, 224]
 */
function parseInputShape(raw) {
    if (!raw || typeof raw !== 'string') return null;
    const cleaned = raw.replace(/[()[\]]/g, '').trim();
    if (!cleaned) return null;
    const parts = cleaned
        .split(',')
        .map((p) => p.trim())
        .filter((p) => p.length > 0); // drop trailing empty from "(128,)" etc.
    /** @type {Shape} */
    const shape = parts.map((p) => {
        if (/^(none|null|b|n|\?)$/i.test(p)) return null;
        const n = Number(p);
        return Number.isFinite(n) ? n : null;
    });
    if (shape.length === 0) return null;
    if (shape[0] !== null) return [null, ...shape];
    return shape;
}

const passthrough = ([s]) => ({ output: s ?? null, errors: [] });

const RULES = {
    'deepflow.io.Input': (_inputs, params) => {
        const shape = parseInputShape(params.shape);
        return { output: shape, errors: [] };
    },
    'deepflow.io.Output': passthrough,
    'deepflow.template.Input': passthrough,
    'deepflow.template.Output': passthrough,

    'torch.nn.Conv2d': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        if (s.length !== 4) {
            return { output: null, errors: [`Conv2d expects 4D (B, C, H, W); got rank ${s.length}`] };
        }
        const inC = numOrNull(p.in_channels);
        const outC = numOrNull(p.out_channels);
        if (inC !== null && s[1] !== null && s[1] !== inC) {
            return { output: null, errors: [`Conv2d in_channels=${inC} but input has ${s[1]} channels`] };
        }
        const k = parseSpatial(p.kernel_size, 3);
        const st = parseSpatial(p.stride, 1);
        const pad = parseSpatial(p.padding, 0);
        const dil = parseSpatial(p.dilation, 1);
        const h = computeConvDim(s[2], k[0], st[0], pad[0], dil[0]);
        const w = computeConvDim(s[3], k[1], st[1], pad[1], dil[1]);
        return { output: [s[0], outC ?? null, h, w], errors: [] };
    },

    'torch.nn.ConvTranspose2d': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        if (s.length !== 4) {
            return { output: null, errors: [`ConvTranspose2d expects 4D; got rank ${s.length}`] };
        }
        const inC = numOrNull(p.in_channels);
        const outC = numOrNull(p.out_channels);
        if (inC !== null && s[1] !== null && s[1] !== inC) {
            return { output: null, errors: [`ConvTranspose2d in_channels=${inC} but input has ${s[1]} channels`] };
        }
        const k = parseSpatial(p.kernel_size, 3);
        const st = parseSpatial(p.stride, 1);
        const pad = parseSpatial(p.padding, 0);
        const op = parseSpatial(p.output_padding, 0);
        const h = computeConvTransposeDim(s[2], k[0], st[0], pad[0], op[0]);
        const w = computeConvTransposeDim(s[3], k[1], st[1], pad[1], op[1]);
        return { output: [s[0], outC ?? null, h, w], errors: [] };
    },

    'torch.nn.MaxPool2d':         poolRule,
    'torch.nn.AvgPool2d':         poolRule,
    'torch.nn.AdaptiveAvgPool2d': adaptivePoolRule,

    'torch.nn.Linear': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const inF = numOrNull(p.in_features);
        const outF = numOrNull(p.out_features);
        const last = s[s.length - 1];
        if (inF !== null && last !== null && last !== inF) {
            return { output: null, errors: [`Linear in_features=${inF} but input has ${last} features`] };
        }
        return { output: [...s.slice(0, -1), outF ?? null], errors: [] };
    },

    'torch.nn.Flatten': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const start = normalizeAxis(numOrNull(p.start_dim) ?? 1, s.length);
        const e = normalizeAxis(numOrNull(p.end_dim) ?? -1, s.length);
        if (start === null || e === null || start > e) return { output: null, errors: ['Flatten dimensions must be ordered and within the input rank'] };
        let prod = 1;
        let unknown = false;
        for (let i = start; i <= e; i++) {
            if (s[i] === null) unknown = true;
            else prod *= s[i];
        }
        const flattened = unknown ? null : prod;
        return { output: [...s.slice(0, start), flattened, ...s.slice(e + 1)], errors: [] };
    },

    'torch.nn.Embedding': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const dim = numOrNull(p.embedding_dim);
        return { output: [...s, dim ?? null], errors: [] };
    },

    'deepflow.nn.PositionalEmbedding': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const dim = numOrNull(p.embedding_dim);
        const maxLen = numOrNull(p.max_len);
        const t = s[s.length - 1];
        const errors = [];
        if (maxLen !== null && t !== null && t > maxLen) {
            errors.push(`Positional Embedding max_len=${maxLen} but sequence length is ${t}`);
        }
        return { output: [...s, dim ?? null], errors };
    },

    'deepflow.nn.CausalSelfAttention': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const errors = [];
        if (s.length !== 3) errors.push(`Causal Self-Attention expects (B, T, D); got rank ${s.length}`);
        const d = numOrNull(p.embed_dim);
        const heads = numOrNull(p.num_heads);
        const last = s[s.length - 1];
        if (d !== null && last !== null && last !== d) {
            errors.push(`Causal Self-Attention embed_dim=${d} but input has ${last} features`);
        }
        if (d !== null && heads !== null && heads > 0 && d % heads !== 0) {
            errors.push(`embed_dim=${d} is not divisible by num_heads=${heads}`);
        }
        return { output: s, errors };
    },
    'torch.nn.MultiheadAttention': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const d = numOrNull(p.embed_dim);
        const last = s[s.length - 1];
        if (d !== null && last !== null && last !== d) {
            return { output: s, errors: [`MultiheadAttention embed_dim=${d} but input has ${last} features`] };
        }
        return { output: s, errors: [] };
    },

    'torch.nn.BatchNorm2d': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        if (s.length !== 4) {
            return { output: s, errors: [`BatchNorm2d expects 4D; got rank ${s.length}`] };
        }
        const f = numOrNull(p.num_features);
        if (f !== null && s[1] !== null && s[1] !== f) {
            return { output: s, errors: [`BatchNorm2d num_features=${f} but input has ${s[1]} channels`] };
        }
        return { output: s, errors: [] };
    },
    'torch.nn.BatchNorm1d': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const f = numOrNull(p.num_features);
        const channelDim = s.length === 2 ? 1 : 1;
        if (f !== null && s[channelDim] !== null && s[channelDim] !== f) {
            return { output: s, errors: [`BatchNorm1d num_features=${f} but input has ${s[channelDim]} channels`] };
        }
        return { output: s, errors: [] };
    },
    'torch.nn.LayerNorm': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const n = numOrNull(typeof p.normalized_shape === 'string' ? p.normalized_shape.replace(/[()[\],]/g, '') : p.normalized_shape);
        const last = s[s.length - 1];
        if (n !== null && last !== null && last !== n) {
            return { output: s, errors: [`LayerNorm normalized_shape=${n} but input has ${last} features`] };
        }
        return { output: s, errors: [] };
    },
    'torch.nn.GroupNorm': ([s], p) => {
        if (!s) return { output: null, errors: [] };
        const c = numOrNull(p.num_channels);
        if (c !== null && s[1] !== null && s[1] !== c) {
            return { output: s, errors: [`GroupNorm num_channels=${c} but input has ${s[1]} channels`] };
        }
        return { output: s, errors: [] };
    },

    'torch.nn.Dropout':    passthrough,
    'torch.nn.ReLU':       passthrough,
    'torch.nn.LeakyReLU':  passthrough,
    'torch.nn.GELU':       passthrough,
    'torch.nn.Sigmoid':    passthrough,
    'torch.nn.Tanh':       passthrough,
    'torch.nn.Softmax':    passthrough,

    'deepflow.op.Parameter': parameterRule,
    'deepflow.op.Arange': arangeRule,
    'deepflow.op.PositionIndices': ([shape]) => ({ output: shape ? shape.slice(0, 2) : null, errors: shape && shape.length !== 2 ? ['Position indices expects tokens with shape (B, T)'] : [] }),
    'deepflow.op.Gather': gatherRule,
    'deepflow.op.Mean': meanRule,
    'deepflow.op.Subtract': elementwiseRule,
    'deepflow.op.AddScalar': passthrough,
    'deepflow.op.Pow': passthrough,
    'deepflow.op.Rsqrt': passthrough,
    'deepflow.op.Exp': passthrough,
    'deepflow.op.Sin': passthrough,
    'deepflow.op.Cos': passthrough,
    'deepflow.op.Unsqueeze': unsqueezeRule,
    'deepflow.op.Slice': sliceRule,
    'deepflow.op.Add':      elementwiseRule,
    'deepflow.op.Multiply': elementwiseRule,
    'deepflow.op.Concat':   concatRule,
    'deepflow.op.Split':    splitRule,
    'deepflow.op.Broadcast': broadcastRule,
    'deepflow.op.Stack': stackRule,
    'deepflow.op.Transpose': transposeRule,
    'deepflow.op.MatMul': matmulRule,
    'deepflow.op.Scale': passthrough,
    'deepflow.op.CausalMask': ([shape]) => ({
        output: shape ?? null,
        errors: shape && shape.length < 2 ? ['Causal Mask expects a tensor of rank at least 2'] : [],
    }),
};

function parseSpatial(value, fallback) {
    if (value === null || value === undefined || value === '') return [fallback, fallback];
    if (typeof value === 'number') return [value, value];
    if (typeof value === 'string') {
        const cleaned = value.replace(/[()[\]]/g, '').trim();
        if (cleaned.includes(',')) {
            const parts = cleaned.split(',').map((p) => Number(p.trim()));
            if (parts.length === 2 && parts.every(Number.isFinite)) return parts;
        }
        const n = Number(cleaned);
        if (Number.isFinite(n)) return [n, n];
    }
    return [fallback, fallback];
}

function computeConvDim(inDim, k, s, p, d) {
    if (inDim === null) return null;
    return Math.floor((inDim + 2 * p - d * (k - 1) - 1) / s + 1);
}

function computeConvTransposeDim(inDim, k, s, p, op) {
    if (inDim === null) return null;
    return (inDim - 1) * s - 2 * p + k + op;
}

function poolRule([s], p) {
    if (!s) return { output: null, errors: [] };
    if (s.length !== 4) {
        return { output: s, errors: [`Pool2d expects 4D; got rank ${s.length}`] };
    }
    const k = parseSpatial(p.kernel_size, 2);
    const st = parseSpatial(p.stride ?? p.kernel_size, k[0]);
    const pad = parseSpatial(p.padding, 0);
    const h = computeConvDim(s[2], k[0], st[0], pad[0], 1);
    const w = computeConvDim(s[3], k[1], st[1], pad[1], 1);
    return { output: [s[0], s[1], h, w], errors: [] };
}

function adaptivePoolRule([s], p) {
    if (!s) return { output: null, errors: [] };
    if (s.length !== 4) {
        return { output: s, errors: [`AdaptiveAvgPool2d expects 4D; got rank ${s.length}`] };
    }
    const target = parseSpatial(p.output_size, 1);
    return { output: [s[0], s[1], target[0], target[1]], errors: [] };
}

function elementwiseRule(inputs) {
    const hasCollection = inputs.some(isShapeCollection);
    inputs = flattenShapeInputs(inputs);
    if (inputs.length < (hasCollection ? 1 : 2)) {
        return { output: inputs[0] ?? null, errors: ['Element-wise op needs at least 2 inputs'] };
    }
    const present = inputs.filter(Boolean);
    // Some inputs have unknown shapes: propagate what we know, no error.
    if (present.length < 2) return { output: present[0] ?? null, errors: [] };
    const rank = Math.max(...present.map((shape) => shape.length));
    const output = [], errors = [];
    for (let offset = 1; offset <= rank; offset++) {
        const dimensions = present.map((shape) => offset <= shape.length ? shape.at(-offset) : 1);
        const known = [...new Set(dimensions.filter((dim) => dim !== null && dim !== 1))];
        if (known.length > 1) errors.push(`Element-wise dim ${rank - offset} mismatch: ${known.join(' vs ')}`);
        output.unshift(known[0] ?? (dimensions.includes(null) ? null : 1));
    }
    return { output, errors };
}

function concatRule(inputs, p) {
    const hasCollection = inputs.some(isShapeCollection);
    inputs = flattenShapeInputs(inputs);
    if (inputs.length < (hasCollection ? 1 : 2)) {
        return { output: inputs[0] ?? null, errors: ['Concat needs at least 2 inputs'] };
    }
    const present = inputs.filter(Boolean);
    if (!present.length) return { output: null, errors: [] };
    const axis = numOrNull(p.dim) ?? 1;
    const ref = present[0];
    if (!present.every((s) => s.length === ref.length)) {
        return { output: ref, errors: ['Concat inputs must have equal rank'] };
    }
    const concatDim = normalizeAxis(axis, ref.length);
    if (concatDim === null) return { output: null, errors: [`Concat dim ${axis} is outside rank ${ref.length}`] };
    const errors = [];
    /** @type {Shape} */
    const out = ref.map((d, i) => {
        if (i === concatDim) {
            let sum = 0;
            for (const s of present) {
                if (s[i] === null) return null;
                sum += s[i];
            }
            return sum;
        }
        const dims = present.map((s) => s[i]).filter((d) => d !== null);
        const unique = new Set(dims);
        if (unique.size > 1) errors.push(`Concat dim ${i} mismatch: ${[...unique].join(' vs ')}`);
        return ref[i];
    });
    return { output: out, errors };
}

/**
 * @param {string} libraryKey
 * @param {Shape[]} inputShapes
 * @param {Object<string, *>} parameters
 * @returns {ShapeResult}
 */
export function inferShape(libraryKey, inputShapes, parameters = {}) {
    const collectionAware = new Set(['deepflow.io.Output', 'deepflow.template.Input', 'deepflow.template.Output', 'deepflow.op.Concat', 'deepflow.op.Add', 'deepflow.op.Multiply', 'deepflow.op.Stack']);
    if (inputShapes.some(isShapeCollection) && !collectionAware.has(libraryKey)) {
        return { output: null, errors: ['This block expects a tensor; reduce the collection with Concat, Stack, or Add first'] };
    }
    const rule = RULES[libraryKey];
    if (!rule) {
        return { output: inputShapes[0] ?? null, errors: [] };
    }
    return rule(inputShapes, parameters);
}

/** @param {Shape | null} shape */
export function formatShape(shape) {
    if (!shape) return '?';
    if (isShapeCollection(shape)) {
        const item = shape.items[0];
        const same = shape.items.every((candidate) => JSON.stringify(candidate) === JSON.stringify(item));
        return same ? `${shape.items.length} × ${formatShape(item)}` : `${shape.items.length} tensors`;
    }
    return '(' + shape.map((d) => (d === null ? '?' : String(d))).join(', ') + ')';
}

function normalizeAxis(raw, rank, insert = false) {
    const limit = rank + (insert ? 1 : 0);
    if (!Number.isInteger(raw)) return null;
    const axis = raw < 0 ? limit + raw : raw;
    return axis >= 0 && axis < limit ? axis : null;
}

function splitRule([shape], parameters) {
    const { count, error } = resolveRepeatCount(parameters.chunks === undefined ? 2 : parameters.chunks);
    if (error) return { output: null, errors: [error.replace('Repeat count', 'Split chunks')] };
    if (!shape) return { output: shapeCollection(Array(count).fill(null)), errors: [] };
    const axis = normalizeAxis(parameters.dim === undefined ? -1 : Number(parameters.dim), shape.length);
    if (axis === null) return { output: null, errors: ['Split dim is outside the input rank'] };
    if (shape[axis] !== null && shape[axis] % count !== 0) {
        return { output: null, errors: [`Split dimension ${shape[axis]} must be divisible by ${count} chunks`] };
    }
    const part = shape.map((size, index) => index === axis && size !== null ? size / count : size);
    return { output: shapeCollection(Array.from({ length: count }, () => [...part])), errors: [] };
}

function broadcastRule([shape], parameters) {
    const { count, error } = resolveRepeatCount(parameters.count === undefined ? 2 : parameters.count);
    if (error) return { output: null, errors: [error.replace('Repeat count', 'Broadcast count')] };
    return { output: shapeCollection(Array.from({ length: count }, () => shape ? [...shape] : null)), errors: [] };
}

function stackRule(inputs, parameters) {
    const flattened = flattenShapeInputs(inputs);
    const present = flattened.filter(Boolean);
    if (!present.length) return { output: null, errors: flattened.length ? [] : ['Stack needs at least one input'] };
    const ref = present[0];
    const errors = present.every((shape) => shape.length === ref.length && shape.every((dim, index) => dim === null || ref[index] === null || dim === ref[index])) ? [] : ['Stack inputs must have the same shape'];
    const axis = normalizeAxis(parameters.dim === undefined ? 0 : Number(parameters.dim), ref.length, true);
    if (axis === null) return { output: null, errors: ['Stack dim is outside the output rank'] };
    return { output: [...ref.slice(0, axis), flattened.length, ...ref.slice(axis)], errors };
}

function transposeRule([shape], parameters) {
    if (!shape) return { output: null, errors: [] };
    const dim0 = normalizeAxis(parameters.dim0 === undefined ? -2 : Number(parameters.dim0), shape.length);
    const dim1 = normalizeAxis(parameters.dim1 === undefined ? -1 : Number(parameters.dim1), shape.length);
    if (dim0 === null || dim1 === null) return { output: null, errors: ['Transpose dimensions must be within the input rank'] };
    const output = [...shape];
    [output[dim0], output[dim1]] = [output[dim1], output[dim0]];
    return { output, errors: [] };
}

function matmulRule(inputs) {
    if (inputs.length !== 2) return { output: null, errors: ['MatMul needs two inputs (a and b)'] };
    const [a, b] = inputs;
    if (!a || !b) return { output: null, errors: [] };
    if (!a.length || !b.length) return { output: null, errors: ['MatMul expects tensors of rank at least 1'] };
    const left = a.length === 1 ? [1, ...a] : a;
    const right = b.length === 1 ? [...b, 1] : b;
    const errors = [];
    const innerA = left.at(-1), innerB = right.at(-2);
    if (innerA !== null && innerB !== null && innerA !== innerB) errors.push(`MatMul inner dimensions mismatch: ${innerA} vs ${innerB}`);
    const batchA = left.slice(0, -2), batchB = right.slice(0, -2);
    const batch = [];
    for (let offset = 1; offset <= Math.max(batchA.length, batchB.length); offset++) {
        const x = offset <= batchA.length ? batchA.at(-offset) : 1;
        const y = offset <= batchB.length ? batchB.at(-offset) : 1;
        if (x !== null && y !== null && x !== y && x !== 1 && y !== 1) errors.push(`MatMul batch dimensions mismatch: ${x} vs ${y}`);
        batch.unshift(x === y ? x : x === 1 ? y : y === 1 ? x : x === null || y === null ? null : x);
    }
    const output = [...batch];
    if (a.length > 1) output.push(left.at(-2));
    if (b.length > 1) output.push(right.at(-1));
    return { output, errors };
}

function parameterRule(_inputs, parameters) {
    const shape = resolveDimensionTuple(parameters.shape);
    const errors = [];
    if (!shape || shape.some((dimension) => !Number.isSafeInteger(dimension) || dimension < 1)) {
        return { output: null, errors: ['Parameter shape must contain positive integer dimensions'] };
    }
    if (parameters.init === 'kaiming_uniform' && shape.length < 2) errors.push('Kaiming initialization needs a parameter of rank at least 2');
    const scale = parameters.scale ?? 1;
    if (typeof scale !== 'number' || !Number.isFinite(scale) || scale < 0) errors.push('Parameter scale must be a nonnegative finite number');
    const index = parameters.padding_idx;
    if (index !== null && index !== undefined && (!Number.isInteger(index) || !shape.length || index < -shape[0] || index >= shape[0])) errors.push('padding_idx is outside the parameter first dimension');
    return { output: shape, errors };
}

function arangeRule(_inputs, parameters) {
    const { start = 0, end = 8, step = 1 } = parameters;
    if (![start, end, step].every((value) => typeof value === 'number' && Number.isFinite(value)) || step === 0 || (end - start) * step < 0) {
        return { output: null, errors: ['Arange needs finite bounds and a nonzero step toward its end'] };
    }
    const length = Math.ceil((end - start) / step);
    return Number.isSafeInteger(length) ? { output: [length], errors: [] } : { output: null, errors: ['Arange length must be a finite integer'] };
}

function gatherRule([indices, weight], parameters) {
    const errors = [];
    if (weight && weight.length !== 2) errors.push('Gather weight must be a two-dimensional table');
    const index = parameters.padding_idx;
    if (weight && index !== undefined && index !== null && (!Number.isInteger(index) || (weight[0] !== null && (index < -weight[0] || index >= weight[0])))) errors.push('Gather padding_idx is outside the weight table');
    return { output: indices && weight?.length === 2 ? [...indices, weight[1]] : null, errors };
}

function meanRule([shape], parameters) {
    const rawDims = resolveDimensionTuple(parameters.dim ?? -1);
    if (!rawDims?.length || rawDims.some((dim) => !Number.isInteger(dim))) return { output: null, errors: ['Mean dim must be an integer or a nonempty tuple of integers'] };
    if (!shape) return { output: null, errors: [] };
    const dimensions = rawDims.map((dim) => normalizeAxis(dim, shape.length));
    if (dimensions.includes(null) || new Set(dimensions).size !== dimensions.length) return { output: null, errors: ['Mean dimensions must be distinct and within the input rank'] };
    const output = parameters.keepdim === false ? shape.filter((_, index) => !dimensions.includes(index)) : shape.map((dim, index) => dimensions.includes(index) ? 1 : dim);
    return { output, errors: [] };
}

function unsqueezeRule([shape], parameters) {
    if (!shape) return { output: null, errors: [] };
    const axis = normalizeAxis(parameters.dim ?? -1, shape.length, true);
    if (axis === null) return { output: null, errors: ['Unsqueeze dim is outside the output rank'] };
    return { output: [...shape.slice(0, axis), 1, ...shape.slice(axis)], errors: [] };
}

function sliceRule([shape], parameters) {
    const { start = 0, end = null, step = 1, dim = -1 } = parameters;
    if (!Number.isInteger(start) || (end !== null && !Number.isInteger(end)) || !Number.isInteger(step) || step < 1) return { output: null, errors: ['Slice needs integer bounds and a positive integer step'] };
    if (!shape) return { output: null, errors: [] };
    const axis = normalizeAxis(dim, shape.length);
    if (axis === null) return { output: null, errors: ['Slice dim is outside the input rank'] };
    const size = shape[axis];
    const bound = (value) => Math.max(0, Math.min(size, value < 0 ? size + value : value));
    const length = size === null ? null : Math.max(0, Math.ceil(((end === null ? size : bound(end)) - bound(start)) / step));
    return { output: shape.map((value, index) => index === axis ? length : value), errors: [] };
}
