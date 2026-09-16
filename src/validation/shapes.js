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
        const start = numOrNull(p.start_dim) ?? 1;
        const end = numOrNull(p.end_dim) ?? -1;
        const e = end < 0 ? s.length + end : end;
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

    'deepflow.op.Add':      elementwiseRule,
    'deepflow.op.Multiply': elementwiseRule,
    'deepflow.op.Concat':   concatRule,
    'deepflow.op.Split':    passthrough,
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
    if (inputs.length < 2) {
        return { output: inputs[0] ?? null, errors: ['Element-wise op needs at least 2 inputs'] };
    }
    const present = inputs.filter(Boolean);
    // Some inputs have unknown shapes: propagate what we know, no error.
    if (present.length < 2) return { output: present[0] ?? null, errors: [] };
    const ranks = new Set(present.map((s) => s.length));
    if (ranks.size > 1) {
        return { output: present[0], errors: [`Element-wise inputs differ in rank: ${[...ranks].join(' vs ')}`] };
    }
    const ref = present[0];
    /** @type {string[]} */
    const errors = [];
    for (let i = 0; i < ref.length; i++) {
        const dims = present.map((s) => s[i]).filter((d) => d !== null);
        const unique = new Set(dims);
        if (unique.size > 1) {
            errors.push(`Element-wise dim ${i} mismatch: ${[...unique].join(' vs ')}`);
        }
    }
    return { output: ref, errors };
}

function concatRule(inputs, p) {
    if (inputs.length < 2) {
        return { output: inputs[0] ?? null, errors: ['Concat needs at least 2 inputs'] };
    }
    const present = inputs.filter(Boolean);
    if (present.length < 2) return { output: null, errors: [] };
    const axis = numOrNull(p.dim) ?? 1;
    const ref = present[0];
    if (!present.every((s) => s.length === ref.length)) {
        return { output: ref, errors: ['Concat inputs must have equal rank'] };
    }
    const concatDim = axis < 0 ? ref.length + axis : axis;
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
export function inferShape(libraryKey, inputShapes, parameters) {
    const rule = RULES[libraryKey];
    if (!rule) {
        return { output: inputShapes[0] ?? null, errors: [] };
    }
    return rule(inputShapes, parameters);
}

/** @param {Shape | null} shape */
export function formatShape(shape) {
    if (!shape) return '?';
    return '(' + shape.map((d) => (d === null ? '?' : String(d))).join(', ') + ')';
}
