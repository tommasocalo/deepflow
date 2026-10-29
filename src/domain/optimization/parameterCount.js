import { buildParameterScope, repeatBindingValue, resolveDimensionTuple, resolveLayerParameters, resolveParameterValue, resolveRepeatCount, withGlobalParameters } from '../expressions.js';
import { effectiveTemplateId } from '../architecture/instanceStructure.js';
import { defaultParameterValues, getEntry } from '../library/index.js';

const sizeOf = (shape) => Array.isArray(shape) && shape.length > 0
    && shape.every((axis) => Number.isSafeInteger(axis) && axis > 0)
    ? shape.reduce((product, axis) => product * axis, 1) : null;
const positive = (value) => Number.isSafeInteger(value) && value > 0 ? value : null;

/** Modules with no learned weights. Anything else unrecognized may hide weights. */
const WEIGHT_FREE_MODULE = /^torch\.nn\.(ReLU|LeakyReLU|GELU|SiLU|Tanh|Sigmoid|Softmax|LogSoftmax|Dropout|Dropout2d|Dropout3d|Flatten|Identity|MaxPool[123]d|AvgPool[123]d|AdaptiveAvgPool[123]d|AdaptiveMaxPool[123]d|Upsample)$/;

/** Spatial rank from the module name: Conv1d → 1, ConvTranspose2d → 2. */
export function spatialRank(libraryKey) {
    return Number(String(libraryKey).match(/([123])d$/)?.[1] ?? 1);
}

/**
 * Kernel elements of a convolution from its raw kernel_size. A scalar applies
 * to every spatial axis, as in PyTorch: Conv2d with kernel_size 3 is 3 × 3.
 * An empty value uses the library default; an unresolved reference stays null.
 */
export function kernelVolume(libraryKey, kernel, scope = {}) {
    const rank = spatialRank(libraryKey);
    const raw = kernel === null || kernel === undefined || kernel === ''
        ? getEntry(libraryKey)?.parameters.find((parameter) => parameter.name === 'kernel_size')?.default
        : kernel;
    let dims = resolveDimensionTuple(raw, scope);
    if (dims?.length === 1 && rank > 1) dims = Array(rank).fill(dims[0]);
    return dims?.length === rank ? sizeOf(dims) : null;
}

/** Library defaults merged under the node's own values, resolved in scope. */
export function resolvedLayerParameters(node, scope = {}) {
    const defaults = defaultParameterValues(getEntry(node.libraryKey));
    return resolveLayerParameters({ ...node, parameters: { ...defaults, ...node.parameters } }, scope).parameters;
}

/**
 * Learned weights of one layer node: { total, trainable }, or null when the
 * size is unknown (an unresolved setting or a module with hidden weights).
 */
export function layerSize(node, scope) {
    const key = node.libraryKey;
    const p = resolvedLayerParameters(node, scope);
    const n = (name) => positive(p[name]);
    if (key === 'deepflow.op.Parameter') {
        if (p.enabled === false) return { total: 0, trainable: 0 };
        const total = sizeOf(resolveDimensionTuple(p.shape, scope));
        return total === null ? null : { total, trainable: p.requires_grad === false ? 0 : total };
    }
    let total = 0;
    if (key === 'torch.nn.Linear') {
        if (!n('in_features') || !n('out_features')) return null;
        total = n('in_features') * n('out_features') + (p.bias === false ? 0 : n('out_features'));
    } else if (key === 'torch.nn.Embedding') {
        if (!n('num_embeddings') || !n('embedding_dim')) return null;
        total = n('num_embeddings') * n('embedding_dim');
    } else if (key === 'torch.nn.LayerNorm') {
        if (p.elementwise_affine !== false) {
            const size = sizeOf(resolveDimensionTuple(p.normalized_shape, scope));
            if (size === null) return null;
            total = size * (p.bias === false ? 1 : 2);
        }
    } else if (/^torch\.nn\.(BatchNorm|InstanceNorm)[123]d$/.test(key)) {
        if (p.affine) { if (!n('num_features')) return null; total = 2 * n('num_features'); }
    } else if (key === 'torch.nn.GroupNorm') {
        if (p.affine !== false) { if (!n('num_channels')) return null; total = 2 * n('num_channels'); }
    } else if (/^torch\.nn\.Conv[123]d$/.test(key)) {
        const kernel = kernelVolume(key, node.parameters?.kernel_size, scope);
        const groups = n('groups') ?? 1;
        // PyTorch requires both channel counts to be divisible by groups.
        if (!n('in_channels') || !n('out_channels') || !kernel || n('in_channels') % groups || n('out_channels') % groups) return null;
        total = n('out_channels') * (n('in_channels') / groups) * kernel + (p.bias === false ? 0 : n('out_channels'));
    } else if (/^torch\.nn\.ConvTranspose[123]d$/.test(key)) {
        // PyTorch stores ConvTranspose weights as (in, out / groups, *kernel).
        const kernel = kernelVolume(key, node.parameters?.kernel_size, scope);
        const groups = n('groups') ?? 1;
        if (!n('in_channels') || !n('out_channels') || !kernel || n('in_channels') % groups || n('out_channels') % groups) return null;
        total = n('in_channels') * (n('out_channels') / groups) * kernel + (p.bias === false ? 0 : n('out_channels'));
    } else if (key === 'deepflow.nn.PositionalEmbedding') {
        if (!n('max_len') || !n('embedding_dim')) return null;
        total = n('max_len') * n('embedding_dim');
    } else if (key?.startsWith('torch.nn.') || key?.startsWith('deepflow.nn.')) {
        // An unrecognized neural module may contain hidden weights.
        if (!WEIGHT_FREE_MODULE.test(key)) return null;
    }
    if (!Number.isSafeInteger(total) || total < 0) return null;
    return { total, trainable: total };
}

/** The exporter expands every repeated occurrence, including local variants. */
export function countArchitectureParameters(graph, templates = {}, outerScope = {}) {
    const unsupported = new Set();
    const budget = { remaining: 20000 };
    const autoCount = (body, node, scope) => {
        const counts = (body.edges ?? []).filter((edge) => edge.target === node.id).map((edge) => {
            const source = body.nodes[edge.source];
            if (!['deepflow.op.Split', 'deepflow.op.Broadcast'].includes(source?.libraryKey)) return null;
            const p = resolveLayerParameters(source, scope).parameters;
            return positive(p.chunks ?? p.count);
        }).filter(Boolean);
        return counts.length && counts.every((count) => count === counts[0]) ? counts[0] : null;
    };
    const visit = (body, parentScope, depth) => {
        const scope = withGlobalParameters(body, parentScope);
        let total = 0;
        let trainable = 0;
        for (const node of Object.values(body?.nodes ?? {})) {
            if (--budget.remaining < 0) { unsupported.add('Expanded graph exceeds 20,000 nodes'); break; }
            if (node.kind === 'layer') {
                const size = layerSize(node, scope);
                if (!size) { unsupported.add(node.libraryKey ?? node.label ?? 'Unknown layer'); continue; }
                total += size.total;
                trainable += size.trainable;
                continue;
            }
            if (!['supernode', 'repeat'].includes(node.kind)) continue;
            if (depth >= 8) { unsupported.add('Template depth exceeds 8'); continue; }
            const count = node.kind === 'repeat' ? node.count === 'auto' ? autoCount(body, node, scope)
                : resolveRepeatCount(node.count, scope).count : 1;
            if (!count) { unsupported.add(`${node.label ?? 'Repeat'}: unresolved count`); continue; }
            for (let index = 0; index < count; index++) {
                const template = templates[effectiveTemplateId(node, index)];
                if (!template?.graph) { unsupported.add(`${node.label ?? 'Component'}: missing template`); continue; }
                const bindings = node.kind === 'repeat' ? node.bindings : node.parameterValues;
                const childScope = buildParameterScope(template.freeParameters,
                    bindings,
                    node.kind === 'repeat' ? { ...scope, [node.iterationVariable || 'i']: index } : scope,
                    node.kind === 'repeat' ? index : undefined);
                // Metric estimation has no input tensors. An explicitly bound
                // derived width still gives the same value as the incoming edge
                // on valid models, and lets us count the editable attention body.
                for (const parameter of template.freeParameters ?? []) {
                    if (!parameter.derived || childScope[parameter.name] != null) continue;
                    const raw = repeatBindingValue(bindings?.[parameter.name], parameter, node.kind === 'repeat' ? index : undefined);
                    if (raw !== undefined) childScope[parameter.name] = resolveParameterValue(raw, scope);
                }
                const child = visit(template.graph, childScope, depth + 1);
                total += child.total;
                trainable += child.trainable;
            }
        }
        return { total, trainable };
    };
    const count = visit(graph, outerScope, 0);
    return { ...count, complete: unsupported.size === 0 && budget.remaining >= 0, unsupported: [...unsupported] };
}
