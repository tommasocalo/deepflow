import { buildParameterScope, repeatBindingValue, resolveDimensionTuple, resolveLayerParameters, resolveParameterValue, resolveRepeatCount, withGlobalParameters } from '../expressions.js';
import { effectiveTemplateId } from '../architecture/instanceStructure.js';
import { defaultParameterValues, getEntry } from '../library/index.js';

const sizeOf = (shape) => Array.isArray(shape) && shape.length > 0
    && shape.every((axis) => Number.isSafeInteger(axis) && axis > 0)
    ? shape.reduce((product, axis) => product * axis, 1) : null;
const positive = (value) => Number.isSafeInteger(value) && value > 0 ? value : null;

/**
 * Kernel axes of a ConvNd. A scalar kernel_size covers every spatial axis, as in PyTorch (Conv2d(k=3) is 3 × 3);
 * tuples are used as given. Counting a scalar as one axis undercounted Conv2d(1→8, k=3) as 32 instead of 80.
 */
function convKernel(value, axes, scope) {
    const resolved = resolveParameterValue(value, scope);
    const scalar = typeof resolved === 'number' || (typeof resolved === 'string' && !/[,([]/.test(resolved));
    const dims = resolveDimensionTuple(value, scope);
    return scalar && dims?.length === 1 ? Array.from({ length: axes }, () => dims[0]) : dims;
}

/** Count only weights whose sizes are known. Unknown modules invalidate the estimate. */
function layerSize(node, scope) {
    const key = node.libraryKey;
    const defaults = defaultParameterValues(getEntry(key));
    const p = resolveLayerParameters({ ...node, parameters: { ...defaults, ...node.parameters } }, scope).parameters;
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
        const kernel = sizeOf(convKernel(p.kernel_size, Number(key.at(-2)), scope));
        const groups = n('groups') ?? 1;
        if (!n('in_channels') || !n('out_channels') || !kernel || n('in_channels') % groups) return null;
        total = n('out_channels') * (n('in_channels') / groups) * kernel + (p.bias === false ? 0 : n('out_channels'));
    } else if (key === 'deepflow.nn.PositionalEmbedding') {
        if (!n('max_len') || !n('embedding_dim')) return null;
        total = n('max_len') * n('embedding_dim');
    } else if (key?.startsWith('torch.nn.') || key?.startsWith('deepflow.nn.')) {
        // An unrecognized neural module may contain hidden weights.
        if (!/^torch\.nn\.(ReLU|GELU|SiLU|Tanh|Sigmoid|Softmax|LogSoftmax|Dropout|Dropout2d|Dropout3d|Flatten|Identity|MaxPool[123]d|AvgPool[123]d|AdaptiveAvgPool[123]d|AdaptiveMaxPool[123]d|Upsample)$/.test(key)) return null;
    }
    if (!Number.isSafeInteger(total) || total < 0) return null;
    return { total, trainable: total };
}

/**
 * The walker shared by the whole-model and per-node counts. Every expanded node uses one budget of 20,000, and
 * every reason that makes a count incomplete is added to the `issues` set passed in.
 */
function createParameterWalker(templates) {
    const budget = { remaining: 20000 };
    const zero = () => ({ total: 0, trainable: 0 });
    const autoCount = (body, node, scope) => {
        const counts = (body.edges ?? []).filter((edge) => edge.target === node.id).map((edge) => {
            const source = body.nodes[edge.source];
            if (!['deepflow.op.Split', 'deepflow.op.Broadcast'].includes(source?.libraryKey)) return null;
            const p = resolveLayerParameters(source, scope).parameters;
            return positive(p.chunks ?? p.count);
        }).filter(Boolean);
        return counts.length && counts.every((count) => count === counts[0]) ? counts[0] : null;
    };
    /** One node of `body`, with templates and repeats expanded. */
    const countNode = (body, node, scope, depth, issues) => {
        if (node.kind === 'layer') {
            const size = layerSize(node, scope);
            if (!size) { issues.add(node.libraryKey ?? node.label ?? 'Unknown layer'); return zero(); }
            return size;
        }
        if (!['supernode', 'repeat'].includes(node.kind)) return zero();
        if (depth >= 8) { issues.add('Template depth exceeds 8'); return zero(); }
        const count = node.kind === 'repeat' ? node.count === 'auto' ? autoCount(body, node, scope)
            : resolveRepeatCount(node.count, scope).count : 1;
        if (!count) { issues.add(`${node.label ?? 'Repeat'}: unresolved count`); return zero(); }
        let total = 0;
        let trainable = 0;
        for (let index = 0; index < count; index++) {
            const template = templates[effectiveTemplateId(node, index)];
            if (!template?.graph) { issues.add(`${node.label ?? 'Component'}: missing template`); continue; }
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
            const child = visit(template.graph, childScope, depth + 1, issues);
            total += child.total;
            trainable += child.trainable;
        }
        return { total, trainable };
    };
    /** Every node of `body`. */
    const visit = (body, parentScope, depth, issues) => {
        const scope = withGlobalParameters(body, parentScope);
        let total = 0;
        let trainable = 0;
        for (const node of Object.values(body?.nodes ?? {})) {
            if (--budget.remaining < 0) { issues.add('Expanded graph exceeds 20,000 nodes'); break; }
            const size = countNode(body, node, scope, depth, issues);
            total += size.total;
            trainable += size.trainable;
        }
        return { total, trainable };
    };
    return { budget, countNode, visit };
}

/** The exporter expands every repeated occurrence, including local variants. */
export function countArchitectureParameters(graph, templates = {}, outerScope = {}) {
    const walker = createParameterWalker(templates);
    const unsupported = new Set();
    const count = walker.visit(graph, outerScope, 0, unsupported);
    return { ...count, complete: unsupported.size === 0 && walker.budget.remaining >= 0, unsupported: [...unsupported] };
}

/**
 * Parameters per root node (docs/edu/SPEC.md §6.11.4): `{ [rootNodeId]: { total, trainable, complete } }`. A
 * component or repeat counts everything it expands to, with the rules of countArchitectureParameters, so the totals
 * add up to the whole-model count. `complete` is false for a node whose size is not fully known.
 */
export function countParametersByNode(graph, templates = {}, outerScope = {}) {
    const walker = createParameterWalker(templates);
    const scope = withGlobalParameters(graph, outerScope);
    const result = {};
    for (const node of Object.values(graph?.nodes ?? {})) {
        const issues = new Set();
        if (--walker.budget.remaining < 0) issues.add('Expanded graph exceeds 20,000 nodes');
        const size = issues.size ? { total: 0, trainable: 0 } : walker.countNode(graph, node, scope, 0, issues);
        result[node.id] = { total: size.total, trainable: size.trainable, complete: issues.size === 0 && walker.budget.remaining >= 0 };
    }
    return result;
}
