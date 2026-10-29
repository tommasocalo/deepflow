import { asFreeParamRef } from '../expressions.js';
import { getEntry } from '../library/index.js';
import { formatParameterValue } from '../parameterDisplay.js';
import { resolvedLayerParameters } from '../optimization/parameterCount.js';
import { fieldAlias } from '../parameterBindings.js';

/**
 * The defining hyperparameters of each row, torchinfo-style: one compact cell
 * such as `768 → 3072` or `×12 · mlp_width 3072`. Everything else (dropout of
 * a block, eps, flags) stays in the node's Settings.
 *
 * A layout is a list of parts: fields `{ field, prefix }` and separators `{ text }`.
 */
const field = (name, extra = {}) => ({ field: name, ...extra });
const text = (value) => ({ text: value });

const CONVOLUTION = [field('in_channels'), text('→'), field('out_channels'), text('·'), field('kernel_size', { prefix: 'k' }), field('stride', { prefix: 's' }), field('padding', { prefix: 'p' }), field('groups', { prefix: 'g', hideDefault: true })];
const POOLING = [field('kernel_size', { prefix: 'k' }), field('stride', { prefix: 's' })];

const LAYER_LAYOUTS = {
    'torch.nn.Linear': [field('in_features'), text('→'), field('out_features')],
    'torch.nn.Embedding': [field('num_embeddings'), text('×'), field('embedding_dim')],
    'deepflow.nn.PositionalEmbedding': [field('max_len'), text('×'), field('embedding_dim')],
    'torch.nn.LayerNorm': [field('normalized_shape')],
    'torch.nn.BatchNorm1d': [field('num_features')],
    'torch.nn.BatchNorm2d': [field('num_features')],
    'torch.nn.GroupNorm': [field('num_groups'), text('/'), field('num_channels')],
    'torch.nn.Dropout': [field('p', { prefix: 'p' })],
    'torch.nn.GELU': [field('approximate')],
    'torch.nn.LeakyReLU': [field('negative_slope', { prefix: 'slope' })],
    'torch.nn.Softmax': [field('dim', { prefix: 'dim' })],
    'torch.nn.Conv1d': CONVOLUTION,
    'torch.nn.Conv2d': CONVOLUTION,
    'torch.nn.ConvTranspose2d': CONVOLUTION,
    'torch.nn.MaxPool1d': POOLING,
    'torch.nn.MaxPool2d': POOLING,
    'torch.nn.AvgPool2d': POOLING,
    'torch.nn.AdaptiveAvgPool2d': [field('output_size')],
    'deepflow.nn.CausalSelfAttention': [field('num_heads', { prefix: 'H' })],
    'torch.nn.MultiheadAttention': [field('num_heads', { prefix: 'H' })],
    'torch.nn.TransformerEncoderLayer': [field('d_model'), text('·'), field('nhead', { prefix: 'H' })],
    'torch.nn.LSTM': [field('input_size'), text('→'), field('hidden_size')],
    'torch.nn.GRU': [field('input_size'), text('→'), field('hidden_size')],
    'deepflow.io.Input': [field('shape')],
    'deepflow.op.Parameter': [field('shape', { readOnly: 'Edit the dimensions in Settings.' })],
    'deepflow.op.Split': [field('chunks', { prefix: 'chunks' })],
    'deepflow.op.Broadcast': [field('count', { prefix: '×' })],
    'deepflow.op.Scale': [field('factor', { prefix: '×' })],
    'deepflow.op.Transpose': [field('dim0'), text('↔'), field('dim1')],
    'deepflow.op.Concat': [field('dim', { prefix: 'dim' })],
    'deepflow.op.Stack': [field('dim', { prefix: 'dim' })],
    'deepflow.op.Mean': [field('dim', { prefix: 'dim' })],
    'deepflow.op.AddScalar': [field('value', { prefix: '+' })],
    'deepflow.op.Pow': [field('exponent', { prefix: '^' })],
};

/** Standard component definitions, by family; their fields are instance bindings. */
const FAMILY_LAYOUTS = {
    linear: [field('in_features'), text('→'), field('out_features')],
    embedding: [field('num_embeddings'), text('×'), field('embedding_dim')],
    'learned-position': [field('max_len'), text('×'), field('embedding_dim')],
    'sinusoidal-position': [field('D')],
    layernorm: [field('normalized_shape')],
    attention: [field('H', { prefix: 'H' })],
    'attention-head': [field('dropout', { prefix: 'p' })],
};

/** Custom repeats and components show at most this many interesting integer bindings. */
const MAX_BINDINGS = 2;
const NUMBER_LITERAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export const COUNT_SPEC = Object.freeze({ name: 'count', type: 'int', required: true, description: 'Number of instances' });

/** Where a field's value is stored on its node. */
export function fieldSource(node, name) {
    if (node.kind === 'layer') return 'parameters';
    if (node.kind === 'repeat') return name === 'count' ? 'count' : 'bindings';
    return 'parameterValues';
}

/** The value as written on the node, before defaults. */
export function fieldRaw(node, source, name) {
    if (!node) return undefined;
    if (source === 'count') return node.count;
    return node[source]?.[name];
}

const unset = (raw) => raw === undefined || raw === null || raw === '';

/** A plain forward such as `d_model: $d_model` restates its parent's value. */
function isForward(name, raw) {
    return typeof raw === 'string' && asFreeParamRef(raw.trim()) === name;
}

/** Spec of a field from the library (layers) or the definition (containers). */
export function fieldSpec(row, name, templates) {
    if (row.kind === 'layer') return getEntry(row.libraryKey)?.parameters.find((parameter) => parameter.name === name) ?? null;
    if (name === 'count' && row.kind === 'repeat') return COUNT_SPEC;
    return templates[row.definitionIds[0]]?.freeParameters?.find((parameter) => parameter.name === name) ?? null;
}

/** Which parts a row's config cell shows. */
export function configLayout(row, node, templates) {
    if (row.kind === 'layer') return LAYER_LAYOUTS[row.libraryKey] ?? [];
    const familyParts = FAMILY_LAYOUTS[row.family];
    const parts = row.kind === 'repeat' ? [field('count', { prefix: '×' })] : [];
    if (familyParts) {
        if (parts.length) parts.push(text('·'));
        return [...parts, ...familyParts];
    }
    // Custom definitions: integer bindings that say something, such as 4 * d_model.
    const template = templates[row.definitionIds[0]];
    const bindings = node.kind === 'repeat' ? node.bindings : node.parameterValues;
    const interesting = (template?.freeParameters ?? []).filter((parameter) => parameter.type === 'int' && !parameter.derived && !parameter.hidden
        && !unset(bindings?.[parameter.name]) && !isForward(parameter.name, bindings?.[parameter.name]));
    for (const parameter of interesting.slice(0, MAX_BINDINGS)) {
        if (parts.length) parts.push(text('·'));
        parts.push(field(parameter.name, { prefix: parameter.name }));
    }
    if (interesting.length > MAX_BINDINGS) {
        parts.push({ note: `+${interesting.length - MAX_BINDINGS}`, title: interesting.slice(MAX_BINDINGS).map((parameter) => parameter.name).join(', ') });
    }
    return parts;
}

/** Display text of a written value. */
export function rawText(raw, spec) {
    if (unset(raw)) {
        const fallback = spec?.default;
        return { text: fallback === null || fallback === undefined ? '—' : formatPlain(fallback), tone: 'default' };
    }
    if (Array.isArray(raw)) return { text: `(${raw.map((part) => String(part)).join(', ')})`, tone: raw.every((part) => typeof part === 'number') ? 'literal' : 'expression' };
    if (typeof raw === 'string') {
        const trimmed = raw.trim();
        const literal = NUMBER_LITERAL.test(trimmed) || ['enum', 'string'].includes(spec?.type) && !trimmed.startsWith('$');
        return { text: trimmed, tone: literal ? 'literal' : 'expression' };
    }
    return { text: formatPlain(raw), tone: 'literal' };
}

function formatPlain(value) {
    if (typeof value === 'number') return formatParameterValue(value);
    if (Array.isArray(value)) return value.map(formatPlain).join(' × ');
    return String(value);
}

/** Display text of a resolved value. */
export function valueText(value, spec) {
    if (value === null || value === undefined) return spec?.nullable ? 'None' : '?';
    if (typeof value === 'string') return value.trim();
    return formatPlain(value);
}

/** Resolved value of a layer field in its level scope, library defaults included. */
export function resolvedLayerField(node, scope, name) {
    return resolvedLayerParameters(node, scope)[name];
}

/**
 * Per-occurrence values of a container binding: each instance's resolved
 * scope (a repeat has one per instance, so formulas in `i` vary).
 */
export function resolvedBinding(entry, name) {
    return entry.instances.map((instance) => instance.scope ? instance.scope[name] ?? null : null);
}

/**
 * One text for the values of every occurrence: a single value, or `3…128`
 * (first…last) when they differ, with a run-length list in the title.
 */
export function summarizeValues(values, spec, labels = []) {
    const texts = values.map((value) => valueText(value, spec));
    if (!texts.length) return { text: '?', varies: false, title: '' };
    const varies = texts.some((value) => value !== texts[0]);
    if (!varies) return { text: texts[0], varies: false, title: '' };
    const runs = [];
    texts.forEach((value, index) => {
        const label = labels[index] || `#${index + 1}`;
        const last = runs.at(-1);
        if (last && last.value === value) last.to = label;
        else runs.push({ value, from: label, to: label });
    });
    const title = runs.slice(0, 12).map((run) => `${run.from === run.to ? run.from : `${run.from}–${run.to.replace(/^i=/, '')}`}: ${run.value}`).join(' · ')
        + (runs.length > 12 ? ' · …' : '');
    return { text: `${texts[0]}…${texts.at(-1)}`, varies: true, title };
}

/** An eye alias: the field shows the alias default, which is where edits go. */
export function aliasFor(templates, ownerId, nodeId, name) {
    return ownerId && templates[ownerId] ? fieldAlias(templates[ownerId], nodeId, name) : null;
}

/** Whether a flag resolves to false for every occurrence (Linear without bias). */
export function resolvesFalse(values) {
    return values.length > 0 && values.every((value) => value === false);
}
