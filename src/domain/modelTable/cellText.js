import { formatExact, formatMetric, formatMetricRange, formatShare } from '../metrics/format.js';
import { runsText } from './configFields.js';

/**
 * The words behind each table cell. Hovers and the status line read the same
 * text, so what a tooltip explains is also reachable from the keyboard.
 */

export const METRIC_UNITS = { params: 'parameters', macs: 'MACs', activation: 'bytes' };
export const MACS_TITLE = 'Multiply-accumulates per sample (B = 1): MatMul, Linear and Conv, and the products inside attention and recurrent modules. Causal attention counts the full T × T product.';
export const ACTIVATION_TITLE = 'Activation memory per sample (B = 1, fp32): the sum of forward op outputs, eager, nothing freed. Standard Linear, Embedding, LayerNorm and position modules count one output; attention internals count individually; views and weights count nothing.';

/** One instance's value: exact, a lower bound, or unknown. */
export const instanceValueText = (value, format = formatExact) => value === null || value === undefined ? '?'
    : typeof value === 'object' ? `≥ ${format(value.atLeast)}` : format(value);

/** Labels of runs of instances: `i=1–11, i=14`. */
export const runLabels = (runs) => runs.map((run) => run.from === run.to ? run.from : `${run.from}–${run.to.replace(/^i=/, '')}`).join(', ');

/** A figure is a lower bound when something inside is unknown or an ancestor was not analysed completely. */
export const isPartial = (summary, row) => summary.unresolved > 0 || !!row.lowerBound;

/** What a partial figure leaves out. */
export function unresolvedLines(summary) {
    if (!summary.unresolved) return [];
    if (summary.reason) return [`Not counted: ${summary.reason}`];
    const items = summary.unresolvedItems.slice(0, 4).map((item) => item.reason ? `${item.label} (${item.reason})` : item.label);
    return [`Not counted: ${items.join('; ')}${summary.unresolvedItems.length > 4 ? '; …' : ''}`];
}

/**
 * The small figure under a metric: one instance, or the range of instances.
 * Only when every instance is known exactly, so it never contradicts the total.
 */
export function eachText(metric, summary) {
    const { each, count, knownCount } = summary;
    if (!each || count < 2 || knownCount < count || !(each.max > 0)) return null;
    return each.min === each.max ? `${formatMetric(metric, each.max, { compact: true })} ea` : formatMetricRange(metric, each.min, each.max);
}

/** A metric cell explained, line by line. */
export function metricLines(metric, summary, row) {
    const unit = METRIC_UNITS[metric];
    const partial = isPartial(summary, row);
    const lines = [];
    const { count, knownCount, each } = summary;
    if (!summary.total && summary.unresolved) lines.push('Unknown');
    else if (count > 1 && knownCount === count && each) {
        const analysed = partial ? ' analysed' : '';
        if (each.min === each.max) lines.push(`${formatExact(each.max)} ${unit} per instance × ${count}${analysed} = ${formatExact(summary.total)}`);
        else {
            lines.push(`${formatExact(each.min)} to ${formatExact(each.max)} ${unit} per instance · ${formatExact(summary.total)} in total`);
            if (summary.perInstance) lines.push(runsText(summary.perInstance, (value) => instanceValueText(value)));
        }
    } else if (count > 1) {
        // Some instances are not known exactly: no "× N", and each instance says what is known.
        lines.push(`At least ${formatExact(summary.total)} ${unit}: ${knownCount ? `exact in ${knownCount} of ${count} instances` : `no instance is known exactly`}`);
        if (summary.perInstance) lines.push(runsText(summary.perInstance, (value) => instanceValueText(value)));
    } else lines.push(`${partial ? 'At least ' : ''}${formatExact(summary.total)} ${unit}`);
    if (row.lowerBound) lines.push(`At least: ${row.lowerBound.label} ${row.lowerBound.reason}`);
    lines.push(...unresolvedLines(summary));
    return lines;
}

/**
 * A share of the whole model. When the model total leaves rows out, an exact
 * row's share is an upper bound (≤); when the row leaves something out too,
 * the share could be larger or smaller, so it reads as unknown.
 */
export function shareInfo(row, lens, modelTotal, modelPartial) {
    const fraction = row.share[lens];
    const text = formatShare(fraction);
    if (!text) return null;
    const summary = row[lens];
    const of = `${formatExact(summary.total)} of ${modelPartial ? 'at least ' : ''}${formatExact(modelTotal)} ${METRIC_UNITS[lens]}`;
    if (modelPartial && isPartial(summary, row)) {
        return { fraction, text: '?', unknown: true, title: `Unknown share: this row and the model total both leave out what cannot be counted (${of}: ${text} of what is counted)` };
    }
    // "<0.1%" already reads as a bound.
    if (modelPartial) return { fraction, text: text.startsWith('<') ? text : `≤${text}`, unknown: false, title: `At most ${text}: ${of}; the model total leaves out rows it cannot count` };
    return { fraction, text, unknown: false, title: `${of} (${text})` };
}

/** The ×N cell: occurrences per forward pass, and how nested repeats multiply. */
export function multiplicityInfo(row) {
    if (row.multiplicity === null) return { text: '?', title: row.incompleteReason ? `Unknown: ${row.incompleteReason}` : 'The count is unresolved', unknown: true };
    if (row.lowerBound) {
        return { text: `≥${row.multiplicity}`, title: `At least ${row.multiplicity} per forward pass: ${row.lowerBound.label} ${row.lowerBound.reason}`, unknown: false };
    }
    // Nested repeats multiply: 144 per forward pass = 12 Transformer blocks × 12 Attention heads.
    const product = row.factors?.reduce((total, factor) => total * factor.count, 1);
    const breakdown = row.factors?.length > 1 && product === row.multiplicity ? `: ${row.factors.map((factor) => `${factor.count} ${factor.label}`).join(' × ')}` : '';
    return { text: row.multiplicity > 1 ? `×${row.multiplicity}` : '', title: `${row.multiplicity} per forward pass${breakdown}`, unknown: false };
}

/** Ops inside a standard module: its one output tensor stands for theirs. */
export const FUSED_NOTE = 'Counted in the module output: standard Linear, Embedding, LayerNorm and position modules keep one output tensor';

/** Why the table leaves a shape unknown while the canvas may still draw one. */
export const SHAPE_GUESS_NOTE = 'The canvas still draws the shape validation carried past this point; the table counts nothing from it.';

/** How the totals treat shape errors, for the error summary. */
export const SHAPE_ERROR_NOTE = 'Totals use the declared settings. A layer whose shape check fails counts no MACs or activation (?). '
    + 'Where the error leaves a shape open, as when the inputs of an Add, Concat or Stack disagree, nothing is sized from that shape: '
    + 'later MACs, activations and widths inferred from it (such as an attention D) read ? too. '
    + 'The canvas still draws the shapes validation carries on with; the table counts nothing from them.';

/** The Output cell explained. */
export function shapeLines(row) {
    if (row.shape.unknown) return [row.shape.title || 'Unknown', SHAPE_GUESS_NOTE];
    return [row.shape.title, row.collectedShape ? `The repeat collects them as ${row.collectedShape}` : null,
        row.kind === 'repeat' && row.multiplicity > 1 && !row.collectedShape ? 'Output of each instance; the last one leaves the repeat' : null].filter(Boolean);
}

/** Which instances of a shared definition fail: `Fails in i=1–11 (11 of 12)`. */
export function failingText(row) {
    if (!row.failing?.count) return null;
    const { count, of, runs } = row.failing;
    return count === of ? `Fails in all ${of} instances` : `Fails in ${runLabels(runs)} (${count} of ${of})`;
}

/** The Layer cell explained: name, type, errors and where they occur. */
export function layerLines(row) {
    return [`${row.label} · ${row.typeName}`, ...row.errors, failingText(row)].filter(Boolean);
}

/**
 * One line for the status bar about the active cell, or null where the row's
 * own description says more (the Layer and Config columns, empty cells).
 */
export function cellStatus(row, column, { lens, modelTotal, modelPartial }) {
    if (!row) return null;
    const metric = lens === 'activation' ? 'activation' : 'macs';
    const join = (lines) => lines.filter(Boolean).join(' · ');
    if (column === 0) return row.errors.length ? join([row.errors[0], failingText(row)]) : null;
    if (column === 1) {
        const count = multiplicityInfo(row);
        return count.text || count.unknown ? `×N: ${count.title}` : null;
    }
    if (column === 2) {
        if (row.shape.unknown) return `Output unknown: ${join([row.shape.title.replace(/^Unknown: /, ''), SHAPE_GUESS_NOTE])}`;
        // A plain shape says nothing the cell does not show.
        const lines = shapeLines(row);
        return lines.length > 1 || row.shape.varies || /\n/.test(row.shape.title ?? '') ? `Output: ${join(lines.flatMap((line) => line.split('\n')))}` : null;
    }
    if (column === 4 || column === 5) {
        const key = column === 4 ? 'params' : metric;
        const summary = row[key];
        if (!summary.total && !summary.unresolved) return key === 'activation' && row.fusedActivation ? FUSED_NOTE : null;
        // The unit names the column: "1,536 parameters per instance × 12 = 18,432".
        return join(metricLines(key, summary, row));
    }
    if (column === 6) {
        const share = shareInfo(row, lens, modelTotal, modelPartial);
        return share ? `Share: ${share.title}` : null;
    }
    return null;
}
