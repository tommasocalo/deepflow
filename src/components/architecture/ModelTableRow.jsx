import { memo, useMemo } from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronRight, GitFork, Repeat } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { formatExact, formatMetric, formatShare } from '../../domain/metrics/index.js';
import { configAccessibleText, runsText } from '../../domain/modelTable/configFields.js';
import { fitShapeText } from '../../domain/modelTable/fitText.js';
import { FAMILY_ASSETS, styleFor } from './categoryStyle.js';
import OperationIcon from './OperationIcon.jsx';
import ModelTableConfigCell from './ModelTableConfigCell.jsx';
import { fieldTargets } from './modelTableEdits.js';
import { partitionParameterErrors } from './inspectorIssues.js';

const UNITS = { params: 'parameters', macs: 'MACs', activation: 'bytes' };

/** The canvas vocabulary: operation marks, standard component icons, container icons. */
function Glyph({ glyph }) {
    if (glyph.libraryKey) {
        const category = getEntry(glyph.libraryKey)?.category;
        // Tint sparingly: module families and standard components only.
        const tinted = glyph.standard || !!FAMILY_ASSETS[category];
        return <span className="df-mt-glyph" style={tinted ? { background: `color-mix(in srgb, ${styleFor(category).accent} 14%, white)` } : undefined}>
            <OperationIcon libraryKey={glyph.libraryKey} size={14} />
        </span>;
    }
    const Icon = glyph.icon === 'parallel' ? GitFork : glyph.icon === 'repeat' ? Repeat : Boxes;
    return <span className="df-mt-glyph"><Icon size={12} strokeWidth={2.5} aria-hidden="true" /></span>;
}

/** What a partial figure leaves out, one line per cause. */
function unresolvedLines(summary) {
    if (!summary.unresolved) return [];
    if (summary.reason) return [`Not counted: ${summary.reason}`];
    const items = summary.unresolvedItems.slice(0, 4).map((item) => item.reason ? `${item.label} (${item.reason})` : item.label);
    return [`Not counted: ${items.join('; ')}${summary.unresolvedItems.length > 4 ? '; …' : ''}`];
}

function metricTitle(metric, summary, row) {
    const unit = UNITS[metric];
    const multiplicity = row.multiplicity;
    const lines = [];
    if (summary.each && multiplicity > 1) {
        if (summary.each.min === summary.each.max) lines.push(`${formatExact(summary.each.max)} ${unit} per instance × ${multiplicity} = ${formatExact(summary.total)}`);
        else {
            lines.push(`${formatExact(summary.each.min)} to ${formatExact(summary.each.max)} ${unit} per instance · ${formatExact(summary.total)} in total`);
            // Which instances differ, in one hover.
            if (summary.perInstance) lines.push(runsText(summary.perInstance, (value) => value === null ? '?' : formatExact(value)));
        }
    } else lines.push(`${formatExact(summary.total)} ${unit}`);
    if (row.lowerBound) lines.push(`At least: ${row.lowerBound.label} ${row.lowerBound.reason}`);
    lines.push(...unresolvedLines(summary));
    return lines.join('\n');
}

/** A per-instance range sharing one unit: 4.73–7.09M, 1.86K–295K. */
function rangeText(metric, min, max) {
    const low = formatMetric(metric, min, { compact: true });
    const high = formatMetric(metric, max, { compact: true });
    const unit = (text) => text?.match(/[A-Za-z ]+$/)?.[0] ?? '';
    return unit(low) && unit(low) === unit(high) ? `${low.slice(0, -unit(low).length)}–${high}` : `${low}–${high}`;
}

/** A number, its compact form, and a small per-instance figure when ×N > 1. */
function MetricCell({ metric, summary, row }) {
    const partial = summary.unresolved > 0 || !!row.lowerBound;
    if (!summary.total && summary.unresolved) {
        const title = ['Unknown', row.lowerBound ? `At least: ${row.lowerBound.label} ${row.lowerBound.reason}` : null, ...unresolvedLines(summary)];
        return <span className="df-mt-unknown" title={title.filter(Boolean).join('\n')}>?</span>;
    }
    if (!summary.total) {
        // Ops inside a standard module: its single output stands for theirs.
        if (metric === 'activation' && row.fusedActivation) {
            return <span className="df-mt-fused" title="Counted in the module output: standard Linear, Embedding, LayerNorm and position modules keep one output tensor">–</span>;
        }
        return null;
    }
    const each = summary.each && row.multiplicity > 1 && summary.each.max > 0 ? summary.each : null;
    return <span className="df-mt-metric" title={metricTitle(metric, summary, row)}>
        <span className="df-mt-number">{partial ? '≥ ' : ''}{formatMetric(metric, summary.total)}</span>
        {each && <span className="df-mt-each">{each.min === each.max ? `${formatMetric(metric, each.max, { compact: true })} ea` : rangeText(metric, each.min, each.max)}</span>}
    </span>;
}

/** Shapes read like the canvas but tighter: B×1024×768, and 12 × [B×1024×64] for collections. */
function CompactShape({ text }) {
    if (!text) return null;
    const collection = text.match(/^(\d+) tensors(?: \[(.*)\])?$/);
    const dims = (value) => value.split(' × ').flatMap((dim, index) => index
        ? [<span key={`x${index}`} className="df-mt-times" aria-hidden="true">×</span>, <span key={index}>{dim}</span>]
        : [<span key={index}>{dim}</span>]);
    if (collection) {
        return <><span>{collection[1]}</span><span className="df-mt-times is-collection" aria-hidden="true">×</span>{collection[2] ? <>[{dims(collection[2])}]</> : 'tensors'}</>;
    }
    return <>{dims(text)}</>;
}

function ShareCell({ row, lens, modelTotal, modelPartial }) {
    const fraction = row.share[lens];
    const text = formatShare(fraction);
    if (!text) return null;
    const summary = row[lens];
    const bound = modelPartial ? `\nAt most ${text}: the model total leaves out rows it cannot count` : '';
    return <span className="df-mt-share" title={`${formatExact(summary.total)} of ${modelPartial ? 'at least ' : ''}${formatExact(modelTotal)} ${UNITS[lens]} (${text})${bound}`}>
        <span className="df-mt-bar" aria-hidden="true"><span style={{ width: `${Math.max(1, Math.min(100, fraction * 100))}%` }} /></span>
        <span className="df-mt-share-text">{text}</span>
    </span>;
}

/** Field messages: layer errors by parameter name, component checks by binding. */
function rowFieldErrors(row) {
    const specs = row.config.filter((part) => part.kind === 'field').map((part) => ({ name: part.name }));
    const fields = row.kind === 'layer' ? partitionParameterErrors(specs, row.errors).fields : {};
    for (const [name, messages] of Object.entries(row.parameterIssues ?? {})) fields[name] = [...(fields[name] ?? []), ...messages];
    return fields;
}

function multiplicityText(row) {
    if (row.multiplicity === null) return { text: '?', title: row.incompleteReason ? `Unknown: ${row.incompleteReason}` : 'The count is unresolved', unknown: true };
    if (row.lowerBound) {
        return { text: `≥${row.multiplicity}`, title: `At least ${row.multiplicity} per forward pass: ${row.lowerBound.label} ${row.lowerBound.reason}`, unknown: false };
    }
    // Nested repeats multiply: 144 per forward pass = 12 Transformer blocks × 12 Attention heads.
    const product = row.factors?.reduce((total, factor) => total * factor.count, 1);
    const breakdown = row.factors?.length > 1 && product === row.multiplicity ? `: ${row.factors.map((factor) => `${factor.count} ${factor.label}`).join(' × ')}` : '';
    return { text: row.multiplicity > 1 ? `×${row.multiplicity}` : '', title: `${row.multiplicity} per forward pass${breakdown}`, unknown: false };
}

function ModelTableRow({ row, index, cells, lens, modelTotal, modelPartial, context, selected, focusColumn, editing, suggestions, budgets, actions }) {
    const targets = fieldTargets(row, cells, context);
    const fieldErrors = useMemo(() => rowFieldErrors(row), [row]);
    const editable = Object.values(targets).some((target) => target.editable);
    const metric = lens === 'activation' ? 'activation' : 'macs';
    const cell = (column) => ({
        'data-col': column,
        tabIndex: focusColumn === column ? 0 : -1,
        onFocus: (event) => { if (event.target === event.currentTarget) actions.focusCell(row, column); },
    });
    const hasErrors = row.errors.length > 0;
    const warnings = row.warnings;
    const editableHints = Object.values(targets).filter((target) => target.editable).map((target) => `${target.field.name}: ${target.hint}`);
    const count = multiplicityText(row);
    const off = row.config.some((part) => part.kind === 'note' && part.text === 'off');
    const shapeText = row.shape.unknown ? null : fitShapeText(row.shape.text, budgets?.output);
    const shapeTitle = [row.shape.title, row.collectedShape ? `The repeat collects them as ${row.collectedShape}` : null,
        row.kind === 'repeat' && row.multiplicity > 1 && !row.collectedShape ? 'Output of each instance; the last one leaves the repeat' : null].filter(Boolean).join('\n');
    return <tr role="row" data-row-key={row.key} aria-rowindex={index + 2} aria-level={row.depth + 1} aria-posinset={row.posInSet} aria-setsize={row.setSize}
        aria-expanded={row.expandable ? row.expanded : undefined} aria-selected={selected}
        className={`df-mt-row${selected ? ' is-selected' : ''}${row.kind !== 'layer' ? ' is-container' : ''}${row.expanded ? ' is-expanded' : ''}${hasErrors ? ' has-error' : ''}${off ? ' is-off' : ''}`}
        onMouseDown={(event) => {
            // A double-click opens the level; it should not select a word.
            if (event.detail > 1 && !event.target.closest?.('input, select')) event.preventDefault();
        }}
        onClick={(event) => actions.click(row, Number(event.target.closest?.('[data-col]')?.dataset.col ?? 0))}
        onDoubleClick={(event) => { if (!event.target.closest?.('.df-mt-editor')) actions.open(row); }}>
        <th role="rowheader" scope="row" {...cell(0)} className="df-mt-layer" aria-invalid={hasErrors || undefined}
            title={`${row.label} · ${row.typeName}${hasErrors ? `\n${row.errors.join('\n')}` : ''}`}>
            <span className="df-mt-layer-inner" style={{ paddingLeft: `${row.depth * 14}px` }}>
                {row.expandable
                    ? <button type="button" className="df-mt-chevron" aria-hidden="true" tabIndex={-1}
                        // Focus goes to the row header, which carries aria-expanded, never to this hidden button.
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={(event) => {
                            event.stopPropagation();
                            event.currentTarget.closest('[data-col]')?.focus({ preventScroll: true });
                            actions.toggle(row);
                        }}><ChevronRight size={12} strokeWidth={3} /></button>
                    : <span className="df-mt-chevron-space" />}
                <Glyph glyph={row.glyph} />
                <span className="df-mt-label">{row.label}</span>
                {hasErrors && <AlertCircle className="df-mt-status is-error" size={11} strokeWidth={3} aria-label="Has errors" />}
                {!hasErrors && warnings.length > 0 && <AlertTriangle className="df-mt-status is-warning" size={11} strokeWidth={3} aria-label={warnings.join('; ')} />}
            </span>
        </th>
        <td role="gridcell" {...cell(1)} className="df-mt-num df-mt-count" title={count.text || count.unknown ? count.title : undefined}>
            {count.unknown ? <span className="df-mt-unknown">?</span> : count.text}
        </td>
        {/* The label reads the shape as words; the drawn digits are hidden from screen readers. */}
        <td role="gridcell" {...cell(2)} className="df-mt-shape" title={shapeTitle || undefined}
            aria-label={row.shape.unknown ? 'unknown' : row.shape.text ?? undefined}>
            {row.shape.unknown ? <span className="df-mt-unknown">?</span> : <span aria-hidden="true"><CompactShape text={shapeText} /></span>}
        </td>
        <td role="gridcell" {...cell(3)} className={`df-mt-config-cell${editing ? ' is-editing' : ''}`} aria-readonly={!editable}
            aria-label={editing ? undefined : configAccessibleText(row, cells) || undefined}
            title={editing ? undefined : editableHints.join('\n') || undefined}>
            <ModelTableConfigCell row={row} cells={cells} targets={targets} context={context} fieldErrors={fieldErrors}
                editing={editing} suggestions={suggestions} budget={budgets?.config} labelled={!!configAccessibleText(row, cells)}
                onStartEdit={(fieldName) => actions.startEdit(row, fieldName)}
                onReadOnly={(reason) => { actions.click(row, 3); actions.message(reason); }}
                onDone={(edits, options) => actions.editDone(row, edits, options)} onMessage={actions.message} />
        </td>
        <td role="gridcell" {...cell(4)} className="df-mt-num"><MetricCell metric="params" summary={row.params} row={row} /></td>
        <td role="gridcell" {...cell(5)} className="df-mt-num"><MetricCell metric={metric} summary={row[metric]} row={row} /></td>
        <td role="gridcell" {...cell(6)} className="df-mt-num"><ShareCell row={row} lens={lens} modelTotal={modelTotal} modelPartial={modelPartial} /></td>
    </tr>;
}

export default memo(ModelTableRow);
