import { memo, useMemo } from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronRight, GitFork, Repeat } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { formatExact, formatMetric, formatShare } from '../../domain/metrics/index.js';
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

function metricTitle(metric, summary, multiplicity) {
    const unit = UNITS[metric];
    const lines = [];
    if (summary.each && multiplicity > 1) {
        lines.push(summary.each.min === summary.each.max
            ? `${formatExact(summary.each.max)} ${unit} per instance × ${multiplicity} = ${formatExact(summary.total)}`
            : `${formatExact(summary.each.min)} to ${formatExact(summary.each.max)} ${unit} per instance · ${formatExact(summary.total)} in total`);
    } else lines.push(`${formatExact(summary.total)} ${unit}`);
    if (summary.unresolved) lines.push(summary.reason ? `Not counted: ${summary.reason}` : `Not counted: ${summary.unresolvedLabels.join(', ')} (unknown size)`);
    return lines.join('\n');
}

/** A number, its unit-free compact form, and a small per-instance figure when ×N > 1. */
function MetricCell({ metric, summary, multiplicity }) {
    if (!summary.total && summary.unresolved) {
        return <span className="df-mt-unknown" title={summary.reason ? `Unknown: ${summary.reason}` : `Unknown: ${summary.unresolvedLabels.join(', ')}`}>?</span>;
    }
    if (!summary.total) return null;
    const each = summary.each && multiplicity > 1 && summary.each.max > 0 ? summary.each : null;
    const compact = (value) => formatMetric(metric, value, { compact: true });
    return <span className="df-mt-metric" title={metricTitle(metric, summary, multiplicity)}>
        <span className="df-mt-number">{summary.unresolved ? '≥ ' : ''}{formatMetric(metric, summary.total)}</span>
        {each && <span className="df-mt-each">{each.min === each.max ? `${compact(each.max)} ea` : `${compact(each.min)}–${compact(each.max)}`}</span>}
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

function ShareCell({ row, lens, modelTotal }) {
    const fraction = row.share[lens];
    const text = formatShare(fraction);
    if (!text) return null;
    const summary = row[lens];
    return <span className="df-mt-share" title={`${formatExact(summary.total)} of ${formatExact(modelTotal)} ${UNITS[lens]} (${text})`}>
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

function ModelTableRow({ row, index, cells, lens, modelTotal, context, selected, focusColumn, editing, suggestions, actions }) {
    const targets = fieldTargets(row, cells, context);
    const fieldErrors = useMemo(() => rowFieldErrors(row), [row]);
    const editable = Object.values(targets).some((target) => target.editable);
    const cell = (column) => ({
        'data-col': column,
        tabIndex: focusColumn === column ? 0 : -1,
        onFocus: (event) => { if (event.target === event.currentTarget) actions.focusCell(row, column); },
    });
    const hasErrors = row.errors.length > 0;
    const warnings = row.warnings;
    const editableHints = Object.values(targets).filter((target) => target.editable).map((target) => `${target.field.name}: ${target.hint}`);
    return <tr role="row" data-row-key={row.key} aria-rowindex={index + 2} aria-level={row.depth + 1} aria-posinset={row.posInSet} aria-setsize={row.setSize}
        aria-expanded={row.expandable ? row.expanded : undefined} aria-selected={selected}
        className={`df-mt-row${selected ? ' is-selected' : ''}${row.kind !== 'layer' ? ' is-container' : ''}${row.expanded ? ' is-expanded' : ''}${hasErrors ? ' has-error' : ''}`}
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
                        onClick={(event) => { event.stopPropagation(); actions.toggle(row); }}><ChevronRight size={12} strokeWidth={3} /></button>
                    : <span className="df-mt-chevron-space" />}
                <Glyph glyph={row.glyph} />
                <span className="df-mt-label">{row.label}</span>
                {hasErrors && <AlertCircle className="df-mt-status is-error" size={11} strokeWidth={3} aria-label="Has errors" />}
                {!hasErrors && warnings.length > 0 && <AlertTriangle className="df-mt-status is-warning" size={11} strokeWidth={3} aria-label={warnings.join('; ')} />}
            </span>
        </th>
        <td role="gridcell" {...cell(1)} className="df-mt-num df-mt-count" title={row.multiplicity === null ? 'The count is unresolved' : `${row.multiplicity} per forward pass`}>
            {row.multiplicity === null ? <span className="df-mt-unknown">?</span> : row.multiplicity > 1 ? `×${row.multiplicity}` : ''}
        </td>
        <td role="gridcell" {...cell(2)} className="df-mt-shape" title={row.shape.title} aria-label={row.shape.text ?? undefined}><CompactShape text={row.shape.text} /></td>
        <td role="gridcell" {...cell(3)} className={`df-mt-config-cell${editing ? ' is-editing' : ''}`} aria-readonly={!editable}
            title={editing ? undefined : editableHints.join('\n') || undefined}>
            <ModelTableConfigCell row={row} cells={cells} targets={targets} context={context} fieldErrors={fieldErrors}
                editing={editing} suggestions={suggestions}
                onStartEdit={(fieldName) => actions.startEdit(row, fieldName)}
                onDone={(edits, options) => actions.editDone(row, edits, options)} onMessage={actions.message} />
        </td>
        <td role="gridcell" {...cell(4)} className="df-mt-num"><MetricCell metric="params" summary={row.params} multiplicity={row.multiplicity} /></td>
        <td role="gridcell" {...cell(5)} className="df-mt-num"><MetricCell metric="macs" summary={row.macs} multiplicity={row.multiplicity} /></td>
        <td role="gridcell" {...cell(6)} className="df-mt-num"><ShareCell row={row} lens={lens} modelTotal={modelTotal} /></td>
    </tr>;
}

export default memo(ModelTableRow);
