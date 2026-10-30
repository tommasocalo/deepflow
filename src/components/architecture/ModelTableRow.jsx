import { memo, useMemo } from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronRight, GitFork, Repeat } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { formatMetric } from '../../domain/metrics/index.js';
import { eachText, FUSED_NOTE, isPartial, layerLines, metricLines, multiplicityInfo, shapeLines, shareInfo } from '../../domain/modelTable/cellText.js';
import { configAccessibleText } from '../../domain/modelTable/configFields.js';
import { fitShapeText } from '../../domain/modelTable/fitText.js';
import { FAMILY_ASSETS, styleFor } from './categoryStyle.js';
import OperationIcon from './OperationIcon.jsx';
import ModelTableConfigCell from './ModelTableConfigCell.jsx';
import { fieldTargets } from './modelTableEdits.js';
import { partitionParameterErrors } from './inspectorIssues.js';

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

/** A number, its compact form, and a small per-instance figure when every instance is known. */
function MetricCell({ metric, summary, row }) {
    if (!summary.total && summary.unresolved) {
        return <span className="df-mt-unknown" title={metricLines(metric, summary, row).join('\n')}>?</span>;
    }
    if (!summary.total) {
        // Ops inside a standard module: its single output stands for theirs.
        if (metric === 'activation' && row.fusedActivation) {
            return <span className="df-mt-fused" title={FUSED_NOTE}>–</span>;
        }
        return null;
    }
    const each = eachText(metric, summary);
    return <span className="df-mt-metric" title={metricLines(metric, summary, row).join('\n')}>
        <span className="df-mt-number">{isPartial(summary, row) ? '≥ ' : ''}{formatMetric(metric, summary.total)}</span>
        {each && <span className="df-mt-each">{each}</span>}
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
    const share = shareInfo(row, lens, modelTotal, modelPartial);
    if (!share) return null;
    if (share.unknown) return <span className="df-mt-unknown" title={share.title}>?</span>;
    return <span className="df-mt-share" title={share.title}>
        <span className="df-mt-bar" aria-hidden="true"><span style={{ width: `${Math.max(1, Math.min(100, share.fraction * 100))}%` }} /></span>
        <span className="df-mt-share-text">{share.text}</span>
    </span>;
}

/** Field messages: layer errors by parameter name, component checks by binding. */
function rowFieldErrors(row) {
    const specs = row.config.filter((part) => part.kind === 'field').map((part) => ({ name: part.name }));
    const fields = row.kind === 'layer' ? partitionParameterErrors(specs, row.errors).fields : {};
    for (const [name, messages] of Object.entries(row.parameterIssues ?? {})) fields[name] = [...(fields[name] ?? []), ...messages];
    return fields;
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
    const count = multiplicityInfo(row);
    const off = row.config.some((part) => part.kind === 'note' && part.text === 'off');
    const shapeText = row.shape.unknown ? null : fitShapeText(row.shape.text, budgets?.output);
    const shapeTitle = shapeLines(row).join('\n');
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
            title={layerLines(row).join('\n')}>
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
