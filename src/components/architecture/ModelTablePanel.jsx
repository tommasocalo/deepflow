import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { isValidLevelPath, sameLevelPath } from '../../domain/architecture/levelNavigation.js';
import { ancestorKeys, buildModelTableRows, rowKey, shapeSummary } from '../../domain/modelTable/rows.js';
import { formatBytes, formatExact, formatMacs, formatMetric } from '../../domain/metrics/index.js';
import { parameterSuggestions } from '../../domain/parameterBindings.js';
import { useModelMetrics } from '../../hooks/useModelMetrics.js';
import ModelTableRow from './ModelTableRow.jsx';
import { applyModelTableEdit, COLUMNS, CONFIG_COLUMN, fieldTargets } from './modelTableEdits.js';
import './model-table.css';

const RENDER_STEP = 300;
const WIDTH_KEY = 'deepflow:model-table-width';
const MIN_WIDTH = 380;
const DEFAULT_WIDTH = 760;
const MIN_HEIGHT = 160;

const DEPTHS = [
    ['root', 'Root', 'Nodes of the main graph'],
    ['blocks', 'Blocks', 'Open repeats and custom components; standard modules stay one row'],
    ['ops', 'Ops', 'Every layer and operation'],
];
const LENSES = [
    ['params', 'Params', 'Share of learned parameters'],
    ['macs', 'MACs', 'Share of multiply-accumulates per sample'],
    ['activation', 'Activations', 'Share of fp32 activation memory per sample'],
];
const CELLS = [
    ['symbols', 'Symbols', 'Settings as written, such as $d_model or 4 * d_model'],
    ['values', 'Values', 'Settings resolved to numbers'],
];
const LENS_NAMES = { params: 'parameters', macs: 'multiply-accumulates', activation: 'activation bytes' };

/** A width the user chose earlier, or null for the default (760px or less of a small window). */
function readWidth() {
    try {
        const saved = Number(localStorage.getItem(WIDTH_KEY));
        return Number.isFinite(saved) && saved >= MIN_WIDTH ? saved : null;
    } catch {
        return null;
    }
}
function saveWidth(width) {
    try {
        if (width === null) localStorage.removeItem(WIDTH_KEY);
        else localStorage.setItem(WIDTH_KEY, String(Math.round(width)));
    } catch {
        // The width is a convenience; storage may be unavailable.
    }
}

/** Three choices in one row; arrow keys move within the group. */
function Segmented({ label, value, options, onChange }) {
    const select = (key, target) => {
        onChange(key);
        requestAnimationFrame(() => target?.focus());
    };
    return <div className="df-mt-segmented" role="radiogroup" aria-label={label}>
        <span className="df-mt-segmented-label" aria-hidden="true">{label}</span>
        <span className="df-mt-segments">{options.map(([key, text, title], index) => <button key={key} type="button" role="radio"
            aria-checked={value === key} tabIndex={value === key ? 0 : -1} title={title} onClick={() => onChange(key)}
            onKeyDown={(event) => {
                const step = ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 0;
                if (!step) return;
                event.preventDefault();
                const next = (index + step + options.length) % options.length;
                select(options[next][0], event.currentTarget.parentElement.children[next]);
            }}>{text}</button>)}</span>
    </div>;
}

function Stat({ label, value, detail, title }) {
    return <div className="df-mt-stat" title={title}>
        <dt>{label}</dt>
        <dd>{value}</dd>
        {detail && <dd className="df-mt-stat-detail">{detail}</dd>}
    </div>;
}

function Footer({ metrics, assumptionsId }) {
    const totals = metrics.totals;
    const unresolved = (metric) => totals.unresolved[metric];
    const partial = (metric) => unresolved(metric).size > 0;
    const unresolvedText = (metric) => `${unresolved(metric).size} row${unresolved(metric).size === 1 ? '' : 's'} unresolved`;
    const unresolvedTitle = (metric) => partial(metric) ? `Not counted (unknown size): ${[...new Set(unresolved(metric).values())].join(', ')}` : '';
    const frozen = totals.params - totals.trainable;
    const largest = totals.largest;
    const inputs = metrics.inputs.map((input) => `${input.label} ${shapeSummary([input.shape]).text ?? '?'}`).join(' · ');
    return <footer className="df-mt-footer">
        <dl className="df-mt-stats">
            <Stat label="Params" value={`${partial('params') ? '≥ ' : ''}${formatExact(totals.params)}`}
                detail={partial('params') ? unresolvedText('params') : frozen ? `trainable ${formatExact(totals.trainable)} · frozen ${formatExact(frozen)}` : 'all trainable'}
                title={unresolvedTitle('params') || 'Learned parameters, every layer counted once per occurrence (no weight tying)'} />
            <Stat label="MACs / sample" value={`${partial('macs') ? '≥ ' : ''}${formatMacs(totals.macs) ?? '0'}`}
                detail={partial('macs') ? unresolvedText('macs') : formatExact(totals.macs)}
                title={unresolvedTitle('macs') || 'Forward multiply-accumulates at B = 1: MatMul, Linear and Conv. Causal attention counts the full T × T product.'} />
            <Stat label="Largest activation" value={largest ? formatBytes(largest.bytes) : '—'}
                detail={largest ? `${largest.label} · ${shapeSummary([largest.shape]).text ?? '?'}` : partial('activation') ? unresolvedText('activation') : null}
                title={`Largest single output tensor at B = 1, fp32. All activations: ${partial('activation') ? '≥ ' : ''}${formatBytes(totals.activation) ?? '0 B'}. Standard Linear, Embedding, LayerNorm and position modules count one output; attention internals count individually; views and weights count nothing.`} />
            <Stat label="Weights" value={`${partial('params') ? '≥ ' : ''}${formatBytes(totals.params * 4) ?? '0 B'}`} detail="fp32, 4 bytes each"
                title={`${formatExact(totals.params * 4)} bytes`} />
        </dl>
        <p id={assumptionsId} className="df-mt-assumptions">
            B = 1{inputs ? ` · ${inputs}` : ''} · fp32 · forward pass · MACs: MatMul, Linear, Conv · no weight tying
        </p>
    </footer>;
}

/** The whole model as a torchinfo-style table beside the canvas of one pane. */
export default function ModelTablePanel() {
    const store = useArchitectureStoreApi();
    const { depth, share: lens, cells } = useArchitectureStore((state) => state.modelTable);
    const currentPath = useArchitectureStore((state) => state.currentPath);
    const selectedNodeIds = useArchitectureStore((state) => state.selectedNodeIds);
    const inspectedIterations = useArchitectureStore((state) => state.inspectedIterations);
    const { graph, templates, metrics } = useModelMetrics();
    const flow = useReactFlow();
    const assumptionsId = useId();
    const [overrides, setOverrides] = useState({});
    const [limit, setLimit] = useState(RENDER_STEP);
    const [active, setActive] = useState({ key: null, column: 0 });
    const [editing, setEditing] = useState(null);
    const [message, setMessage] = useState('');
    const [width, setWidth] = useState(readWidth);
    const [height, setHeight] = useState(null);
    const [sheet, setSheet] = useState(false);
    const asideRef = useRef(null);
    const bodyRef = useRef(null);
    const dragRef = useRef(null);
    const rowsRef = useRef(new Map());

    const ready = !!metrics?.root;
    const rows = useMemo(() => ready ? buildModelTableRows({ metrics, rootGraph: graph, templates, depth, overrides }) : [], [ready, metrics, graph, templates, depth, overrides]);
    const byKey = useMemo(() => new Map(rows.map((row, index) => [row.key, { row, index }])), [rows]);
    const context = useMemo(() => ({ graph, templates, templateUses: metrics?.templateUses ?? {} }), [graph, templates, metrics]);
    const selectedKeys = useMemo(() => new Set(selectedNodeIds.map((id) => rowKey([...currentPath, id]))), [selectedNodeIds, currentPath]);
    const activeEntry = byKey.get(active.key) ?? (rows.length ? { row: rows[0], index: 0 } : null);
    const activeRow = activeEntry?.row ?? null;
    const activeColumn = byKey.has(active.key) ? active.column : 0;
    const editingRow = editing ? byKey.get(editing.key)?.row ?? null : null;
    const suggestions = useMemo(() => editingRow && isValidLevelPath(graph, templates, editingRow.levelPath, inspectedIterations)
        ? parameterSuggestions(graph, editingRow.levelPath, templates, inspectedIterations, editingRow.scope) : [], [editingRow, graph, templates, inspectedIterations]);

    useLayoutEffect(() => {
        rowsRef.current = byKey;
    }, [byKey]);

    const rowElement = useCallback((key) => [...(bodyRef.current?.children ?? [])].find((element) => element.dataset.rowKey === key), []);
    const focusCell = useCallback((key, column) => rowElement(key)?.querySelector(`[data-col="${column}"]`)?.focus({ preventScroll: false }), [rowElement]);

    /** Pan the canvas only when the node is outside it. */
    const reveal = useCallback((nodeId, delay = 0) => {
        const run = () => {
            const canvas = asideRef.current?.parentElement?.querySelector('[data-architecture-canvas]');
            const bounds = canvas?.getBoundingClientRect?.();
            const state = store.getState();
            const node = resolveGraphLevel(state.graph, state.currentPath, useTemplatesStore.getState().templates, state.inspectedIterations).graph.nodes[nodeId];
            if (!node || !bounds?.width || !bounds?.height) return;
            try {
                const { x, y, zoom } = flow.getViewport();
                const left = node.position.x * zoom + x;
                const top = (node.position.y + 20) * zoom + y;
                if (left < 48 || top < 48 || left > bounds.width - 48 || top > bounds.height - 48) {
                    flow.setCenter(node.position.x, node.position.y + 20, { zoom, duration: 200 });
                }
            } catch {
                // The canvas can unmount while a reveal waits.
            }
        };
        if (delay) setTimeout(run, delay);
        else requestAnimationFrame(run);
    }, [flow, store]);

    /** Open the row's level, select its node, and optionally its settings. */
    const openRow = useCallback((row, { settings = false } = {}) => {
        if (!row) return;
        const templatesNow = useTemplatesStore.getState().templates;
        const shows = (state) => isValidLevelPath(state.graph, templatesNow, row.levelPath, state.inspectedIterations)
            && !!resolveGraphLevel(state.graph, row.levelPath, templatesNow, state.inspectedIterations).graph.nodes[row.nodeId];
        // A node of one local version only: inspect the instance that has it.
        if (!shows(store.getState())) for (const step of row.firstInstance) store.getState().setInspectedIteration(step.path, step.index);
        const sameLevel = sameLevelPath(store.getState().currentPath, row.levelPath);
        if (!store.getState().openLevel(row.levelPath)) return;
        store.getState().selectOnly(row.nodeId);
        if (settings) store.getState().setExpanded(row.nodeId);
        reveal(row.nodeId, sameLevel ? 0 : 450);
    }, [reveal, store]);

    const toggle = useCallback((row) => {
        setOverrides((current) => ({ ...current, [row.key]: !row.expanded }));
    }, []);

    const actions = useMemo(() => ({
        focusCell: (row, column) => setActive((current) => current.key === row.key && current.column === column ? current : { key: row.key, column }),
        click: (row, column) => {
            setActive({ key: row.key, column });
            setMessage('');
            if (sameLevelPath(row.levelPath, store.getState().currentPath)) {
                store.getState().selectOnly(row.nodeId);
                reveal(row.nodeId);
            }
        },
        open: (row) => openRow(row),
        toggle,
        startEdit: (row, fieldName = null, seed = null) => {
            setActive({ key: row.key, column: CONFIG_COLUMN });
            setMessage('');
            setEditing({ key: row.key, fieldName, seed });
        },
        editDone: (row, edits, { refocus = true } = {}) => {
            const written = edits?.length ? applyModelTableEdit(store, row, edits) : 0;
            setEditing(null);
            setMessage(written ? `Set ${row.label} ${edits.map((edit) => edit.field.name).join(', ')} · ⌘Z / Ctrl+Z to undo` : edits ? 'No change' : 'Edit cancelled');
            if (refocus) focusCell(row.key, CONFIG_COLUMN);
        },
        message: setMessage,
    }), [focusCell, openRow, reveal, store, toggle]);

    // Canvas → table: reveal and scroll to the selected node's row. The table
    // never takes focus from the canvas.
    useEffect(() => store.subscribe((state, previous) => {
        if (state.selectedNodeIds === previous.selectedNodeIds && state.currentPath === previous.currentPath) return;
        const id = state.selectedNodeIds[0];
        if (!id) return;
        const key = rowKey([...state.currentPath, id]);
        setActive((current) => current.key === key ? current : { key, column: current.column });
        if (!rowsRef.current.has(key)) {
            const ancestors = ancestorKeys([...state.currentPath, id]);
            setOverrides((current) => ancestors.every((ancestor) => current[ancestor] === true) ? current
                : { ...current, ...Object.fromEntries(ancestors.map((ancestor) => [ancestor, true])) });
        }
        requestAnimationFrame(() => requestAnimationFrame(() => {
            const index = rowsRef.current.get(key)?.index;
            if (index === undefined) return;
            setLimit((current) => index < current ? current : Math.ceil((index + 1) / RENDER_STEP) * RENDER_STEP);
            requestAnimationFrame(() => rowElement(key)?.scrollIntoView?.({ block: 'nearest' }));
        }));
    }), [rowElement, store]);

    // The panel docks at the right, or becomes a bottom sheet in narrow panes.
    useEffect(() => {
        const workspace = asideRef.current?.parentElement;
        if (!workspace || typeof ResizeObserver === 'undefined') return undefined;
        const observer = new ResizeObserver(() => setSheet(getComputedStyle(workspace).flexDirection === 'column'));
        observer.observe(workspace);
        return () => observer.disconnect();
    }, []);

    const setDepth = (next) => {
        store.getState().setModelTable({ depth: next });
        setOverrides({});
        setLimit(RENDER_STEP);
        setEditing(null);
    };
    const close = () => {
        store.getState().setShowModelTable(false);
        requestAnimationFrame(() => document.querySelector('[aria-label="Show model table"]')?.focus());
    };

    const onGridKeyDown = (event) => {
        if (editing || event.target.closest?.('.df-mt-editor')) return;
        const cellElement = event.target.closest?.('[data-col]');
        const entry = byKey.get(cellElement?.closest('tr')?.dataset.rowKey);
        if (!cellElement || !entry) return;
        const { row, index } = entry;
        const column = Number(cellElement.dataset.col);
        const rendered = Math.min(rows.length, limit);
        const go = (nextIndex, nextColumn = column) => {
            const targetIndex = Math.max(0, Math.min(rendered - 1, nextIndex));
            const targetColumn = Math.max(0, Math.min(COLUMNS.length - 1, nextColumn));
            const target = rows[targetIndex];
            setActive({ key: target.key, column: targetColumn });
            focusCell(target.key, targetColumn);
        };
        const editable = Object.values(fieldTargets(row, cells, context)).some((target) => target.editable);
        const modifier = event.ctrlKey || event.metaKey;
        switch (event.key) {
            case 'ArrowDown': go(index + 1); break;
            case 'ArrowUp': go(index - 1); break;
            case 'PageDown': go(index + 10); break;
            case 'PageUp': go(index - 10); break;
            case 'Home': if (modifier) go(0); else go(index, 0); break;
            case 'End': if (modifier) go(rendered - 1); else go(index, COLUMNS.length - 1); break;
            case 'ArrowRight':
                if (column === 0 && row.expandable && !row.expanded) toggle(row);
                else go(index, column + 1);
                break;
            case 'ArrowLeft':
                if (column > 0) go(index, column - 1);
                else if (row.expanded) toggle(row);
                else if (row.parentKey && byKey.has(row.parentKey)) go(byKey.get(row.parentKey).index, 0);
                break;
            case 'Enter':
                if (column === CONFIG_COLUMN && editable) actions.startEdit(row);
                else openRow(row);
                break;
            case 'F2':
                if (editable) actions.startEdit(row);
                break;
            case ' ':
                if (column === 0 && row.expandable) toggle(row);
                break;
            default:
                // Spreadsheet style: typing on a setting starts editing it.
                if (column === CONFIG_COLUMN && editable && event.key.length === 1 && !modifier && !event.altKey) actions.startEdit(row, null, event.key);
                else return;
        }
        event.preventDefault();
    };

    const renderedWidth = () => width ?? asideRef.current?.getBoundingClientRect().width ?? DEFAULT_WIDTH;
    const resizeBy = (delta) => {
        if (sheet) {
            const current = height ?? asideRef.current?.getBoundingClientRect().height ?? 320;
            setHeight(Math.max(MIN_HEIGHT, current + delta));
            return;
        }
        const limitWidth = (asideRef.current?.parentElement?.getBoundingClientRect().width ?? 0) * 0.72 || Infinity;
        const next = Math.max(MIN_WIDTH, Math.min(limitWidth, renderedWidth() + delta));
        setWidth(next);
        saveWidth(next);
    };

    const totals = metrics?.totals;
    const modelTotal = totals?.[lens] ?? 0;
    const lensTotal = formatMetric(lens, modelTotal) ?? '0';
    const visibleRows = rows.slice(0, limit);
    const breadcrumb = activeRow ? ['main', ...ancestorKeys(activeRow.path).map((key) => byKey.get(key)?.row.label ?? '…'), activeRow.label].join(' › ') : '';
    const sameLevel = activeRow && sameLevelPath(activeRow.levelPath, currentPath);
    const owner = activeRow?.ownerIds.length === 1 ? templates[activeRow.ownerIds[0]] : null;
    const uses = owner ? metrics?.templateUses?.[owner.id] ?? 0 : 0;
    const focusText = message || (editing ? 'Enter apply · Esc cancel · Tab next field' : activeRow
        ? [breadcrumb, activeRow.typeName, activeRow.multiplicity > 1 ? `×${activeRow.multiplicity}` : null,
            owner ? `in the shared ${owner.name} definition${uses > 1 ? ` (${uses} uses)` : ''}` : null,
            sameLevel ? 'on this canvas' : 'Enter or double-click opens its level'].filter(Boolean).join(' · ')
        : 'Select a row');

    return <aside ref={asideRef} className={`df-model-table nokey${sheet ? ' is-sheet' : ''}`} aria-label="Model table"
        style={{ ...(width ? { '--model-table-width': `${width}px` } : {}), ...(height ? { '--model-table-height': `${height}px` } : {}) }}
        onKeyDown={(event) => {
            // Plain keys stay in the table: Delete, Backspace, Space and "/"
            // never reach the canvas or the ribbon. Undo still works: its
            // listener runs in the capture phase.
            if (!event.ctrlKey && !event.metaKey && !event.altKey) event.stopPropagation();
        }}
        onCopy={(event) => {
            if (event.target.closest?.('input, textarea, select') || window.getSelection?.()?.toString()) return;
            const text = event.target.closest?.('[data-col]')?.textContent?.trim();
            if (text && event.clipboardData) event.clipboardData.setData('text/plain', text);
            event.preventDefault();
        }}
        onPaste={(event) => { if (!event.target.closest?.('input, textarea, select')) event.preventDefault(); }}>
        <div className="df-mt-resizer" role="separator" tabIndex={0} aria-label="Resize model table"
            aria-orientation={sheet ? 'horizontal' : 'vertical'} aria-valuenow={Math.round(sheet ? height ?? 0 : width ?? DEFAULT_WIDTH)} aria-valuemin={sheet ? MIN_HEIGHT : MIN_WIDTH}
            onPointerDown={(event) => {
                dragRef.current = { x: event.clientX, y: event.clientY, width: renderedWidth(), height: asideRef.current?.getBoundingClientRect().height ?? 0 };
                event.currentTarget.setPointerCapture?.(event.pointerId);
                event.preventDefault();
            }}
            onPointerMove={(event) => {
                const drag = dragRef.current;
                if (!drag) return;
                if (sheet) setHeight(Math.max(MIN_HEIGHT, drag.height + drag.y - event.clientY));
                else {
                    const limitWidth = (asideRef.current?.parentElement?.getBoundingClientRect().width ?? 0) * 0.72 || Infinity;
                    setWidth(Math.max(MIN_WIDTH, Math.min(limitWidth, drag.width + drag.x - event.clientX)));
                }
            }}
            onPointerUp={(event) => {
                dragRef.current = null;
                event.currentTarget.releasePointerCapture?.(event.pointerId);
                if (!sheet && width) saveWidth(width);
            }}
            onPointerCancel={() => { dragRef.current = null; }}
            onDoubleClick={() => { if (sheet) setHeight(null); else { setWidth(null); saveWidth(null); } }}
            onKeyDown={(event) => {
                const keys = sheet ? { ArrowUp: 16, ArrowDown: -16 } : { ArrowLeft: 16, ArrowRight: -16 };
                if (keys[event.key]) { event.preventDefault(); resizeBy(keys[event.key]); }
                else if (event.key === 'Home' || event.key === 'End') {
                    event.preventDefault();
                    resizeBy(event.key === 'Home' ? -100000 : 100000);
                }
            }} />
        <div className="df-mt-toolbar">
            <h2 className="df-mt-title">Model table</h2>
            <Segmented label="Depth" value={depth} options={DEPTHS} onChange={setDepth} />
            <Segmented label="Share" value={lens} options={LENSES} onChange={(next) => store.getState().setModelTable({ share: next })} />
            <Segmented label="Cells" value={cells} options={CELLS} onChange={(next) => { setEditing(null); store.getState().setModelTable({ cells: next }); }} />
            <button type="button" className="df-mt-close" aria-label="Close model table" title="Close model table" onClick={close}><X size={13} strokeWidth={3} /></button>
        </div>
        {metrics?.unavailable ? <p className="df-mt-empty" role="status">The table could not be computed: {metrics.error}</p> : <>
            <div className="df-mt-scroll">
                <table className="df-mt-table" role="treegrid" aria-label="Model layers" aria-readonly="true" aria-rowcount={rows.length + 1} aria-describedby={assumptionsId}>
                    <colgroup>
                        <col className="df-mt-col-layer" /><col className="df-mt-col-count" /><col className="df-mt-col-output" />
                        <col className="df-mt-col-config" /><col className="df-mt-col-params" /><col className="df-mt-col-macs" /><col className="df-mt-col-share" />
                    </colgroup>
                    <thead>
                        <tr role="row" aria-rowindex={1}>
                            <th role="columnheader" scope="col" className="df-mt-layer">Layer</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Occurrences per forward pass: repeat and parallel counts multiply">×N</th>
                            <th role="columnheader" scope="col" title="Output shape of each occurrence; B = batch">Output</th>
                            <th role="columnheader" scope="col" title="Defining settings. Symbols: as written · Values: resolved. Click a setting to edit it.">Config</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Learned parameters of every occurrence; the small figure is one instance">Params</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Multiply-accumulates per sample (B = 1): MatMul, Linear and Conv. Causal attention counts the full T × T product.">MACs</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title={`Share of the whole model: ${formatExact(modelTotal)} ${LENS_NAMES[lens]}`}>% of {lensTotal}</th>
                        </tr>
                    </thead>
                    <tbody ref={bodyRef} onKeyDown={onGridKeyDown}>
                        {visibleRows.map((row, index) => <ModelTableRow key={row.key} row={row} index={index} cells={cells} lens={lens} modelTotal={modelTotal}
                            context={context} selected={selectedKeys.has(row.key)} focusColumn={row === activeRow ? activeColumn : null}
                            editing={editing?.key === row.key ? editing : null} suggestions={editing?.key === row.key ? suggestions : null} actions={actions} />)}
                        {rows.length > limit && <tr className="df-mt-more"><td colSpan={COLUMNS.length}>
                            <button type="button" onClick={() => setLimit((current) => current + RENDER_STEP)}>
                                {rows.length - limit > RENDER_STEP ? `Show ${RENDER_STEP} more (${rows.length - limit} left)` : `Show ${rows.length - limit} more`}
                            </button>
                        </td></tr>}
                    </tbody>
                </table>
            </div>
            <div className="df-mt-focus">
                <span className="df-mt-focus-text" aria-live="polite" title={focusText}>{focusText}</span>
                <button type="button" disabled={!activeRow} onClick={() => openRow(activeRow)} title="Open this row's level and select its node">Show on canvas</button>
                <button type="button" disabled={!activeRow} onClick={() => openRow(activeRow, { settings: true })} title="Open the node's settings on the canvas">Settings</button>
            </div>
            {ready && <Footer metrics={metrics} assumptionsId={assumptionsId} />}
        </>}
    </aside>;
}
