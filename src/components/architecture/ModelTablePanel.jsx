import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { isValidLevelPath, sameLevelPath } from '../../domain/architecture/levelNavigation.js';
import { ancestorKeys, buildModelTableRows, rowKey } from '../../domain/modelTable/rows.js';
import { ACTIVATION_TITLE, cellStatus, knownShapeMark, MACS_TITLE } from '../../domain/modelTable/cellText.js';
import { drawnShapeLength } from '../../domain/modelTable/fitText.js';
import { formatExact, formatMetric } from '../../domain/metrics/index.js';
import { parameterSuggestions } from '../../domain/parameterBindings.js';
import { useModelMetrics } from '../../hooks/useModelMetrics.js';
import { useValidation } from '../../hooks/useValidation.js';
import { groupGraphIssues } from './graphIssues.js';
import ModelTableFooter from './ModelTableFooter.jsx';
import ModelTableRow from './ModelTableRow.jsx';
import { applyModelTableEdit, COLUMNS, CONFIG_COLUMN } from './modelTableEdits.js';
import { takeModelTableFocus } from './modelTableLoader.js';
import { useDockResize } from './useDockResize.js';
import { useGridKeyboard } from './useGridKeyboard.js';
import './model-table.css';

const RENDER_STEP = 300;
/** Monospace widths until the probes are measured (10.5px settings, 10px shapes in Chromium). */
const FALLBACK_CHARS = { config: 6.32, shape: 6.02 };
const CELL_PADDING = 12;
const PROBE_TEXT = '0'.repeat(20);

const DEPTHS = [
    ['root', 'Root', 'Nodes of the main graph'],
    ['blocks', 'Blocks', 'Open repeats and custom components; standard modules stay one row'],
    ['ops', 'Ops', 'Every layer and operation'],
];
const LENSES = [
    ['params', 'Params', 'Share of learned parameters'],
    ['macs', 'MACs', 'Share of multiply-accumulates per sample'],
    ['activation', 'Activations', 'Share of activation memory per sample: the sum of forward op outputs (eager, no freeing), fp32. The MACs column shows bytes instead.', 'Act.'],
];
const CELLS = [
    ['symbols', 'Symbols', 'Settings as written, such as $d_model or 4 * d_model'],
    ['values', 'Values', 'Settings resolved to numbers'],
];
const LENS_NAMES = { params: 'parameters', macs: 'multiply-accumulates', activation: 'activation bytes' };

/** Three choices in one row; arrow keys move within the group. */
function Segmented({ label, value, options, onChange }) {
    const select = (key, target) => {
        onChange(key);
        requestAnimationFrame(() => target?.focus());
    };
    return <div className="df-mt-segmented" role="radiogroup" aria-label={label}>
        <span className="df-mt-segmented-label" aria-hidden="true">{label}</span>
        <span className="df-mt-segments">{options.map(([key, text, title, short], index) => <button key={key} type="button" role="radio"
            aria-checked={value === key} tabIndex={value === key ? 0 : -1} title={title} onClick={() => onChange(key)}
            // The narrowest panels draw the short form; the name stays whole.
            data-short={short} aria-label={short ? text : undefined}
            onKeyDown={(event) => {
                const step = ['ArrowRight', 'ArrowDown'].includes(event.key) ? 1 : ['ArrowLeft', 'ArrowUp'].includes(event.key) ? -1 : 0;
                if (!step) return;
                event.preventDefault();
                const next = (index + step + options.length) % options.length;
                select(options[next][0], event.currentTarget.parentElement.children[next]);
            }}>{text}</button>)}</span>
    </div>;
}

/** The whole model as a torchinfo-style table beside the canvas of one pane. */
export default function ModelTablePanel({ paneId = 'main' }) {
    const store = useArchitectureStoreApi();
    const { depth, share: lens, cells } = useArchitectureStore((state) => state.modelTable);
    const currentPath = useArchitectureStore((state) => state.currentPath);
    const selectedNodeIds = useArchitectureStore((state) => state.selectedNodeIds);
    const inspectedIterations = useArchitectureStore((state) => state.inspectedIterations);
    const { graph, templates, metrics } = useModelMetrics();
    // What the canvas badge of this pane counts: the errors of the level it shows.
    const levelValidation = useValidation();
    const flow = useReactFlow();
    const assumptionsId = useId();
    const detailsId = useId();
    const [overrides, setOverrides] = useState({});
    const [limit, setLimit] = useState(RENDER_STEP);
    const [active, setActive] = useState({ key: null, column: 0 });
    const [editing, setEditing] = useState(null);
    const [message, setMessage] = useState('');
    const [scrolledX, setScrolledX] = useState(false);
    // The column sliding under the sticky Layer column: its cut text is hidden.
    const [covered, setCovered] = useState(null);
    const [columnWidths, setColumnWidths] = useState({ output: null, config: null, chars: FALLBACK_CHARS });
    // Details of the totals: open in a wide panel beside the canvas, folded in
    // narrow panels and bottom sheets, until the user chooses.
    const [detailsChoice, setDetailsChoice] = useState(null);
    const asideRef = useRef(null);
    const bodyRef = useRef(null);
    const outputHeaderRef = useRef(null);
    const configHeaderRef = useRef(null);
    const configProbeRef = useRef(null);
    const shapeProbeRef = useRef(null);
    const rowsRef = useRef(new Map());
    const contextRef = useRef({ templateUses: {} });
    const { sheet, width: panelWidth, separator, style } = useDockResize(asideRef);

    const ready = !!metrics?.root;
    const rows = useMemo(() => ready ? buildModelTableRows({ metrics, rootGraph: graph, templates, depth, overrides }) : [], [ready, metrics, graph, templates, depth, overrides]);
    const byKey = useMemo(() => new Map(rows.map((row, index) => [row.key, { row, index }])), [rows]);
    const context = useMemo(() => ({ graph, templates, templateUses: metrics?.templateUses ?? {} }), [graph, templates, metrics]);
    const selectedKeys = useMemo(() => new Set(selectedNodeIds.map((id) => rowKey([...currentPath, id]))), [selectedNodeIds, currentPath]);
    const levelErrorCount = useMemo(() => {
        const viewGraph = isValidLevelPath(graph, templates, currentPath, inspectedIterations) ? resolveGraphLevel(graph, currentPath, templates, inspectedIterations).graph : null;
        return viewGraph ? groupGraphIssues(levelValidation, viewGraph, templates).reduce((count, group) => count + group.errors.length, 0) : null;
    }, [levelValidation, graph, templates, currentPath, inspectedIterations]);
    // Characters that fit in the Output and Config columns, from the fonts as rendered; unknown until laid out.
    const budgets = useMemo(() => ({
        output: columnWidths.output ? Math.floor((columnWidths.output - CELL_PADDING) / columnWidths.chars.shape) : Infinity,
        config: columnWidths.config ? Math.floor((columnWidths.config - CELL_PADDING) / columnWidths.chars.config) : Infinity,
        // Shape separators are drawn with margins (see .df-mt-times and .df-mt-known).
        glyphs: { times: 1 + 2 / columnWidths.chars.shape, collection: 1 + 5 / columnWidths.chars.shape, mark: 4 / columnWidths.chars.shape },
    }), [columnWidths]);
    const activeEntry = byKey.get(active.key) ?? (rows.length ? { row: rows[0], index: 0 } : null);
    const activeRow = activeEntry?.row ?? null;
    const activeColumn = byKey.has(active.key) ? active.column : 0;
    const editingRow = editing ? byKey.get(editing.key)?.row ?? null : null;
    const suggestions = useMemo(() => editingRow && isValidLevelPath(graph, templates, editingRow.levelPath, inspectedIterations)
        ? parameterSuggestions(graph, editingRow.levelPath, templates, inspectedIterations, editingRow.scope) : [], [editingRow, graph, templates, inspectedIterations]);

    useLayoutEffect(() => {
        rowsRef.current = byKey;
        contextRef.current = context;
    }, [byKey, context]);

    const rowElement = useCallback((key) => [...(bodyRef.current?.children ?? [])].find((element) => element.dataset.rowKey === key), []);
    const focusCell = useCallback((key, column) => rowElement(key)?.querySelector(`[data-col="${column}"]`)?.focus({ preventScroll: false }), [rowElement]);
    const canvasElement = useCallback(() => asideRef.current?.parentElement?.querySelector('[data-architecture-canvas]'), []);

    /** Pan the canvas only when the node is outside it. */
    const reveal = useCallback((nodeId, delay = 0) => {
        const run = () => {
            const bounds = canvasElement()?.getBoundingClientRect?.();
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
    }, [canvasElement, flow, store]);

    /** Escape from the grid: back to the canvas, on the selected node when it is drawn. */
    const focusCanvas = useCallback(() => {
        const canvas = canvasElement();
        if (!canvas) return;
        const selected = store.getState().selectedNodeIds;
        const node = [...canvas.querySelectorAll('.react-flow__node')].find((element) => selected.includes(element.dataset.id));
        (node ?? canvas).focus({ preventScroll: true });
        setMessage('');
    }, [canvasElement, store]);

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

    /** Bring a row into view: open its ancestors, extend the rendered rows, scroll to it. */
    const revealKey = useCallback((key, { focus = false } = {}) => {
        if (!rowsRef.current.has(key)) {
            const ancestors = ancestorKeys(JSON.parse(key));
            setOverrides((current) => ancestors.every((ancestor) => current[ancestor] === true) ? current
                : { ...current, ...Object.fromEntries(ancestors.map((ancestor) => [ancestor, true])) });
        }
        setActive((current) => current.key === key ? current : { key, column: focus ? 0 : current.column });
        requestAnimationFrame(() => requestAnimationFrame(() => {
            const index = rowsRef.current.get(key)?.index;
            if (index === undefined) return;
            setLimit((current) => index < current ? current : Math.ceil((index + 1) / RENDER_STEP) * RENDER_STEP);
            requestAnimationFrame(() => {
                const element = rowElement(key);
                element?.scrollIntoView?.({ block: 'nearest' });
                if (focus) element?.querySelector('[data-col="0"]')?.focus({ preventScroll: true });
            });
        }));
    }, [rowElement]);

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
            const result = edits?.length ? applyModelTableEdit(store, row, edits, { templateUses: contextRef.current.templateUses }) : { written: 0 };
            setEditing(null);
            // The confirmation names the value that changed, as the undo step does.
            setMessage(result.written ? `${result.label} · ⌘Z / Ctrl+Z to undo` : edits ? 'No change' : 'Edit cancelled');
            if (refocus) focusCell(row.key, CONFIG_COLUMN);
        },
        message: setMessage,
    }), [focusCell, openRow, reveal, store, toggle]);

    const onGridKeyDown = useGridKeyboard({ rows, byKey, limit, editing, cells, context, actions, openRow, focusCell, setActive, setMessage, focusCanvas });

    // Canvas → table: reveal and scroll to the selected node's row. The table
    // never takes focus from the canvas.
    useEffect(() => store.subscribe((state, previous) => {
        if (state.selectedNodeIds === previous.selectedNodeIds && state.currentPath === previous.currentPath) return;
        const id = state.selectedNodeIds[0];
        if (!id) return;
        revealKey(rowKey([...state.currentPath, id]));
    }), [revealKey, store]);

    // Opened from the toggle: focus the active cell, so the grid is one step
    // away. The cells exist once mounted; focus already inside stays put.
    useEffect(() => {
        if (!takeModelTableFocus(paneId)) return;
        const aside = asideRef.current;
        if (!aside || aside.contains(document.activeElement)) return;
        (aside.querySelector('[role="treegrid"] [data-col][tabindex="0"]') ?? aside.querySelector('[role="radio"][aria-checked="true"]'))?.focus();
    }, [paneId]);

    // Column widths and the fonts as rendered decide how much of a setting or a shape fits.
    useEffect(() => {
        const output = outputHeaderRef.current;
        const config = configHeaderRef.current;
        if (!output || !config || typeof ResizeObserver === 'undefined') return undefined;
        const charWidth = (probe, fallback) => (probe?.getBoundingClientRect().width ?? 0) / PROBE_TEXT.length || fallback;
        const update = () => {
            const next = {
                output: output.getBoundingClientRect().width || null,
                config: config.getBoundingClientRect().width || null,
                chars: { config: charWidth(configProbeRef.current, FALLBACK_CHARS.config), shape: charWidth(shapeProbeRef.current, FALLBACK_CHARS.shape) },
            };
            setColumnWidths((current) => current.output === next.output && current.config === next.config
                && current.chars.config === next.chars.config && current.chars.shape === next.chars.shape ? current : next);
        };
        const observer = new ResizeObserver(update);
        observer.observe(output);
        observer.observe(config);
        // A web font that arrives late changes every width.
        let live = true;
        document.fonts?.ready?.then(() => { if (live) update(); });
        return () => {
            live = false;
            observer.disconnect();
        };
    }, [ready]);

    const setDepth = (next) => {
        store.getState().setModelTable({ depth: next });
        setOverrides({});
        setLimit(RENDER_STEP);
        setEditing(null);
        setMessage('');
    };
    const close = () => {
        store.getState().setShowModelTable(false);
        // Back to the Table toggle in the ribbon, which every pane shares.
        requestAnimationFrame(() => document.querySelector('[data-model-table-toggle]')?.focus());
    };

    const totals = metrics?.totals;
    const modelTotal = totals?.[lens] ?? 0;
    // Shares of a partial total are upper bounds; the header says the denominator is a lower one.
    const modelPartial = (totals?.unresolved?.[lens]?.size ?? 0) > 0;
    // With nothing countable, the denominator is unknown rather than zero.
    const lensTotal = modelPartial && !modelTotal ? '?' : `${modelPartial ? '≥' : ''}${formatMetric(lens, modelTotal) ?? '0'}`;
    const visibleRows = useMemo(() => rows.slice(0, limit), [rows, limit]);
    // The Output column grows (within bounds, see the CSS) to fit its longest label and mark.
    const outputNeed = useMemo(() => visibleRows.reduce((longest, row) => {
        if (row.shape.unknown) return longest;
        const mark = knownShapeMark(row);
        return Math.max(longest, drawnShapeLength(row.shape.text, budgets.glyphs) + (mark ? mark.length + budgets.glyphs.mark : 0));
    }, 0), [visibleRows, budgets]);
    const breadcrumb = activeRow ? ['main', ...ancestorKeys(activeRow.path).map((key) => byKey.get(key)?.row.label ?? '…'), activeRow.label].join(' › ') : '';
    const sameLevel = activeRow && sameLevelPath(activeRow.levelPath, currentPath);
    const owner = activeRow?.ownerIds.length === 1 ? templates[activeRow.ownerIds[0]] : null;
    const uses = owner ? metrics?.templateUses?.[owner.id] ?? 0 : 0;
    // The active cell's details (per-instance figures, why a value is unknown) read here as well as on hover.
    const detail = activeRow ? cellStatus(activeRow, activeColumn, { lens, modelTotal, modelPartial }) : null;
    const rowText = activeRow ? [breadcrumb, activeRow.typeName, activeRow.multiplicity > 1 ? `×${activeRow.multiplicity}` : null,
        owner ? `in the shared ${owner.name} definition${uses > 1 ? ` (${uses} uses)` : ''}` : null,
        sameLevel ? 'on this canvas' : 'Enter or double-click opens its level'].filter(Boolean).join(' · ') : 'Select a row';
    const focusText = message || (editing ? 'Enter apply · Esc cancel · Tab next field'
        : detail ? `${activeColumn === 0 ? breadcrumb : activeRow.label} · ${detail}` : rowText);
    const metricColumn = lens === 'activation' ? 'activation' : 'macs';
    const detailsOpen = detailsChoice ?? (!sheet && (panelWidth ?? Infinity) >= 700);

    return <aside ref={asideRef} id={`df-model-table-${paneId}`} className={`df-model-table nokey${sheet ? ' is-sheet' : ''}`} aria-label="Model table" style={style}
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
        <div className="df-mt-toolbar">
            <h2 className="df-mt-title">Model table</h2>
            <Segmented label="Depth" value={depth} options={DEPTHS} onChange={setDepth} />
            <Segmented label="Share" value={lens} options={LENSES} onChange={(next) => { setMessage(''); store.getState().setModelTable({ share: next }); }} />
            <Segmented label="Cells" value={cells} options={CELLS} onChange={(next) => { setEditing(null); setMessage(''); store.getState().setModelTable({ cells: next }); }} />
            <button type="button" className="df-mt-close" aria-label="Close model table" title="Close model table" onClick={close}><X size={13} strokeWidth={3} /></button>
        </div>
        {metrics?.unavailable ? <p className="df-mt-empty" role="status">The table could not be computed: {metrics.error}</p> : <>
            <div className={`df-mt-scroll${scrolledX ? ' is-scrolled-x' : ''}`} data-covered={covered ?? undefined}
                onScroll={(event) => {
                    const element = event.currentTarget;
                    const next = element.scrollLeft > 0;
                    setScrolledX((current) => current === next ? current : next);
                    // A cell half under the Layer column would show digits of another number.
                    const heads = element.querySelectorAll('thead th');
                    const edge = element.scrollLeft + (heads[0]?.offsetWidth ?? 0);
                    let under = null;
                    for (let index = 1; index < heads.length; index++) {
                        if (next && heads[index].offsetLeft < edge && heads[index].offsetLeft + heads[index].offsetWidth > edge) under = index;
                    }
                    setCovered((current) => current === under ? current : under);
                }}>
                <table className={`df-mt-table is-${metricColumn}`} role="treegrid" aria-label="Model layers" aria-rowcount={rows.length + 1} aria-describedby={assumptionsId}
                    // Whole characters, so the budget measured from the column covers the longest label.
                    style={{ '--mt-output-need': `${Math.ceil(Math.ceil(outputNeed) * columnWidths.chars.shape) + CELL_PADDING + 1}px` }}>
                    <colgroup>
                        <col className="df-mt-col-layer" /><col className="df-mt-col-count" /><col className="df-mt-col-output" />
                        <col className="df-mt-col-config" /><col className="df-mt-col-params" /><col className="df-mt-col-metric" /><col className="df-mt-col-share" />
                    </colgroup>
                    <thead>
                        <tr role="row" aria-rowindex={1}>
                            <th role="columnheader" scope="col" className="df-mt-layer" title="Esc returns to the canvas">Layer</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Occurrences per forward pass: repeat and parallel counts multiply">×N</th>
                            <th ref={outputHeaderRef} role="columnheader" scope="col" title="Output shape of each occurrence; B = batch">Output</th>
                            <th ref={configHeaderRef} role="columnheader" scope="col" title="Defining settings. Symbols: as written · Values: resolved. Click a setting to edit it.">Config</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Learned parameters of every occurrence; the small figure is one instance">Params</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title={metricColumn === 'activation' ? ACTIVATION_TITLE : MACS_TITLE}>
                                {/* The sum of op outputs, not what training keeps for backward. */}
                                {metricColumn === 'activation' ? 'Σ outputs' : 'MACs'}
                            </th>
                            <th role="columnheader" scope="col" className="df-mt-num df-mt-share-head"
                                title={`Share of the whole model: ${modelPartial ? 'at least ' : ''}${formatExact(modelTotal)} ${LENS_NAMES[lens]}${modelPartial ? '. Rows the table cannot count are left out, so shares of exact rows are upper bounds (≤) and shares of partial rows are unknown.' : ''}`}>
                                % of {lensTotal}
                            </th>
                        </tr>
                    </thead>
                    <tbody ref={bodyRef} onKeyDown={onGridKeyDown}>
                        {visibleRows.map((row, index) => <ModelTableRow key={row.key} row={row} index={index} cells={cells} lens={lens} modelTotal={modelTotal} modelPartial={modelPartial}
                            context={context} selected={selectedKeys.has(row.key)} focusColumn={row === activeRow ? activeColumn : null} budgets={budgets}
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
                <span className="df-mt-focus-text" title={focusText}>{focusText}</span>
                <button type="button" disabled={!activeRow} onClick={() => openRow(activeRow)} title="Open this row's level and select its node">Show on canvas</button>
                <button type="button" disabled={!activeRow} onClick={() => openRow(activeRow, { settings: true })} title="Open the node's settings on the canvas">Settings</button>
            </div>
            {/* Results of edits and explanations are announced; moving between cells is not. */}
            <span className="sr-only" role="status" aria-live="polite">{message}</span>
            {ready && <ModelTableFooter metrics={metrics} levelErrorCount={levelErrorCount} assumptionsId={assumptionsId} detailsId={detailsId}
                detailsOpen={detailsOpen} onToggleDetails={() => setDetailsChoice(!detailsOpen)}
                onShowError={(error) => {
                    revealKey(rowKey(error.path), { focus: true });
                    setMessage(`${error.label}: ${error.messages.join('; ')}`);
                }} />}
        </>}
        {/* Probes for the monospace widths the cells are drawn in. */}
        <span ref={configProbeRef} className="df-mt-probe df-mt-config-cell" aria-hidden="true">{PROBE_TEXT}</span>
        <span ref={shapeProbeRef} className="df-mt-probe df-mt-shape" aria-hidden="true">{PROBE_TEXT}</span>
        {/* Last in the Tab order: the toolbar and the grid come first. */}
        <div className="df-mt-resizer" {...separator}><span className="df-mt-grip" aria-hidden="true" /></div>
    </aside>;
}
