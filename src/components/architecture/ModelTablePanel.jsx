import { useCallback, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, AlertTriangle, X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { isValidLevelPath, sameLevelPath } from '../../domain/architecture/levelNavigation.js';
import { ancestorKeys, buildModelTableRows, rowKey, shapeSummary } from '../../domain/modelTable/rows.js';
import { ANALYSIS_LIMITS, formatBytes, formatExact, formatMacs, formatMetric } from '../../domain/metrics/index.js';
import { parameterSuggestions } from '../../domain/parameterBindings.js';
import { useModelMetrics } from '../../hooks/useModelMetrics.js';
import ModelTableRow from './ModelTableRow.jsx';
import { applyModelTableEdit, COLUMNS, CONFIG_COLUMN, fieldTargets } from './modelTableEdits.js';
import { takeModelTableFocus } from './modelTableLoader.js';
import './model-table.css';

const RENDER_STEP = 300;
const WIDTH_KEY = 'deepflow:model-table-width';
const MIN_WIDTH = 380;
const DEFAULT_WIDTH = 760;
const MIN_HEIGHT = 160;
/** Character widths of the table's monospace fonts (10.5px settings, 10px shapes), measured in Chromium. */
const CONFIG_CHAR = 6.32;
const SHAPE_CHAR = 6.02;
const CELL_PADDING = 12;

const DEPTHS = [
    ['root', 'Root', 'Nodes of the main graph'],
    ['blocks', 'Blocks', 'Open repeats and custom components; standard modules stay one row'],
    ['ops', 'Ops', 'Every layer and operation'],
];
const LENSES = [
    ['params', 'Params', 'Share of learned parameters'],
    ['macs', 'MACs', 'Share of multiply-accumulates per sample'],
    ['activation', 'Activations', 'Share of activation memory per sample: the sum of forward op outputs (eager, no freeing), fp32. The MACs column shows bytes instead.'],
];
const CELLS = [
    ['symbols', 'Symbols', 'Settings as written, such as $d_model or 4 * d_model'],
    ['values', 'Values', 'Settings resolved to numbers'],
];
const LENS_NAMES = { params: 'parameters', macs: 'multiply-accumulates', activation: 'activation bytes' };
const MACS_TITLE = 'Multiply-accumulates per sample (B = 1): MatMul, Linear and Conv, and the products inside attention and recurrent modules. Causal attention counts the full T × T product.';
const ACTIVATION_TITLE = 'Activation memory per sample (B = 1, fp32): the sum of forward op outputs, eager, nothing freed. Standard Linear, Embedding, LayerNorm and position modules count one output; views and weights count nothing.';

/** A width the user chose earlier, or null for the default. */
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

/** Where the largest activation sits, in full: `Conv2d in ResNet stages i=2`. */
function largestLocation(largest) {
    const steps = (largest.where ?? []).filter((step) => step.repeat && step.count > 1);
    return [largest.label, ...steps.map((step) => `${step.label} i=${step.index}`)].join(' in ');
}

/** The same, compact for the footer: `Conv2d (i=2) · B×256×32×32`. */
function largestSummary(largest) {
    const indices = (largest.where ?? []).filter((step) => step.repeat && step.count > 1).map((step) => step.index);
    const occurrence = indices.length === 1 ? ` (i=${indices[0]})` : indices.length ? ` (${indices.join('.')})` : '';
    return `${largest.label}${occurrence} · ${(shapeSummary([largest.shape]).text ?? '?').replace(/ × /g, '×')}`;
}

function Footer({ metrics, assumptionsId, onShowError }) {
    const { totals, issues } = metrics;
    const unresolved = (metric) => totals.unresolved[metric];
    const partial = (metric) => unresolved(metric).size > 0;
    const unresolvedText = (metric) => `${unresolved(metric).size} row${unresolved(metric).size === 1 ? '' : 's'} unresolved`;
    const unresolvedTitle = (metric) => {
        if (!partial(metric)) return '';
        const items = [...new Map([...unresolved(metric).values()].map((item) => [`${item.label}\u001f${item.reason}`, item])).values()];
        return `Not counted:\n${items.slice(0, 8).map((item) => `${item.label}${item.reason ? `: ${item.reason}` : ''}`).join('\n')}${items.length > 8 ? '\n…' : ''}`;
    };
    // A partial total is a lower bound; with nothing counted it is unknown.
    const bounded = (metric, text) => partial(metric) ? (totals[metric] > 0 ? `≥ ${text}` : '?') : text;
    const frozen = totals.params - totals.trainable;
    const largest = totals.largest;
    const inputs = metrics.inputs.map((input) => `${input.label} ${shapeSummary([input.shape]).text ?? '?'}`).join(' · ');
    const errors = issues?.errors ?? [];
    const errorCount = errors.length + (issues?.global?.length ?? 0);
    const firstError = errors[0];
    const errorList = [...(issues?.global ?? []), ...errors.map((error) => `${error.label}: ${error.messages.join('; ')}`)];
    const limits = issues?.limits ?? {};
    // Attention and recurrent modules count their products in closed form.
    const macModules = (metrics.macModules ?? []).map((kind) => kind === 'attention' ? ', attention' : ', RNN').join('');
    return <footer className="df-mt-footer">
        {errorCount > 0 && <p className="df-mt-alert">
            <AlertCircle size={12} strokeWidth={2.6} aria-hidden="true" />
            <span className="df-mt-alert-text" title={`${errorList.slice(0, 12).join('\n')}${errorList.length > 12 ? '\n…' : ''}\n\nTotals use the declared settings. A layer with a shape error of its own, and anything whose size depends on an unknown value, is not counted (?).`}>
                <strong>{errorCount} error{errorCount === 1 ? '' : 's'}</strong> · totals use declared settings
                <span className="df-mt-alert-detail">{firstError ? ` · ${firstError.label}: ${firstError.messages[0]}` : ` · ${issues.global[0]}`}</span>
            </span>
            {firstError && <button type="button" onClick={() => onShowError(firstError)} title={`Show ${firstError.label} in the table`}>Show</button>}
        </p>}
        {(limits.budget || limits.depth) && <p className="df-mt-alert is-limit">
            <AlertTriangle size={12} strokeWidth={2.6} aria-hidden="true" />
            <span className="df-mt-alert-text" title="Validation expands every repeat and component; past this limit nothing is analysed, so counts (×N) and totals inside are lower bounds.">
                <strong>{limits.budget ? `Analysis stopped at ${ANALYSIS_LIMITS.nodes.toLocaleString('en-US')} expanded nodes` : `Components nested deeper than ${ANALYSIS_LIMITS.depth} levels are not analysed`}</strong> · counts and totals are lower bounds (≥)
            </span>
        </p>}
        <dl className="df-mt-stats">
            <Stat label="Params" value={bounded('params', formatExact(totals.params))}
                detail={partial('params') ? unresolvedText('params') : frozen ? `trainable ${formatExact(totals.trainable)} · frozen ${formatExact(frozen)}` : 'all trainable'}
                title={unresolvedTitle('params') || 'Learned parameters, every layer counted once per occurrence (no weight tying)'} />
            <Stat label="MACs / sample" value={bounded('macs', formatMacs(totals.macs) ?? '0')}
                detail={partial('macs') ? unresolvedText('macs') : formatExact(totals.macs)}
                title={unresolvedTitle('macs') || MACS_TITLE} />
            <Stat label="Largest activation" value={largest ? bounded('activation', formatBytes(largest.bytes)) : partial('activation') ? '?' : '—'}
                detail={largest ? largestSummary(largest) : partial('activation') ? unresolvedText('activation') : null}
                title={`Largest single output tensor at B = 1, fp32${largest ? `: ${largestLocation(largest)}, ${shapeSummary([largest.shape]).text ?? '?'}` : ''}.${partial('activation') ? ` ${unresolvedText('activation')}, so a larger one may exist.` : ''} All activations: ${partial('activation') ? '≥ ' : ''}${formatBytes(totals.activation) ?? '0 B'}, the sum of forward op outputs (eager, no freeing). Standard Linear, Embedding, LayerNorm and position modules count one output; attention internals count individually; views and weights count nothing.`} />
            <Stat label="Weights" value={bounded('params', formatBytes(totals.params * 4) ?? '0 B')} detail="fp32, 4 bytes each"
                title={`${formatExact(totals.params * 4)} bytes`} />
        </dl>
        <p id={assumptionsId} className="df-mt-assumptions">
            B = 1{inputs ? ` · ${inputs}` : ''} · fp32 · forward pass · MACs: MatMul, Linear, Conv{macModules} · activations: Σ op outputs, standard modules count one output · no weight tying
        </p>
    </footer>;
}

/** The whole model as a torchinfo-style table beside the canvas of one pane. */
export default function ModelTablePanel({ paneId = 'main' }) {
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
    const [scrolledX, setScrolledX] = useState(false);
    // The column sliding under the sticky Layer column: its cut text is hidden.
    const [covered, setCovered] = useState(null);
    const [columnWidths, setColumnWidths] = useState({ output: null, config: null });
    const asideRef = useRef(null);
    const bodyRef = useRef(null);
    const outputHeaderRef = useRef(null);
    const configHeaderRef = useRef(null);
    const dragRef = useRef(null);
    const rowsRef = useRef(new Map());

    const ready = !!metrics?.root;
    const rows = useMemo(() => ready ? buildModelTableRows({ metrics, rootGraph: graph, templates, depth, overrides }) : [], [ready, metrics, graph, templates, depth, overrides]);
    const byKey = useMemo(() => new Map(rows.map((row, index) => [row.key, { row, index }])), [rows]);
    const context = useMemo(() => ({ graph, templates, templateUses: metrics?.templateUses ?? {} }), [graph, templates, metrics]);
    const selectedKeys = useMemo(() => new Set(selectedNodeIds.map((id) => rowKey([...currentPath, id]))), [selectedNodeIds, currentPath]);
    // Characters that fit in the Output and Config columns; unknown until laid out.
    const budgets = useMemo(() => ({
        output: columnWidths.output ? Math.floor((columnWidths.output - CELL_PADDING) / SHAPE_CHAR) : Infinity,
        config: columnWidths.config ? Math.floor((columnWidths.config - CELL_PADDING) / CONFIG_CHAR) : Infinity,
    }), [columnWidths]);
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

    // The panel docks at the right, or becomes a bottom sheet in narrow panes.
    useEffect(() => {
        const workspace = asideRef.current?.parentElement;
        if (!workspace || typeof ResizeObserver === 'undefined') return undefined;
        const observer = new ResizeObserver(() => setSheet(getComputedStyle(workspace).flexDirection === 'column'));
        observer.observe(workspace);
        return () => observer.disconnect();
    }, []);

    // Column widths decide how much of a setting or a shape fits.
    useEffect(() => {
        const output = outputHeaderRef.current;
        const config = configHeaderRef.current;
        if (!output || !config || typeof ResizeObserver === 'undefined') return undefined;
        const observer = new ResizeObserver(() => {
            const next = { output: output.getBoundingClientRect().width || null, config: config.getBoundingClientRect().width || null };
            setColumnWidths((current) => current.output === next.output && current.config === next.config ? current : next);
        });
        observer.observe(output);
        observer.observe(config);
        return () => observer.disconnect();
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
            // An explanation belongs to the cell it was given for.
            setMessage('');
            focusCell(target.key, targetColumn);
        };
        const targets = Object.values(fieldTargets(row, cells, context));
        const editable = targets.some((target) => target.editable);
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
                // A setting that cannot be edited here says where to change it.
                else if (column === CONFIG_COLUMN && targets.length) setMessage(targets.map((target) => `${target.field.name}: ${target.hint}`).join(' · '));
                else openRow(row);
                break;
            case 'F2':
                if (editable) actions.startEdit(row);
                else if (targets.length) setMessage(targets.map((target) => `${target.field.name}: ${target.hint}`).join(' · '));
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
    // Shares of a partial total are upper bounds; the header says the denominator is a lower one.
    const modelPartial = (totals?.unresolved?.[lens]?.size ?? 0) > 0;
    // With nothing countable, the denominator is unknown rather than zero.
    const lensTotal = modelPartial && !modelTotal ? '?' : `${modelPartial ? '≥' : ''}${formatMetric(lens, modelTotal) ?? '0'}`;
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
    const metricColumn = lens === 'activation' ? 'activation' : 'macs';

    return <aside ref={asideRef} id={`df-model-table-${paneId}`} className={`df-model-table nokey${sheet ? ' is-sheet' : ''}`} aria-label="Model table"
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
                <table className="df-mt-table" role="treegrid" aria-label="Model layers" aria-rowcount={rows.length + 1} aria-describedby={assumptionsId}>
                    <colgroup>
                        <col className="df-mt-col-layer" /><col className="df-mt-col-count" /><col className="df-mt-col-output" />
                        <col className="df-mt-col-config" /><col className="df-mt-col-params" /><col className="df-mt-col-metric" /><col className="df-mt-col-share" />
                    </colgroup>
                    <thead>
                        <tr role="row" aria-rowindex={1}>
                            <th role="columnheader" scope="col" className="df-mt-layer">Layer</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Occurrences per forward pass: repeat and parallel counts multiply">×N</th>
                            <th ref={outputHeaderRef} role="columnheader" scope="col" title="Output shape of each occurrence; B = batch">Output</th>
                            <th ref={configHeaderRef} role="columnheader" scope="col" title="Defining settings. Symbols: as written · Values: resolved. Click a setting to edit it.">Config</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title="Learned parameters of every occurrence; the small figure is one instance">Params</th>
                            <th role="columnheader" scope="col" className="df-mt-num" title={metricColumn === 'activation' ? ACTIVATION_TITLE : MACS_TITLE}>
                                {metricColumn === 'activation' ? 'Act. mem' : 'MACs'}
                            </th>
                            <th role="columnheader" scope="col" className="df-mt-num df-mt-share-head"
                                title={`Share of the whole model: ${modelPartial ? 'at least ' : ''}${formatExact(modelTotal)} ${LENS_NAMES[lens]}${modelPartial ? '. Rows the table cannot count are left out, so shares are upper bounds.' : ''}`}>
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
            {ready && <Footer metrics={metrics} assumptionsId={assumptionsId} onShowError={(error) => {
                revealKey(rowKey(error.path), { focus: true });
                setMessage(`${error.label}: ${error.messages.join('; ')}`);
            }} />}
        </>}
    </aside>;
}
