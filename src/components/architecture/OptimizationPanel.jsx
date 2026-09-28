import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { ArrowRight, BarChart3, Check, CircleAlert, Plus, RotateCcw, SlidersHorizontal, Trash2, X } from 'lucide-react';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { bestArchitectureCandidate, evaluateArchitectureCandidate, MAX_CANDIDATES, MAX_SEARCH_AXES, prepareArchitectureSearch } from '../../domain/optimization/search.js';
import './optimization-panel.css';

const integer = new Intl.NumberFormat('en-US');
const compact = new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 });
const formatCount = (value) => value === null ? 'Unknown' : integer.format(value);
const formatCompact = (value) => value === null ? '—' : compact.format(value);
const cancelRun = (counter) => { counter.current += 1; };

function suggestedAxis(parameter) {
    const value = Number(parameter?.default);
    const fallback = parameter?.type === 'float' ? 0.1 : 4;
    const center = Number.isFinite(value) && value > 0 ? value : fallback;
    const lower = parameter?.type === 'int' ? Math.max(1, Math.floor(center / 2)) : Number((center / 2).toPrecision(4));
    const step = parameter?.type === 'int' ? Math.max(1, center - lower) : Number((center / 2).toPrecision(4));
    const upper = parameter?.type === 'int' ? center + step : Number((center * 1.5).toPrecision(4));
    return { name: parameter?.name ?? '', mode: 'range', start: String(lower), end: String(upper), step: String(step), values: [...new Set([lower, center, upper])].join(', ') };
}

function AxisEditor({ axis, index, parameters, usedNames, onChange, onRemove, disabled }) {
    const parameter = parameters.find((item) => item.name === axis.name);
    const set = (patch) => onChange(index, { ...axis, ...patch });
    return <section className="df-opt-axis" aria-label={`Search parameter ${index + 1}`}>
        <div className="df-opt-axis-head"><span className="df-opt-axis-index">{String(index + 1).padStart(2, '0')}</span><strong>Parameter</strong>
            {index > 0 && <button type="button" className="df-opt-icon-button" aria-label={`Remove search parameter ${index + 1}`} title="Remove parameter" disabled={disabled} onClick={() => onRemove(index)}><Trash2 size={13} /></button>}
        </div>
        <select aria-label={`Search parameter ${index + 1}`} value={axis.name} disabled={disabled}
            onChange={(event) => onChange(index, suggestedAxis(parameters.find((item) => item.name === event.target.value)))}>
            {parameters.map((item) => <option key={item.name} value={item.name} disabled={usedNames.includes(item.name) && axis.name !== item.name}>${item.name}</option>)}
        </select>
        <div className="df-opt-axis-meta"><span>{parameter?.type === 'int' ? 'Whole number' : 'Decimal'} · current {String(parameter?.default ?? '—')}</span>
            <div className="df-opt-segment" role="group" aria-label={`Values mode for ${axis.name}`}>
                <button type="button" aria-pressed={axis.mode === 'range'} disabled={disabled} onClick={() => set({ mode: 'range' })}>Range</button>
                <button type="button" aria-pressed={axis.mode === 'list'} disabled={disabled} onClick={() => set({ mode: 'list' })}>List</button>
            </div>
        </div>
        {axis.mode === 'range' ? <div className="df-opt-range">
            <label>From<input aria-label={`${axis.name} range start`} type="number" step="any" value={axis.start} disabled={disabled} onChange={(event) => set({ start: event.target.value })} /></label>
            <label>To<input aria-label={`${axis.name} range end`} type="number" step="any" value={axis.end} disabled={disabled} onChange={(event) => set({ end: event.target.value })} /></label>
            <label>Step<input aria-label={`${axis.name} range step`} type="number" step="any" value={axis.step} disabled={disabled} onChange={(event) => set({ step: event.target.value })} /></label>
        </div> : <label className="df-opt-list-label">Values, separated by commas
            <input aria-label={`${axis.name} values`} value={axis.values} disabled={disabled} onChange={(event) => set({ values: event.target.value })} placeholder="384, 768, 1152" />
        </label>}
    </section>;
}

function CandidateRow({ row, index, maxSize, baselineSize, selected, best, budget, onClick }) {
    const withinBudget = budget === null || row.parameters !== null && row.parameters <= budget;
    const status = !row.valid ? 'Shape issue' : row.parameters === null ? 'Size unknown' : !withinBudget ? 'Over limit' : 'Valid';
    const width = row.parameters !== null && maxSize > 0 ? Math.max(2, row.parameters / maxSize * 100) : 0;
    const delta = row.parameters !== null && baselineSize ? (row.parameters / baselineSize - 1) * 100 : null;
    return <button type="button" className={`df-opt-candidate${selected ? ' is-selected' : ''}${best ? ' is-best' : ''}`} aria-pressed={selected} onClick={onClick}>
        <span className="df-opt-candidate-num">{String(index + 1).padStart(2, '0')}</span>
        <span className="df-opt-candidate-main"><span className="df-opt-candidate-settings">{Object.entries(row.values).map(([name, value]) => <span key={name}><small>${name}</small><strong>{String(value)}</strong></span>)}</span>
            <span className="df-opt-size-track"><i style={{ width: `${width}%` }} /></span></span>
        <span className="df-opt-candidate-measure"><strong>{formatCompact(row.parameters)}</strong><small>{delta === null ? 'parameters' : `${delta >= 0 ? '+' : ''}${delta.toFixed(1)}% vs current`}</small></span>
        <span className={`df-opt-candidate-status ${row.valid && withinBudget && row.parameters !== null ? 'is-valid' : 'is-issue'}`}>{best ? 'Best fit' : status}</span>
    </button>;
}

/** Bounded structural comparison. It never edits the graph until Apply. */
export default function OptimizationPanel({ open, onClose }) {
    const titleId = useId();
    const closeRef = useRef(null);
    const runId = useRef(0);
    const onCloseRef = useRef(onClose);
    const store = useArchitectureStoreApi();
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    const numeric = useMemo(() => (graph.parameters ?? []).filter((parameter) => ['int', 'float'].includes(parameter.type)), [graph.parameters]);
    const [axes, setAxes] = useState(() => {
        const first = numeric.find((parameter) => parameter.name === 'd_model') ?? numeric[0];
        return first ? [suggestedAxis(first)] : [];
    });
    const [objective, setObjective] = useState('smallest');
    const [budgetMillions, setBudgetMillions] = useState('');
    const [rows, setRows] = useState([]);
    const [selectedKey, setSelectedKey] = useState(null);
    const [filter, setFilter] = useState('all');
    const [running, setRunning] = useState(false);
    const [runTotal, setRunTotal] = useState(0);
    const [error, setError] = useState('');
    const [notice, setNotice] = useState('');
    const [completed, setCompleted] = useState(false);
    const [searchedAxes, setSearchedAxes] = useState('');
    const [source, setSource] = useState(null);

    const activeAxes = axes.length && axes.every((axis) => numeric.some((parameter) => parameter.name === axis.name))
        ? axes : numeric.length ? [suggestedAxis(numeric.find((parameter) => parameter.name === 'd_model') ?? numeric[0])] : [];
    const axesKey = JSON.stringify(activeAxes);
    const prepared = activeAxes.length ? prepareArchitectureSearch(graph, activeAxes) : { candidates: [], error: '' };
    const baseline = useMemo(() => evaluateArchitectureCandidate(graph, templates, {}), [graph, templates]);
    const budgetValue = budgetMillions.trim() === '' ? null : Number(budgetMillions) * 1e6;
    const budget = Number.isFinite(budgetValue) && budgetValue > 0 ? budgetValue : null;
    const budgetError = budgetMillions.trim() !== '' && budget === null ? 'Enter a positive parameter limit.'
        : objective === 'largest-within-budget' && budget === null ? 'Set a parameter limit for this goal.' : '';
    const referenceValues = Object.fromEntries((graph.parameters ?? []).map((parameter) => [parameter.name, parameter.default]));
    const stale = rows.length > 0 && (source?.graph !== graph || source?.templates !== templates || searchedAxes !== axesKey);
    const best = bestArchitectureCandidate(rows, { objective, budget, referenceValues });
    const eligible = rows.filter((row) => row.valid && row.parameters !== null && (budget === null || row.parameters <= budget));
    const issues = rows.length - eligible.length;
    const displayed = rows.filter((row) => filter === 'all' || (filter === 'valid' ? eligible.includes(row) : !eligible.includes(row)));
    const selected = displayed.find((row) => JSON.stringify(row.values) === selectedKey)
        ?? (best && displayed.includes(best) ? best : displayed[0]) ?? null;
    const maxSize = Math.max(baseline.parameters ?? 0, ...rows.map((row) => row.parameters ?? 0));

    useEffect(() => { onCloseRef.current = onClose; }, [onClose]);
    useEffect(() => {
        if (!open) return undefined;
        const previousFocus = document.activeElement;
        requestAnimationFrame(() => closeRef.current?.focus());
        const onEscape = (event) => {
            if (event.key !== 'Escape') return;
            event.preventDefault();
            event.stopPropagation();
            cancelRun(runId);
            setRunning(false);
            onCloseRef.current();
        };
        document.addEventListener('keydown', onEscape, true);
        return () => {
            cancelRun(runId);
            document.removeEventListener('keydown', onEscape, true);
            previousFocus?.focus?.({ preventScroll: true });
        };
    }, [open]);

    function changeAxis(index, next) {
        setAxes(activeAxes.map((axis, position) => position === index ? next : axis));
        setError('');
    }
    function addAxis() {
        const next = numeric.find((item) => !activeAxes.some((axis) => axis.name === item.name));
        if (next) setAxes([...activeAxes, suggestedAxis(next)]);
    }
    function close() {
        cancelRun(runId);
        setRunning(false);
        onClose();
    }
    async function runSearch() {
        if (prepared.error || !prepared.candidates.length || budgetError) { setError(prepared.error || budgetError || 'Add a search parameter.'); return; }
        const currentRun = ++runId.current;
        const inputGraph = graph;
        const inputTemplates = templates;
        setSource({ graph: inputGraph, templates: inputTemplates });
        setSearchedAxes(axesKey);
        setRows([]);
        setSelectedKey(null);
        setFilter('all');
        setRunTotal(prepared.candidates.length);
        setError('');
        setNotice('');
        setCompleted(false);
        setRunning(true);
        await new Promise((resolve) => requestAnimationFrame(resolve));
        const results = [];
        try {
            for (const values of prepared.candidates) {
                if (runId.current !== currentRun) return;
                const result = evaluateArchitectureCandidate(inputGraph, inputTemplates, values);
                results.push(result);
                setRows([...results]);
                await new Promise((resolve) => setTimeout(resolve, 0));
            }
            if (runId.current !== currentRun) return;
            const initial = Object.fromEntries((inputGraph.parameters ?? []).map((parameter) => [parameter.name, parameter.default]));
            const suggestion = bestArchitectureCandidate(results, { objective, budget, referenceValues: initial });
            setSelectedKey(JSON.stringify((suggestion ?? results[0])?.values));
            setCompleted(true);
            setRunning(false);
        } catch (failure) {
            if (runId.current !== currentRun) return;
            setError(`Could not compare these configurations: ${failure.message}`);
            setRunning(false);
        }
    }
    function cancelSearch() {
        cancelRun(runId);
        setRunning(false);
        setNotice(`Stopped after ${rows.length} of ${runTotal} combinations.`);
    }
    function applyCandidate() {
        if (!selected || stale || budgetError || !selected.valid || selected.parameters === null || budget !== null && selected.parameters > budget) return;
        if (store.getState().graph !== source?.graph || useTemplatesStore.getState().templates !== source?.templates) {
            setError('The architecture changed. Compare again before applying a result.'); return;
        }
        recordArchitectureEdit('Apply architecture search result', () => {
            for (const [name, value] of Object.entries(selected.values)) store.getState().patchGlobalParameter(name, { default: value });
        });
        setNotice('Values applied to the architecture. Undo restores the previous settings.');
        setError('');
    }
    function onDialogKeyDown(event) {
        if (event.key !== 'Tab') return;
        const focusable = [...event.currentTarget.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), [href]')];
        if (!focusable.length) return;
        const first = focusable[0], last = focusable.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }

    if (!open) return null;
    return createPortal(<div className="df-opt-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}>
        <section className="df-opt-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} onKeyDown={onDialogKeyDown}>
            <header className="df-opt-header">
                <span className="df-opt-mark"><BarChart3 size={22} strokeWidth={2} /></span>
                <div><span className="df-opt-kicker">ARCHITECTURE / OPTIMIZATION</span><h2 id={titleId}>Explore configurations</h2>
                    <p>Compare shape validity and model size before changing the graph.</p></div>
                <button ref={closeRef} type="button" className="df-opt-close" aria-label="Close optimization" onClick={close}><X size={18} /></button>
            </header>
            <div className="df-opt-summary" aria-label="Comparison summary">
                <div><small>Current graph</small><strong>{formatCompact(baseline.parameters)} <span>parameters</span></strong><em className={baseline.valid ? 'is-valid' : 'is-issue'}>{baseline.valid ? 'Shapes valid' : 'Shape issues'}</em></div>
                <div><small>Search space</small><strong>{prepared.error ? '—' : prepared.candidates.length} <span>combinations</span></strong><em>Up to {MAX_CANDIDATES} per run</em></div>
                <div><small>Comparison</small><strong>{running ? `${rows.length} / ${runTotal}` : completed ? `${eligible.length} fit` : 'Ready'}</strong><em>{running ? 'Checking graph shapes…' : completed ? `${issues} outside constraints or with issues` : 'No graph changes until Apply'}</em></div>
            </div>
            <div className="df-opt-workspace">
                <aside className="df-opt-controls" aria-label="Search setup">
                    <div className="df-opt-section-heading"><SlidersHorizontal size={15} /><div><strong>Search space</strong><span>Choose up to {MAX_SEARCH_AXES} global parameters</span></div></div>
                    {!numeric.length ? <p className="df-opt-empty">Add a numeric global parameter in Parameters to compare configurations.</p> : <>
                        <div className="df-opt-axes">{activeAxes.map((axis, index) => <AxisEditor key={`${index}-${axis.name}`} axis={axis} index={index} parameters={numeric} usedNames={activeAxes.map((item) => item.name)} onChange={changeAxis}
                            onRemove={(position) => setAxes(activeAxes.filter((_, i) => i !== position))} disabled={running} />)}</div>
                        {activeAxes.length < Math.min(MAX_SEARCH_AXES, numeric.length) && <button type="button" className="df-opt-add-axis" disabled={running} onClick={addAxis}><Plus size={13} /> Add parameter</button>}
                        {prepared.error && <p className="df-opt-inline-error" role="alert">{prepared.error}</p>}
                        {!prepared.error && <p className="df-opt-preview">{prepared.candidates.length} combination{prepared.candidates.length === 1 ? '' : 's'} · {prepared.axes?.map((axis) => `${axis.name}: ${axis.values.length}`).join(' × ')}</p>}
                    </>}
                    <div className="df-opt-section-heading df-opt-goal-heading"><BarChart3 size={15} /><div><strong>Goal &amp; limit</strong><span>Rank valid, countable configurations</span></div></div>
                    <div className="df-opt-goals" role="group" aria-label="Optimization goal">
                        <button type="button" aria-pressed={objective === 'smallest'} disabled={running} onClick={() => setObjective('smallest')}><strong>Smaller model</strong><span>Fewest parameters</span></button>
                        <button type="button" aria-pressed={objective === 'largest-within-budget'} disabled={running} onClick={() => setObjective('largest-within-budget')}><strong>Use budget</strong><span>Largest below limit</span></button>
                    </div>
                    <label className="df-opt-budget">Parameter limit <span>(millions)</span><input type="number" min="0.001" step="any" aria-label="Parameter limit in millions" placeholder={objective === 'largest-within-budget' ? 'Required, e.g. 200' : 'Optional'} value={budgetMillions} disabled={running} onChange={(event) => setBudgetMillions(event.target.value)} /></label>
                    {budgetError && <p className="df-opt-inline-error" role="alert">{budgetError}</p>}
                    <p className="df-opt-method">This checks tensor dimensions and estimates parameter count. It does not train candidates or predict their quality.</p>
                </aside>
                <main className="df-opt-results" aria-label="Search results">
                    {!rows.length && !running ? <div className="df-opt-start"><span><BarChart3 size={29} /></span><strong>Ready to compare</strong><p>Results will show each configuration’s shape status, estimated parameters and difference from the current graph.</p></div> : <>
                        <div className="df-opt-results-head"><div><span className="df-opt-kicker">RESULTS</span><h3>{running ? 'Checking configurations…' : 'Structural comparison'}</h3><p>{eligible.length} fit · {issues} with issues or above limit</p></div>
                            {best && <div className="df-opt-best"><small>Best fit</small><strong>{formatCompact(best.parameters)}</strong><span>{Object.entries(best.values).map(([name, value]) => `${name} ${value}`).join(' · ')}</span></div>}
                        </div>
                        {running && <div className="df-opt-progress" role="progressbar" aria-valuenow={rows.length} aria-valuemin="0" aria-valuemax={runTotal} aria-label="Configurations checked"><span style={{ width: `${runTotal ? rows.length / runTotal * 100 : 0}%` }} /></div>}
                        <div className="df-opt-filters" role="group" aria-label="Filter results">
                            <button type="button" aria-pressed={filter === 'all'} onClick={() => setFilter('all')}>All <span>{rows.length}</span></button>
                            <button type="button" aria-pressed={filter === 'valid'} onClick={() => setFilter('valid')}>Fits <span>{eligible.length}</span></button>
                            <button type="button" aria-pressed={filter === 'issues'} onClick={() => setFilter('issues')}>Issues <span>{issues}</span></button>
                            {stale && <span className="df-opt-stale"><CircleAlert size={12} /> Search space or graph changed. Compare again.</span>}
                        </div>
                        <div className="df-opt-candidates" aria-label="Candidate configurations">{displayed.length ? displayed.map((row) => <CandidateRow key={JSON.stringify(row.values)} row={row}
                            index={rows.indexOf(row)} maxSize={maxSize} baselineSize={baseline.parameters} selected={selected === row} best={best === row} budget={budget}
                            onClick={() => setSelectedKey(JSON.stringify(row.values))} />) : <p className="df-opt-no-results">No results in this filter.</p>}</div>
                        {selected && <section className="df-opt-detail" aria-label="Selected candidate details"><div className="df-opt-detail-heading"><strong>Selected configuration</strong><span>{selected.valid ? selected.parameters === null ? 'Size unavailable' : budget !== null && selected.parameters > budget ? 'Above limit' : 'Ready to apply' : 'Shape issues'}</span></div>
                            <div className="df-opt-detail-values">{Object.entries(selected.values).map(([name, value]) => <div key={name}><small>${name}</small><strong>{String(value)}</strong><span>from {String(graph.parameters.find((parameter) => parameter.name === name)?.default ?? '—')}</span></div>)}
                                <div><small>Parameters</small><strong>{formatCount(selected.parameters)}</strong><span>estimated tensors</span></div></div>
                            {!!selected.issues.length && <ul className="df-opt-issues">{selected.issues.map((issue) => <li key={issue}><CircleAlert size={12} />{issue}</li>)}</ul>}
                            {!!selected.unsupported.length && <p className="df-opt-unsupported">Size estimate unavailable: {selected.unsupported.join(', ')}</p>}
                        </section>}
                    </>}
                </main>
            </div>
            <footer className="df-opt-footer">
                <div aria-live="polite">{error ? <span className="is-error"><CircleAlert size={14} />{error}</span> : notice ? <span className="is-success"><Check size={14} />{notice}</span> : <span>Comparison is local; Apply records one Undo step.</span>}</div>
                {running ? <button type="button" className="df-opt-secondary" onClick={cancelSearch}>Stop</button>
                    : <button type="button" className="df-opt-secondary" disabled={!!prepared.error || !prepared.candidates.length || !!budgetError || !numeric.length} onClick={runSearch}>{rows.length ? <RotateCcw size={14} /> : <BarChart3 size={14} />}{rows.length ? 'Compare again' : 'Compare configurations'}</button>}
                <button type="button" className="df-opt-primary" disabled={!selected || running || stale || !!budgetError || !selected.valid || selected.parameters === null || budget !== null && selected.parameters > budget} onClick={applyCandidate}>Apply selected <ArrowRight size={14} /></button>
            </footer>
        </section>
    </div>, document.body);
}
