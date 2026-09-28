import { useEffect, useMemo, useRef, useState } from 'react';
import { Search, X } from 'lucide-react';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { bestArchitectureCandidate, evaluateArchitectureCandidate, prepareArchitectureSearch } from '../../domain/optimization/search.js';
import './optimization-panel.css';

const formatCount = (value) => value === null ? '—' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 }).format(value / 1e6) + 'M';
const suggestedValues = (parameter) => {
    const value = Number(parameter?.default);
    if (!Number.isFinite(value) || value === 0) return '1, 2, 4';
    const lower = parameter.type === 'int' ? Math.max(1, Math.round(value / 2)) : Number((value / 2).toPrecision(4));
    const upper = parameter.type === 'int' ? Math.round(value * 1.5) : Number((value * 1.5).toPrecision(4));
    return [...new Set([lower, value, upper])].join(', ');
};

/** A local structural search: validate a small grid and compare model size. */
export default function OptimizationPanel({ onClose }) {
    const store = useArchitectureStoreApi();
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    const numeric = useMemo(() => (graph.parameters ?? []).filter((parameter) => ['int', 'float'].includes(parameter.type)), [graph.parameters]);
    const [first, setFirst] = useState(() => numeric.find((parameter) => parameter.name === 'd_model')?.name ?? numeric[0]?.name ?? '');
    const [second, setSecond] = useState('');
    const [firstValues, setFirstValues] = useState(() => suggestedValues(numeric.find((parameter) => parameter.name === 'd_model') ?? numeric[0]));
    const [secondValues, setSecondValues] = useState('');
    const [objective, setObjective] = useState('smallest');
    const [budgetMillions, setBudgetMillions] = useState('200');
    const [rows, setRows] = useState([]);
    const [error, setError] = useState('');
    const [notice, setNotice] = useState('');
    const [running, setRunning] = useState(false);
    const runId = useRef(0);
    const sourceGraph = useRef(null);
    useEffect(() => () => { runId.current++; }, []);
    const budget = objective === 'largest-within-budget' ? Number(budgetMillions) * 1e6 : null;
    const best = bestArchitectureCandidate(rows, { objective, budget: Number.isFinite(budget) && budget > 0 ? budget : null });

    async function runSearch() {
        const axes = [{ name: first, values: firstValues }, ...(second ? [{ name: second, values: secondValues }] : [])];
        const prepared = prepareArchitectureSearch(graph, axes);
        if (prepared.error) { setError(prepared.error); return; }
        if (objective === 'largest-within-budget' && (!Number.isFinite(budget) || budget <= 0)) {
            setError('Enter a positive parameter budget in millions.'); return;
        }
        const currentRun = ++runId.current;
        sourceGraph.current = graph;
        setRows([]);
        setError('');
        setNotice('');
        setRunning(true);
        await new Promise((resolve) => requestAnimationFrame(resolve));
        for (const values of prepared.candidates) {
            if (runId.current !== currentRun) return;
            const result = evaluateArchitectureCandidate(graph, templates, values);
            setRows((previous) => [...previous, result]);
            // Give the panel and the main graph a chance to repaint between
            // candidates rather than validating the full search in one frame.
            await new Promise((resolve) => setTimeout(resolve, 0));
        }
        if (runId.current === currentRun) setRunning(false);
    }

    function applyCandidate(row) {
        if (store.getState().graph !== sourceGraph.current) {
            setError('The architecture changed. Run the search again before applying a result.');
            return;
        }
        recordArchitectureEdit('Apply architecture search result', () => {
            for (const [name, value] of Object.entries(row.values)) store.getState().patchGlobalParameter(name, { default: value });
        });
        sourceGraph.current = store.getState().graph;
        setError('');
        setNotice('Applied to the architecture. Undo restores the previous values.');
    }

    return <div className="df-opt-panel" onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}>
        <header><Search size={16} aria-hidden="true" /><strong>Optimization · structure search</strong>
            <button type="button" aria-label="Close optimization" onClick={onClose}><X size={14} /></button></header>
        <p className="df-opt-intro">Compare global parameter settings by tensor validity and model size.</p>
        {!numeric.length ? <p className="df-opt-empty">Add a numeric global parameter in Parameters to create a search space.</p> : <>
            <div className="df-opt-grid">
                <label>Parameter<select aria-label="First search parameter" value={first} onChange={(event) => {
                    setFirst(event.target.value);
                    setFirstValues(suggestedValues(numeric.find((parameter) => parameter.name === event.target.value)));
                }}>{numeric.map((parameter) => <option key={parameter.name} value={parameter.name}>${parameter.name}</option>)}</select></label>
                <label>Values<input aria-label="First search values" value={firstValues} onChange={(event) => setFirstValues(event.target.value)} placeholder="384, 768, 1152" /></label>
                <label>Second parameter<select aria-label="Second search parameter" value={second} onChange={(event) => {
                    setSecond(event.target.value);
                    setSecondValues(event.target.value ? suggestedValues(numeric.find((parameter) => parameter.name === event.target.value)) : '');
                }}><option value="">None</option>{numeric.map((parameter) => <option key={parameter.name} value={parameter.name}>${parameter.name}</option>)}</select></label>
                <label>Values<input aria-label="Second search values" value={secondValues} onChange={(event) => setSecondValues(event.target.value)} disabled={!second} placeholder="6, 12, 18" /></label>
            </div>
            <div className="df-opt-action-row">
                <label>Goal<select aria-label="Search goal" value={objective} onChange={(event) => setObjective(event.target.value)}>
                    <option value="smallest">Smallest valid model</option><option value="largest-within-budget">Largest under budget</option>
                </select></label>
                {objective === 'largest-within-budget' && <label>Budget (M)<input aria-label="Parameter budget in millions" type="number" min="0.001" step="any" value={budgetMillions} onChange={(event) => setBudgetMillions(event.target.value)} /></label>}
                <button type="button" className="df-opt-run" disabled={running} onClick={runSearch}>{running ? `Checking ${rows.length}…` : 'Compare'}</button>
            </div>
            {error && <p className="df-opt-feedback is-error" role="alert">{error}</p>}
            {notice && <p className="df-opt-feedback" role="status">{notice}</p>}
            {rows.length > 0 && <div className="df-opt-results"><table><thead><tr><th>Settings</th><th>Parameters</th><th>Graph</th><th /></tr></thead>
                <tbody>{rows.map((row) => <tr key={JSON.stringify(row.values)} className={best === row ? 'is-best' : ''}>
                    <td>{Object.entries(row.values).map(([name, value]) => <span key={name}><code>{name}</code> {value}</span>)}</td>
                    <td title={row.unsupported.join('\n') || undefined}>{formatCount(row.parameters)}</td>
                    <td title={row.issues.join('\n') || row.unsupported.join('\n') || undefined}>{row.valid ? 'Valid' : 'Issues'}</td>
                    <td><button type="button" onClick={() => applyCandidate(row)} disabled={!row.valid}>Apply</button></td>
                </tr>)}</tbody></table></div>}
            {rows.length > 0 && !running && <p className="df-opt-footnote">{best ? `Suggested: ${Object.entries(best.values).map(([name, value]) => `${name} ${value}`).join(' · ')}` : 'No valid result with a complete size estimate.'} Size counts separate tensors in this graph; quality requires training.</p>}
        </>}
    </div>;
}
