import { useId, useState } from 'react';
import { GitBranch, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { resolveGradientFactor } from '../../domain/gradientFactor.js';
import { useConnectionPlacement } from './useInspectorPlacement.js';
import VariableInput from './VariableInput.jsx';
import './gradient-flow.css';

const RULES = [
    { value: 'none', label: 'Flow', icon: 'flow', hint: 'Keep the normal backward path.' },
    { value: 'stop', label: 'Stop', icon: 'stop', hint: 'Block gradients through this connection; forward values stay unchanged.' },
    { value: 'detach', label: 'Detach', icon: 'detach', hint: 'Detach the tensor from autograd. Same backward effect as Stop.' },
    { value: 'reverse', label: 'Reverse', icon: 'reverse', hint: 'Multiply the backward gradient by −λ; forward values stay unchanged.' },
];

function ReverseFactor({ factor, scope, globals, onCommit }) {
    const [draft, setDraft] = useState(null);
    const [editing, setEditing] = useState(false);
    const shown = draft ?? (factor === '' ? '' : String(factor));
    const resolved = shown ? resolveGradientFactor(shown, scope) : null;
    const symbolic = typeof shown === 'string' && /[A-Za-z_$]/.test(shown);
    const suggestions = Object.entries(scope ?? {})
        .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
        .map(([name, value]) => ({ name, value, owner: globals.has(name) ? 'Global' : 'Current level' }));
    const commit = () => {
        if (draft === null) return;
        const raw = draft.trim();
        if (!raw) { setDraft(null); return; }
        const checked = resolveGradientFactor(raw, scope);
        if (checked.error) return;
        onCommit(/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(raw) ? checked.value : raw);
        setDraft(null);
    };
    return <div className="df-gradient-factor">
        <span className="df-gradient-factor-label" title="Backward gradient is multiplied by −λ; the forward value stays unchanged">Factor λ</span>
        <VariableInput aria-label="Gradient reversal factor" type="text" inputMode="decimal" autoComplete="off" spellCheck={false} floatingSuggestions
            placeholder={factor === '' ? 'Mixed' : '1'} value={shown} suggestions={suggestions}
            aria-invalid={!!resolved?.error && !editing} onFocus={() => { setEditing(true); setDraft((current) => current ?? (factor === '' ? '' : String(factor))); }}
            onChange={(event) => setDraft(event.target.value)} onBlur={() => { setEditing(false); commit(); }}
            onKeyDown={(event) => { if (event.key === 'Enter' && event.currentTarget.getAttribute('aria-expanded') !== 'true') event.currentTarget.blur(); }} />
        {symbolic && !resolved?.error && <output className="df-gradient-factor-value" title="Resolved factor">= {resolved.value}</output>}
        {resolved?.error && !editing && <small className="df-gradient-factor-error" role="alert">{resolved.error}</small>}
    </div>;
}

/** One card edits every selected connection: shared values show, differing ones read Mixed. */
function ConnectionDetail({ graph, edges, report, anchor, onClose }) {
    const patchEdge = useArchitectureStore((s) => s.patchEdge);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const groupId = useId();
    const lead = edges.at(-1);
    const panelRef = useConnectionPlacement(lead.id, anchor);
    const many = edges.length > 1;
    const title = many ? `${edges.length} connections` : `${graph.nodes[lead.source]?.label} → ${graph.nodes[lead.target]?.label}`;
    const rules = new Set(edges.map((edge) => edge.gradientOp ?? 'none'));
    const rule = rules.size === 1 ? [...rules][0] : '';
    const factors = new Set(edges.map((edge) => edge.gradientOpLambda ?? 1));
    const factor = factors.size === 1 ? [...factors][0] : '';
    const patchAll = (patch) => recordArchitectureEdit('Change gradient flow', () => {
        for (const edge of edges) patchEdge(edge.id, patch);
    });
    const globals = new Set((rootGraph.parameters ?? []).map((parameter) => parameter.name));
    return <aside ref={panelRef} className="df-inspector df-panel-subtle df-connection-inspector nodrag nopan nowheel"
        aria-label="Connection gradient settings" onKeyDown={(event) => {
            // ReactFlow tracks additive selection from these key events even
            // when focus is still in the card after editing a connection.
            if (!['Meta', 'Control', 'Shift'].includes(event.key)) event.stopPropagation();
        }}>
        <header className="df-inspector-header">
            <GitBranch size={18} strokeWidth={2.4} aria-hidden="true" />
            <strong className="df-inspector-title" title={many ? edges.map((edge) => `${graph.nodes[edge.source]?.label} → ${graph.nodes[edge.target]?.label}`).join('\n') : title}>{title}</strong>
            <button type="button" className="df-inspector-close" onClick={onClose} aria-label="Close connection settings"><X size={13} strokeWidth={3} /></button>
        </header>
        <div className="df-inspector-body">
            <fieldset className="df-gradient-rules" aria-label="Connection gradient rule">
                {RULES.map(({ value, label, icon, hint }) => <label key={value} className="df-gradient-choice" title={hint}>
                    <input type="radio" name={groupId} value={value} checked={rule === value} onChange={() => patchAll({ gradientOp: value })} aria-label={label} aria-describedby={`${groupId}-${value}`} />
                    <img src={`/assets/gradients/${icon}.png`} width={30} height={30} alt="" draggable={false} />
                    <span>{label}</span>
                    <span id={`${groupId}-${value}`} className="df-gradient-sr-only">{hint}</span>
                </label>)}
            </fieldset>
            {rule === 'reverse' && <ReverseFactor factor={factor} scope={report?.scope ?? {}} globals={globals}
                onCommit={(value) => patchAll({ gradientOpLambda: value })} />}
            {rule === '' && <p className="df-gradient-mixed">Mixed rules</p>}
        </div>
    </aside>;
}

export default function GradientFlowControls({ graph, edges = [], report, anchor, onClose }) {
    return edges.length > 0 && <ConnectionDetail key={edges.map((edge) => edge.id).join('|')} graph={graph} edges={edges} report={report} anchor={anchor} onClose={onClose} />;
}
