import { useId } from 'react';
import { GitBranch, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { useConnectionPlacement } from './useInspectorPlacement.js';
import './gradient-flow.css';

const RULES = [
    { value: 'none', label: 'Flow', icon: 'flow', hint: 'Keep the normal backward path.' },
    { value: 'stop', label: 'Stop', icon: 'stop', hint: 'Block gradients through this connection; forward values stay unchanged.' },
    { value: 'detach', label: 'Detach', icon: 'detach', hint: 'Detach the tensor from autograd. Same backward effect as Stop.' },
    { value: 'reverse', label: 'Reverse', icon: 'reverse', hint: 'Multiply the backward gradient by −λ; forward values stay unchanged.' },
];
const STATUS_LABELS = { flow: 'Differentiable', reverse: 'Reversed', stopped: 'Stopped', constant: 'No gradient', unknown: 'Unknown' };

/** One card edits every selected connection: shared values show, differing ones read Mixed. */
function ConnectionDetail({ graph, edges, report, anchor, onClose }) {
    const patchEdge = useArchitectureStore((s) => s.patchEdge);
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
    const statuses = new Map();
    for (const edge of edges) {
        const { status = 'unknown', reason = 'Gradient path is unknown' } = report?.edges[edge.id] ?? {};
        const reasons = statuses.get(status) ?? [];
        reasons.push(many ? `${graph.nodes[edge.source]?.label} → ${graph.nodes[edge.target]?.label}: ${reason}` : reason);
        statuses.set(status, reasons);
    }
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
            <div className="df-gradient-footer">
                {rule === 'reverse' && <label className="df-connection-field"><span title="Gradient reversal factor">Factor λ</span><input aria-label="Gradient reversal factor" type="number" min="0" step="0.1" value={factor} placeholder="Mixed" onChange={(event) => {
                    const value = Number(event.target.value);
                    if (event.target.value !== '' && Number.isFinite(value) && value >= 0) patchAll({ gradientOpLambda: value });
                }} /></label>}
                <div className="df-gradient-statuses" aria-label="Gradient path status" aria-live="polite">
                    {rule === '' && <span className="df-gradient-mixed">Mixed rules</span>}
                    {[...statuses].map(([status, reasons]) => <span key={status} className={`df-gradient-status is-${status}`} title={reasons.join('\n')}>
                        <i aria-hidden="true" />{many && `${reasons.length} `}{STATUS_LABELS[status] ?? status}
                    </span>)}
                </div>
            </div>
        </div>
    </aside>;
}

export default function GradientFlowControls({ graph, edges = [], report, anchor, onClose }) {
    return edges.length > 0 && <ConnectionDetail key={edges.map((edge) => edge.id).join('|')} graph={graph} edges={edges} report={report} anchor={anchor} onClose={onClose} />;
}
