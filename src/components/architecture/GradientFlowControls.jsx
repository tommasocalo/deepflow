import { GitBranch, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { recordArchitectureEdit } from '../../stores/architectureHistory.js';
import { useConnectionPlacement } from './useInspectorPlacement.js';
import './gradient-flow.css';

const RULES = [['none', 'Flow'], ['stop', 'Stop gradient'], ['detach', 'Detach'], ['reverse', 'Reverse']];
const STATUS_LABELS = { flow: 'differentiable', reverse: 'reversed', stopped: 'stopped', constant: 'no gradient', unknown: 'unknown' };

/** One card edits every selected connection: shared values show, differing ones read Mixed. */
function ConnectionDetail({ graph, edges, report, anchor, onClose }) {
    const patchEdge = useArchitectureStore((s) => s.patchEdge);
    const currentPath = useArchitectureStore((s) => s.currentPath);
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
    const counts = {};
    for (const edge of edges) {
        const status = report?.edges[edge.id]?.status ?? 'unknown';
        counts[status] = (counts[status] ?? 0) + 1;
    }
    const status = Object.keys(counts).length === 1 ? Object.keys(counts)[0] : 'mixed';
    const reason = many
        ? Object.entries(counts).map(([key, count]) => `${count} ${STATUS_LABELS[key] ?? key}`).join(' · ')
        : report?.edges[lead.id]?.reason ?? 'Gradient path is unknown';
    return <aside ref={panelRef} className="df-inspector df-panel-subtle df-connection-inspector nodrag nopan nowheel"
        aria-label="Connection gradient settings" onKeyDown={(event) => event.stopPropagation()}>
        <header className="df-inspector-header">
            <GitBranch size={18} strokeWidth={2.4} aria-hidden="true" />
            <strong className="df-inspector-title" title={many ? edges.map((edge) => `${graph.nodes[edge.source]?.label} → ${graph.nodes[edge.target]?.label}`).join('\n') : title}>{title}</strong>
            <button type="button" className="df-inspector-close" onClick={onClose} aria-label="Close connection settings"><X size={13} strokeWidth={3} /></button>
        </header>
        <div className="df-inspector-body">
            <label className="df-connection-field">Backward<select aria-label="Connection gradient rule" value={rule} onChange={(event) => patchAll({ gradientOp: event.target.value })}>
                {rule === '' && <option value="" disabled>Mixed</option>}
                {RULES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select></label>
            {rule === 'reverse' && <label className="df-connection-field">Factor<input aria-label="Gradient reversal factor" type="number" min="0" step="0.1" value={factor} placeholder="Mixed" onChange={(event) => {
                const value = Number(event.target.value);
                if (event.target.value !== '' && Number.isFinite(value) && value >= 0) patchAll({ gradientOpLambda: value });
            }} /></label>}
            <p className={`df-gradient-reason is-${status}`}>{reason}</p>
            {many && <small className="df-connection-note">Shift- or ⌘-click connections to add or remove them</small>}
            {currentPath.length > 0 && <small className="df-connection-note">Shared template · applies to every instance</small>}
        </div>
    </aside>;
}

export default function GradientFlowControls({ graph, edges = [], report, anchor, onClose }) {
    return <>
        <div className="df-gradient-legend" role="note" aria-label="Gradient path legend">
            <strong>Gradient paths</strong><span title="Structural analysis. This view does not show measured gradient values.">Structure</span>
            <div><i className="is-flow" />Differentiable<i className="is-constant" />No gradient<i className="is-stopped" />Stopped<i className="is-unknown" />Unknown</div>
        </div>
        {edges.length > 0 && <ConnectionDetail key={edges.map((edge) => edge.id).join('|')} graph={graph} edges={edges} report={report} anchor={anchor} onClose={onClose} />}
    </>;
}
