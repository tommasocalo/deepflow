import { GitBranch, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useConnectionPlacement } from './useInspectorPlacement.js';
import './gradient-flow.css';

/** A selected connection opens its detail beside it, like a node's settings. */
function ConnectionDetail({ graph, edge, report, anchor, onClose }) {
    const patchEdge = useArchitectureStore((s) => s.patchEdge);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const panelRef = useConnectionPlacement(edge.id, anchor);
    const route = `${graph.nodes[edge.source]?.label} → ${graph.nodes[edge.target]?.label}`;
    const status = report?.edges[edge.id]?.status ?? 'unknown';
    return <aside ref={panelRef} className="df-inspector df-panel-subtle df-connection-inspector nodrag nopan nowheel"
        aria-label="Connection gradient settings" onKeyDown={(event) => event.stopPropagation()}>
        <header className="df-inspector-header">
            <GitBranch size={18} strokeWidth={2.4} aria-hidden="true" />
            <strong className="df-inspector-title" title={route}>{route}</strong>
            <button type="button" className="df-inspector-close" onClick={onClose} aria-label="Close connection settings"><X size={13} strokeWidth={3} /></button>
        </header>
        <div className="df-inspector-body">
            <label className="df-connection-field">Backward<select aria-label="Connection gradient rule" value={edge.gradientOp ?? 'none'} onChange={(event) => patchEdge(edge.id, { gradientOp: event.target.value })}>
                <option value="none">Flow</option><option value="stop">Stop gradient</option><option value="detach">Detach</option><option value="reverse">Reverse</option>
            </select></label>
            {edge.gradientOp === 'reverse' && <label className="df-connection-field">Factor<input aria-label="Gradient reversal factor" type="number" min="0" step="0.1" value={edge.gradientOpLambda ?? 1} onChange={(event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value) && value >= 0) patchEdge(edge.id, { gradientOpLambda: value });
            }} /></label>}
            <p className={`df-gradient-reason is-${status}`}>{report?.edges[edge.id]?.reason ?? 'Gradient path is unknown'}</p>
            {currentPath.length > 0 && <small className="df-connection-note">Shared template · applies to every instance</small>}
        </div>
    </aside>;
}

export default function GradientFlowControls({ graph, edge, report, anchor, onClose }) {
    return <>
        <div className="df-gradient-legend" role="note" aria-label="Gradient path legend">
            <strong>Gradient paths</strong><span title="Structural analysis. This view does not show measured gradient values.">Structure</span>
            <div><i className="is-flow" />Differentiable<i className="is-constant" />No gradient<i className="is-stopped" />Stopped<i className="is-unknown" />Unknown</div>
        </div>
        {edge && <ConnectionDetail key={edge.id} graph={graph} edge={edge} report={report} anchor={anchor} onClose={onClose} />}
    </>;
}
