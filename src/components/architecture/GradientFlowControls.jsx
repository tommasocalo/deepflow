import { X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import './gradient-flow.css';

export default function GradientFlowControls({ graph, edge, report, onClose }) {
    const patchEdge = useArchitectureStore((s) => s.patchEdge);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    return <>
        <div className="df-gradient-legend" role="note" aria-label="Gradient path legend">
            <strong>Gradient paths</strong><span title="Structural analysis. This view does not show measured gradient values.">Structure</span>
            <div><i className="is-flow" />Differentiable<i className="is-constant" />No gradient<i className="is-stopped" />Stopped<i className="is-unknown" />Unknown</div>
        </div>
        {edge && <aside className="df-gradient-inspector nodrag nopan nowheel" aria-label="Connection gradient settings" onKeyDown={(event) => event.stopPropagation()}>
            <header><strong>Connection</strong><button type="button" onClick={onClose} aria-label="Close connection settings"><X size={12} /></button></header>
            <p className="df-gradient-endpoints" title={`${graph.nodes[edge.source]?.label} → ${graph.nodes[edge.target]?.label}`}>{graph.nodes[edge.source]?.label} → {graph.nodes[edge.target]?.label}</p>
            <label>Backward<select aria-label="Connection gradient rule" value={edge.gradientOp ?? 'none'} onChange={(event) => patchEdge(edge.id, { gradientOp: event.target.value })}>
                <option value="none">Flow</option><option value="stop">Stop gradient</option><option value="detach">Detach</option><option value="reverse">Reverse</option>
            </select></label>
            {edge.gradientOp === 'reverse' && <label>Factor<input aria-label="Gradient reversal factor" type="number" min="0" step="0.1" value={edge.gradientOpLambda ?? 1} onChange={(event) => {
                const value = Number(event.target.value);
                if (Number.isFinite(value) && value >= 0) patchEdge(edge.id, { gradientOpLambda: value });
            }} /></label>}
            <p className={`df-gradient-reason is-${report?.edges[edge.id]?.status ?? 'unknown'}`}>{report?.edges[edge.id]?.reason ?? 'Gradient path is unknown'}</p>
            {currentPath.length > 0 && <small>Shared template · applies to every instance</small>}
        </aside>}
    </>;
}
