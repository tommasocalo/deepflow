import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Boxes, ChevronRight, GitFork, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { resolveRepeatCount } from '../../domain/expressions.js';
import { isInteractive } from './nodeGeometry.js';
import { inputPortsFor } from './nodeInputPorts.js';
import { useOpenToSideNode } from './OpenToSideContext.jsx';
import NodeInputPorts from './NodeInputPorts.jsx';
import PortShapeLabels from './PortShapeLabels.jsx';

function ParallelRepeatNodeView({ id, data, selected }) {
    const node = data.node;
    const templates = useTemplatesStore((s) => s.templates);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const enterRepeatIteration = useArchitectureStore((s) => s.enterRepeatIteration);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const context = useValidationContext();
    const report = useValidation();
    const { count } = resolveRepeatCount(node.count, context?.scope);
    const ports = inputPortsFor(node, templates);
    const template = templates[node.templateId];
    const sideMenuProps = useOpenToSideNode([...currentPath, id], { disabled: !template });
    const tone = report.errors[id]?.length ? 'error' : 'ok';
    const shown = Math.min(count || 0, 24);
    const width = Math.max(ports.length * 220, Math.min(12, shown) * 49 + 24, 260);
    const open = (event, index) => { event.stopPropagation(); enterRepeatIteration(id, index); };
    return (
        <div {...sideMenuProps} className={`df-parallel df-parallel--${tone}${selected ? ' is-selected' : ''}${expandedNodeId === id ? ' is-active' : ''}`}
            style={{ width }} onClick={(event) => { if (!isInteractive(event.target)) toggleExpanded(id); }}
            role="button" tabIndex={0} aria-label={`Inspect ${node.label}`} aria-expanded={expandedNodeId === id}
            onKeyDown={(event) => {
                sideMenuProps.onKeyDown?.(event);
                if (!event.defaultPrevented && event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) { event.preventDefault(); event.stopPropagation(); toggleExpanded(id); }
            }}
            onDoubleClick={(event) => { if (!isInteractive(event.target)) open(event, 0); }}>
            <div className="df-parallel-header">
                <GitFork size={14} strokeWidth={2.7} /><strong>{node.label}</strong>
                <span className="df-parallel-mode">Parallel</span>
                <span className="df-repeat-count">{typeof node.count === 'string' && node.count.startsWith('$') ? `${node.count} = ` : '× '}{count ?? '?'}</span>
                <button className="nodrag df-container-open" aria-label={`Open ${node.label}`} title="Open definition" disabled={!template} onClick={(event) => open(event, 0)}><ChevronRight size={14} strokeWidth={3} /></button>
            </div>
            <div className="df-parallel-branches" style={{ gridTemplateColumns: `repeat(${Math.min(12, shown) || 1}, 1fr)` }}>
                {Array.from({ length: shown }, (_, index) => <button type="button" key={index} className="nodrag df-parallel-instance" aria-label={`Open ${node.label} instance ${index + 1}`} title={`${template?.name ?? 'Template'} · instance ${index + 1} · formula index ${node.iterationVariable || 'i'} = ${index}`} onMouseDown={(event) => event.stopPropagation()} onClick={(event) => open(event, index)}><Boxes size={14} strokeWidth={2.4} /><b>{index + 1}</b></button>)}
                {!count && <span className="df-parallel-invalid">Set a valid count</span>}
            </div>
            <div className="df-parallel-footer">{count > shown ? `${shown} of ${count} shown · open a branch to inspect more` : 'Shared structure · separate weights'}<span>Tensor collection ↓</span></div>
            <button aria-label={`Delete ${node.label}`} onClick={(event) => { event.stopPropagation(); removeNode(id); }} className="nodrag df-block-delete"><X size={10} strokeWidth={3} /></button>
            <PortShapeLabels node={node} shapes={report.shapes} hideInput={ports.length > 1} />
            <NodeInputPorts node={node} shapes={report.shapes} />
            <Handle type="source" position={Position.Bottom} className="df-handle" />
        </div>
    );
}

export default memo(ParallelRepeatNodeView);
