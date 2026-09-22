import { useComponentTemplate } from '../../hooks/useComponentTemplate.js';
import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Boxes, ChevronRight, GitFork, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { resolveRepeatCount } from '../../domain/expressions.js';
import { resolveComponentContext } from '../../domain/componentContext.js';
import { isInteractive } from './nodeGeometry.js';
import { inputPortsFor } from './nodeInputPorts.js';
import { useOpenToSideNode } from './OpenToSideContext.jsx';
import NodeInputPorts from './NodeInputPorts.jsx';
import PortShapeLabels from './PortShapeLabels.jsx';
import './parallel-preview.css';

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
    const ports = inputPortsFor(node, templates);
    const template = useComponentTemplate(node);
    const { count } = template && context?.graph
        ? resolveComponentContext(node, template, context.graph, report.shapes, context.scope)
        : resolveRepeatCount(node.count, context?.scope);
    const sideMenuProps = useOpenToSideNode([...currentPath, id], { disabled: !template });
    const tone = report.errors[id]?.length ? 'error' : 'ok';
    const localCount = Object.entries(node.structureOverrides ?? {}).filter(([index, override]) => override?.enabled && Number(index) < count).length;
    const instances = count > 3 ? [0, 1, count - 1] : Array.from({ length: count || 0 }, (_, index) => index);
    // The drawing represents the collection; its footprint is independent of
    // instance count. Open the definition to inspect any individual instance.
    const width = Math.max(316, ports.length * 116);
    const open = (event, index) => { event.stopPropagation(); enterRepeatIteration(id, index); };
    return (
        <div {...sideMenuProps} className={`df-parallel df-parallel--compact df-parallel--${tone}${selected ? ' is-selected' : ''}${expandedNodeId === id ? ' is-active' : ''}`}
            style={{ width, '--parallel-port-width': `${width / Math.max(1, ports.length)}px` }} onClick={(event) => { if (!isInteractive(event.target)) toggleExpanded(id); }}
            role="button" tabIndex={0} aria-label={`Inspect ${node.label}`} aria-expanded={expandedNodeId === id}
            onKeyDown={(event) => {
                sideMenuProps.onKeyDown?.(event);
                if (!event.defaultPrevented && event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) { event.preventDefault(); event.stopPropagation(); toggleExpanded(id); }
            }}
            onDoubleClick={(event) => { if (!isInteractive(event.target)) open(event, 0); }}>
            <div className="df-parallel-header">
                <GitFork size={14} strokeWidth={2.7} /><strong title={node.label}>{node.label}</strong>
                <span className="df-repeat-count" title={`${count ?? 'Unresolved'} parallel instances${typeof node.count === 'string' ? ` · ${node.count}` : ''}`}>× {count ?? '?'}</span>
                <button className="nodrag df-container-open" aria-label={`Open ${node.label}`} title="Open definition" disabled={!template} onClick={(event) => open(event, 0)}><ChevronRight size={14} strokeWidth={3} /></button>
            </div>
            <div className="df-parallel-preview" aria-label={`${count ?? 'Unresolved'} parallel instances`}>
                {instances.map((index, order) => <span key={index} className="df-parallel-preview-item">
                    {count > 3 && order === 2 && <span className="df-parallel-ellipsis" aria-hidden="true">···</span>}
                    <button type="button" className="nodrag df-parallel-instance" disabled={!template}
                        style={{ '--parallel-copy-offset': `${order * 3}px` }}
                        aria-label={`Open ${node.label} instance ${index + 1}`}
                        title={`${template?.name ?? 'Template'} · instance ${index + 1}. Open the definition to inspect any instance.`}
                        onMouseDown={(event) => event.stopPropagation()} onClick={(event) => open(event, index)}>
                        <Boxes size={15} strokeWidth={2.4} /><b>{index + 1}</b>
                    </button>
                </span>)}
                {!count && <span className="df-parallel-invalid">Set a valid count</span>}
            </div>
            <div className="df-parallel-footer"><span>{localCount ? `Parallel · ${localCount} local` : 'Parallel · shared structure'}</span><span>Collection ↓</span></div>
            <button aria-label={`Delete ${node.label}`} onClick={(event) => { event.stopPropagation(); removeNode(id); }} className="nodrag df-block-delete"><X size={10} strokeWidth={3} /></button>
            <PortShapeLabels node={node} shapes={report.shapes} hideInput={ports.length > 1} />
            <NodeInputPorts node={node} shapes={report.shapes} />
            <Handle type="source" position={Position.Bottom} className="df-handle" />
        </div>
    );
}

export default memo(ParallelRepeatNodeView);
