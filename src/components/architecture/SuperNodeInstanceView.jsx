import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Boxes, ChevronRight, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { CONTAINER_WIDTH, isInteractive } from './nodeGeometry.js';

/** A supernode instance: hatched container, label, "open" affordance. */
const SuperNodeInstanceView = ({ id, data, selected }) => {
    const node = data.node;
    const template = useTemplatesStore((s) => s.templates[node.templateId]);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const validation = useValidation();
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const tone = errors.length ? 'error' : warnings.length ? 'warning' : 'ok';
    const innerCount = template
        ? Object.values(template.graph.nodes).filter((n) => !(n.kind === 'layer' && n.libraryKey.startsWith('deepflow.template.'))).length
        : 0;
    const onClick = useCallback((e) => {
        if (!isInteractive(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);

    return (
        <div
            className={['df-container', `df-container--${tone}`, selected ? 'is-selected' : '', expandedNodeId === id ? 'is-active' : ''].join(' ')}
            style={{ width: CONTAINER_WIDTH }}
            onClick={onClick}
            onDoubleClick={(e) => { if (!isInteractive(e.target)) { e.stopPropagation(); enterSupernode(id); } }}
            title={template ? `${template.name} — double-click to open` : 'Template missing'}
        >
            <Boxes size={15} strokeWidth={2.5} className="df-container-icon" />
            <span className="df-container-label">{node.label}</span>
            <span className="df-container-meta">{template ? `${innerCount} block${innerCount === 1 ? '' : 's'}` : 'missing'}</span>
            <button
                className="nodrag df-container-open"
                aria-label={`Open ${node.label}`}
                title="Open"
                disabled={!template}
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => { e.stopPropagation(); enterSupernode(id); }}
            >
                <ChevronRight size={14} strokeWidth={3} />
            </button>
            <button
                aria-label={`Delete ${node.label}`}
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className="nodrag df-block-delete"
            >
                <X size={10} strokeWidth={3} />
            </button>
            <Handle type="target" position={Position.Top} className="df-handle" />
            <Handle type="source" position={Position.Bottom} className="df-handle" />
        </div>
    );
};

export default memo(SuperNodeInstanceView);
