import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { ChevronRight, Repeat, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { isInteractive } from './nodeGeometry.js';
import PortShapeLabels from './PortShapeLabels.jsx';

/**
 * A repeated template: the container is drawn as a short stack so the
 * repetition is visible; the count is the only number shown.
 */
const RepeatNodeView = ({ id, data, selected }) => {
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
    const onClick = useCallback((e) => {
        if (!isInteractive(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);
    const count = Number(node.count) || 0;
    const ghosts = Math.min(3, Math.max(0, count - 1));

    return (
        <div
            className={['df-repeat', `df-repeat--${tone}`, selected ? 'is-selected' : '', expandedNodeId === id ? 'is-active' : ''].join(' ')}
            style={{ '--repeat-stack-depth': `${ghosts * 5}px` }}
            onClick={onClick}
            onDoubleClick={(e) => { if (!isInteractive(e.target)) { e.stopPropagation(); enterSupernode(id); } }}
            title={template ? `${template.name} × ${count} — double-click to open` : 'Template missing'}
        >
            {Array.from({ length: ghosts }, (_, i) => (
                <div key={i} className="df-repeat-ghost" style={{ transform: `translate(${(ghosts - i) * 5}px, ${(ghosts - i) * 5}px)`, zIndex: -1 - i }} />
            ))}
            <div className="df-repeat-face">
                <Repeat size={14} strokeWidth={2.75} className="df-container-icon" />
                <span className="df-container-label" title={node.label}>{node.label}</span>
                <span className="df-repeat-count">× {count}</span>
                <button
                    className="nodrag df-container-open"
                    aria-label={`Open ${node.label}`}
                    title="Open template"
                    disabled={!template}
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); enterSupernode(id); }}
                >
                    <ChevronRight size={14} strokeWidth={3} />
                </button>
            </div>
            <button
                aria-label={`Delete ${node.label}`}
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className="nodrag df-block-delete"
            >
                <X size={10} strokeWidth={3} />
            </button>
            <PortShapeLabels node={node} shapes={validation.shapes} />
            <Handle type="target" position={Position.Top} className="df-handle" />
            <Handle type="source" position={Position.Bottom} className="df-handle" />
        </div>
    );
};

export default memo(RepeatNodeView);
