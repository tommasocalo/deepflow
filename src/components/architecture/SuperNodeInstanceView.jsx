import { useComponentTemplate } from '../../hooks/useComponentTemplate.js';
import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Boxes, ChevronRight, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { isInteractive } from './nodeGeometry.js';
import PortShapeLabels from './PortShapeLabels.jsx';
import { useOpenToSideNode } from './OpenToSideContext.jsx';
import NodeInputPorts from './NodeInputPorts.jsx';
import { inputPortsFor } from './nodeInputPorts.js';
import { STANDARD_COMPONENTS } from '../../domain/components/index.js';
import { getEntry } from '../../domain/library/index.js';
import { styleFor } from './categoryStyle.js';
import OperationIcon from './OperationIcon.jsx';

/** A supernode instance: hatched container, label, "open" affordance. */
const SuperNodeInstanceView = ({ id, data, selected }) => {
    const node = data.node;
    const template = useComponentTemplate(node);
    const standard = STANDARD_COMPONENTS.find((component) => component.key === (template?.standardKey ?? template?.componentStyleKey));
    const accent = standard ? styleFor(getEntry(standard.libraryKey)?.category).accent : undefined;
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const sideMenuProps = useOpenToSideNode([...currentPath, id], { disabled: !template });
    const validation = useValidation();
    const templates = useTemplatesStore((s) => s.templates);
    const multiInput = inputPortsFor(node, templates).length > 1;
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
            {...sideMenuProps}
            className={['df-container', standard ? 'df-container--standard' : '', `df-container--${tone}`, selected ? 'is-selected' : '', expandedNodeId === id ? 'is-active' : ''].join(' ')}
            style={{ '--node-accent': accent, ...(multiInput ? { minWidth: 220 * inputPortsFor(node, templates).length, maxWidth: 'none' } : {}) }}
            onClick={onClick}
            role="button" tabIndex={0} aria-label={`Inspect ${node.label}`} aria-expanded={expandedNodeId === id}
            onKeyDown={(event) => {
                sideMenuProps.onKeyDown?.(event);
                if (!event.defaultPrevented && event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) { event.preventDefault(); event.stopPropagation(); toggleExpanded(id); }
            }}
            onDoubleClick={(e) => { if (!isInteractive(e.target)) { e.stopPropagation(); enterSupernode(id); } }}
            title={template ? `${template.name} — double-click to open` : 'Template missing'}
        >
            {standard ? <OperationIcon libraryKey={standard.libraryKey} size={24} /> : <Boxes size={15} strokeWidth={2.5} className="df-container-icon" />}
            <span className="df-container-label" title={node.label}>{node.label}</span>
            {!standard && <span className="df-container-meta">{template ? `${innerCount} block${innerCount === 1 ? '' : 's'}` : 'missing'}</span>}
            <button
                className="nodrag df-container-open"
                aria-label={`Open ${node.label}`}
                title="Open definition"
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
            <PortShapeLabels node={node} shapes={validation.shapes} hideInput={multiInput} />
            <NodeInputPorts node={node} shapes={validation.shapes} />
            <Handle type="source" position={Position.Bottom} className="df-handle" />
        </div>
    );
};

export default memo(SuperNodeInstanceView);
