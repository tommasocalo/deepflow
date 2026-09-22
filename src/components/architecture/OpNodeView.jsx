import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { OP_SIZE, isInteractive } from './nodeGeometry.js';
import OperationIcon from './OperationIcon.jsx';
import PortShapeLabels from './PortShapeLabels.jsx';
import { getEntry } from '../../domain/library/index.js';
import NodeInputPorts from './NodeInputPorts.jsx';
import { getOperationInputIds } from '../../domain/graphInputs.js';
// Keys are listed in nodeGeometry.js (isOpKey) so the canvas can route without importing a component file.

/** Element-wise / combine operators are glyphs on the line, not blocks. */

const OpNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const title = getEntry(node.libraryKey)?.name ?? node.label;
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const validation = useValidation();
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const tone = errors.length ? 'error' : warnings.length ? 'warning' : 'ok';
    const onClick = useCallback((e) => {
        if (!isInteractive(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);

    return (
        <div
            className={['df-op', ['deepflow.op.Add', 'deepflow.op.Multiply'].includes(node.libraryKey) ? '' : 'df-op--routing', `df-op--${tone}`, selected ? 'is-selected' : '', expandedNodeId === id ? 'is-active' : ''].join(' ')}
            style={{ width: OP_SIZE, height: OP_SIZE }}
            title={`${title}${node.label && node.label !== title ? ` — ${node.label}` : ''}`}
            onClick={onClick}
            role="button"
            tabIndex={0}
            aria-label={`Inspect ${node.label}`}
            aria-expanded={expandedNodeId === id}
            onKeyDown={(event) => {
                if (event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) {
                    event.preventDefault();
                    event.stopPropagation();
                    toggleExpanded(id);
                }
            }}
        >
            <OperationIcon libraryKey={node.libraryKey} size={22} />
            <span className="sr-only">{title}</span>
            <PortShapeLabels node={node} shapes={validation.shapes} hideInput={(getOperationInputIds(node)?.length ?? 0) > 1} />
            {node.libraryKey !== 'deepflow.op.Arange' && <NodeInputPorts node={node} shapes={validation.shapes} />}
            <Handle type="source" position={Position.Bottom} className="df-handle df-handle--op" />
        </div>
    );
};

export default memo(OpNodeView);
