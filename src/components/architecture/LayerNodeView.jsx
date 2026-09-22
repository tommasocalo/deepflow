import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { styleFor } from './categoryStyle.js';
import { isInteractive } from './nodeGeometry.js';

import OperationIcon from './OperationIcon.jsx';
import PortShapeLabels from './PortShapeLabels.jsx';

const INPUT_ONLY = new Set(['deepflow.io.Input', 'deepflow.template.Input', 'deepflow.op.Parameter', 'deepflow.op.Arange']);
const OUTPUT_ONLY = new Set(['deepflow.io.Output', 'deepflow.template.Output']);

/**
 * The operation mark and user-assigned name set the card width. Click to configure.
 */
const LayerNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const entry = getEntry(node.libraryKey);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const validation = useValidation();
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const style = styleFor(entry?.category ?? 'IO');
    const isBoundary = node.libraryKey.startsWith('deepflow.template.');
    const tone = errors.length ? 'error' : warnings.length ? 'warning' : 'ok';
    const active = expandedNodeId === id;

    const onClick = useCallback((e) => {
        if (!isInteractive(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);

    return (
        <div
            className={[
                'df-block', style.card,
                `df-block--${tone}`,
                selected ? 'is-selected' : '',
                active ? 'is-active' : '',
                isBoundary ? 'df-block--boundary' : '',
            ].join(' ')}
            style={{ '--node-accent': style.accent }}
            onClick={onClick}
            role="button"
            tabIndex={0}
            aria-label={`Inspect ${node.label}`}
            aria-expanded={active}
            onKeyDown={(event) => {
                if (event.target === event.currentTarget && ['Enter', ' '].includes(event.key)) {
                    event.preventDefault();
                    event.stopPropagation();
                    toggleExpanded(id);
                }
            }}
            title={`${node.label} · ${entry?.name ?? node.libraryKey}\n${entry?.description ?? ''}`}
        >
            <OperationIcon libraryKey={node.libraryKey} size={24} />
            <span className="df-block-name">{node.label || entry?.name || 'Untitled'}</span>
            <PortShapeLabels node={node} shapes={validation.shapes} />
            {tone === 'error' && <AlertCircle size={12} strokeWidth={3} className="df-block-status df-block-icon--error" />}
            {tone === 'warning' && <AlertTriangle size={12} strokeWidth={3} className="df-block-status df-block-icon--warning" />}

            {!INPUT_ONLY.has(node.libraryKey) && (
                <Handle type="target" position={Position.Top} className="df-handle" />
            )}
            {!OUTPUT_ONLY.has(node.libraryKey) && (
                <Handle type="source" position={Position.Bottom} className="df-handle" />
            )}
        </div>
    );
};

export default memo(LayerNodeView);
