import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle, SlidersHorizontal, X } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { styleFor } from './categoryStyle.js';
import { isInteractive, activationLabel } from './nodeGeometry.js';

import OperationIcon from './OperationIcon.jsx';

const INPUT_ONLY = new Set(['deepflow.io.Input', 'deepflow.template.Input']);
const OUTPUT_ONLY = new Set(['deepflow.io.Output', 'deepflow.template.Output']);

/**
 * Compact operation mark and output dimensions. Configuration lives in the inspector.
 */
const LayerNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const entry = getEntry(node.libraryKey);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const validation = useValidation();
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const outputShape = validation.shapes[id];
    const style = styleFor(entry?.category ?? 'IO');
    const isIO = entry?.category === 'IO';
    const isBoundary = node.libraryKey.startsWith('deepflow.template.');
    const shapeLabel = activationLabel(outputShape) ?? '—';
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
            title={`${node.label} · ${entry?.name ?? node.libraryKey}\n${entry?.description ?? ''}`}
        >
            <OperationIcon libraryKey={node.libraryKey} size={34} />
            <div className="df-block-content">
                <span className="sr-only">{node.label}</span>
                <span className="df-block-caption">{isIO ? (INPUT_ONLY.has(node.libraryKey) ? 'input' : 'output') : 'output'}</span>
                <span className="df-block-shape" title={`Output dimensions: ${shapeLabel}`}>{shapeLabel}</span>
            </div>
            {tone === 'error' && <AlertCircle size={12} strokeWidth={3} className="df-block-status df-block-icon--error" />}
            {tone === 'warning' && <AlertTriangle size={12} strokeWidth={3} className="df-block-status df-block-icon--warning" />}
            <button
                type="button"
                className="nodrag df-block-inspect"
                aria-label={`Inspect ${node.label}`}
                aria-expanded={active}
                title={`Configure ${node.label}`}
                onMouseDown={(event) => event.stopPropagation()}
                onClick={(event) => { event.stopPropagation(); toggleExpanded(id); }}
            ><SlidersHorizontal size={13} strokeWidth={2.5} aria-hidden="true" /></button>

            <button
                aria-label={`Delete ${node.label}`}
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className="nodrag df-block-delete"
            >
                <X size={10} strokeWidth={3} />
            </button>

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
