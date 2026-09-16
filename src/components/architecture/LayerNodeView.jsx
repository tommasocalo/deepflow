import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle, X } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { styleFor } from './categoryStyle.js';
import { BLOCK_HEIGHT, isInteractive, widthForShape } from './nodeGeometry.js';

const INPUT_ONLY = new Set(['deepflow.io.Input', 'deepflow.template.Input']);
const OUTPUT_ONLY = new Set(['deepflow.io.Output', 'deepflow.template.Output']);

/**
 * Compact block: category colour, label, nothing else. Parameters live in
 * the NodeInspector; activations live on the edges.
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
    const width = isIO ? 220 : widthForShape(outputShape, 200);
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
            style={{ width, height: BLOCK_HEIGHT }}
            onClick={onClick}
            title={entry?.description}
        >
            {tone === 'error' && <AlertCircle size={14} strokeWidth={3} className="df-block-icon df-block-icon--error" />}
            {tone === 'warning' && <AlertTriangle size={14} strokeWidth={3} className="df-block-icon df-block-icon--warning" />}
            <span className="df-block-label">{node.label}</span>
            {isIO && !isBoundary && (
                <span className="df-block-tag">{INPUT_ONLY.has(node.libraryKey) ? 'in' : 'out'}</span>
            )}

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
