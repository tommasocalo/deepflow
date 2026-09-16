import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle, GripVertical, SlidersHorizontal, X } from 'lucide-react';
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
    const tensorFlow = data.tensorFlow || node.visualization === 'tensor-flow';
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
                tensorFlow ? 'df-block--tensor-flow' : '',
                `df-block--${tone}`,
                selected ? 'is-selected' : '',
                active ? 'is-active' : '',
                isBoundary ? 'df-block--boundary' : '',
            ].join(' ')}
            style={{ width, height: BLOCK_HEIGHT, ...(tensorFlow ? { '--tensor-accent': style.accent } : {}) }}
            onClick={onClick}
            title={tensorFlow ? `${entry?.description ?? node.label} · Drag to move, click to inspect` : entry?.description}
        >
            {tone === 'error' && <AlertCircle size={14} strokeWidth={3} className="df-block-icon df-block-icon--error" />}
            {tone === 'warning' && <AlertTriangle size={14} strokeWidth={3} className="df-block-icon df-block-icon--warning" />}
            <span className="df-block-label">{node.label}</span>
            {tensorFlow && entry?.name.toLowerCase() !== node.label.toLowerCase() && <span className="df-tensor-layer-kind">{entry?.name}</span>}
            {tensorFlow && <GripVertical size={12} className="df-tensor-drag" aria-hidden="true" />}
            {isIO && !isBoundary && (
                <span className="df-block-tag">{INPUT_ONLY.has(node.libraryKey) ? 'in' : 'out'}</span>
            )}

            <button
                type="button"
                className="nodrag df-block-inspect"
                aria-label={`Inspect ${node.label}`}
                aria-expanded={active}
                title="Inspect block"
                onMouseDown={(event) => event.stopPropagation()}
                onClick={(event) => { event.stopPropagation(); toggleExpanded(id); }}
            ><SlidersHorizontal size={12} aria-hidden="true" /></button>

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
