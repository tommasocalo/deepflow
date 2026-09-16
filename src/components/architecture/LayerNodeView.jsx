import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle, SlidersHorizontal, X } from 'lucide-react';
import { getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { styleFor } from './categoryStyle.js';
import { isInteractive, activationLabel } from './nodeGeometry.js';

import TensorPreview from './TensorPreview.jsx';
import { featurePreview } from './featurePreview.js';

const INPUT_ONLY = new Set(['deepflow.io.Input', 'deepflow.template.Input']);
const OUTPUT_ONLY = new Set(['deepflow.io.Output', 'deepflow.template.Output']);

/**
 * Each operation contains a bounded preview of its output feature vector.
 * Configuration remains in the separate inspector.
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
    const channelFirst = ['Convolution', 'Pooling'].includes(entry?.category) && outputShape?.length > 2;
    const featureAxis = channelFirst ? 1 : outputShape?.length - 1;
    const preview = isIO ? null : featurePreview(outputShape, featureAxis);
    const isBoundary = node.libraryKey.startsWith('deepflow.template.');
    const width = preview ? preview.width + 6 : 184;
    const tone = errors.length ? 'error' : warnings.length ? 'warning' : 'ok';
    const active = expandedNodeId === id;

    const onClick = useCallback((e) => {
        if (!isInteractive(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);

    return (
        <div
            className={[
                'df-block', style.card,
                preview ? 'df-block--features' : '',
                `df-block--${tone}`,
                selected ? 'is-selected' : '',
                active ? 'is-active' : '',
                isBoundary ? 'df-block--boundary' : '',
            ].join(' ')}
            style={{ width, '--tensor-accent': style.accent }}
            onClick={onClick}
            title={entry?.description}
        >
            <div className={preview ? "df-feature-tools" : "df-block-heading"}>
            {tone === 'error' && <AlertCircle size={14} strokeWidth={3} className="df-block-icon df-block-icon--error" />}
            {tone === 'warning' && <AlertTriangle size={14} strokeWidth={3} className="df-block-icon df-block-icon--warning" />}
            <span className={preview ? "df-feature-name" : "df-block-label"}>{node.label}</span>
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
            ><SlidersHorizontal size={13} strokeWidth={2.5} aria-hidden="true" /></button>
            </div>
            {preview ? <TensorPreview shape={outputShape} featureAxis={featureAxis} featureLabel={channelFirst ? 'channels' : 'features'} /> : <div className="df-block-shape">{activationLabel(outputShape) ?? 'Shape unresolved'}</div>}

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
