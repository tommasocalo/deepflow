import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { OP_SIZE, isInteractive } from './nodeGeometry.js';
// Keys are listed in nodeGeometry.js (isOpKey) so the canvas can route without importing a component file.

/** Element-wise / combine operators are glyphs on the line, not blocks. */
const GLYPHS = {
    'deepflow.op.Add':      { symbol: '+', title: 'Add' },
    'deepflow.op.Multiply': { symbol: '×', title: 'Multiply' },
    'deepflow.op.Concat':   { symbol: '⧺', title: 'Concatenate' },
    'deepflow.op.Split':    { symbol: '⑂', title: 'Split' },
};

const OpNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const glyph = GLYPHS[node.libraryKey] ?? { symbol: '?', title: node.label };
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
            className={['df-op', `df-op--${tone}`, selected ? 'is-selected' : '', expandedNodeId === id ? 'is-active' : ''].join(' ')}
            style={{ width: OP_SIZE, height: OP_SIZE }}
            title={`${glyph.title}${node.label && node.label !== glyph.title ? ` — ${node.label}` : ''}`}
            onClick={onClick}
        >
            <span className="df-op-symbol" aria-hidden="true">{glyph.symbol}</span>
            <span className="sr-only">{glyph.title}</span>
            <Handle type="target" position={Position.Top} className="df-handle df-handle--op" />
            <Handle type="source" position={Position.Bottom} className="df-handle df-handle--op" />
        </div>
    );
};

export default memo(OpNodeView);
