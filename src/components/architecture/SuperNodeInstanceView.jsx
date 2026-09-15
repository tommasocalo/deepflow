import React, { memo, useCallback, useEffect } from 'react';
import { Handle, Position, useUpdateNodeInternals } from '@xyflow/react';
import { Boxes, ChevronDown, ChevronUp, Maximize2, Repeat, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { formatShape } from '../../validation/index.js';
import ParameterField from './ParameterField.jsx';

function shouldToggleOnCardClick(target) {
    if (!target || !target.closest) return true;
    return !target.closest('input, button, select, textarea, label, a, summary, .react-flow__handle');
}

const SuperNodeInstanceView = ({ id, data, selected }) => {
    const node = data.node;
    const template = useTemplatesStore((s) => s.templates[node.templateId]);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const convertInstanceToRepeat = useArchitectureStore((s) => s.convertInstanceToRepeat);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const validation = useValidation();
    const onCardClick = useCallback((e) => {
        if (shouldToggleOnCardClick(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);
    const outShape = validation.shapes[id];
    const warnings = validation.warnings[id] ?? [];
    const errors = validation.errors[id] ?? [];

    const innerCount = template ? Object.keys(template.graph.nodes).length : 0;
    const bindings = node.parameterValues ?? {};
    const freeParams = template?.freeParameters ?? [];
    const showExpanded = expandedNodeId === id;
    const updateNodeInternals = useUpdateNodeInternals();
    useEffect(() => {
        // Re-measure handle positions after the expanded content is laid out.
        const frame = requestAnimationFrame(() => updateNodeInternals(id));
        return () => cancelAnimationFrame(frame);
    }, [id, showExpanded, updateNodeInternals]);

    const cardStyle = {
        width: showExpanded ? 320 : 260,
        background: 'repeating-linear-gradient(135deg, #fef3c7, #fef3c7 8px, #fde68a 8px, #fde68a 16px)',
        border: `2px dashed ${errors.length ? '#dc2626' : '#000'}`,
        boxShadow: selected ? '5px 5px 0px 0px #000000' : '3px 3px 0px 0px #000000',
        transition: 'box-shadow 120ms ease',
    };

    return (
        <div
            className="df-architecture-card rounded-xl relative group p-3 pr-9"
            style={cardStyle}
            onClick={onCardClick}
            onDoubleClick={(e) => { if (shouldToggleOnCardClick(e.target)) { e.stopPropagation(); enterSupernode(id); } }}
        >
            <div className="flex items-center justify-between gap-2">
                {showExpanded ? (
                    <input
                        aria-label="Block name"
                        value={node.label}
                        onChange={(e) => patchNode(id, { label: e.target.value })}
                        className="text-sm font-bold tracking-tight bg-transparent outline-none border-b border-transparent focus:border-current min-w-0 flex-1 nodrag flex items-center gap-1.5"
                    />
                ) : (
                    <h3 className="df-node-title text-sm font-bold tracking-tight truncate flex items-center gap-1.5 min-w-0">
                        <Boxes size={14} strokeWidth={2.5} />
                        <span className="truncate">{node.label}</span>
                    </h3>
                )}
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-70 whitespace-nowrap shrink-0">
                    Template
                </span>
            </div>

            <div className="mt-1 text-[11px] opacity-80">
                {template ? `${template.name} — ${innerCount} block${innerCount !== 1 ? 's' : ''}` : 'broken — template missing'}
            </div>

            {/* Collapsed mode: show first 2 bindings inline */}
            {!showExpanded && freeParams
                .filter((p) => bindings[p.name] !== null && bindings[p.name] !== undefined && bindings[p.name] !== '')
                .slice(0, 2)
                .map((p) => (
                    <div key={p.name} className="flex justify-between items-center text-[10px] opacity-80 font-mono mt-0.5">
                        <span>${p.name}</span>
                        <span className="font-bold min-w-0 max-w-[55%] truncate" title={String(bindings[p.name])}>{String(bindings[p.name])}</span>
                    </div>
                ))}

            {/* Expanded: bindings + actions */}
            {showExpanded && (
                <div className="mt-2 nopan nowheel" onKeyDown={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()} onClick={(e) => e.stopPropagation()}>
                    <div className="flex gap-1.5 mb-2">
                        <button
                            disabled={!template}
                            onClick={(e) => { e.stopPropagation(); enterSupernode(id); }}
                            className="flex-1 bg-emerald-200 border-2 border-black rounded text-[11px] font-bold py-1 hover:bg-emerald-300 nodrag flex items-center justify-center gap-1"
                        >
                            <Maximize2 size={11} strokeWidth={3} /> Open
                        </button>
                        <button
                            disabled={!template}
                            onClick={(e) => { e.stopPropagation(); convertInstanceToRepeat(id, 2); }}
                            className="flex-1 bg-yellow-200 border-2 border-black rounded text-[11px] font-bold py-1 hover:bg-yellow-300 nodrag flex items-center justify-center gap-1"
                            title="Convert into a Repeat that unrolls this template N times"
                        >
                            <Repeat size={11} strokeWidth={3} /> Repeat
                        </button>
                    </div>

                    {freeParams.length > 0 ? (
                        <div className="space-y-0">
                            {freeParams.map((p) => (
                                <ParameterField
                                    key={p.name}
                                    param={p}
                                    value={bindings[p.name]}
                                    onChange={(v) => patchNode(id, {
                                        parameterValues: { ...bindings, [p.name]: v },
                                    })}
                                />
                            ))}
                        </div>
                    ) : (
                        <div className="text-[11px] italic opacity-60">
                            No free parameters declared. Open the template to add some.
                        </div>
                    )}

                    {errors.length > 0 && (
                        <div className="mt-2 border-2 border-red-600 rounded p-1.5 bg-red-50/90">
                            <ul className="text-[11px] text-red-800 list-disc pl-4">
                                {errors.map((error, index) => <li key={index}>{error}</li>)}
                            </ul>
                        </div>
                    )}
                    {warnings.length > 0 && (
                        <div className="mt-2 border-2 border-amber-500 rounded p-1.5 bg-amber-50/80">
                            <ul className="text-[11px] text-amber-800 list-disc pl-4">
                                {warnings.map((w, i) => <li key={i}>{w}</li>)}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {outShape && (
                <div className="mt-2 pt-1.5 border-t border-black/20 text-[10px] opacity-70 font-mono truncate flex items-center justify-between gap-2">
                    <span>out: {formatShape(outShape)}</span>
                </div>
            )}
            {!showExpanded && (
                <div className="mt-1 text-[9px] opacity-50 italic">double-click to open</div>
            )}

            <button
                type="button"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={(e) => { e.stopPropagation(); toggleExpanded(id); }}
                className="nodrag absolute right-2 bottom-2 z-10 p-0.5 hover:bg-black/10 rounded"
                title={showExpanded ? 'Collapse' : 'Expand'}
                aria-label={`${showExpanded ? 'Collapse' : 'Expand'} ${node.label}`}
                aria-expanded={showExpanded}
            >
                {showExpanded ? <ChevronUp size={12} strokeWidth={3} /> : <ChevronDown size={12} strokeWidth={3} />}
            </button>

            <button
                aria-label="Delete supernode"
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className="nodrag absolute -right-1.5 -top-1.5 z-20 h-4 w-4 rounded-full border border-black/15 bg-white text-black transition-all inline-flex items-center justify-center hover:bg-red-100 hover:border-red-400"
            >
                <X size={10} strokeWidth={3} />
            </button>

            <Handle
                type="target"
                position={Position.Top}
                className="df-architecture-handle w-3! h-3! bg-white! border-2! border-black! rounded-full!"
                style={{ top: 0 }}
            />
            <Handle
                type="source"
                position={Position.Bottom}
                className="df-architecture-handle w-3! h-3! bg-white! border-2! border-black! rounded-full!"
                style={{ bottom: 0 }}
            />
        </div>
    );
};

export default memo(SuperNodeInstanceView);
