import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { ChevronUp, Maximize2, Repeat, RotateCcw, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { formatShape } from '../../validation/index.js';
import ParameterField from './ParameterField.jsx';

function shouldToggleOnCardClick(target) {
    if (!target || !target.closest) return true;
    return !target.closest('input, button, select, textarea, label, a');
}

const RepeatNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const template = useTemplatesStore((s) => s.templates[node.templateId]);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const patchRepeatNode = useArchitectureStore((s) => s.patchRepeatNode);
    const patchRepeatBinding = useArchitectureStore((s) => s.patchRepeatBinding);
    const convertRepeatToInstance = useArchitectureStore((s) => s.convertRepeatToInstance);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const validation = useValidation();
    const onCardClick = useCallback((e) => {
        if (shouldToggleOnCardClick(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);
    const outShape = validation.shapes[id];
    const warnings = validation.warnings[id] ?? [];
    const errors = validation.errors[id] ?? [];

    const free = template?.freeParameters ?? [];
    const iterVar = node.iterationVariable || 'i';
    const showExpanded = expandedNodeId === id;

    const borderColor = errors.length ? '#dc2626' : warnings.length ? '#f59e0b' : '#000';
    const shadow = errors.length
        ? `${selected ? '5px 5px' : '3px 3px'} 0px 0px #dc2626`
        : warnings.length
            ? `${selected ? '5px 5px' : '3px 3px'} 0px 0px #f59e0b`
            : `${selected ? '5px 5px' : '3px 3px'} 0px 0px #000000`;

    return (
        <div
            className="relative group"
            style={{ width: showExpanded ? 340 : 268, transition: 'width 180ms ease' }}
            onClick={onCardClick}
            onDoubleClick={(e) => { e.stopPropagation(); enterSupernode(id); }}
        >
            <div
                className="absolute rounded-xl"
                style={{
                    inset: 0,
                    transform: 'translate(8px, 8px)',
                    background: 'repeating-linear-gradient(135deg, #fde68a, #fde68a 8px, #fcd34d 8px, #fcd34d 16px)',
                    border: '2px dashed #000',
                    opacity: 0.6,
                    zIndex: 0,
                }}
            />
            <div
                className="absolute rounded-xl"
                style={{
                    inset: 0,
                    transform: 'translate(4px, 4px)',
                    background: 'repeating-linear-gradient(135deg, #fef3c7, #fef3c7 8px, #fde68a 8px, #fde68a 16px)',
                    border: '2px dashed #000',
                    opacity: 0.85,
                    zIndex: 0,
                }}
            />
            <div
                className="relative rounded-xl p-3"
                style={{
                    background: 'repeating-linear-gradient(135deg, #ffffff, #ffffff 8px, #fef3c7 8px, #fef3c7 16px)',
                    border: `2px dashed ${borderColor}`,
                    boxShadow: shadow,
                    zIndex: 1,
                    transition: 'box-shadow 120ms ease',
                }}
            >
                <div className="flex items-center justify-between gap-2">
                    {showExpanded ? (
                        <input
                            value={node.label}
                            onChange={(e) => patchRepeatNode(id, { label: e.target.value })}
                            className="text-sm font-bold tracking-tight bg-transparent outline-none border-b border-transparent focus:border-current min-w-0 flex-1 nodrag"
                        />
                    ) : (
                        <h3 className="text-sm font-bold tracking-tight truncate flex items-center gap-1.5 min-w-0">
                            <Repeat size={14} strokeWidth={2.5} />
                            <span className="truncate">{node.label}</span>
                        </h3>
                    )}
                    <span className="text-[10px] font-bold uppercase tracking-wider whitespace-nowrap bg-black text-white px-1.5 py-0.5 rounded shrink-0">
                        × {node.count}
                    </span>
                </div>
                <div className="mt-0.5 text-[11px] opacity-70 truncate">
                    {template ? `${template.name}` : 'broken — template missing'}
                </div>

                {!showExpanded && free
                    .filter((p) => node.bindings?.[p.name] !== undefined && node.bindings?.[p.name] !== null && node.bindings?.[p.name] !== '')
                    .slice(0, 2)
                    .map((p) => (
                        <div key={p.name} className="flex justify-between items-center text-[10px] opacity-80 font-mono mt-0.5">
                            <span>${p.name}</span>
                            <span className="font-bold truncate max-w-[140px]">{String(node.bindings[p.name])}</span>
                        </div>
                    ))}

                {showExpanded && (
                    <div className="mt-2 nopan nowheel" onMouseDown={(e) => e.stopPropagation()}>
                        <div className="flex gap-1.5 mb-2">
                            <button
                                onClick={(e) => { e.stopPropagation(); enterSupernode(id); }}
                                className="flex-1 bg-emerald-200 border-2 border-black rounded text-[11px] font-bold py-1 hover:bg-emerald-300 nodrag flex items-center justify-center gap-1"
                            >
                                <Maximize2 size={11} strokeWidth={3} /> Open
                            </button>
                            <button
                                onClick={(e) => { e.stopPropagation(); convertRepeatToInstance(id); }}
                                className="flex-1 bg-stone-200 border-2 border-black rounded text-[11px] font-bold py-1 hover:bg-stone-300 nodrag flex items-center justify-center gap-1"
                                title="Convert back to a single supernode instance"
                            >
                                <RotateCcw size={11} strokeWidth={3} /> Single
                            </button>
                        </div>

                        <div className="grid grid-cols-2 gap-2 mb-2">
                            <label className="flex items-center justify-between gap-1 text-[11px]">
                                <span className="text-text-secondary font-mono">N</span>
                                <input
                                    type="number" min="1"
                                    value={node.count}
                                    onChange={(e) => patchRepeatNode(id, {
                                        count: Math.max(1, Number.parseInt(e.target.value, 10) || 1),
                                    })}
                                    className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                                    style={{ width: '5ch' }}
                                />
                            </label>
                            <label className="flex items-center justify-between gap-1 text-[11px]">
                                <span className="text-text-secondary font-mono">var</span>
                                <input
                                    type="text"
                                    value={iterVar}
                                    onChange={(e) => patchRepeatNode(id, { iterationVariable: e.target.value || 'i' })}
                                    className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                                    style={{ width: '5ch' }}
                                />
                            </label>
                        </div>

                        {free.length > 0 ? (
                            <div className="space-y-0">
                                {free.map((p) => (
                                    <label key={p.name} className="flex items-center justify-between gap-2 py-0.5">
                                        <span className="text-[11px] font-mono text-text-secondary" title={p.description}>
                                            ${p.name}
                                        </span>
                                        <input
                                            type="text"
                                            placeholder={`e.g. 64 * 2 ** ${iterVar}`}
                                            value={node.bindings?.[p.name] ?? ''}
                                            onChange={(e) => {
                                                const raw = e.target.value;
                                                patchRepeatBinding(id, p.name, raw === '' ? null : raw);
                                            }}
                                            className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                                            style={{ width: '14ch' }}
                                        />
                                    </label>
                                ))}
                            </div>
                        ) : (
                            <div className="text-[11px] italic opacity-60">No free parameters; open the template to declare some.</div>
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
                        {showExpanded && (
                            <button
                                onMouseDown={(e) => e.stopPropagation()}
                                onClick={(e) => { e.stopPropagation(); toggleExpanded(id); }}
                                className="nodrag p-0.5 hover:bg-black/10 rounded"
                                title="Collapse"
                                aria-label="Collapse node"
                            >
                                <ChevronUp size={11} strokeWidth={3} />
                            </button>
                        )}
                    </div>
                )}
                {!showExpanded && (
                    <div className="mt-1 text-[9px] opacity-50 italic">double-click to open template</div>
                )}
            </div>

            <button
                aria-label="Delete repeat"
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className="nodrag absolute -right-1.5 -top-1.5 z-20 h-4 w-4 rounded-full border border-black/15 bg-white text-black transition-all inline-flex items-center justify-center hover:bg-red-100 hover:border-red-400"
            >
                <X size={10} strokeWidth={3} />
            </button>

            <Handle
                type="target"
                position={Position.Top}
                className="w-3! h-3! bg-white! border-2! border-black! rounded-full!"
                style={{ top: -7, zIndex: 2 }}
            />
            <Handle
                type="source"
                position={Position.Bottom}
                className="w-3! h-3! bg-white! border-2! border-black! rounded-full!"
                style={{ bottom: -15, zIndex: 2 }}
            />
        </div>
    );
};

export default memo(RepeatNodeView);
