import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { ChevronUp, Layers3, X } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore.js';

function shouldToggleOnCardClick(target) {
    if (!target || !target.closest) return true;
    return !target.closest('input, button, select, textarea, label, a');
}

const PhaseNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const graph = useTrainingStore((s) => s.graph);
    const removeNode = useTrainingStore((s) => s.removeNode);
    const patchNode = useTrainingStore((s) => s.patchNode);
    const expandedNodeId = useTrainingStore((s) => s.expandedNodeId);
    const toggleExpanded = useTrainingStore((s) => s.toggleExpanded);
    const showExpanded = expandedNodeId === id;
    const onCardClick = useCallback((e) => {
        if (shouldToggleOnCardClick(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);

    const optimizers = Object.values(graph.nodes).filter((n) => n.kind === 'optimizer');
    const losses = Object.values(graph.nodes).filter((n) => n.kind === 'loss');
    const optSet = new Set(node.optimizerIds ?? []);
    const lossSet = new Set(node.lossIds ?? []);
    const togglePhaseOptimizer = (oid) => {
        const next = new Set(optSet);
        if (next.has(oid)) next.delete(oid); else next.add(oid);
        patchNode(id, { optimizerIds: [...next] });
    };
    const togglePhaseLoss = (lid) => {
        const next = new Set(lossSet);
        if (next.has(lid)) next.delete(lid); else next.add(lid);
        patchNode(id, { lossIds: [...next] });
    };

    return (
        <div
            className="rounded-md relative group p-3 bg-emerald-100"
            style={{
                width: showExpanded ? 320 : 240,
                border: '2px solid #000',
                boxShadow: selected ? '5px 5px 0px 0px #000000' : '3px 3px 0px 0px #000000',
                transition: 'width 180ms ease, box-shadow 120ms ease',
            }}
            onClick={onCardClick}
        >
            <div className="flex items-center justify-between gap-2">
                {showExpanded ? (
                    <input
                        value={node.label}
                        onChange={(e) => patchNode(id, { label: e.target.value })}
                        className="text-sm font-bold tracking-tight bg-transparent outline-none border-b border-transparent focus:border-current min-w-0 flex-1 nodrag"
                    />
                ) : (
                    <h3 className="text-sm font-bold tracking-tight truncate flex items-center gap-1.5 min-w-0">
                        <Layers3 size={14} strokeWidth={2.5} />
                        <span className="truncate">{node.label}</span>
                    </h3>
                )}
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-70 whitespace-nowrap shrink-0">
                    Phase
                </span>
            </div>

            {!showExpanded && (
                <div className="mt-2 space-y-0.5 text-[11px] opacity-80">
                    <div className="flex justify-between"><span>optimizers</span><span className="font-bold font-mono">{optSet.size}</span></div>
                    <div className="flex justify-between"><span>losses</span><span className="font-bold font-mono">{lossSet.size}</span></div>
                    <div className="flex justify-between"><span>every</span><span className="font-bold font-mono">{node.everyNSteps} step{node.everyNSteps !== 1 ? 's' : ''}</span></div>
                </div>
            )}

            {showExpanded && (
                <div className="mt-2 nopan nowheel" onMouseDown={(e) => e.stopPropagation()}>
                    <div className="grid grid-cols-2 gap-2 mb-2">
                        <label className="flex items-center justify-between gap-1 text-[11px]">
                            <span className="text-text-secondary font-mono">every N</span>
                            <input
                                type="number" min="1"
                                value={node.everyNSteps ?? 1}
                                onChange={(e) => patchNode(id, { everyNSteps: Math.max(1, Number.parseInt(e.target.value, 10) || 1) })}
                                className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                                style={{ width: '5ch' }}
                            />
                        </label>
                        <label className="flex items-center justify-between gap-1 text-[11px]">
                            <span className="text-text-secondary font-mono">cycle</span>
                            <input
                                type="number" min="1"
                                value={node.stepsPerCycle ?? 1}
                                onChange={(e) => patchNode(id, { stepsPerCycle: Math.max(1, Number.parseInt(e.target.value, 10) || 1) })}
                                className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                                style={{ width: '5ch' }}
                            />
                        </label>
                    </div>
                    <div>
                        <div className="text-[10px] font-bold uppercase tracking-wider text-text-secondary mb-1">
                            Optimizers ({optSet.size})
                        </div>
                        <div className="border-2 border-black rounded max-h-24 overflow-y-auto bg-white mb-2">
                            {optimizers.length === 0 && (
                                <div className="p-1.5 text-[10px] italic opacity-60">No optimizers yet.</div>
                            )}
                            {optimizers.map((o) => (
                                <label key={o.id} className="flex items-center gap-1.5 px-1.5 py-0.5 text-[11px] hover:bg-stone-100 cursor-pointer nodrag">
                                    <input
                                        type="checkbox"
                                        checked={optSet.has(o.id)}
                                        onChange={() => togglePhaseOptimizer(o.id)}
                                        className="w-3 h-3 accent-black"
                                    />
                                    <span className="truncate">{o.label}</span>
                                </label>
                            ))}
                        </div>
                        <div className="text-[10px] font-bold uppercase tracking-wider text-text-secondary mb-1">
                            Losses ({lossSet.size})
                        </div>
                        <div className="border-2 border-black rounded max-h-24 overflow-y-auto bg-white">
                            {losses.length === 0 && (
                                <div className="p-1.5 text-[10px] italic opacity-60">No losses yet.</div>
                            )}
                            {losses.map((l) => (
                                <label key={l.id} className="flex items-center gap-1.5 px-1.5 py-0.5 text-[11px] hover:bg-stone-100 cursor-pointer nodrag">
                                    <input
                                        type="checkbox"
                                        checked={lossSet.has(l.id)}
                                        onChange={() => togglePhaseLoss(l.id)}
                                        className="w-3 h-3 accent-black"
                                    />
                                    <span className="truncate">{l.label}</span>
                                </label>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {showExpanded && (
                <div className="mt-2 pt-1.5 border-t border-black/20 flex justify-end">
                    <button
                        onMouseDown={(e) => e.stopPropagation()}
                        onClick={(e) => { e.stopPropagation(); toggleExpanded(id); }}
                        className="nodrag p-0.5 hover:bg-black/10 rounded"
                        title="Collapse"
                        aria-label="Collapse node"
                    >
                        <ChevronUp size={11} strokeWidth={3} />
                    </button>
                </div>
            )}

            <button
                aria-label="Delete phase"
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className="nodrag absolute -right-1.5 -top-1.5 z-20 h-4 w-4 rounded-full border border-black/15 bg-white text-black transition-all inline-flex items-center justify-center hover:bg-red-100 hover:border-red-400"
            >
                <X size={10} strokeWidth={3} />
            </button>

            <Handle type="target" position={Position.Top}    className="w-3! h-3! bg-white! border-2! border-black! rounded-full!" style={{ top: -7 }} />
            <Handle type="source" position={Position.Bottom} className="w-3! h-3! bg-white! border-2! border-black! rounded-full!" style={{ bottom: -7 }} />
        </div>
    );
};

export default memo(PhaseNodeView);
