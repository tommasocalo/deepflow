import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { ChevronUp, Cog, X } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';

function shouldToggleOnCardClick(target) {
    if (!target || !target.closest) return true;
    return !target.closest('input, button, select, textarea, label, a');
}

const OPTIMIZER_TYPES = ['Adam', 'AdamW', 'SGD', 'RMSprop'];

const OptimizerNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const removeNode = useTrainingStore((s) => s.removeNode);
    const patchNode = useTrainingStore((s) => s.patchNode);
    const patchHyperparameters = useTrainingStore((s) => s.patchHyperparameters);
    const expandedNodeId = useTrainingStore((s) => s.expandedNodeId);
    const toggleExpanded = useTrainingStore((s) => s.toggleExpanded);
    const archGraph = useArchitectureStore((s) => s.graph);

    const lr = node.hyperparameters?.learning_rate;
    const showExpanded = expandedNodeId === id;
    const onCardClick = useCallback((e) => {
        if (shouldToggleOnCardClick(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);
    const candidates = Object.values(archGraph.nodes ?? {}).filter(
        (n) => n.kind === 'layer' || n.kind === 'supernode' || n.kind === 'repeat',
    );
    const groupSet = new Set(node.parameterGroup ?? []);
    const toggleGroup = (nodeId) => {
        const next = new Set(groupSet);
        if (next.has(nodeId)) next.delete(nodeId);
        else next.add(nodeId);
        patchNode(id, { parameterGroup: [...next] });
    };

    return (
        <div
            className="rounded-md relative group p-3 bg-blue-100"
            style={{
                width: showExpanded ? 300 : 240,
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
                        <Cog size={14} strokeWidth={2.5} />
                        <span className="truncate">{node.label}</span>
                    </h3>
                )}
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-70 whitespace-nowrap shrink-0">
                    Optimizer
                </span>
            </div>

            {!showExpanded && (
                <div className="mt-2 space-y-0.5 text-[11px] opacity-80">
                    <div className="flex justify-between"><span>type</span><span className="font-bold font-mono">{node.type}</span></div>
                    {lr !== undefined && (
                        <div className="flex justify-between"><span>lr</span><span className="font-bold font-mono">{lr}</span></div>
                    )}
                    <div className="flex justify-between">
                        <span>params</span>
                        <span className="font-bold font-mono">
                            {node.parameterGroup?.length ? `${node.parameterGroup.length} group${node.parameterGroup.length !== 1 ? 's' : ''}` : 'all'}
                        </span>
                    </div>
                </div>
            )}

            {showExpanded && (
                <div className="mt-2 nopan nowheel" onMouseDown={(e) => e.stopPropagation()}>
                    <label className="flex items-center justify-between gap-2 py-0.5">
                        <span className="text-[11px] font-mono text-text-secondary">type</span>
                        <select
                            value={node.type}
                            onChange={(e) => patchNode(id, { type: e.target.value })}
                            className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white nodrag"
                        >
                            {OPTIMIZER_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
                        </select>
                    </label>
                    {Object.entries(node.hyperparameters ?? {}).map(([k, v]) => {
                        const isNum = typeof v === 'number';
                        // Tuples come in as strings like "(0.9, 0.999)" — give them more room.
                        const widthCh = isNum ? 9 : 14;
                        return (
                            <label key={k} className="flex items-center justify-between gap-2 py-0.5">
                                <span className="text-[11px] font-mono text-text-secondary">{k}</span>
                                <input
                                    type={isNum ? 'number' : 'text'}
                                    step="any"
                                    value={String(v ?? '')}
                                    onChange={(e) => {
                                        const raw = e.target.value;
                                        const next = isNum ? Number.parseFloat(raw) : raw;
                                        patchHyperparameters(id, { [k]: Number.isFinite(next) ? next : raw });
                                    }}
                                    className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                                    style={{ width: `${widthCh}ch` }}
                                />
                            </label>
                        );
                    })}

                    <div className="mt-2 border-t border-black/20 pt-2">
                        <div className="text-[10px] font-bold uppercase tracking-wider text-text-secondary mb-1">
                            Parameter group ({groupSet.size === 0 ? 'all' : `${groupSet.size}`})
                        </div>
                        <div className="border-2 border-black rounded max-h-32 overflow-y-auto bg-white">
                            {candidates.length === 0 && (
                                <div className="p-1.5 text-[10px] italic opacity-60">No architecture nodes.</div>
                            )}
                            {candidates.map((n) => (
                                <label
                                    key={n.id}
                                    className="flex items-center gap-1.5 px-1.5 py-0.5 text-[11px] hover:bg-stone-100 cursor-pointer nodrag"
                                >
                                    <input
                                        type="checkbox"
                                        checked={groupSet.has(n.id)}
                                        onChange={() => toggleGroup(n.id)}
                                        className="w-3 h-3 accent-black"
                                    />
                                    <span className="font-mono opacity-50 text-[9px]">{n.kind}</span>
                                    <span className="truncate">{n.label}</span>
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
                aria-label="Delete optimizer"
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

export default memo(OptimizerNodeView);
