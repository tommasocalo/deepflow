import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertTriangle, ChevronUp, Target, X } from 'lucide-react';
import { useTrainingStore } from '../../stores/trainingStore.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { formatLossLabel, LOSS_DEFAULTS } from '../../domain/training/index.js';

function shouldToggleOnCardClick(target) {
    if (!target || !target.closest) return true;
    return !target.closest('input, button, select, textarea, label, a');
}

const LOSS_TYPES = ['cross_entropy', 'bce', 'mse', 'l1', 'custom'];
const defaultLabel = (type) => `${formatLossLabel(type)} loss`;
/** The option text: the readable name with the PyTorch-style id the code and the lessons use. */
const optionText = (type) => (formatLossLabel(type) === type ? type : `${formatLossLabel(type)} (${type})`);

const LossNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const removeNode = useTrainingStore((s) => s.removeNode);
    const patchNode = useTrainingStore((s) => s.patchNode);
    const patchHyperparameters = useTrainingStore((s) => s.patchHyperparameters);
    const expandedNodeId = useTrainingStore((s) => s.expandedNodeId);
    const toggleExpanded = useTrainingStore((s) => s.toggleExpanded);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const showExpanded = expandedNodeId === id;
    const onCardClick = useCallback((e) => {
        if (shouldToggleOnCardClick(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);

    const targetNode = node.architectureOutputId ? rootGraph.nodes[node.architectureOutputId] : null;
    const targetLabel = targetNode ? targetNode.label : '(none)';
    const outputCandidates = Object.values(rootGraph.nodes ?? {}).filter(
        (n) => n.kind === 'layer' && n.libraryKey === 'deepflow.io.Output',
    );

    const borderColor = !node.architectureOutputId ? '#f59e0b' : '#000';
    const shadow = !node.architectureOutputId
        ? `${selected ? '5px 5px' : '3px 3px'} 0px 0px #f59e0b`
        : `${selected ? '5px 5px' : '3px 3px'} 0px 0px #000000`;

    return (
        <div
            className="rounded-md relative group p-3 bg-pink-100"
            style={{
                width: showExpanded ? 300 : 240,
                border: `2px solid ${borderColor}`,
                boxShadow: shadow,
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
                        <Target size={14} strokeWidth={2.5} />
                        {!node.architectureOutputId && <AlertTriangle size={12} className="text-amber-600 shrink-0" strokeWidth={3} />}
                        <span className="truncate">{node.label}</span>
                    </h3>
                )}
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-70 whitespace-nowrap shrink-0">
                    Loss
                </span>
            </div>

            {!showExpanded && (
                <div className="mt-2 space-y-0.5 text-[11px] opacity-80">
                    <div className="flex justify-between"><span>type</span><span className="font-bold font-mono">{node.type}</span></div>
                    <div className="flex justify-between"><span>weight</span><span className="font-bold font-mono">{node.weight}</span></div>
                    <div className="flex justify-between"><span>output</span><span className={`font-bold truncate max-w-[120px] ${targetNode ? '' : 'text-amber-700'}`}>{targetLabel}</span></div>
                </div>
            )}

            {showExpanded && (
                <div className="mt-2 nopan nowheel" onMouseDown={(e) => e.stopPropagation()}>
                    <label className="flex items-center justify-between gap-2 py-0.5">
                        <span className="text-[11px] font-mono text-text-secondary">type</span>
                        <select
                            value={node.type}
                            onChange={(e) => {
                                const type = e.target.value;
                                // A default label follows the type ("BCE loss" → "Cross-Entropy loss"); a custom one stays.
                                patchNode(id, {
                                    type,
                                    ...(node.label === defaultLabel(node.type) ? { label: defaultLabel(type) } : {}),
                                    hyperparameters: { ...(LOSS_DEFAULTS[type] ?? {}) },
                                });
                            }}
                            className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white nodrag"
                        >
                            {LOSS_TYPES.map((t) => <option key={t} value={t}>{optionText(t)}</option>)}
                        </select>
                    </label>
                    <label className="flex items-center justify-between gap-2 py-0.5">
                        <span className="text-[11px] font-mono text-text-secondary">output</span>
                        <select
                            value={node.architectureOutputId ?? ''}
                            onChange={(e) => patchNode(id, { architectureOutputId: e.target.value || null })}
                            className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white nodrag max-w-[160px]"
                        >
                            <option value="">— pick —</option>
                            {outputCandidates.map((o) => (
                                <option key={o.id} value={o.id}>{o.label}</option>
                            ))}
                        </select>
                    </label>
                    <label className="flex items-center justify-between gap-2 py-0.5">
                        <span className="text-[11px] font-mono text-text-secondary">label key</span>
                        <input
                            type="text"
                            value={node.labelKey ?? ''}
                            onChange={(e) => patchNode(id, { labelKey: e.target.value || 'labels' })}
                            className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                            style={{ width: '10ch' }}
                        />
                    </label>
                    <label className="flex items-center justify-between gap-2 py-0.5">
                        <span className="text-[11px] font-mono text-text-secondary">weight</span>
                        <input
                            type="number"
                            step="any"
                            value={node.weight ?? 1}
                            onChange={(e) => patchNode(id, { weight: Number.parseFloat(e.target.value) })}
                            className="border-2 border-black rounded px-1.5 py-0.5 text-[11px] bg-white font-mono nodrag text-right"
                            style={{ width: '6ch' }}
                        />
                    </label>
                    {Object.entries(node.hyperparameters ?? {}).map(([k, v]) => {
                        const isNum = typeof v === 'number';
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
                                    style={{ width: isNum ? '9ch' : '14ch' }}
                                />
                            </label>
                        );
                    })}
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
                aria-label="Delete loss"
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

export default memo(LossNodeView);
