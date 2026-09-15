import React, { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { AlertCircle, AlertTriangle, ChevronUp, X } from 'lucide-react';
import { PYTORCH_BLOCKS, defaultParameterValues, getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { formatShape } from '../../validation/index.js';
import { styleFor } from './categoryStyle.js';
import ParameterField from './ParameterField.jsx';

const SHAPE_PRIORITY = [
    'in_channels', 'out_channels',
    'in_features', 'out_features',
    'embed_dim', 'num_heads',
    'd_model', 'nhead',
    'hidden_size', 'input_size',
    'num_features',
    'kernel_size',
    'p',
    'shape', 'name',
];

function summaryFields(entry, parameters) {
    if (!entry) return [];
    const ordered = entry.parameters
        .map((p) => p.name)
        .sort((a, b) => {
            const ai = SHAPE_PRIORITY.indexOf(a);
            const bi = SHAPE_PRIORITY.indexOf(b);
            if (ai === -1 && bi === -1) return 0;
            if (ai === -1) return 1;
            if (bi === -1) return -1;
            return ai - bi;
        })
        .filter((name) => parameters[name] !== null && parameters[name] !== undefined && parameters[name] !== '');
    return ordered.slice(0, 2).map((name) => ({ name, value: parameters[name] }));
}

function shouldToggleOnCardClick(target) {
    if (!target || !target.closest) return true;
    return !target.closest('input, button, select, textarea, label, a');
}

const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);

const ParamFields = ({ params, node, errors, onPatch }) => (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        {params.map((p) => (
            <ParameterField
                key={p.name}
                param={p}
                value={node.parameters[p.name]}
                onChange={(v) => onPatch({ [p.name]: v })}
                error={errors.some((e) => e.toLowerCase().includes(p.name.toLowerCase()))}
            />
        ))}
    </div>
);

const ParamGroup = ({ title, params, node, errors, onPatch, collapsed = false }) => {
    if (!params.length) return null;
    if (collapsed) {
        return (
            <details className="pt-1 border-t border-black/10">
                <summary className="cursor-pointer list-none text-[9px] font-bold uppercase tracking-wider opacity-60 hover:opacity-90">
                    {title} <span className="font-mono">({params.length})</span>
                </summary>
                <div className="mt-0.5">
                    <ParamFields params={params} node={node} errors={errors} onPatch={onPatch} />
                </div>
            </details>
        );
    }
    return (
        <section className="pt-1 border-t border-black/10 first:border-t-0 first:pt-0">
            <div className="mb-0.5 text-[9px] font-bold uppercase tracking-wider opacity-60">
                {title}
            </div>
            <ParamFields params={params} node={node} errors={errors} onPatch={onPatch} />
        </section>
    );
};

const LayerNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const entry = getEntry(node.libraryKey);
    const removeNode = useArchitectureStore((s) => s.removeNode);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const patchLayerParameters = useArchitectureStore((s) => s.patchLayerParameters);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const toggleExpanded = useArchitectureStore((s) => s.toggleExpanded);
    const validation = useValidation();
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const outputShape = validation.shapes[id];
    const style = styleFor(entry?.category ?? 'IO');
    const showExpanded = expandedNodeId === id;
    const isDarkCard = entry?.category === 'IO';
    const moduleOptions = entry
        ? PYTORCH_BLOCKS.filter((candidate) =>
            candidate.category === entry.category && !TEMPLATE_ONLY.has(candidate.key),
        )
        : [];

    const borderColor = errors.length ? '#dc2626' : warnings.length ? '#f59e0b' : '#000000';
    const shadow = errors.length
        ? `${selected ? '5px 5px' : '3px 3px'} 0px 0px #dc2626`
        : warnings.length
            ? `${selected ? '5px 5px' : '3px 3px'} 0px 0px #f59e0b`
            : `${selected ? '5px 5px' : '3px 3px'} 0px 0px #000000`;

    const collapsedFields = summaryFields(entry, node.parameters);
    const params = entry?.parameters ?? [];
    const requiredParams = params.filter((p) => p.required);
    const shapeParams = params.filter((p) => !p.required && SHAPE_PRIORITY.includes(p.name));
    const optionParams = params.filter((p) => !p.required && !SHAPE_PRIORITY.includes(p.name));
    const onCardClick = useCallback((e) => {
        if (shouldToggleOnCardClick(e.target)) toggleExpanded(id);
    }, [id, toggleExpanded]);
    const switchModule = useCallback((libraryKey) => {
        const nextEntry = getEntry(libraryKey);
        if (!nextEntry || nextEntry.key === node.libraryKey) return;
        patchNode(id, {
            libraryKey: nextEntry.key,
            label: nextEntry.name,
            parameters: defaultParameterValues(nextEntry),
        });
    }, [id, node.libraryKey, patchNode]);

    return (
        <div
            className={`${style.card} rounded-md relative group p-3 pr-9 ${showExpanded ? 'pb-6' : ''}`}
            style={{
                width: showExpanded ? 320 : 240,
                border: `2px solid ${borderColor}`,
                boxShadow: shadow,
                transition: 'width 180ms ease, box-shadow 120ms ease',
            }}
            onClick={onCardClick}
        >
            <button
                aria-label="Delete node"
                onClick={(e) => { e.stopPropagation(); removeNode(id); }}
                onMouseDown={(e) => e.stopPropagation()}
                className={[
                    'nodrag absolute -right-1.5 -top-1.5 z-20 h-4 w-4 rounded-full transition-colors inline-flex items-center justify-center',
                    isDarkCard
                        ? 'bg-white text-black hover:bg-red-200'
                        : 'bg-white text-black border border-black/15 hover:bg-red-100 hover:border-red-400',
                ].join(' ')}
            >
                <X size={10} strokeWidth={3} />
            </button>

            <div className="flex items-center gap-2">
                {showExpanded ? (
                    <input
                        value={node.label}
                        onChange={(e) => patchNode(id, { label: e.target.value })}
                        onClick={(e) => e.stopPropagation()}
                        className="text-sm font-bold tracking-tight bg-transparent outline-none border-b border-transparent focus:border-current min-w-0 flex-1 nodrag"
                    />
                ) : (
                    <h3 className="text-sm font-bold tracking-tight truncate flex flex-1 items-center gap-1 min-w-0">
                        {errors.length > 0 && <AlertCircle size={14} className="text-red-600 shrink-0" strokeWidth={3} />}
                        {errors.length === 0 && warnings.length > 0 && <AlertTriangle size={14} className="text-amber-600 shrink-0" strokeWidth={3} />}
                        <span className="truncate">{node.label}</span>
                    </h3>
                )}
                <span className="text-[10px] font-bold uppercase tracking-wider opacity-70 whitespace-nowrap shrink-0">
                    {entry?.category ?? 'unknown'}
                </span>
            </div>

            {!showExpanded && collapsedFields.length > 0 && (
                <div className="mt-2 space-y-0.5">
                    {collapsedFields.map((f) => (
                        <div key={f.name} className="flex justify-between items-center text-[11px] opacity-80">
                            <span>{f.name}</span>
                            <span className="font-bold font-mono">{String(f.value)}</span>
                        </div>
                    ))}
                </div>
            )}

            {showExpanded && (
                <div
                    className="mt-2 nopan nowheel"
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                >
                    {moduleOptions.length > 1 && (
                        <label className="mb-2 inline-flex items-center gap-1.5">
                            <span className="text-[9px] font-bold uppercase tracking-wider opacity-60">
                                module
                            </span>
                            <select
                                value={node.libraryKey}
                                onChange={(e) => switchModule(e.target.value)}
                                onClick={(e) => e.stopPropagation()}
                                className="nodrag df-input h-6 px-1.5 text-[11px] font-semibold"
                                style={{ maxWidth: '15ch' }}
                            >
                                {moduleOptions.map((option) => (
                                    <option key={option.key} value={option.key}>{option.name}</option>
                                ))}
                            </select>
                        </label>
                    )}
                    {entry?.description && (
                        <p className="text-[10px] opacity-70 mb-2 leading-snug">{entry.description}</p>
                    )}
                    {params.length ? (
                        <div className="space-y-1.5">
                            <ParamGroup
                                title="Required"
                                params={requiredParams}
                                node={node}
                                errors={errors}
                                onPatch={(patch) => patchLayerParameters(id, patch)}
                            />
                            <ParamGroup
                                title={requiredParams.length ? 'Shape' : 'Parameters'}
                                params={shapeParams}
                                node={node}
                                errors={errors}
                                onPatch={(patch) => patchLayerParameters(id, patch)}
                            />
                            <ParamGroup
                                title="Options"
                                params={optionParams}
                                node={node}
                                errors={errors}
                                onPatch={(patch) => patchLayerParameters(id, patch)}
                                collapsed={requiredParams.length > 0 || shapeParams.length > 0}
                            />
                        </div>
                    ) : (
                        <div className="text-[11px] italic opacity-60">No parameters.</div>
                    )}

                    {errors.length > 0 && (
                        <div className="mt-2 border border-red-200 border-l-4 border-l-red-600 rounded-md bg-red-50/80 px-3 py-2">
                            <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-red-700 mb-1">
                                <AlertCircle size={10} strokeWidth={3} /> Errors
                            </div>
                            <ul className="text-[11px] text-red-700 list-disc pl-3.5 leading-snug">
                                {errors.map((e, i) => <li key={i}>{e}</li>)}
                            </ul>
                        </div>
                    )}
                    {warnings.length > 0 && (
                        <div className="mt-2 border border-amber-200 border-l-4 border-l-amber-500 rounded-md bg-amber-50/80 px-3 py-2">
                            <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-wider text-amber-700 mb-1">
                                <AlertTriangle size={10} strokeWidth={3} /> Warnings
                            </div>
                            <ul className="text-[11px] text-amber-800 list-disc pl-3.5 leading-snug">
                                {warnings.map((w, i) => <li key={i}>{w}</li>)}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {outputShape && (
                <div className="mt-2 pt-1.5 border-t border-black/20 text-[10px] opacity-70 font-mono truncate">
                    out: {formatShape(outputShape)}
                </div>
            )}

            {showExpanded && (
                <button
                    onMouseDown={(e) => e.stopPropagation()}
                    onClick={(e) => { e.stopPropagation(); toggleExpanded(id); }}
                    className="nodrag absolute right-2 bottom-2 p-0.5 hover:bg-black/10 rounded"
                    title="Collapse"
                    aria-label="Collapse node"
                >
                    <ChevronUp size={12} strokeWidth={3} />
                </button>
            )}

            <Handle
                type="target"
                position={Position.Top}
                className="w-3! h-3! bg-white! border-2! border-black! rounded-full!"
                style={{ top: -7 }}
            />
            <Handle
                type="source"
                position={Position.Bottom}
                className="w-3! h-3! bg-white! border-2! border-black! rounded-full!"
                style={{ bottom: -7 }}
            />
        </div>
    );
};

export default memo(LayerNodeView);
