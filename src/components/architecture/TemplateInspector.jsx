import React, { useState } from 'react';
import { Plus, Trash2, Variable } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';

const PARAM_TYPES = ['int', 'float', 'string', 'bool', 'tuple', 'enum'];

/**
 * Visible only when the user has drilled into a template AND has no
 * single node selected. Lets the author declare free parameters that
 * instances and repeats can bind. Internal layer parameters can reference
 * a free parameter as `$name` (or use it inside an arithmetic expression).
 */
const TemplateInspector = () => {
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const getLevel = useArchitectureStore((s) => s.getLevel);
    const template = useTemplatesStore((s) => {
        const lvl = getLevel();
        return lvl.isRoot ? null : s.templates[lvl.templateId];
    });
    const patchTemplate = useTemplatesStore((s) => s.patchTemplate);
    const addFreeParameter = useTemplatesStore((s) => s.addFreeParameter);
    const patchFreeParameter = useTemplatesStore((s) => s.patchFreeParameter);
    const removeFreeParameter = useTemplatesStore((s) => s.removeFreeParameter);

    const [draftName, setDraftName] = useState('');

    if (currentPath.length === 0 || selectedNodeIds.length > 0) return null;
    if (!template) return null;

    const free = template.freeParameters ?? [];

    const handleAdd = () => {
        const name = draftName.trim();
        if (!name) return;
        addFreeParameter(template.id, { name, type: 'int', description: '', default: null });
        setDraftName('');
    };

    return (
        <div
            className="df-panel-subtle absolute pointer-events-auto overflow-hidden flex flex-col"
            style={{ left: '4rem', top: '1rem', bottom: '1rem', width: '320px', zIndex: 40 }}
        >
            <div className="p-3 border-b border-black/15 bg-amber-50">
                <div className="text-[10px] font-bold uppercase tracking-wider opacity-70 flex items-center gap-1">
                    <Variable size={12} /> Template
                </div>
                <input
                    type="text"
                    value={template.name}
                    onChange={(e) => patchTemplate(template.id, { name: e.target.value })}
                    className="mt-0.5 w-full text-base font-bold tracking-tight bg-transparent outline-none border-b border-transparent focus:border-black"
                />
                <input
                    type="text"
                    placeholder="Description (optional)"
                    value={template.description}
                    onChange={(e) => patchTemplate(template.id, { description: e.target.value })}
                    className="mt-1 w-full text-xs bg-transparent outline-none border-b border-transparent focus:border-black placeholder:text-text-secondary"
                />
                <div className="mt-1 text-[10px] font-mono opacity-60 truncate">{template.id}</div>
            </div>

            <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-white">
                <div>
                    <div className="text-[11px] font-bold uppercase tracking-wider text-text-secondary mb-2">
                        Free parameters
                    </div>
                    <div className="space-y-2">
                        {free.length === 0 && (
                            <div className="text-xs text-text-secondary italic">
                                None yet. Declare a parameter and reference it from internal nodes as
                                <span className="font-mono"> $name</span>.
                            </div>
                        )}
                        {free.map((p) => (
                            <div key={p.name} className="border border-black/15 rounded-md p-2 bg-stone-50 space-y-1">
                                <div className="flex items-center justify-between gap-2">
                                    <span className="font-mono text-xs font-bold">${p.name}</span>
                                    <button
                                        onClick={() => removeFreeParameter(template.id, p.name)}
                                        className="bg-white border border-black/20 rounded p-0.5 hover:bg-red-50"
                                        aria-label={`Remove ${p.name}`}
                                    >
                                        <Trash2 size={10} strokeWidth={3} />
                                    </button>
                                </div>
                                <select
                                    value={p.type}
                                    onChange={(e) => patchFreeParameter(template.id, p.name, { type: e.target.value })}
                                    className="df-input w-full px-1 py-0.5 text-xs bg-white"
                                >
                                    {PARAM_TYPES.map((t) => (
                                        <option key={t} value={t}>{t}</option>
                                    ))}
                                </select>
                                <input
                                    type="text"
                                    placeholder="default (optional)"
                                    value={p.default ?? ''}
                                    onChange={(e) => patchFreeParameter(template.id, p.name, { default: e.target.value || null })}
                                    className="df-input w-full px-1 py-0.5 text-xs bg-white font-mono"
                                />
                                <input
                                    type="text"
                                    placeholder="description"
                                    value={p.description ?? ''}
                                    onChange={(e) => patchFreeParameter(template.id, p.name, { description: e.target.value })}
                                    className="df-input w-full px-1 py-0.5 text-[11px] bg-white"
                                />
                            </div>
                        ))}
                    </div>

                    <div className="mt-2 flex items-center gap-1">
                        <input
                            type="text"
                            placeholder="new param name"
                            value={draftName}
                            onChange={(e) => setDraftName(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
                            className="df-input flex-1 px-2 py-1 text-xs font-mono"
                        />
                        <button
                            onClick={handleAdd}
                            className="df-button df-button-primary px-2"
                            aria-label="Add parameter"
                        >
                            <Plus size={12} strokeWidth={3} />
                        </button>
                    </div>
                </div>

                <div className="border-t-2 border-black/20 pt-2 text-[11px] text-text-secondary">
                    Tip: in any internal-node parameter, type
                    <span className="font-mono"> $param </span>
                    or an expression like
                    <span className="font-mono"> 64 * 2 ** $stage </span>
                    to bind to free parameters.
                </div>
            </div>
        </div>
    );
};

export default TemplateInspector;
