import React, { useMemo, useState } from 'react';
import { Code2, Copy, Download, X } from 'lucide-react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';
import { generateArchitectureModule, generateTrainingScript } from '../codegen/index.js';

const TABS = [
    { id: 'model', label: 'model.py' },
    { id: 'train', label: 'train.py' },
];

const CodePanel = ({ open, onClose }) => {
    const archGraph = useArchitectureStore((s) => s.graph);
    const templates = useTemplatesStore((s) => s.templates);
    const trainGraph = useTrainingStore((s) => s.graph);
    const [tab, setTab] = useState('model');

    const sources = useMemo(() => {
        try {
            return {
                model: generateArchitectureModule(archGraph, { templates, className: 'Model' }),
                train: generateTrainingScript(archGraph, trainGraph, { templates }),
            };
        } catch (err) {
            return {
                model: `# codegen error: ${err.message}\n${err.stack ?? ''}`,
                train: `# codegen error: ${err.message}\n${err.stack ?? ''}`,
            };
        }
    }, [archGraph, templates, trainGraph]);

    if (!open) return null;
    const code = sources[tab] ?? '';

    const copy = async () => {
        try { await navigator.clipboard.writeText(code); } catch {}
    };
    const download = () => {
        const blob = new Blob([code], { type: 'text/x-python' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = TABS.find((t) => t.id === tab)?.label ?? 'code.py';
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    };

    return (
        <div
            className="df-panel-subtle absolute flex flex-col overflow-hidden pointer-events-auto"
            style={{
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)',
                width: 'min(900px, 92vw)',
                height: 'min(700px, 88vh)',
                zIndex: 60,
            }}
        >
            <div className="p-2 border-b border-black/15 bg-stone-50 flex items-center gap-2">
                <Code2 size={14} strokeWidth={2.5} />
                <span className="text-xs font-bold uppercase tracking-wider mr-2">Generated Code</span>
                <div className="flex gap-1">
                    {TABS.map((t) => (
                        <button
                            key={t.id}
                            onClick={() => setTab(t.id)}
                            className={`px-2 py-0.5 text-xs font-bold border border-black/20 rounded-md ${tab === t.id ? 'bg-black text-white border-black' : 'bg-white hover:bg-stone-100'}`}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
                <div className="ml-auto flex items-center gap-1">
                    <button
                        onClick={copy}
                        className="df-button h-7 px-2 text-[11px]"
                    >
                        <Copy size={11} strokeWidth={3} /> copy
                    </button>
                    <button
                        onClick={download}
                        className="df-button h-7 px-2 text-[11px]"
                    >
                        <Download size={11} strokeWidth={3} /> download
                    </button>
                    <button
                        onClick={onClose}
                        className="df-button h-7 w-7 p-0 hover:bg-red-50"
                        aria-label="Close code panel"
                    >
                        <X size={12} strokeWidth={3} />
                    </button>
                </div>
            </div>
            <div className="flex-1 overflow-auto bg-[#0d1117]">
                <pre className="text-[12px] leading-[1.5] p-4 font-mono text-stone-100 whitespace-pre">
{code}
                </pre>
            </div>
        </div>
    );
};

export default CodePanel;
