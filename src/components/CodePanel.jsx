import React, { useCallback, useMemo, useState, useSyncExternalStore } from 'react';
import { Code2, Copy, Download, NotebookText, X } from 'lucide-react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';
import { generateArchitectureModule, generateTrainingScript } from '../codegen/index.js';
import { trainingNotebookText } from '../codegen/edu/notebook.js';
import { onSourceRegistered, readSource, subscribeSource } from '../edu/bridge.js';
import { getDataset } from '../edu/data/registry.js';

const TABS = [
    { id: 'model', label: 'model.py' },
    { id: 'train', label: 'train.py' },
];

/**
 * A bridge source's value, re-read when the source changes or registers. `select` picks the part that matters; it
 * is compared as JSON, so busy sources (the trainer publishes progress) only re-render when that part changes.
 */
function useBridgeValue(name, select) {
    const subscribe = useCallback((onChange) => {
        let unsubscribe = subscribeSource(name, onChange);
        const stopListening = onSourceRegistered((registered) => {
            if (registered !== name) return;
            unsubscribe();
            unsubscribe = subscribeSource(name, onChange);
            onChange();
        });
        return () => { unsubscribe(); stopListening(); };
    }, [name]);
    const snapshot = () => JSON.stringify(select(readSource(name)) ?? null);
    const json = useSyncExternalStore(subscribe, snapshot, snapshot);
    return useMemo(() => JSON.parse(json), [json]);
}

const selectBinding = (data) => (data?.datasetId
    ? { datasetId: data.datasetId, options: data.options ?? {}, split: data.split ?? {}, dataSeed: data.dataSeed ?? 1 }
    : null);
const selectSettings = (trainer) => (trainer?.settings
    ? { epochs: trainer.settings.epochs, batchSize: trainer.settings.batchSize, seed: trainer.settings.seed }
    : null);

function saveFile(text, name, type) {
    const blob = new Blob([text], { type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

/**
 * Generated PyTorch code (docs/edu/SPEC.md §3.6, §6.11.3). Controlled: `tab` and `onTabChange` come from the Edu UI
 * store, so lessons can open train.py; without them the panel keeps its own tab. When a dataset is bound (bridge
 * source 'data'), train.py is the complete script with loaders and an evaluation loop; otherwise it is DeepFlow's
 * stub as before.
 */
const CodePanel = ({ open, onClose, tab, onTabChange }) => {
    const archGraph = useArchitectureStore((s) => s.graph);
    const templates = useTemplatesStore((s) => s.templates);
    const trainGraph = useTrainingStore((s) => s.graph);
    const [localTab, setLocalTab] = useState('model');
    const activeTab = TABS.some((t) => t.id === tab) ? tab : localTab;
    const selectTab = (id) => {
        setLocalTab(id);
        onTabChange?.(id);
    };

    const binding = useBridgeValue('data', selectBinding);
    const settings = useBridgeValue('trainer', selectSettings);
    const data = useMemo(() => {
        const spec = binding ? getDataset(binding.datasetId) : null;
        return spec ? { spec, options: binding.options, split: binding.split, dataSeed: binding.dataSeed, ...(settings ? { settings } : {}) } : undefined;
    }, [binding, settings]);

    // Code is generated only while the panel is open.
    const sources = useMemo(() => {
        if (!open) return null;
        try {
            return {
                model: generateArchitectureModule(archGraph, { templates, className: 'Model' }),
                train: generateTrainingScript(archGraph, trainGraph, { templates, data }),
                ok: true,
            };
        } catch (err) {
            return {
                model: `# codegen error: ${err.message}\n${err.stack ?? ''}`,
                train: `# codegen error: ${err.message}\n${err.stack ?? ''}`,
                ok: false,
            };
        }
    }, [open, archGraph, templates, trainGraph, data]);

    if (!open || !sources) return null;
    const code = sources[activeTab] ?? '';
    const fileName = TABS.find((t) => t.id === activeTab)?.label ?? 'code.py';
    const notebook = activeTab === 'train' && !!data && sources.ok;

    const copy = async () => {
        try { await navigator.clipboard.writeText(code); } catch {
            // Clipboard access may be unavailable; the code can still be downloaded.
        }
    };

    return (
        <div
            className="df-panel-subtle absolute flex flex-col overflow-hidden pointer-events-auto"
            role="dialog"
            aria-label="Generated code"
            style={{
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)',
                width: 'min(900px, 92vw)',
                height: 'min(700px, 88vh)',
                // Stay inside the canvas area, which is narrower when the lesson panel is docked.
                maxWidth: 'calc(100% - 16px)',
                maxHeight: 'calc(100% - 16px)',
                zIndex: 60,
            }}
        >
            <div className="relative p-2 pr-11 border-b border-black/15 bg-stone-50 flex flex-wrap items-center gap-2">
                <Code2 size={14} strokeWidth={2.5} aria-hidden="true" />
                <span className="text-xs font-bold uppercase tracking-wider mr-2">Generated Code</span>
                <div className="flex gap-1" role="tablist" aria-label="Files">
                    {TABS.map((t) => (
                        <button
                            key={t.id}
                            type="button"
                            role="tab"
                            aria-selected={activeTab === t.id}
                            data-edu-target={`code-tab-${t.id}`}
                            onClick={() => selectTab(t.id)}
                            className={`px-2 py-0.5 text-xs font-bold border border-black/20 rounded-md ${activeTab === t.id ? 'bg-black text-white border-black' : 'bg-white hover:bg-stone-100'}`}
                        >
                            {t.label}
                        </button>
                    ))}
                </div>
                <div className="ml-auto flex flex-wrap items-center gap-1">
                    <button type="button" onClick={copy} className="df-button h-7 px-2 text-[11px]">
                        <Copy size={11} strokeWidth={3} aria-hidden="true" /> copy
                    </button>
                    <button type="button" onClick={() => saveFile(code, fileName, 'text/x-python')} className="df-button h-7 px-2 text-[11px]">
                        <Download size={11} strokeWidth={3} aria-hidden="true" /> download
                    </button>
                    {notebook && (
                        <button
                            type="button"
                            onClick={() => saveFile(trainingNotebookText(sources.train, { name: 'train.ipynb' }), 'train.ipynb', 'application/x-ipynb+json')}
                            className="df-button h-7 px-2 text-[11px]"
                            data-edu-target="code-notebook"
                        >
                            <NotebookText size={11} strokeWidth={3} aria-hidden="true" /> Download .ipynb
                        </button>
                    )}
                </div>
                <button
                    type="button"
                    onClick={onClose}
                    className="df-button absolute top-2 right-2 h-7 w-7 p-0 hover:bg-red-50"
                    aria-label="Close code panel"
                >
                    <X size={12} strokeWidth={3} aria-hidden="true" />
                </button>
            </div>
            {activeTab === 'train' && data && (
                <p className="px-3 py-1.5 border-b border-black/10 bg-white text-[11px] leading-snug text-stone-700" data-edu-target="code-colab-note">
                    <strong>Run in Colab:</strong> download the .ipynb, open colab.research.google.com, choose File → Upload notebook and run both cells.
                    Or upload train.py and run <code className="font-mono">!python train.py</code>. A GPU is optional. Trains on {data.spec.name}.
                </p>
            )}
            <div className="flex-1 overflow-auto bg-[#0d1117]" role="tabpanel" aria-label={fileName} tabIndex={0}>
                <pre className="text-[12px] leading-[1.5] p-4 font-mono text-stone-100 whitespace-pre">
{code}
                </pre>
            </div>
        </div>
    );
};

export default CodePanel;
