import React from 'react';
import { Code2, FileJson, RotateCcw, Upload } from 'lucide-react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';

const TABS = [
    { id: 'architecture', label: 'Architecture' },
    { id: 'data',         label: 'Data',     disabled: true },
    { id: 'training',     label: 'Training' },
];

function downloadBlob(name, content, mime = 'application/json') {
    const blob = new Blob([content], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const ToolbarStrip = ({ activeTab, onChange, onShowCode }) => {
    const archGraph = useArchitectureStore((s) => s.graph);
    const archResetGraph = useArchitectureStore((s) => s.resetCurrentGraph);
    const trainingGraph = useTrainingStore((s) => s.graph);
    const trainingReset = useTrainingStore((s) => s.resetGraph);
    const templates = useTemplatesStore((s) => s.templates);

    const handleSave = () => {
        const payload = {
            version: 'deepflow-v2',
            architecture: archGraph,
            templates,
            training: trainingGraph,
        };
        downloadBlob('deepflow-graph.json', JSON.stringify(payload, null, 2));
    };
    const handleLoad = () => {
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = 'application/json';
        inp.onchange = async () => {
            const file = inp.files?.[0];
            if (!file) return;
            try {
                const data = JSON.parse(await file.text());
                if (data?.architecture) {
                    useArchitectureStore.setState((s) => ({ ...s, graph: data.architecture }));
                }
                if (data?.templates) {
                    useTemplatesStore.setState((s) => ({ ...s, templates: data.templates }));
                }
                if (data?.training) {
                    useTrainingStore.setState((s) => ({ ...s, graph: data.training }));
                }
            } catch (err) {
                alert('Could not load file: ' + (err.message ?? err));
            }
        };
        inp.click();
    };
    const handleReset = () => {
        if (!window.confirm('Reset architecture and training graphs to empty?')) return;
        archResetGraph();
        trainingReset();
    };

    return (
        <div
            className="df-toolbar w-full flex items-center justify-between gap-4"
            style={{ padding: '0.5rem 1rem' }}
        >
            <div className="inline-flex items-center gap-7">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        disabled={tab.disabled}
                        onClick={() => !tab.disabled && onChange(tab.id)}
                        className={[
                            'df-tab',
                            activeTab === tab.id ? 'df-tab-active' : 'hover:text-black',
                            tab.disabled ? 'opacity-40 cursor-not-allowed hover:text-text-secondary' : 'cursor-pointer',
                        ].join(' ')}
                    >
                        <span>{tab.label}</span>
                        {tab.disabled && (
                            <span className="ml-1 text-[8px] uppercase tracking-wider opacity-60">soon</span>
                        )}
                    </button>
                ))}
            </div>

            <div className="inline-flex items-center gap-5">
                <button
                    onClick={onShowCode}
                    className="df-button min-w-[128px] px-5"
                    title="Generated PyTorch source"
                >
                    <Code2 size={13} strokeWidth={2.5} />
                    Code
                </button>
                <div className="inline-flex border border-black/20 rounded-[7px] overflow-hidden bg-white">
                    <ActionSegment icon={FileJson}  label="Save"  onClick={handleSave}  isFirst />
                    <ActionSegment icon={Upload}    label="Load"  onClick={handleLoad} />
                    <ActionSegment icon={RotateCcw} label="Reset" onClick={handleReset} />
                </div>
            </div>
        </div>
    );
};

const ActionSegment = ({ icon: Icon, label, onClick, isFirst, title }) => (
    <button
        onClick={onClick}
        title={title ?? label}
        className={[
            'inline-flex items-center justify-center gap-1.5 h-8 px-3 text-xs font-bold',
            'transition-colors hover:bg-stone-100',
            isFirst ? '' : 'border-l border-black/20',
        ].join(' ')}
    >
        <Icon size={13} strokeWidth={2.5} />
        {label}
    </button>
);

export default ToolbarStrip;
