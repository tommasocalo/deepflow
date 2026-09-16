import React from 'react';
import { Code2, FolderOpen, RotateCcw, Save } from 'lucide-react';
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
    const goToLevel = useArchitectureStore((s) => s.goToLevel);
    const trainingGraph = useTrainingStore((s) => s.graph);
    const trainingReset = useTrainingStore((s) => s.resetGraph);
    const templates = useTemplatesStore((s) => s.templates);

    const handleSave = () => {
        const payload = { version: 'deepflow-v2', architecture: archGraph, templates, training: trainingGraph };
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
                goToLevel(0);
                if (data?.architecture) useArchitectureStore.setState((s) => ({ ...s, graph: data.architecture }));
                if (data?.templates) useTemplatesStore.setState((s) => ({ ...s, templates: data.templates }));
                if (data?.training) useTrainingStore.setState((s) => ({ ...s, graph: data.training }));
            } catch (err) {
                alert('Could not load file: ' + (err.message ?? err));
            }
        };
        inp.click();
    };
    const handleReset = () => {
        if (!window.confirm('Reset architecture and training graphs to empty?')) return;
        goToLevel(0);
        archResetGraph();
        trainingReset();
    };

    return (
        <div className="df-toolbar">
            <div className="df-toolbar-tabs">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        disabled={tab.disabled}
                        onClick={() => !tab.disabled && onChange(tab.id)}
                        className={`df-tab${activeTab === tab.id ? ' df-tab-active' : ''}`}
                    >
                        <span>{tab.label}</span>
                        {tab.disabled && <span className="df-tab-soon">soon</span>}
                    </button>
                ))}
            </div>

            <div className="df-toolbar-actions">
                <div className="df-action-group" role="group" aria-label="Project actions">
                    <ActionSegment icon={Code2}      label="Code"  onClick={onShowCode}  title="Generated PyTorch source" />
                    <ActionSegment icon={Save}       label="Save"  onClick={handleSave}  title="Download graph as JSON" />
                    <ActionSegment icon={FolderOpen} label="Load"  onClick={handleLoad}  title="Load a graph JSON" />
                    <ActionSegment icon={RotateCcw}  label="Reset" onClick={handleReset} title="Clear both graphs" />
                </div>
            </div>
        </div>
    );
};

const ActionSegment = ({ icon: Icon, label, onClick, title }) => (
    <button type="button" onClick={onClick} title={title ?? label} className="df-action-segment">
        {Icon ? <Icon size={12} strokeWidth={2.5} /> : null}
        <span className="df-action-label">{label}</span>
    </button>
);

export default ToolbarStrip;
