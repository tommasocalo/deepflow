import React from 'react';
import { Code2, Database, FileJson, Network, RotateCcw, SlidersHorizontal, Upload } from 'lucide-react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';

const TABS = [
    { id: 'architecture', label: 'Architecture', icon: Network },
    { id: 'data', label: 'Data', icon: Database, disabled: true },
    { id: 'training', label: 'Training', icon: SlidersHorizontal },
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
                    useArchitectureStore.setState({ graph: data.architecture, currentPath: [], selectedNodeIds: [], expandedNodeId: null });
                }
                if (data?.templates) {
                    useTemplatesStore.setState((s) => ({ ...s, templates: data.templates }));
                }
                if (data?.training) {
                    useTrainingStore.setState({ graph: data.training, selectedNodeIds: [], expandedNodeId: null });
                }
            } catch (err) {
                alert('Could not load file: ' + (err.message ?? err));
            }
        };
        inp.click();
    };
    const handleReset = () => {
        if (!window.confirm('Reset architecture and training graphs to empty?')) return;
        useArchitectureStore.getState().goToLevel(0);
        archResetGraph();
        trainingReset();
    };

    return (
        <div className="df-toolbar">
            <nav className="df-toolbar-tabs" aria-label="Workspace views">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        disabled={tab.disabled}
                        aria-current={activeTab === tab.id ? 'page' : undefined}
                        onClick={() => !tab.disabled && onChange(tab.id)}
                        className={`df-tab ${activeTab === tab.id ? 'df-tab-active' : ''}`}
                    >
                        <tab.icon size={14} />
                        <span>{tab.label}</span>
                        {tab.disabled && (
                            <span className="df-tab-soon">Soon</span>
                        )}
                    </button>
                ))}
            </nav>

            <div className="df-toolbar-actions">
                <button
                    onClick={onShowCode}
                    className="df-button df-code-button"
                    title="Generated PyTorch source"
                >
                    <Code2 size={13} strokeWidth={2.5} />
                    Code
                </button>
                <div className="df-action-group">
                    <ActionSegment icon={FileJson} label="Save" onClick={handleSave} title="Download your workspace as JSON" />
                    <ActionSegment icon={Upload} label="Load" onClick={handleLoad} title="Open a saved workspace" />
                    <ActionSegment icon={RotateCcw} label="Reset" onClick={handleReset} />
                </div>
            </div>
        </div>
    );
};

const ActionSegment = ({ icon, label, onClick, title }) => {
    const Icon = icon;
    return (
    <button
        onClick={onClick}
        title={title ?? label}
        aria-label={label}
        className="df-action-segment"
    >
        <Icon size={13} strokeWidth={2.5} />
        <span className="df-action-label">{label}</span>
    </button>
    );
};

export default ToolbarStrip;
