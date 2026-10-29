import React, { useState } from 'react';
import { Code2, FolderOpen, RotateCcw, Save } from 'lucide-react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useArchitectureHistory } from '../stores/architectureHistory.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { useTrainingStore } from '../stores/trainingStore.js';
import { importEditableArchitecture } from '../stores/importArchitecture.js';
import { buildEditableOnnxArchitecture } from '../domain/onnx/editable.js';
import OnnxImport from './OnnxImport.jsx';
import { t } from '../edu/lessons/engine/i18n.js';
import { useLessonSession } from '../edu/lessons/engine/session.js';
import { TABS } from '../edu/shell/uiStore.js';
import { applyLoadedPayload, buildSavePayload, restoreEduSlices } from '../edu/shell/workspaceFile.js';

function downloadBlob(name, content, mime = 'application/json') {
    const blob = new Blob([content], { type: mime });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const ToolbarStrip = ({ activeTab, onChange, onShowCode, codeOpen = false }) => {
    const archGraph = useArchitectureStore((s) => s.graph);
    const archResetGraph = useArchitectureStore((s) => s.resetCurrentGraph);
    const goToLevel = useArchitectureStore((s) => s.goToLevel);
    const trainingGraph = useTrainingStore((s) => s.graph);
    const trainingReset = useTrainingStore((s) => s.resetGraph);
    const templates = useTemplatesStore((s) => s.templates);
    const [onnxFile, setOnnxFile] = useState(null);

    const lessonMode = useLessonSession((s) => s.mode === 'lesson');

    const handleSave = () => {
        // Edu slices (dataset binding, trainer settings) travel with the graphs; DeepFlow ignores them.
        const payload = buildSavePayload({ architecture: archGraph, templates, training: trainingGraph });
        downloadBlob('deepflow-graph.json', JSON.stringify(payload, null, 2));
    };
    const handleLoad = () => {
        const inp = document.createElement('input');
        inp.type = 'file';
        inp.accept = '.json,.onnx,application/json';
        inp.onchange = async () => {
            const file = inp.files?.[0];
            if (!file) return;
            if (/\.onnx$/i.test(file.name)) {
                setOnnxFile(file);
                return;
            }
            try {
                applyLoadedPayload(JSON.parse(await file.text()));
            } catch (err) {
                alert('Could not load file: ' + (err.message ?? err));
            }
        };
        inp.click();
    };
    const handleOnnxImport = (operatorGraph, fileName) => {
        const source = importEditableArchitecture(buildEditableOnnxArchitecture(operatorGraph, { fileName }));
        onChange('architecture');
        return source;
    };
    const handleReset = () => {
        if (!window.confirm(t(lessonMode ? 'shell.resetLesson' : 'shell.resetSandbox'))) return;
        goToLevel(0);
        archResetGraph();
        trainingReset();
        restoreEduSlices({});
        useArchitectureHistory.getState().clear();
    };

    return (
        <div className="df-toolbar">
            <div className="df-toolbar-tabs">
                {TABS.map((tab) => (
                    <button
                        key={tab}
                        type="button"
                        data-tab={tab}
                        aria-current={activeTab === tab ? 'page' : undefined}
                        onClick={() => onChange(tab)}
                        className={`df-tab${activeTab === tab ? ' df-tab-active' : ''}`}
                    >
                        <span>{t(`tab.${tab}`)}</span>
                    </button>
                ))}
            </div>

            <div className="df-toolbar-actions">
                <div className="df-action-group" role="group" aria-label="Project actions">
                    <ActionSegment icon={Code2}      label="Code"  onClick={onShowCode}  title="Generated PyTorch source" expanded={codeOpen} />
                    <ActionSegment icon={Save}       label="Save"  onClick={handleSave}  title="Download graph as JSON" />
                    <ActionSegment icon={FolderOpen} label="Load"  onClick={handleLoad}  title="Load a DeepFlow JSON or ONNX model" />
                    <ActionSegment icon={RotateCcw}  label="Reset" onClick={handleReset} title="Clear both graphs" />
                </div>
            </div>
            <OnnxImport open={!!onnxFile} initialFile={onnxFile} onClose={() => setOnnxFile(null)} onImport={handleOnnxImport} />
        </div>
    );
};

const ActionSegment = ({ icon: Icon, label, onClick, title, disabled, expanded }) => (
    <button type="button" onClick={onClick} title={title ?? label} aria-label={label} aria-expanded={expanded} disabled={disabled} className="df-action-segment">
        {Icon ? <Icon size={12} strokeWidth={2.5} /> : null}
        <span className="df-action-label">{label}</span>
    </button>
);

export default ToolbarStrip;
