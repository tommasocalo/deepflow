import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Boxes, Braces, Check, ChevronDown, Columns2, Copy, FileUp, GitBranch, Group, Maximize, Redo2, Ruler, Search, Undo2, Variable, X } from 'lucide-react';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { importHuggingFaceArchitecture } from '../../domain/huggingFace/index.js';
import { buildEditableOnnxArchitecture } from '../../domain/onnx/editable.js';
import { effectiveTemplateId, inspectedInstanceIndex } from '../../domain/architecture/instanceStructure.js';
import { resolveComponentContext } from '../../domain/componentContext.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useTrainingStore } from '../../stores/trainingStore.js';
import { recordArchitectureEdit, startArchitectureSession, useArchitectureHistory } from '../../stores/architectureHistory.js';
import HuggingFaceImport from '../HuggingFaceImport.jsx';
import OnnxImport from '../OnnxImport.jsx';
import { useOpenToSide } from './OpenToSideContext.jsx';
import BlockPalette from './BlockPalette.jsx';
import OptimizationPanel from './OptimizationPanel.jsx';
import TemplateInspector, { GlobalParameters } from './TemplateInspector.jsx';
import './ribbon.css';

const MENU_WIDTHS = { operations: 350, components: 360, templates: 310, parameters: 350, group: 280 };
function RepeatIcon({ parallel }) {
    return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="square" aria-hidden="true">
        {parallel ? <><path d="M12 2v3H4v4m8-4h8v4M4 15v4h16v-4m-8 4v3" /><rect x="1" y="9" width="6" height="6" rx="1" /><rect x="17" y="9" width="6" height="6" rx="1" /></>
            : <><rect x="7" y="1" width="10" height="6" rx="1" /><rect x="7" y="17" width="10" height="6" rx="1" /><path d="M12 8v7m-3-3 3 3 3-3" /></>}
    </svg>;
}
function RibbonButton({ icon: Icon, children, className = '', ...props }) {
    return <button type="button" className={`df-ribbon-button ${className}`} {...props}>{Icon && <Icon size={21} strokeWidth={1.8} aria-hidden="true" />}<span>{children}</span></button>;
}
function RibbonGroup({ label, children, className = '' }) {
    return <section className={`df-ribbon-group ${className}`} aria-label={label}><div className="df-ribbon-actions">{children}</div><h2>{label}</h2></section>;
}

/** Commands target the last focused pane; menus are attached to their trigger. */
export default function ArchitectureRibbon({ paneId, split, onInsert, onFitView }) {
    const architectureStore = useArchitectureStoreApi();
    const graph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const showGradientFlow = useArchitectureStore((s) => s.showGradientFlow);
    const showDimensions = useArchitectureStore((s) => s.showDimensions) !== false;
    const canUndo = useArchitectureHistory((s) => s.canUndo);
    const canRedo = useArchitectureHistory((s) => s.canRedo);
    const undoLabel = useArchitectureHistory((s) => s.undoLabel);
    const redoLabel = useArchitectureHistory((s) => s.redoLabel);
    const undo = useArchitectureHistory((s) => s.undo);
    const redo = useArchitectureHistory((s) => s.redo);
    const templates = useTemplatesStore((s) => s.templates);
    const validation = useValidation();
    const validationContext = useValidationContext();
    const openToSide = useOpenToSide();
    const [popup, setPopup] = useState(null);
    const [hfOpen, setHfOpen] = useState(false);
    const [onnxOpen, setOnnxOpen] = useState(false);
    const [optimizationOpen, setOptimizationOpen] = useState(false);
    const [popupPosition, setPopupPosition] = useState({ left: 8, top: 56, width: 310 });
    const [parameterScope, setParameterScope] = useState('global');
    const [groupName, setGroupName] = useState('Component');
    const [notice, setNotice] = useState('');
    const [error, setError] = useState('');
    const ribbonRef = useRef(null);
    const triggerRef = useRef(null);
    const operationsButtonRef = useRef(null);
    const level = resolveGraphLevel(graph, currentPath, templates, inspectedIterations);
    const selection = selectedNodeIds.map((id) => level.graph.nodes[id]).filter(Boolean);
    const selectedComponent = selection.length === 1 && ['supernode', 'repeat'].includes(selection[0].kind) ? selection[0] : null;
    const selectedIteration = selectedComponent ? inspectedInstanceIndex(selectedComponent, [...currentPath, selectedComponent.id], inspectedIterations) : 0;
    const templateId = selectedComponent ? effectiveTemplateId(selectedComponent, selectedIteration) : level.templateId;
    const template = templates[templateId];
    const selectedRepeatMode = selectedComponent?.kind === 'repeat' ? selectedComponent.mode ?? 'sequential' : null;
    const activeParameterScope = parameterScope === 'component' && template ? 'component' : 'global';

    const closePopup = useCallback((restoreFocus = false) => {
        setPopup(null);
        setError('');
        if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
    }, []);
    const closeHf = useCallback(() => setHfOpen(false), []);
    const closeOnnx = useCallback(() => setOnnxOpen(false), []);
    const closeOptimization = useCallback(() => setOptimizationOpen(false), []);
    const importHf = useCallback((preview) => {
        const previous = {
            architecture: useArchitectureStore.getState(),
            templates: useTemplatesStore.getState(),
            training: useTrainingStore.getState(),
            pane: architectureStore.getState(),
        };
        const source = startArchitectureSession(() => {
            try {
                architectureStore.getState().goToLevel(0);
                return importHuggingFaceArchitecture(preview, {
                    architectureStore: useArchitectureStore,
                    templatesStore: useTemplatesStore,
                    trainingStore: useTrainingStore,
                });
            } catch (error) {
                useTemplatesStore.setState(previous.templates, true);
                useArchitectureStore.setState(previous.architecture, true);
                useTrainingStore.setState(previous.training, true);
                architectureStore.setState(previous.pane, true);
                throw error;
            }
        });
        const blocks = source.configuration?.nLayers;
        setNotice(`Imported ${source.repoId}${blocks ? ` · ${blocks} blocks` : ''} · config only`);
        requestAnimationFrame(() => requestAnimationFrame(() => onFitView?.()));
        return source;
    }, [architectureStore, onFitView]);
    const importOnnx = useCallback((operatorGraph, fileName) => {
        const mapped = buildEditableOnnxArchitecture(operatorGraph, { fileName });
        const previous = {
            architecture: useArchitectureStore.getState(),
            templates: useTemplatesStore.getState(),
            training: useTrainingStore.getState(),
            pane: architectureStore.getState(),
        };
        const source = startArchitectureSession(() => {
            try {
                architectureStore.getState().goToLevel(0);
                useArchitectureStore.setState({
                    graph: mapped.graph, currentPath: [], openPaths: [], selectedNodeIds: [],
                    expandedNodeId: null, inspectedIterations: {}, repeatSelections: {}, repeatEditModes: {}, repeatEditError: null,
                });
                if (architectureStore !== useArchitectureStore) architectureStore.setState({
                    currentPath: [], openPaths: [], selectedNodeIds: [], expandedNodeId: null,
                    inspectedIterations: {}, repeatSelections: {}, repeatEditModes: {}, repeatEditError: null,
                });
                useTrainingStore.getState().resetGraph();
                return mapped.source;
            } catch (error) {
                useTemplatesStore.setState(previous.templates, true);
                useArchitectureStore.setState(previous.architecture, true);
                useTrainingStore.setState(previous.training, true);
                architectureStore.setState(previous.pane, true);
                throw error;
            }
        });
        setNotice(`Imported ${source.modelName} · ${source.opCount} ONNX operators mapped`);
        requestAnimationFrame(() => requestAnimationFrame(() => onFitView?.()));
        return source;
    }, [architectureStore, onFitView]);
    const positionPopup = useCallback((name, trigger) => {
        const ribbon = ribbonRef.current?.getBoundingClientRect();
        const anchor = trigger?.getBoundingClientRect();
        if (!ribbon || !anchor) return;
        const width = Math.min(MENU_WIDTHS[name], Math.max(0, ribbon.width - 16), window.innerWidth - 16);
        setPopupPosition({
            left: Math.max(8, Math.min(anchor.left - ribbon.left, ribbon.width - width - 8)),
            top: anchor.bottom - ribbon.top + 3,
            width,
        });
    }, []);
    const showPopup = useCallback((name, trigger) => {
        positionPopup(name, trigger);
        triggerRef.current = trigger;
        if (name === 'parameters') setParameterScope(template ? 'component' : 'global');
        setNotice('');
        setError('');
        setPopup(name);
    }, [positionPopup, template]);
    useEffect(() => {
        if (!popup) return;
        const resize = () => positionPopup(popup, triggerRef.current);
        window.addEventListener('resize', resize);
        return () => window.removeEventListener('resize', resize);
    }, [popup, positionPopup]);
    useEffect(() => {
        if (!popup) return;
        const dismissOutside = (event) => { if (!ribbonRef.current?.contains(event.target)) closePopup(); };
        document.addEventListener('pointerdown', dismissOutside, true);
        return () => document.removeEventListener('pointerdown', dismissOutside, true);
    }, [popup, closePopup]);
    useEffect(() => {
        const keyboardSearch = (event) => {
            const editing = event.target?.closest?.('input, textarea, select, [contenteditable="true"]');
            if ((!editing && event.key === '/') || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
                event.preventDefault();
                showPopup('operations', operationsButtonRef.current);
                requestAnimationFrame(() => ribbonRef.current?.querySelector('[aria-label="Search operations"]')?.focus());
            }
        };
        window.addEventListener('keydown', keyboardSearch);
        return () => window.removeEventListener('keydown', keyboardSearch);
    }, [showPopup]);
    useEffect(() => {
        if (!notice) return;
        const timer = setTimeout(() => setNotice(''), 2800);
        return () => clearTimeout(timer);
    }, [notice]);

    function togglePopup(name, event) {
        if (popup === name) closePopup();
        else showPopup(name, event.currentTarget);
    }
    function addEntry(entry, isTemplate = false) {
        const id = onInsert(entry, isTemplate);
        if (id) { setNotice(`${entry.name} added`); closePopup(true); }
    }
    function groupNodes(name) {
        const state = architectureStore.getState();
        const id = state.groupSelection([...state.selectedNodeIds], name.trim() || 'Component');
        if (!id) { setError('Select blocks with a single input and output boundary to group them.'); return null; }
        const latest = architectureStore.getState();
        const current = resolveGraphLevel(latest.graph, latest.currentPath, useTemplatesStore.getState().templates, latest.inspectedIterations).graph;
        const instance = Object.values(current.nodes).find((node) => node.kind === 'supernode' && node.templateId === id);
        if (instance) latest.selectOnly(instance.id);
        return instance?.id ?? null;
    }
    function confirmGroup(event) {
        event.preventDefault();
        if (selection.length < 2) return;
        if (!groupNodes(groupName)) return;
        setNotice(`${groupName.trim() || 'Component'} created`);
        closePopup(true);
    }
    function repeatSelection(mode) {
        setError('');
        const disableRepeat = selectedRepeatMode === mode;
        let explicitCount;
        if (!disableRepeat && mode === 'sequential' && (selectedComponent?.count === 'auto' || selectedComponent?.repeatSettings?.count === 'auto')) {
            const repeated = { ...selectedComponent, kind: 'repeat', count: 'auto', mode: 'parallel' };
            const result = resolveComponentContext(repeated, template, level.graph, validation.shapes, validationContext.scope);
            if (!result.count) { setError(result.error ?? 'Connect the component before choosing a sequential count.'); return; }
            explicitCount = result.count;
        }
        const label = disableRepeat ? 'Use single instance' : `Repeat ${mode === 'parallel' ? 'in parallel' : 'sequentially'}`;
        const id = recordArchitectureEdit(label, () => {
            const id = selectedComponent?.id ?? groupNodes(selection.length === 1 ? selection[0].label : 'Component');
            if (!id) return null;
            const state = architectureStore.getState();
            if (disableRepeat) {
                const issue = state.getSingleConversionIssue(id);
                if (issue) { setError(issue); return null; }
                if (!state.convertRepeatToInstance(id)) return null;
            }
            else if (selectedComponent?.kind === 'repeat') state.patchRepeatNode(id, { mode, ...(explicitCount !== undefined ? { count: explicitCount } : {}) });
            else if (!state.convertInstanceToRepeat(id, explicitCount, mode)) return null;
            state.selectOnly(id);
            state.setExpanded(id);
            return id;
        });
        if (!id) return;
        closePopup();
        setNotice(disableRepeat ? 'Single instance' : `${mode === 'parallel' ? 'Parallel' : 'Sequential'} repeat selected`);
    }
    const targetLabel = split ? `${paneId === 'side' ? 'Right' : 'Left'} pane` : 'Current graph';
    const menu = ['operations', 'components', 'templates'].includes(popup);
    const popoverStyle = popupPosition;

    return <div ref={ribbonRef} className="df-architecture-ribbon" role="region" aria-label={`Architecture ribbon — ${targetLabel}`} data-active-pane={paneId}
        onKeyDown={(event) => { if (event.key === 'Escape' && popup) { event.preventDefault(); event.stopPropagation(); closePopup(true); } }}>
        <div className="df-ribbon-row">
            <RibbonGroup label="Insert" className="df-ribbon-insert">
                <RibbonButton icon={Boxes} aria-label="Browse components" aria-expanded={popup === 'components'} aria-haspopup="menu" onClick={(event) => togglePopup('components', event)} title="All editable components and layer modules">Components <ChevronDown size={10} /></RibbonButton>
                <button ref={operationsButtonRef} type="button" className="df-ribbon-button" aria-label="Insert operations" aria-keyshortcuts="/ Meta+k Control+k"
                    aria-expanded={popup === 'operations'} aria-haspopup="menu" onClick={(event) => togglePopup('operations', event)} title="Math, routing, sources and outputs (⌘K / Ctrl+K)">
                    <Braces size={21} strokeWidth={1.8} /><span>Operations <ChevronDown size={10} /></span>
                </button>
                <RibbonButton icon={Copy} aria-label="Browse templates" aria-expanded={popup === 'templates'} aria-haspopup="menu" onClick={(event) => togglePopup('templates', event)} title="Insert a saved project template">Templates <ChevronDown size={10} /></RibbonButton>
                <RibbonButton icon={Search} aria-label="Import from Hugging Face" onClick={() => { closePopup(); setHfOpen(true); }} title="Search Hugging Face model configs and import a supported architecture">Hugging Face</RibbonButton>
                <RibbonButton icon={FileUp} aria-label="Import ONNX file" onClick={() => { closePopup(); setOnnxOpen(true); }} title="Inspect a local ONNX operator graph and import a fully mapped architecture">ONNX</RibbonButton>
            </RibbonGroup>
            <RibbonGroup label="Structure">
                <RibbonButton icon={Group} disabled={selection.length < 2} aria-label="Group selection into component" aria-expanded={popup === 'group' && selection.length > 1} onClick={(event) => { setGroupName('Component'); togglePopup('group', event); }} title={selection.length > 1 ? 'Group selected blocks into a reusable component' : 'Select at least two blocks to group'}>Group</RibbonButton>
                <button type="button" className="df-ribbon-button" disabled={!selection.length} aria-label="Repeat selection sequentially" aria-pressed={selectedRepeatMode === 'sequential'} onClick={() => repeatSelection('sequential')} title={selectedRepeatMode === 'sequential' ? 'Return to a single instance' : 'Repeat in sequence; each output feeds the next'}><RepeatIcon /><span>Sequential</span></button>
                <button type="button" className="df-ribbon-button" disabled={!selection.length} aria-label="Repeat selection in parallel" aria-pressed={selectedRepeatMode === 'parallel'} onClick={() => repeatSelection('parallel')} title={selectedRepeatMode === 'parallel' ? 'Return to a single instance' : 'Repeat in parallel; combine outputs with Concat, Add or Stack'}><RepeatIcon parallel /><span>Parallel</span></button>
            </RibbonGroup>
            <RibbonGroup label="Parameters" className="df-ribbon-component">
                <RibbonButton icon={Variable} aria-label="Edit parameters" aria-expanded={popup === 'parameters'} aria-haspopup="dialog" onClick={(event) => togglePopup('parameters', event)} title="Edit global parameters or parameters of the current component">Parameters <ChevronDown size={10} /></RibbonButton>
            </RibbonGroup>
            <RibbonGroup label="Optimization" className="df-ribbon-optimization">
                <RibbonButton icon={Search} aria-label="Explore architecture settings" aria-expanded={optimizationOpen} aria-haspopup="dialog" onClick={() => { closePopup(); setOptimizationOpen(true); }} title="Compare global parameter settings by shape validity and model size">Explore</RibbonButton>
            </RibbonGroup>
            <RibbonGroup label="View">
                <RibbonButton icon={Columns2} onClick={() => { closePopup(); openToSide?.(selectedComponent ? [...currentPath, selectedComponent.id] : currentPath); }} title="Open selected component or current level to the side" aria-label="Open to the side">To side</RibbonButton>
                <RibbonButton icon={Maximize} onClick={onFitView} title={`Fit graph in ${targetLabel.toLowerCase()}`} aria-label="Fit active graph">Fit</RibbonButton>
                <div className="df-ribbon-view-options">
                    <button type="button" aria-label="Show dimensions" aria-pressed={showDimensions} onClick={() => architectureStore.getState().setShowDimensions(!showDimensions)} title="Show tensor dimensions at input and output ports"><Ruler size={14} />Dimensions</button>
                    <button type="button" aria-label="Show gradient flow" aria-pressed={!!showGradientFlow} onClick={() => architectureStore.getState().setShowGradientFlow(!showGradientFlow)} title="Show differentiable paths and stopped gradients"><GitBranch size={14} />Gradients</button>
                </div>
            </RibbonGroup>
            <RibbonGroup label="History" className="df-ribbon-history">
                <RibbonButton icon={Undo2} aria-label="Undo" aria-keyshortcuts="Meta+z Control+z" disabled={!canUndo} onClick={undo} title={canUndo ? `Undo ${undoLabel} (⌘Z / Ctrl+Z)` : 'Undo (⌘Z / Ctrl+Z)'}>Undo</RibbonButton>
                <RibbonButton icon={Redo2} aria-label="Redo" aria-keyshortcuts="Meta+Shift+z Control+Shift+z Control+y" disabled={!canRedo} onClick={redo} title={canRedo ? `Redo ${redoLabel} (⌘⇧Z / Ctrl+Shift+Z)` : 'Redo (⌘⇧Z / Ctrl+Shift+Z)'}>Redo</RibbonButton>
            </RibbonGroup>
            {split && <span className="df-ribbon-pane-cue" aria-live="polite"><ArrowUpRight size={11} />{targetLabel}</span>}
        </div>
        {menu && <div className="df-ribbon-dropdown" style={popoverStyle} data-ribbon-menu={popup} onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}>
            <BlockPalette key={popup} section={popup} onAdd={addEntry} onClose={() => closePopup(true)} />
        </div>}
        {popup === 'group' && selection.length > 1 && <div className="df-ribbon-popover df-ribbon-popover-group" style={popoverStyle} role="dialog" aria-label="Create component">
            <header><strong>Group {selection.length} {selection.length === 1 ? 'block' : 'blocks'}</strong><button type="button" onClick={() => closePopup(true)} aria-label="Close group editor"><X size={14} /></button></header>
            <form onSubmit={confirmGroup}><label>Component name<input autoFocus value={groupName} onChange={(event) => setGroupName(event.target.value)} /></label><p>The selection becomes a reusable template.</p>{error && <p role="alert" className="df-ribbon-error">{error}</p>}<button className="df-ribbon-primary" type="submit">Create component</button></form>
        </div>}
        {popup === 'parameters' && <div className="df-ribbon-popover df-ribbon-popover-parameters" style={popoverStyle} role="dialog"
            aria-label={activeParameterScope === 'component' ? `${template.name} parameters` : 'Global parameters'} onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}>
            <div className="df-ribbon-parameter-bar">
                <div className="df-ribbon-parameter-tabs" role="tablist" aria-label="Parameter scope">
                    <button type="button" role="tab" aria-selected={activeParameterScope === 'global'} onClick={() => setParameterScope('global')}>Global</button>
                    <button type="button" role="tab" aria-selected={activeParameterScope === 'component'} disabled={!template} onClick={() => setParameterScope('component')}
                        title={template ? `Parameters of ${template.name}` : 'Open or select a component'}>Component</button>
                </div>
                <button type="button" className="df-ribbon-parameter-close" onClick={() => closePopup(true)} aria-label="Close parameters"><X size={14} /></button>
            </div>
            {activeParameterScope === 'component' ? <TemplateInspector templateId={templateId} /> : <GlobalParameters />}
        </div>}
        <OptimizationPanel open={optimizationOpen} onClose={closeOptimization} />
        <HuggingFaceImport open={hfOpen} onClose={closeHf} onImport={importHf} currentArchitecture={{
            nLayers: graph.parameters?.find((parameter) => parameter.name === 'n_layers')?.default,
            dModel: graph.parameters?.find((parameter) => parameter.name === 'd_model')?.default,
            nHeads: graph.parameters?.find((parameter) => parameter.name === 'n_heads')?.default,
            source: graph.importSource?.repoId,
        }} />
        <OnnxImport open={onnxOpen} onClose={closeOnnx} onImport={importOnnx}
            currentSource={graph.importSource?.provider === 'onnx' ? graph.importSource : null} />
        {(notice || (error && popup !== 'group')) && <div className={`df-ribbon-notice${error ? ' is-error' : ''}`} role={error ? 'alert' : 'status'}>{!error && <Check size={12} />}<span>{error || notice}</span><button type="button" aria-label="Dismiss message" onClick={() => { setNotice(''); setError(''); }}><X size={12} /></button></div>}
    </div>;
}
