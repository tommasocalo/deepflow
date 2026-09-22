import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Boxes, Braces, Check, ChevronDown, Columns2, Copy, GitBranch, Group, Maximize, Pencil, Ruler, Variable, X } from 'lucide-react';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useOpenToSide } from './OpenToSideContext.jsx';
import { DEFAULT_PINS, PINNABLE_ENTRIES, setInsertionDragData } from './ribbonEntries.js';
import BlockPalette from './BlockPalette.jsx';
import TemplateInspector from './TemplateInspector.jsx';
import OperationIcon from './OperationIcon.jsx';
import './ribbon.css';

const PINS_KEY = 'deepflow:ribbon-pins';
const SHORT_NAMES = { Normalization: 'Norm', Convolution: 'Conv', Pooling: 'Pool', 'Learned positions': 'Positions', 'Sinusoidal positions': 'Sinusoidal' };
const MENU_WIDTHS = { operations: 350, components: 360, templates: 310, preferences: 280, parameters: 310, group: 280 };
function readPins() {
    try {
        const saved = JSON.parse(localStorage.getItem(PINS_KEY));
        if (Array.isArray(saved)) return [...new Set(saved)].filter((key) => PINNABLE_ENTRIES.some((entry) => entry.key === key)).slice(0, 6);
    } catch { /* Personal ribbon preferences are optional. */ }
    return DEFAULT_PINS;
}
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
    const showGradientFlow = useArchitectureStore((s) => s.showGradientFlow);
    const showDimensions = useArchitectureStore((s) => s.showDimensions) !== false;
    const templates = useTemplatesStore((s) => s.templates);
    const openToSide = useOpenToSide();
    const [popup, setPopup] = useState(null);
    const [popupLeft, setPopupLeft] = useState(0);
    const [pins, setPins] = useState(readPins);
    const [groupName, setGroupName] = useState('Component');
    const [notice, setNotice] = useState('');
    const [error, setError] = useState('');
    const ribbonRef = useRef(null);
    const triggerRef = useRef(null);
    const operationsButtonRef = useRef(null);
    const level = resolveGraphLevel(graph, currentPath, templates);
    const selection = selectedNodeIds.map((id) => level.graph.nodes[id]).filter(Boolean);
    const selectedComponent = selection.length === 1 && ['supernode', 'repeat'].includes(selection[0].kind) ? selection[0] : null;
    const templateId = selectedComponent?.templateId ?? level.templateId;
    const template = templates[templateId];
    const selectedRepeatMode = selectedComponent?.kind === 'repeat' ? selectedComponent.mode ?? 'sequential' : null;
    const quickEntries = pins.map((key) => PINNABLE_ENTRIES.find((entry) => entry.key === key)).filter(Boolean);

    const closePopup = useCallback((restoreFocus = false) => {
        setPopup(null);
        setError('');
        if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
    }, []);
    const showPopup = useCallback((name, trigger) => {
        const ribbon = ribbonRef.current?.getBoundingClientRect();
        const anchor = trigger?.getBoundingClientRect();
        if (ribbon && anchor) setPopupLeft(Math.max(8, Math.min(anchor.left - ribbon.left, ribbon.width - MENU_WIDTHS[name] - 8)));
        triggerRef.current = trigger;
        setNotice('');
        setError('');
        setPopup(name);
    }, []);
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
    function savePins(next) {
        setPins(next);
        try { localStorage.setItem(PINS_KEY, JSON.stringify(next)); } catch { /* Still usable without persistence. */ }
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
        const current = resolveGraphLevel(latest.graph, latest.currentPath, useTemplatesStore.getState().templates).graph;
        const instance = Object.values(current.nodes).find((node) => node.kind === 'supernode' && node.templateId === id);
        if (instance) latest.selectOnly(instance.id);
        return instance?.id ?? null;
    }
    function confirmGroup(event) {
        event.preventDefault();
        if (!groupNodes(groupName)) return;
        setNotice(`${groupName.trim() || 'Component'} created`);
        closePopup(true);
    }
    function repeatSelection(mode) {
        setError('');
        const id = selectedComponent?.id ?? groupNodes(selection.length === 1 ? selection[0].label : 'Component');
        if (!id) return;
        const state = architectureStore.getState();
        if (selectedComponent?.kind === 'repeat') state.patchRepeatNode(id, { mode });
        else state.convertInstanceToRepeat(id, 2, mode);
        state.selectOnly(id);
        state.setExpanded(id);
        closePopup();
        setNotice(`${mode === 'parallel' ? 'Parallel' : 'Sequential'} repeat selected`);
    }
    function editDefinition() {
        closePopup();
        if (selectedComponent) architectureStore.getState().enterSupernode(selectedComponent.id);
    }
    function makeCopy() {
        const state = architectureStore.getState();
        const id = selectedComponent ? state.cloneComponent(selectedComponent.id) : state.cloneCurrentComponent();
        if (id) { closePopup(); setNotice('Independent definition created'); }
    }
    const targetLabel = split ? `${paneId === 'side' ? 'Right' : 'Left'} pane` : 'Current graph';
    const menu = ['operations', 'components', 'templates'].includes(popup);
    const popoverStyle = { left: popupLeft, width: MENU_WIDTHS[popup] };

    return <div ref={ribbonRef} className="df-architecture-ribbon" role="region" aria-label={`Architecture ribbon — ${targetLabel}`} data-active-pane={paneId}
        onKeyDown={(event) => { if (event.key === 'Escape' && popup) { event.preventDefault(); event.stopPropagation(); closePopup(true); } }}>
        <div className="df-ribbon-row">
            <RibbonGroup label="Insert" className="df-ribbon-insert">
                {quickEntries.map((entry) => <button type="button" key={entry.key} className="df-ribbon-button df-ribbon-family" data-library-key={entry.key}
                    draggable onDragStart={(event) => setInsertionDragData(event, entry)} onClick={() => addEntry(entry)}
                    title={`${entry.name}: click to add, or drag into a pane. ${entry.description}`} aria-label={`Add ${entry.name}`}>
                    <OperationIcon libraryKey={entry.libraryKey ?? entry.key} size={25} /><span>{SHORT_NAMES[entry.name] ?? entry.name}</span>
                </button>)}
                <RibbonButton icon={Boxes} aria-label="Browse components" aria-expanded={popup === 'components'} aria-haspopup="menu" onClick={(event) => togglePopup('components', event)} title="All editable components and layer modules">Components <ChevronDown size={10} /></RibbonButton>
                <button ref={operationsButtonRef} type="button" className="df-ribbon-button" aria-label="Insert operations" aria-keyshortcuts="/ Meta+k Control+k"
                    aria-expanded={popup === 'operations'} aria-haspopup="menu" onClick={(event) => togglePopup('operations', event)} title="Math, routing, sources and outputs (⌘K / Ctrl+K)">
                    <Braces size={21} strokeWidth={1.8} /><span>Operations <ChevronDown size={10} /></span>
                </button>
                <RibbonButton icon={Copy} aria-label="Browse templates" aria-expanded={popup === 'templates'} aria-haspopup="menu" onClick={(event) => togglePopup('templates', event)} title="Insert a saved project template">Templates <ChevronDown size={10} /></RibbonButton>
            </RibbonGroup>
            <RibbonGroup label="Structure">
                <RibbonButton icon={Group} disabled={!selection.length} aria-label="Group selection into component" aria-expanded={popup === 'group'} onClick={(event) => { setGroupName('Component'); togglePopup('group', event); }} title={selection.length ? 'Group selected blocks into a reusable component' : 'Select blocks to create a component'}>Group</RibbonButton>
                <button type="button" className="df-ribbon-button" disabled={!selection.length} aria-label="Repeat selection sequentially" aria-pressed={selectedRepeatMode === 'sequential'} onClick={() => repeatSelection('sequential')} title="Repeat in sequence; each output feeds the next"><RepeatIcon /><span>Sequential</span></button>
                <button type="button" className="df-ribbon-button" disabled={!selection.length} aria-label="Repeat selection in parallel" aria-pressed={selectedRepeatMode === 'parallel'} onClick={() => repeatSelection('parallel')} title="Repeat in parallel; each receives the same input"><RepeatIcon parallel /><span>Parallel</span></button>
            </RibbonGroup>
            <RibbonGroup label="Component" className="df-ribbon-component">
                <div className="df-ribbon-command-list">
                    <button type="button" disabled={!selectedComponent || !template} onClick={editDefinition} title={selectedComponent ? `Open ${template?.name} for editing` : 'Select a component to open its definition'}><Pencil size={12} />Edit definition</button>
                    <button type="button" disabled={!template} onClick={makeCopy} title="Give this instance its own independent definition"><Copy size={12} />Make copy</button>
                    <button type="button" disabled={!template} aria-label="Edit template parameters" aria-expanded={popup === 'parameters'} onClick={(event) => togglePopup('parameters', event)} title={template ? `Edit shared parameters of ${template.name}` : 'Open or select a component'}><Variable size={12} />Parameters<ChevronDown size={9} /></button>
                </div>
            </RibbonGroup>
            <RibbonGroup label="View">
                <RibbonButton icon={Columns2} onClick={() => { closePopup(); openToSide?.(selectedComponent ? [...currentPath, selectedComponent.id] : currentPath); }} title="Open selected component or current level to the side" aria-label="Open to the side">To side</RibbonButton>
                <RibbonButton icon={Maximize} onClick={onFitView} title={`Fit graph in ${targetLabel.toLowerCase()}`} aria-label="Fit active graph">Fit</RibbonButton>
                <div className="df-ribbon-view-options">
                    <button type="button" aria-label="Show dimensions" aria-pressed={showDimensions} onClick={() => architectureStore.getState().setShowDimensions(!showDimensions)} title="Show tensor dimensions at input and output ports"><Ruler size={14} />Dimensions</button>
                    <button type="button" aria-label="Show gradient flow" aria-pressed={!!showGradientFlow} onClick={() => architectureStore.getState().setShowGradientFlow(!showGradientFlow)} title="Show differentiable paths and stopped gradients"><GitBranch size={14} />Gradients</button>
                </div>
            </RibbonGroup>
            {split && <span className="df-ribbon-pane-cue" aria-live="polite"><ArrowUpRight size={11} />{targetLabel}</span>}
        </div>
        {menu && <div className="df-ribbon-dropdown" style={popoverStyle} data-ribbon-menu={popup} onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}>
            <BlockPalette key={popup} section={popup} onAdd={addEntry} onClose={() => closePopup(true)}
                onCustomize={popup === 'components' ? () => showPopup('preferences', triggerRef.current) : undefined} />
        </div>}
        {popup === 'preferences' && <div className="df-ribbon-popover df-ribbon-preferences" style={popoverStyle} role="dialog" aria-label="Pinned components">
            <header><strong>Pinned components</strong><button type="button" onClick={() => closePopup(true)} aria-label="Close ribbon preferences"><X size={14} /></button></header>
            <p>Choose up to six shortcuts.</p><div className="df-ribbon-pin-list">{PINNABLE_ENTRIES.map((entry) => <label key={entry.key}><input type="checkbox" checked={pins.includes(entry.key)} disabled={!pins.includes(entry.key) && pins.length >= 6}
                onChange={(event) => savePins(event.target.checked ? [...pins, entry.key] : pins.filter((key) => key !== entry.key))} /><OperationIcon libraryKey={entry.libraryKey ?? entry.key} size={19} /><span>{entry.name}</span></label>)}</div>
            <button type="button" className="df-ribbon-restore-pins" onClick={() => savePins(DEFAULT_PINS)}>Restore defaults</button>
        </div>}
        {popup === 'group' && <div className="df-ribbon-popover df-ribbon-popover-group" style={popoverStyle} role="dialog" aria-label="Create component">
            <header><strong>Group {selection.length} {selection.length === 1 ? 'block' : 'blocks'}</strong><button type="button" onClick={() => closePopup(true)} aria-label="Close group editor"><X size={14} /></button></header>
            <form onSubmit={confirmGroup}><label>Component name<input autoFocus value={groupName} onChange={(event) => setGroupName(event.target.value)} /></label><p>The selection becomes a reusable template.</p>{error && <p role="alert" className="df-ribbon-error">{error}</p>}<button className="df-ribbon-primary" type="submit">Create component</button></form>
        </div>}
        {popup === 'parameters' && template && <div className="df-ribbon-popover df-ribbon-popover-parameters" style={popoverStyle} role="dialog" aria-label={`${template.name} template parameters`} onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}>
            <header><strong>{template.name}</strong><button type="button" onClick={() => closePopup(true)} aria-label="Close template parameters"><X size={14} /></button></header><TemplateInspector templateId={templateId} embedded initiallyOpen />
        </div>}
        {(notice || (error && popup !== 'group')) && <div className={`df-ribbon-notice${error ? ' is-error' : ''}`} role={error ? 'alert' : 'status'}>{!error && <Check size={12} />}<span>{error || notice}</span><button type="button" aria-label="Dismiss message" onClick={() => { setNotice(''); setError(''); }}><X size={12} /></button></div>}
    </div>;
}
