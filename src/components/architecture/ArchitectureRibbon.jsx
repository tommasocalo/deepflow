import { useCallback, useEffect, useRef, useState } from 'react';
import { Boxes, Check, ChevronDown, Columns2, Group, Maximize, Search, Variable, X } from 'lucide-react';
import { getPaletteEntries, NODE_FAMILIES } from '../../domain/library/nodeFamilies.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore, useArchitectureStoreApi } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useOpenToSide } from './OpenToSideContext.jsx';
import BlockPalette from './BlockPalette.jsx';
import TemplateInspector from './TemplateInspector.jsx';
import OperationIcon from './OperationIcon.jsx';
import './ribbon.css';

const entries = getPaletteEntries();
const QUICK_ENTRIES = NODE_FAMILIES.map((family) => entries.find((entry) => entry.key === family.defaultKey));
const SHORT_NAMES = { Normalization: 'Norm', Convolution: 'Conv', Pooling: 'Pool' };

function RepeatIcon({ parallel }) {
    return <svg width="23" height="23" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="square" aria-hidden="true">
        {parallel ? <><path d="M12 2v3H4v4m8-4h8v4M4 15v4h16v-4m-8 4v3" /><rect x="1" y="9" width="6" height="6" rx="1" /><rect x="17" y="9" width="6" height="6" rx="1" /></>
            : <><rect x="7" y="1" width="10" height="6" rx="1" /><rect x="7" y="17" width="10" height="6" rx="1" /><path d="M12 8v7m-3-3 3 3 3-3" /></>}
    </svg>;
}

function RibbonButton({ icon: Icon, children, className = '', ...props }) {
    return <button type="button" className={`df-ribbon-button ${className}`} {...props}>{Icon && <Icon size={22} strokeWidth={1.8} aria-hidden="true" />}<span>{children}</span></button>;
}

function RibbonGroup({ label, children, className = '' }) {
    return <section className={`df-ribbon-group ${className}`} aria-label={label}><div className="df-ribbon-actions">{children}</div><h2>{label}</h2></section>;
}

/** One command surface, bound to the store of the last focused graph pane. */
export default function ArchitectureRibbon({ paneId, split, onInsert, onFitView }) {
    const architectureStore = useArchitectureStoreApi();
    const graph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const templates = useTemplatesStore((s) => s.templates);
    const openToSide = useOpenToSide();
    const [popup, setPopup] = useState(null);
    const [groupName, setGroupName] = useState('Component');
    const [notice, setNotice] = useState('');
    const [error, setError] = useState('');
    const ribbonRef = useRef(null);
    const triggerRef = useRef(null);
    const libraryButtonRef = useRef(null);
    const level = resolveGraphLevel(graph, currentPath, templates);
    const selection = selectedNodeIds.map((id) => level.graph.nodes[id]).filter(Boolean);
    const selectedComponent = selection.length === 1 && ['supernode', 'repeat'].includes(selection[0].kind) ? selection[0] : null;
    const templateId = selectedComponent?.templateId ?? level.templateId;
    const template = templates[templateId];
    const selectedRepeatMode = selectedComponent?.kind === 'repeat' ? selectedComponent.mode ?? 'sequential' : null;

    const closePopup = useCallback((restoreFocus = false) => {
        setPopup(null);
        setError('');
        if (restoreFocus) triggerRef.current?.focus({ preventScroll: true });
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
                triggerRef.current = libraryButtonRef.current;
                setNotice('');
                setError('');
                setPopup('library');
                requestAnimationFrame(() => ribbonRef.current?.querySelector('[aria-label="Search blocks"]')?.focus());
            }
        };
        window.addEventListener('keydown', keyboardSearch);
        return () => window.removeEventListener('keydown', keyboardSearch);
    }, []);

    useEffect(() => {
        if (!notice) return;
        const timer = setTimeout(() => setNotice(''), 2800);
        return () => clearTimeout(timer);
    }, [notice]);

    function togglePopup(name, event) {
        triggerRef.current = event.currentTarget;
        setNotice('');
        setError('');
        setPopup((value) => value === name ? null : name);
    }

    function addEntry(entry, isTemplate = false) {
        const id = onInsert(entry, isTemplate);
        if (id) {
            setNotice(`${entry.name} added`);
            closePopup(true);
        }
    }

    function groupNodes(name) {
        const state = architectureStore.getState();
        const id = state.groupSelection([...state.selectedNodeIds], name.trim() || 'Component');
        if (!id) {
            setError('Select blocks with a single input and output boundary to group them.');
            return null;
        }
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

    const targetLabel = split ? `${paneId === 'side' ? 'Right' : 'Left'} pane` : 'Current level';

    return <div ref={ribbonRef} className="df-architecture-ribbon" role="region" aria-label={`Architecture ribbon — ${targetLabel}`} data-active-pane={paneId}
        onKeyDown={(event) => {
            if (event.key === 'Escape' && popup) { event.preventDefault(); event.stopPropagation(); closePopup(true); }
        }}>
        <div className="df-ribbon-row">
            <RibbonGroup label="Insert" className="df-ribbon-insert">
                {QUICK_ENTRIES.map((entry) => <button type="button" key={entry.key} className="df-ribbon-button df-ribbon-family" data-library-key={entry.key}
                    draggable onDragStart={(event) => { event.dataTransfer.setData('application/deepflow-lib-key', entry.key); event.dataTransfer.effectAllowed = 'copy'; }}
                    onClick={() => addEntry(entry)} title={`${entry.name}: click to add, or drag into a pane. ${entry.description}`} aria-label={`Add ${entry.name}`}>
                    <OperationIcon libraryKey={entry.key} size={25} /><span>{SHORT_NAMES[entry.name] ?? entry.name}</span>
                </button>)}
                <button ref={libraryButtonRef} type="button" className="df-ribbon-button df-ribbon-menu-button" aria-label="Search all blocks" aria-keyshortcuts="/ Meta+k Control+k"
                    aria-expanded={popup === 'library'} aria-controls={popup === 'library' ? 'df-ribbon-library-popup' : undefined} onClick={(event) => togglePopup('library', event)} title="Search all blocks (/ or ⌘K / Ctrl+K)">
                    <Search size={21} strokeWidth={1.8} /><span>More <ChevronDown size={10} /></span>
                </button>
                <RibbonButton icon={Boxes} aria-label="Browse templates" aria-expanded={popup === 'templates'} aria-controls={popup === 'templates' ? 'df-ribbon-library-popup' : undefined}
                    onClick={(event) => togglePopup('templates', event)} title="Insert a reusable component template">Templates</RibbonButton>
            </RibbonGroup>
            <RibbonGroup label="Structure">
                <RibbonButton icon={Group} disabled={!selection.length} aria-label="Group selection into component" aria-expanded={popup === 'group'} onClick={(event) => { setGroupName('Component'); togglePopup('group', event); }} title={selection.length ? 'Group selected blocks into a reusable component' : 'Select blocks to group into a component'}>Group</RibbonButton>
                <button type="button" className="df-ribbon-button" disabled={!selection.length} aria-label="Repeat selection sequentially" aria-pressed={selectedRepeatMode === 'sequential'} onClick={() => repeatSelection('sequential')} title={selection.length ? 'Repeat selected blocks in sequence; each output feeds the next' : 'Select blocks to repeat in sequence'}><RepeatIcon /><span>Sequential</span></button>
                <button type="button" className="df-ribbon-button" disabled={!selection.length} aria-label="Repeat selection in parallel" aria-pressed={selectedRepeatMode === 'parallel'} onClick={() => repeatSelection('parallel')} title={selection.length ? 'Repeat selected blocks in parallel; each receives the same input' : 'Select blocks to repeat in parallel'}><RepeatIcon parallel /><span>Parallel</span></button>
            </RibbonGroup>
            <RibbonGroup label="Parameters">
                <RibbonButton icon={Variable} disabled={!template} aria-label="Edit template parameters" aria-expanded={popup === 'parameters'}
                    onClick={(event) => togglePopup('parameters', event)} title={template ? `Edit shared parameters of ${template.name}` : 'Open or select a component to edit its parameters'}>Edit <ChevronDown size={10} /></RibbonButton>
            </RibbonGroup>
            <RibbonGroup label="View">
                <RibbonButton icon={Columns2} onClick={() => { closePopup(); openToSide?.(selectedComponent ? [...currentPath, selectedComponent.id] : currentPath); }} title={selectedComponent ? `Open ${selectedComponent.label} to the side` : 'Open current level to the side'} aria-label="Open to the side">To side</RibbonButton>
                <RibbonButton icon={Maximize} onClick={onFitView} title={`Fit graph in ${targetLabel.toLowerCase()}`} aria-label="Fit active graph">Fit</RibbonButton>
            </RibbonGroup>
            <div className="df-ribbon-context" aria-live="polite"><span>{targetLabel}</span><strong title={level.label || 'main'}>{level.label || 'main'}</strong><small>{selection.length ? `${selection.length} selected` : 'Select blocks to structure'}</small></div>
        </div>
        {(popup === 'library' || popup === 'templates') && <div id="df-ribbon-library-popup" className="df-ribbon-popover df-ribbon-popover-library" role="dialog" aria-label="Block library" onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}><BlockPalette key={popup} initialSection={popup === 'templates' ? 'templates' : 'blocks'} onAdd={addEntry} onClose={() => closePopup(true)} /></div>}
        {popup === 'group' && <div className="df-ribbon-popover df-ribbon-popover-group" role="dialog" aria-label="Create component">
            <header><strong>Group {selection.length} {selection.length === 1 ? 'block' : 'blocks'}</strong><button type="button" onClick={() => closePopup(true)} aria-label="Close group editor"><X size={14} /></button></header>
            <form onSubmit={confirmGroup}><label>Component name<input autoFocus value={groupName} onChange={(event) => setGroupName(event.target.value)} /></label><p>The selection becomes a reusable template.</p>{error && <p role="alert" className="df-ribbon-error">{error}</p>}<button className="df-ribbon-primary" type="submit">Create component</button></form>
        </div>}
        {popup === 'parameters' && template && <div className="df-ribbon-popover df-ribbon-popover-parameters" role="dialog" aria-label={`${template.name} template parameters`} onKeyDown={(event) => { if (event.key !== 'Escape') event.stopPropagation(); }}>
            <header><strong>{template.name}</strong><button type="button" onClick={() => closePopup(true)} aria-label="Close template parameters"><X size={14} /></button></header><TemplateInspector templateId={templateId} embedded initiallyOpen />
        </div>}
        {(notice || (error && popup !== 'group')) && <div className={`df-ribbon-notice${error ? ' is-error' : ''}`} role={error ? 'alert' : 'status'}>{!error && <Check size={12} />}<span>{error || notice}</span><button type="button" aria-label="Dismiss message" onClick={() => { setNotice(''); setError(''); }}><X size={12} /></button></div>}
    </div>;
}
