import { useEffect, useMemo, useRef, useState } from 'react';
import { Boxes, Check, ChevronDown, ChevronRight, Layers3, PanelLeftClose, PanelLeftOpen, Plus, Search, X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { PYTORCH_BLOCKS, getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { styleFor } from './categoryStyle.js';
import TemplateInspector from './TemplateInspector.jsx';
import './palette.css';

const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);
const CATEGORIES = ['IO', 'Linear', 'Attention', 'Normalization', 'Activation', 'Regularization', 'Convolution', 'Pooling', 'Reshape', 'Recurrent', 'Element-wise', 'Combine'];
const LABELS = { IO: 'Inputs & outputs', Linear: 'Linear & embedding', 'Element-wise': 'Math operations', Combine: 'Merge & split', Reshape: 'Tensor shape' };
const normalize = (value) => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const matches = (entry, query) => query.trim().split(/\s+/).every((word) => normalize(`${entry.name} ${entry.category} ${entry.description} ${entry.key}`).includes(normalize(word)));

function containsTemplate(templateId, targetId, templates, visited = new Set()) {
    if (templateId === targetId) return true;
    if (visited.has(templateId)) return false;
    visited.add(templateId);
    return Object.values(templates[templateId]?.graph?.nodes ?? {}).some((node) => node.templateId && containsTemplate(node.templateId, targetId, templates, visited));
}

function LibraryRow({ entry, onAdd, template = false, disabled = false }) {
    const accent = template ? '#7c3aed' : styleFor(entry.category).accent;
    return (
        <button type="button" className="df-library-row" data-library-key={entry.key} disabled={disabled} draggable={!disabled}
            onDragStart={(event) => {
                event.dataTransfer.setData(template ? 'application/deepflow-template-id' : 'application/deepflow-lib-key', entry.key);
                event.dataTransfer.effectAllowed = 'copy';
            }}
            onClick={() => onAdd(entry, template)} title={disabled ? 'This template would contain itself.' : entry.description}
            aria-label={`Add ${entry.name}${template ? ' template' : ''}`}>
            <span className="df-library-symbol" style={{ '--entry-accent': accent }}>{template ? <Boxes size={16} /> : <Layers3 size={15} />}</span>
            <span className="df-library-copy"><span className="df-library-name">{entry.name}</span><span className="df-library-description">{entry.description || 'Reusable graph template'}</span></span>
            <Plus className="df-library-add" size={15} aria-hidden="true" />
        </button>
    );
}

export default function BlockPalette() {
    const [query, setQuery] = useState('');
    const [tab, setTab] = useState('layers');
    const [collapsed, setCollapsed] = useState(false);
    const [openGroups, setOpenGroups] = useState(['IO', 'Linear', 'Attention']);
    const [notice, setNotice] = useState('');
    const panelRef = useRef(null);
    const searchRef = useRef(null);
    const noticeTimer = useRef(null);
    const { screenToFlowPosition, getNodes } = useReactFlow();
    const templates = useTemplatesStore((s) => s.templates);
    const recentKeys = useArchitectureStore((s) => s.recentLibraryKeys);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const addLayer = useArchitectureStore((s) => s.addLayerByLibraryKey);
    const addTemplate = useArchitectureStore((s) => s.addSuperNodeInstance);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const groups = useMemo(() => CATEGORIES.map((category) => ({ category, items: PYTORCH_BLOCKS.filter((entry) => entry.category === category && !TEMPLATE_ONLY.has(entry.key) && matches(entry, query)) })).filter((group) => group.items.length), [query]);
    const entries = groups.flatMap((group) => group.items);
    const templateEntries = Object.values(templates).map((template) => ({ key: template.id, name: template.name, category: 'Template', description: template.description || `${Object.keys(template.graph.nodes).length} nodes · ${template.freeParameters?.length ?? 0} parameters` })).filter((entry) => matches(entry, query));
    const recentEntries = recentKeys.map(getEntry).filter((entry) => entry && !TEMPLATE_ONLY.has(entry.key)).slice(0, 4);
    let currentGraph = rootGraph;
    let editingTemplateId = null;
    for (const id of currentPath) {
        const node = currentGraph.nodes[id];
        if (!node?.templateId || !templates[node.templateId]) break;
        editingTemplateId = node.templateId;
        currentGraph = templates[node.templateId].graph;
    }

    useEffect(() => {
        const focusSearch = (event) => {
            const editing = event.target?.closest?.('input, textarea, select, [contenteditable="true"]');
            if ((!editing && event.key === '/') || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
                event.preventDefault();
                setCollapsed(false);
                requestAnimationFrame(() => searchRef.current?.focus());
            }
        };
        window.addEventListener('keydown', focusSearch);
        return () => { window.removeEventListener('keydown', focusSearch); clearTimeout(noticeTimer.current); };
    }, []);

    function addEntry(entry, isTemplate) {
        const bounds = panelRef.current?.parentElement?.querySelector('[data-architecture-canvas]')?.getBoundingClientRect();
        if (!bounds) return;
        const point = screenToFlowPosition({ x: bounds.left + bounds.width * 0.5, y: bounds.top + bounds.height * 0.42 });
        const topLeft = screenToFlowPosition({ x: bounds.left + 28, y: bounds.top + 70 });
        const bottomRight = screenToFlowPosition({ x: bounds.right - 28, y: bounds.bottom - 70 });
        const clamp = (candidate) => ({
            x: Math.max(topLeft.x, Math.min(candidate.x, bottomRight.x - 340)),
            y: Math.max(topLeft.y, Math.min(candidate.y, bottomRight.y - 200)),
        });
        let position = { x: Math.round(point.x - 140), y: Math.round(point.y - 65) };
        position = clamp(position);
        const nodes = getNodes();
        for (let attempt = 0; attempt < 12; attempt++) {
            if (!nodes.some((node) => Math.abs(node.position.x - position.x) < 220 && Math.abs(node.position.y - position.y) < 130)) break;
            position = clamp({ x: position.x + 36, y: position.y + 40 });
        }
        const id = isTemplate ? addTemplate(entry.key, position, entry.name) : addLayer(entry.key, position);
        if (!id) return;
        selectOnly(id);
        setExpanded(id);
        setNotice(`${entry.name} added`);
        clearTimeout(noticeTimer.current);
        noticeTimer.current = setTimeout(() => setNotice(''), 2200);
    }

    function onSearchKeyDown(event) {
        if (event.key === 'Escape') { if (query) setQuery(''); else searchRef.current?.blur(); }
        if (event.key === 'ArrowDown') { event.preventDefault(); panelRef.current?.querySelector('.df-library-row:not(:disabled)')?.focus(); }
        if (event.key === 'Enter' && query.trim()) {
            const first = tab === 'layers' ? entries[0] : templateEntries.find((entry) => !editingTemplateId || !containsTemplate(entry.key, editingTemplateId, templates));
            if (first) addEntry(first, tab === 'templates');
        }
    }

    if (collapsed) return <aside className="df-library-rail" aria-label="Block library"><button className="df-icon-button" onClick={() => { setCollapsed(false); requestAnimationFrame(() => searchRef.current?.focus()); }} title="Open block library (⌘K / Ctrl+K)" aria-label="Open block library"><PanelLeftOpen size={19} /></button><span className="df-library-rail-label">Blocks</span></aside>;

    return (
        <aside className="df-library" ref={panelRef} aria-label="Block library">
            <header className="df-library-header"><div><h2><Layers3 size={18} /> Blocks</h2><p>The building blocks of your model</p></div><button className="df-icon-button" onClick={() => setCollapsed(true)} aria-label="Collapse block library" title="Collapse library"><PanelLeftClose size={17} /></button></header>
            <div className="df-library-tabs" role="tablist" aria-label="Library type" onKeyDown={(event) => {
                if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
                event.preventDefault();
                const next = event.key === 'Home' ? 'layers' : event.key === 'End' ? 'templates' : tab === 'layers' ? 'templates' : 'layers';
                setTab(next);
                setQuery('');
                panelRef.current?.querySelector(`#${next}-tab`)?.focus();
            }}>
                <button role="tab" tabIndex={tab === 'layers' ? 0 : -1} aria-selected={tab === 'layers'} aria-controls="library-content" id="layers-tab" onClick={() => { setTab('layers'); setQuery(''); }}>Layers <span>{PYTORCH_BLOCKS.filter((entry) => !TEMPLATE_ONLY.has(entry.key)).length}</span></button>
                <button role="tab" tabIndex={tab === 'templates' ? 0 : -1} aria-selected={tab === 'templates'} aria-controls="library-content" id="templates-tab" onClick={() => { setTab('templates'); setQuery(''); }}>Templates <span>{Object.keys(templates).length}</span></button>
            </div>
            <div className="df-library-search"><Search size={16} /><input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={onSearchKeyDown} placeholder={tab === 'layers' ? 'Find a layer…' : 'Find a template…'} aria-label={tab === 'layers' ? 'Search layers' : 'Search templates'} autoComplete="off" spellCheck={false} />{query ? <button onClick={() => { setQuery(''); searchRef.current?.focus(); }} className="df-search-clear" aria-label="Clear search"><X size={14} /></button> : <kbd>/</kbd>}</div>
            <div className="df-library-scroll" id="library-content" role="tabpanel" aria-labelledby={tab === 'layers' ? 'layers-tab' : 'templates-tab'}>
                {currentPath.length > 0 && <TemplateInspector embedded />}
                {tab === 'layers' && !query && recentEntries.length > 0 && <section className="df-library-group df-library-recent"><div className="df-library-section-label">Recently used</div>{recentEntries.map((entry) => <LibraryRow key={entry.key} entry={entry} onAdd={addEntry} />)}</section>}
                {query && <div className="df-library-result-count" role="status">{tab === 'layers' ? entries.length : templateEntries.length} results</div>}
                {tab === 'layers' && groups.map(({ category, items }) => {
                    const open = !!query || openGroups.includes(category);
                    const groupId = `library-${category.toLowerCase().replace(/[^a-z]/g, '-')}`;
                    return <section key={category} className="df-library-group"><button className="df-library-category" aria-expanded={open} aria-controls={groupId} onClick={() => setOpenGroups((previous) => previous.includes(category) ? previous.filter((item) => item !== category) : [...previous, category])}>{open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}<span>{LABELS[category] ?? category}</span><span className="df-library-category-count">{items.length}</span></button>{open && <div id={groupId} className="df-library-group-items">{items.map((entry) => <LibraryRow key={entry.key} entry={entry} onAdd={addEntry} />)}</div>}</section>;
                })}
                {tab === 'templates' && templateEntries.map((entry) => <LibraryRow key={entry.key} entry={entry} onAdd={addEntry} template disabled={!!editingTemplateId && containsTemplate(entry.key, editingTemplateId, templates)} />)}
                {(tab === 'layers' ? !entries.length : !templateEntries.length) && <div className="df-library-empty">{tab === 'templates' && !query ? <Boxes size={26} /> : <Search size={26} />}<h3>{query ? 'No matches yet' : 'Your reusable blocks'}</h3><p>{query ? 'Try a layer name or operation, such as “attention” or “norm”.' : 'Select connected nodes on the canvas and choose Group to create a template.'}</p>{query && <button className="df-button" onClick={() => setQuery('')}>Clear search</button>}</div>}
            </div>
            <footer className="df-library-footer" role="status" aria-live="polite">{notice ? <><Check size={14} />{notice}</> : <><Plus size={14} />Click to add · drag to place</>}</footer>
        </aside>
    );
}
