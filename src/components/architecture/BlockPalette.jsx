import { useEffect, useMemo, useRef, useState } from 'react';
import { Boxes, Check, ChevronDown, ChevronRight, Layers3, Plus, Search, X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { PYTORCH_BLOCKS, getEntry } from '../../domain/library/index.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { styleFor } from './categoryStyle.js';
import TemplateInspector from './TemplateInspector.jsx';
import './palette.css';

const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);
const CATEGORIES = ['IO', 'Linear', 'Attention', 'Normalization', 'Activation', 'Regularization', 'Convolution', 'Pooling', 'Reshape', 'Recurrent', 'Element-wise', 'Combine'];
const FAVORITES = ['torch.nn.Linear', 'torch.nn.Conv2d', 'torch.nn.ReLU', 'torch.nn.LayerNorm'];
const normalize = (value) => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const matches = (entry, query) => query.trim().split(/\s+/).every((word) => normalize(`${entry.name} ${entry.category} ${entry.description} ${entry.key}`).includes(normalize(word)));

function containsTemplate(templateId, targetId, templates, visited = new Set()) {
    if (templateId === targetId) return true;
    if (visited.has(templateId)) return false;
    visited.add(templateId);
    return Object.values(templates[templateId]?.graph?.nodes ?? {}).some((node) => node.templateId && containsTemplate(node.templateId, targetId, templates, visited));
}

function LibraryRow({ entry, onAdd, template = false, disabled = false }) {
    const accent = template ? '#a855f7' : styleFor(entry.category).accent;
    return (
        <button type="button" className="df-library-row" data-library-key={entry.key} disabled={disabled} draggable={!disabled}
            onDragStart={(event) => {
                event.dataTransfer.setData(template ? 'application/deepflow-template-id' : 'application/deepflow-lib-key', entry.key);
                event.dataTransfer.effectAllowed = 'copy';
            }}
            onClick={() => onAdd(entry, template)} title={disabled ? 'This template would contain itself.' : entry.description}
            aria-label={`Add ${entry.name}${template ? ' template' : ''}`}>
            <span className="df-library-dot" style={{ background: accent }} />
            {template ? <Boxes size={11} aria-hidden="true" /> : <Layers3 size={11} aria-hidden="true" />}
            <span className="df-library-name">{entry.name}</span>
            <Plus className="df-library-add" size={11} aria-hidden="true" />
        </button>
    );
}

export default function BlockPalette() {
    const [query, setQuery] = useState('');
    const [collapsed, setCollapsed] = useState(false);
    const [openGroups, setOpenGroups] = useState([]);
    const [notice, setNotice] = useState('');
    const panelRef = useRef(null);
    const searchRef = useRef(null);
    const noticeTimer = useRef(null);
    const { screenToFlowPosition, getNodes } = useReactFlow();
    const templates = useTemplatesStore((s) => s.templates);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const addLayer = useArchitectureStore((s) => s.addLayerByLibraryKey);
    const addTemplate = useArchitectureStore((s) => s.addSuperNodeInstance);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const editingTemplateId = resolveGraphLevel(rootGraph, currentPath, templates).templateId;
    const groups = useMemo(() => CATEGORIES.map((category) => ({ category, items: PYTORCH_BLOCKS.filter((entry) => entry.category === category && !TEMPLATE_ONLY.has(entry.key) && matches(entry, query)) })).filter((group) => group.items.length), [query]);
    const entries = groups.flatMap((group) => group.items);
    const templateEntries = Object.values(templates).map((template) => ({ key: template.id, name: template.name, category: 'Template', description: template.description || 'Reusable graph template' })).filter((entry) => matches(entry, query));
    const favorites = FAVORITES.map(getEntry).filter(Boolean);
    const templateDisabled = (entry) => !!editingTemplateId && containsTemplate(entry.key, editingTemplateId, templates);

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
        const bounds = panelRef.current?.closest('[data-architecture-canvas]')?.getBoundingClientRect();
        if (!bounds) return;
        // Place new blocks in the visible canvas, clear of the floating palette.
        const availableWidth = Math.max(0, bounds.width - 272);
        const point = screenToFlowPosition({ x: bounds.left + availableWidth * 0.5, y: bounds.top + bounds.height * 0.42 });
        const topLeft = screenToFlowPosition({ x: bounds.left + 28, y: bounds.top + 70 });
        const bottomRight = screenToFlowPosition({ x: bounds.left + availableWidth - 28, y: bounds.bottom - 70 });
        // Node positions identify their top-centre. Reserve half the widest
        // card on each side, rather than treating x as its left edge.
        const halfWidth = 150;
        const minX = topLeft.x + halfWidth;
        const maxX = bottomRight.x - halfWidth;
        const clamp = (candidate) => ({
            x: minX <= maxX ? Math.max(minX, Math.min(candidate.x, maxX)) : (topLeft.x + bottomRight.x) / 2,
            y: Math.max(topLeft.y, Math.min(candidate.y, bottomRight.y - 72)),
        });
        let position = clamp({ x: Math.round(point.x), y: Math.round(point.y) });
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
            const first = entries[0] ?? templateEntries.find((entry) => !templateDisabled(entry));
            if (first) addEntry(first, !entries.length);
        }
    }

    function toggleGroup(category) {
        setOpenGroups((previous) => previous.includes(category) ? previous.filter((item) => item !== category) : [...previous, category]);
    }

    if (collapsed) return <aside className="df-library df-library-collapsed df-panel-subtle" aria-label="Block library"><button className="df-library-header" onClick={() => { setCollapsed(false); requestAnimationFrame(() => searchRef.current?.focus()); }} title="Open block library (⌘K / Ctrl+K)" aria-label="Open block library"><Layers3 size={12} /><span>Architecture</span><ChevronDown size={13} /></button></aside>;

    return (
        <aside className="df-library df-panel-subtle nodrag nopan nowheel" ref={panelRef} aria-label="Block library">
            <header className="df-library-header"><h2>Architecture</h2><span className="df-library-kind">blocks</span><button className="df-library-collapse" onClick={() => setCollapsed(true)} aria-label="Collapse block library" title="Collapse library"><ChevronDown size={13} /></button></header>
            <div className="df-library-search"><Search size={12} /><input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={onSearchKeyDown} placeholder="Search blocks…" aria-label="Search blocks" autoComplete="off" spellCheck={false} />{query ? <button onClick={() => { setQuery(''); searchRef.current?.focus(); }} className="df-search-clear" aria-label="Clear search"><X size={12} /></button> : <kbd>/</kbd>}</div>
            <div className="df-library-scroll">
                {currentPath.length > 0 && <TemplateInspector embedded />}
                {!query && <section className="df-library-group"><div className="df-library-section-label">Frequently used</div>{favorites.map((entry) => <LibraryRow key={entry.key} entry={entry} onAdd={addEntry} />)}</section>}
                {groups.map(({ category, items }) => {
                    const open = !!query || openGroups.includes(category);
                    const groupId = `library-${category.toLowerCase().replace(/[^a-z]/g, '-')}`;
                    return <section key={category} className="df-library-group"><button className="df-library-category" aria-expanded={open} aria-controls={groupId} onClick={() => toggleGroup(category)}>{open ? <ChevronDown size={11} /> : <ChevronRight size={11} />}<span>{category === 'IO' ? 'Inputs & outputs' : category}</span><span className="df-library-category-count">{items.length}</span></button>{open && <div id={groupId} className="df-library-group-items">{items.map((entry) => <LibraryRow key={entry.key} entry={entry} onAdd={addEntry} />)}</div>}</section>;
                })}
                {templateEntries.length > 0 && <section className="df-library-group"><button className="df-library-category" aria-expanded={!!query || openGroups.includes('Templates')} aria-controls="library-templates" onClick={() => toggleGroup('Templates')}>{query || openGroups.includes('Templates') ? <ChevronDown size={11} /> : <ChevronRight size={11} />}<span>Templates</span><span className="df-library-category-count">{templateEntries.length}</span></button>{(query || openGroups.includes('Templates')) && <div id="library-templates" className="df-library-group-items">{templateEntries.map((entry) => <LibraryRow key={entry.key} entry={entry} onAdd={addEntry} template disabled={templateDisabled(entry)} />)}</div>}</section>}
                {!entries.length && !templateEntries.length && <p className="df-library-empty">No matching blocks.</p>}
            </div>
            <footer className="df-library-footer" role="status" aria-live="polite">{notice ? <><Check size={11} />{notice}</> : 'Click to add · drag to place'}</footer>
        </aside>
    );
}
