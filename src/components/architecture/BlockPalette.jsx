import { useEffect, useMemo, useRef, useState } from 'react';
import { Boxes, Check, ChevronDown, Layers3, Search, X } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { getPaletteEntries, NODE_FAMILIES } from '../../domain/library/nodeFamilies.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { containsTemplate } from '../../domain/architecture/templateReferences.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import OperationIcon from './OperationIcon.jsx';
import './palette.css';

const FAMILY_ORDER = NODE_FAMILIES.map((family) => family.key);
const PALETTE_ENTRIES = getPaletteEntries().map((entry) => entry.key === 'torch.nn.TransformerEncoderLayer'
    ? { ...entry, name: 'Transformer', aliases: [entry.name] } : entry);
const normalize = (value) => String(value ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
const matches = (entry, query) => query.trim().split(/\s+/).every((word) => normalize(`${entry.name} ${entry.category} ${entry.description} ${entry.key} ${(entry.aliases ?? []).join(' ')}`).includes(normalize(word)));

function LibraryTile({ entry, onAdd, template = false }) {
    return (
        <button type="button" className="df-library-tile" data-library-key={entry.key} draggable
            onDragStart={(event) => {
                event.dataTransfer.setData(template ? 'application/deepflow-template-id' : 'application/deepflow-lib-key', entry.key);
                event.dataTransfer.effectAllowed = 'copy';
            }}
            onClick={() => onAdd(entry, template)} title={entry.description}
            aria-label={`Add ${entry.name}${template ? ' template' : ''}`}>
            {template ? <Boxes size={28} aria-hidden="true" /> : <OperationIcon libraryKey={entry.key} size={28} className="df-library-icon" />}
            <span className="df-library-name">{entry.name}</span>
        </button>
    );
}

export default function BlockPalette({ initiallyCollapsed = false, keyboardActive = true }) {
    const [query, setQuery] = useState('');
    const [collapsed, setCollapsed] = useState(initiallyCollapsed);
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
    const clearExpanded = useArchitectureStore((s) => s.clearExpanded);
    const editingTemplateId = resolveGraphLevel(rootGraph, currentPath, templates).templateId;
    const matchingEntries = useMemo(() => PALETTE_ENTRIES.filter((entry) => matches(entry, query)), [query]);
    const layers = matchingEntries.filter((entry) => !['IO', 'Operations'].includes(entry.category))
        .sort((a, b) => (a.family ? FAMILY_ORDER.indexOf(a.family.key) : a.category === 'Regularization' ? 7 : 8)
            - (b.family ? FAMILY_ORDER.indexOf(b.family.key) : b.category === 'Regularization' ? 7 : 8));
    const operations = matchingEntries.filter((entry) => ['IO', 'Operations'].includes(entry.category));
    const groups = [{ name: 'Layers', items: layers }, { name: 'Inputs & operations', items: operations }];
    const entries = [...layers, ...operations];
    const templateDisabled = (entry) => !!editingTemplateId && containsTemplate(entry.key, editingTemplateId, templates);
    const templateEntries = Object.values(templates).map((template) => ({ key: template.id, name: template.name, category: 'Template', description: template.description || 'Reusable graph template' })).filter((entry) => !templateDisabled(entry) && matches(entry, query));

    useEffect(() => {
        const focusSearch = (event) => {
            if (!keyboardActive) return;
            const editing = event.target?.closest?.('input, textarea, select, [contenteditable="true"]');
            if ((!editing && event.key === '/') || ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k')) {
                event.preventDefault();
                setCollapsed(false);
                requestAnimationFrame(() => searchRef.current?.focus());
            }
        };
        window.addEventListener('keydown', focusSearch);
        return () => { window.removeEventListener('keydown', focusSearch); clearTimeout(noticeTimer.current); };
    }, [keyboardActive]);

    function addEntry(entry, isTemplate) {
        const bounds = panelRef.current?.closest('[data-architecture-canvas]')?.getBoundingClientRect();
        if (!bounds) return;
        // Place new blocks in the visible canvas, clear of the floating palette.
        const availableWidth = Math.max(0, bounds.width - (panelRef.current.getBoundingClientRect().width + 32));
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
        clearExpanded();
        setNotice(`${entry.name} added`);
        clearTimeout(noticeTimer.current);
        noticeTimer.current = setTimeout(() => setNotice(''), 2200);
    }

    function onSearchKeyDown(event) {
        if (event.key === 'Escape') { if (query) setQuery(''); else searchRef.current?.blur(); }
        if (event.key === 'ArrowDown') { event.preventDefault(); panelRef.current?.querySelector('.df-library-tile:not(:disabled)')?.focus(); }
        if (event.key === 'Enter' && query.trim()) {
            const first = entries[0] ?? templateEntries[0];
            if (first) addEntry(first, !entries.length);
        }
    }

    if (collapsed) return <aside className="df-library df-library-collapsed df-panel-subtle" aria-label="Block library"><button className="df-library-header" onClick={() => { setCollapsed(false); requestAnimationFrame(() => searchRef.current?.focus()); }} title="Open block library (⌘K / Ctrl+K)" aria-label="Open block library"><Layers3 size={12} /><span>Architecture</span><ChevronDown size={13} /></button></aside>;

    return (
        <aside className="df-library df-panel-subtle nodrag nopan nowheel" ref={panelRef} aria-label="Block library">
            <header className="df-library-header"><h2>Architecture</h2><span className="df-library-kind">blocks</span><button className="df-library-collapse" onClick={() => setCollapsed(true)} aria-label="Collapse block library" title="Collapse library"><ChevronDown size={13} /></button></header>
            <div className="df-library-search"><Search size={12} /><input ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={onSearchKeyDown} placeholder="Search blocks…" aria-label="Search blocks" autoComplete="off" spellCheck={false} />{query ? <button onClick={() => { setQuery(''); searchRef.current?.focus(); }} className="df-search-clear" aria-label="Clear search"><X size={12} /></button> : <kbd>/</kbd>}</div>
            <div className="df-library-scroll">
                {groups.filter(({ items }) => items.length).map(({ name, items }) => (
                    <section key={name} className="df-library-group" aria-label={name}>
                        <h3 className="df-library-section-label">{name}</h3>
                        <div className="df-library-grid">{items.map((entry) => <LibraryTile key={entry.key} entry={entry} onAdd={addEntry} />)}</div>
                    </section>
                ))}
                {templateEntries.length > 0 && <section className="df-library-group" aria-label="Templates"><h3 className="df-library-section-label">Templates</h3><div className="df-library-grid">{templateEntries.map((entry) => <LibraryTile key={entry.key} entry={entry} onAdd={addEntry} template />)}</div></section>}
                {!entries.length && !templateEntries.length && <p className="df-library-empty">No matching blocks.</p>}
            </div>
            <footer className="df-library-footer" role="status" aria-live="polite">{notice ? <><Check size={11} />{notice}</> : 'Click to add · drag to place'}</footer>
        </aside>
    );
}
