import { useMemo, useRef, useState } from 'react';
import { Boxes, Search, X } from 'lucide-react';
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

function LibraryTile({ entry, onAdd, onDragEnd, template = false }) {
    return (
        <button type="button" className="df-library-tile" data-library-key={entry.key} draggable
            onDragStart={(event) => {
                event.dataTransfer.setData(template ? 'application/deepflow-template-id' : 'application/deepflow-lib-key', entry.key);
                event.dataTransfer.effectAllowed = 'copy';
            }}
            onDragEnd={onDragEnd}
            onClick={() => onAdd(entry, template)} title={entry.description}
            aria-label={`Add ${entry.name}${template ? ' template' : ''}`}>
            {template ? <Boxes size={28} aria-hidden="true" /> : <OperationIcon libraryKey={entry.key} size={28} className="df-library-icon" />}
            <span className="df-library-name">{entry.name}</span>
        </button>
    );
}

/** The ribbon's transient library; insertion is routed to the focused canvas. */
export default function BlockPalette({ onAdd, onClose, initialSection = 'blocks' }) {
    const [query, setQuery] = useState('');
    const [section, setSection] = useState(initialSection);
    const panelRef = useRef(null);
    const searchRef = useRef(null);
    const templates = useTemplatesStore((s) => s.templates);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const editingTemplateId = resolveGraphLevel(rootGraph, currentPath, templates).templateId;
    const matchingEntries = useMemo(() => PALETTE_ENTRIES.filter((entry) => matches(entry, query)), [query]);
    const layers = matchingEntries.filter((entry) => !['IO', 'Operations'].includes(entry.category))
        .sort((a, b) => (a.family ? FAMILY_ORDER.indexOf(a.family.key) : a.category === 'Regularization' ? 7 : 8)
            - (b.family ? FAMILY_ORDER.indexOf(b.family.key) : b.category === 'Regularization' ? 7 : 8));
    const operations = matchingEntries.filter((entry) => ['IO', 'Operations'].includes(entry.category));
    const templateEntries = Object.values(templates)
        .filter((template) => !editingTemplateId || !containsTemplate(template.id, editingTemplateId, templates))
        .map((template) => ({ key: template.id, name: template.name, category: 'Template', description: template.description || 'Reusable graph template' }))
        .filter((entry) => matches(entry, query));
    const groups = section === 'templates'
        ? [{ name: 'Reusable components', items: templateEntries, template: true }]
        : [{ name: 'Layers', items: layers }, { name: 'Inputs & operations', items: operations }];
    const entries = groups.flatMap(({ items }) => items);

    function onSearchKeyDown(event) {
        if (event.key === 'Escape' && query) { event.preventDefault(); event.stopPropagation(); setQuery(''); }
        if (event.key === 'ArrowDown') { event.preventDefault(); panelRef.current?.querySelector('.df-library-tile:not(:disabled)')?.focus(); }
        if (event.key === 'Enter' && query.trim() && entries[0]) {
            event.preventDefault();
            onAdd(entries[0], section === 'templates');
        }
    }

    return (
        <div className="df-library df-ribbon-library" ref={panelRef}>
            <header className="df-library-header"><h2>Insert {section === 'templates' ? 'template' : 'block'}</h2><button type="button" className="df-library-collapse" onClick={onClose} aria-label="Close block library"><X size={14} /></button></header>
            <div className="df-ribbon-library-tabs" role="group" aria-label="Library category">
                <button type="button" aria-pressed={section === 'blocks'} onClick={() => setSection('blocks')}>Blocks</button>
                <button type="button" aria-pressed={section === 'templates'} onClick={() => setSection('templates')}>Templates <span>{templateEntries.length}</span></button>
            </div>
            <div className="df-library-search"><Search size={13} /><input autoFocus ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={onSearchKeyDown} placeholder={section === 'templates' ? 'Search templates…' : 'Search blocks…'} aria-label="Search blocks" autoComplete="off" spellCheck={false} />{query ? <button type="button" onClick={() => { setQuery(''); searchRef.current?.focus(); }} className="df-search-clear" aria-label="Clear search"><X size={12} /></button> : <kbd>/</kbd>}</div>
            <div className="df-library-scroll">
                {groups.filter(({ items }) => items.length).map(({ name, items, template }) => (
                    <section key={name} className="df-library-group" aria-label={name}>
                        <h3 className="df-library-section-label">{name}</h3>
                        <div className="df-library-grid">{items.map((entry) => <LibraryTile key={entry.key} entry={entry} onAdd={onAdd} template={template} onDragEnd={onClose} />)}</div>
                    </section>
                ))}
                {!entries.length && <p className="df-library-empty">{section === 'templates' && !query ? 'Group blocks into a component to create a reusable template.' : 'No matching blocks.'}</p>}
            </div>
            <footer className="df-library-footer">Click to add · drag into either pane</footer>
        </div>
    );
}
