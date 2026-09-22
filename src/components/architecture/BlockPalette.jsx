import { useRef, useState } from 'react';
import { Boxes, Search, SlidersHorizontal, X } from 'lucide-react';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { containsTemplate } from '../../domain/architecture/templateReferences.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { MODULE_ENTRIES, OPERATION_ENTRIES, STANDARD_ENTRIES, matchesEntry, setInsertionDragData } from './ribbonEntries.js';
import OperationIcon from './OperationIcon.jsx';

const GROUPS = [
    { name: 'Math', keys: ['Add', 'Multiply', 'Subtract', 'Divide', 'MatMul', 'Scale', 'AddScalar', 'Pow', 'Rsqrt', 'Exp', 'Sin', 'Cos', 'Mean', 'Variance', 'ReduceMean', 'ReduceSum', 'Softmax', 'CausalMask'] },
    { name: 'Shape & routing', keys: ['Concat', 'Split', 'Broadcast', 'Stack', 'Transpose', 'Reshape', 'Flatten', 'Gather', 'Lookup', 'Select', 'Unsqueeze', 'Slice'] },
    { name: 'Sources & outputs', keys: ['Input', 'Output', 'Parameter', 'PositionIndices', 'Arange'] },
];

function LibraryItem({ entry, onAdd, onDragEnd, template = false }) {
    return <button type="button" role="menuitem" className="df-insert-item" data-library-key={entry.key} draggable
        onDragStart={(event) => setInsertionDragData(event, entry, template)} onDragEnd={onDragEnd}
        onClick={() => onAdd(entry, template)} title={entry.description} aria-label={`Add ${entry.name}${template ? ' template' : ''}`}>
        {template ? <Boxes size={19} strokeWidth={1.8} aria-hidden="true" /> : <OperationIcon libraryKey={entry.libraryKey ?? entry.key} size={22} />}
        <span>{entry.name}</span>
    </button>;
}

/** An anchored ribbon menu: each trigger has one purpose and one catalog. */
export default function BlockPalette({ onAdd, onClose, section = 'operations', onCustomize }) {
    const [query, setQuery] = useState('');
    const panelRef = useRef(null);
    const searchRef = useRef(null);
    const templates = useTemplatesStore((s) => s.templates);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const rootGraph = useArchitectureStore((s) => s.graph);
    const editingTemplateId = resolveGraphLevel(rootGraph, currentPath, templates, inspectedIterations).templateId;
    const templateEntries = Object.values(templates)
        .filter((template) => !template.standardKey && !template.structureVariant)
        .filter((template) => !editingTemplateId || !containsTemplate(template.id, editingTemplateId, templates))
        .map((template) => ({ key: template.id, name: template.name, category: 'Template', description: template.description || 'Reusable graph template' }));
    const categorized = new Set(GROUPS.flatMap((group) => group.keys));
    const operationGroups = [
        ...GROUPS.map((group) => ({ name: group.name, items: OPERATION_ENTRIES.filter((entry) => group.keys.includes(entry.key.split('.').at(-1))) })),
        { name: 'Other operations', items: OPERATION_ENTRIES.filter((entry) => !categorized.has(entry.key.split('.').at(-1))) },
    ];
    const groups = (section === 'templates' ? [{ name: 'Saved templates', items: templateEntries, template: true }]
        : section === 'components' ? [{ name: 'Editable components', items: STANDARD_ENTRIES }, { name: 'Layer modules', items: MODULE_ENTRIES }]
            : operationGroups).map((group) => ({ ...group, items: group.items.filter((entry) => matchesEntry(entry, query)) }));
    const visibleEntries = groups.flatMap(({ items }) => items);
    const label = section === 'components' ? 'components' : section === 'templates' ? 'templates' : 'operations';

    function searchKeyDown(event) {
        if (event.key === 'Escape' && query) { event.preventDefault(); event.stopPropagation(); setQuery(''); }
        if (event.key === 'ArrowDown') { event.preventDefault(); panelRef.current?.querySelector('[role="menuitem"]')?.focus(); }
        if (event.key === 'Enter' && query.trim() && visibleEntries[0]) { event.preventDefault(); onAdd(visibleEntries[0], section === 'templates'); }
    }
    function menuKeyDown(event) {
        if (event.target === searchRef.current || !['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault();
        const items = [...panelRef.current.querySelectorAll('[role="menuitem"]')];
        const index = items.indexOf(document.activeElement);
        const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        items[next]?.focus();
    }

    return <div ref={panelRef} className={`df-insert-menu df-insert-menu--${section}`} onKeyDown={menuKeyDown}>
        <div className="df-insert-search"><Search size={13} /><input autoFocus ref={searchRef} value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={searchKeyDown}
            placeholder={`Find ${label}…`} aria-label={`Search ${label}`} autoComplete="off" spellCheck={false} />
            <button type="button" onClick={() => { if (query) { setQuery(''); searchRef.current?.focus(); } else onClose(); }} aria-label={query ? 'Clear search' : `Close ${label} menu`}><X size={12} /></button>
        </div>
        <div className="df-insert-menu-scroll" role="menu" aria-label={`Insert ${label}`}>
            {groups.filter(({ items }) => items.length).map(({ name, items, template }) => <section key={name} className="df-insert-section" aria-label={name}>
                <h3>{name}</h3><div className="df-insert-items">{items.map((entry) => <LibraryItem key={entry.key} entry={entry} onAdd={onAdd} onDragEnd={onClose} template={template} />)}</div>
            </section>)}
            {!visibleEntries.length && <p className="df-insert-empty">{section === 'templates' && !query ? 'Group blocks to create a reusable template.' : 'No matches.'}</p>}
        </div>
        {onCustomize && <button type="button" className="df-insert-customize" onClick={onCustomize}><SlidersHorizontal size={13} />Customize pinned components</button>}
    </div>;
}
