import { createContext, useContext, useState } from 'react';
import { ArrowUpRight, Link2, Unlink, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { resolveComponentContext } from '../../domain/componentContext.js';
import { availableParameters, formatParameterValue, parameterConsumers, parameterFrames, parameterOrigin, parameterReferences, resolvedFieldValue } from '../../domain/parameterBindings.js';
import './parameter-binding.css';

const BindingContext = createContext(null);

export function ParameterBindingsProvider({ node, children }) {
    const graph = useArchitectureStore((s) => s.graph);
    const path = useArchitectureStore((s) => s.currentPath);
    const openLevel = useArchitectureStore((s) => s.openLevel);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const templates = useTemplatesStore((s) => s.templates);
    const context = useValidationContext();
    const report = useValidation();
    const level = resolveGraphLevel(graph, path, templates);
    const template = templates[node.templateId];
    const component = template ? resolveComponentContext(node, template, context.graph, report.shapes, context.scope) : null;
    const frames = parameterFrames(graph, path, templates);
    const navigate = (targetPath, id) => {
        openLevel(targetPath);
        selectOnly(id);
        setExpanded(id);
    };
    return <BindingContext.Provider value={{ node, path, frames, context, component, template, owner: templates[level.templateId], navigate }}>{children}</BindingContext.Provider>;
}

export function DerivedParameter({ param }) {
    const binding = useContext(BindingContext);
    if (!binding) return null;
    const input = binding.template?.graph.nodes[param.derived.inputId];
    const label = param.name === 'd_head' ? 'Head width' : param.name === 'D' ? 'Width' : param.name.replaceAll('_', ' ');
    return <div className="df-derived-parameter" data-parameter={param.name}>
        <span>{label}</span><strong>{formatParameterValue(binding.component?.scope[param.name])}</strong>
        <small>From {input?.label ?? 'input'} · {param.derived.axis === -1 ? 'last dimension' : `axis ${param.derived.axis}`}</small>
    </div>;
}

/** Compact field-level authoring: local value, existing reference, or an explicit exposed control. */
export default function ParameterBinding({ param, value, onChange, children }) {
    const binding = useContext(BindingContext);
    const exposeField = useTemplatesStore((s) => s.exposeField);
    const [open, setOpen] = useState(false);
    const [draft, setDraft] = useState(param.name);
    if (!binding || !['int', 'float', 'bool', 'tuple'].includes(param.type) || param.derived) return children;
    const { node, frames, context, component, template, owner, path, navigate } = binding;
    const available = availableParameters(frames);
    const scope = param.name === 'count' ? context.scope : component?.scope ?? context.scope;
    const raw = value ?? param.default;
    const references = parameterReferences(raw).filter((name) => available.some((p) => p.name === name) || String(raw).includes(`$${name}`));
    const resolved = resolvedFieldValue(raw, scope);
    const origins = [...new Map(references.map((name) => parameterOrigin(name, frames)).filter(Boolean).map((origin) => [`${JSON.stringify(origin.path)}:${origin.nodeId}:${origin.name}`, origin])).values()];
    const consumers = template && param.name !== 'count' ? parameterConsumers(template, param.name) : [];
    const name = draft.trim();
    const validName = /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !owner?.freeParameters?.some((p) => p.name === name);
    const expose = (event) => {
        event.preventDefault();
        if (!owner || !validName) return;
        if (exposeField(owner.id, { nodeId: node.id, field: param.name, name, spec: param, defaultValue: resolved })) setOpen(false);
    };
    const canDetach = references.length > 0 && resolved != null && (!Array.isArray(resolved) || resolved.every((part) => part != null));
    return <div className={`df-field-binding${open ? ' is-open' : ''}`}>
        {children}
        <button type="button" className="df-field-link" aria-label={`Parameter options for ${param.name}`} title="Link or expose this field" aria-expanded={open} onClick={() => setOpen(!open)}><Link2 size={11} /></button>
        {references.length > 0 && <div className="df-parameter-source">
            <strong>{formatParameterValue(resolved)}</strong>
            {origins.length ? origins.map((origin, index) => <button type="button" key={`${origin.nodeId}-${origin.name}-${index}`} title={`Open ${origin.label} · ${origin.name}`} onClick={() => navigate(origin.path, origin.nodeId)}>← {origin.label} · {origin.name}<ArrowUpRight size={9} /></button>) : <span>Unresolved reference</span>}
        </div>}
        {open && <div className="df-parameter-menu" role="group" aria-label={`${param.name} parameter options`}>
            <header><strong>Parameter</strong><button type="button" aria-label={`Close ${param.name} parameter options`} onClick={() => setOpen(false)}><X size={11} /></button></header>
            {available.length > 0 && <label>Link to
                <select aria-label={`Link ${param.name} to parameter`} value="" onChange={(event) => { if (event.target.value) { onChange(`$${event.target.value}`); setOpen(false); } }}>
                    <option value="">Choose parameter…</option>
                    {available.filter((p) => p.type === param.type || ['int', 'float'].includes(p.type) && ['int', 'float'].includes(param.type)).map((p) => <option key={p.name} value={p.name}>{p.name} · {p.owner}</option>)}
                </select>
            </label>}
            {canDetach && <button type="button" className="df-parameter-action" onClick={() => { onChange(resolved); setOpen(false); }}><Unlink size={11} /> Use local value {formatParameterValue(resolved)}</button>}
            {owner && <form onSubmit={expose}>
                <label>Expose in {owner.name}<input aria-label={`Exposed name for ${param.name}`} value={draft} onChange={(event) => setDraft(event.target.value)} spellCheck={false} /></label>
                <button className="df-parameter-action" type="submit" disabled={!validName}>Expose in component</button>
                {!validName && name && <small>Choose an unused parameter name.</small>}
            </form>}
            {consumers.length > 0 && <div className="df-parameter-consumers"><strong>Used by</strong>{consumers.map((use) => use.parameter ? <span key={`${use.label}-${use.field}`}>{use.label} · default</span> : <button type="button" key={`${use.nodeId}-${use.field}`} onClick={() => navigate([...path, node.id], use.nodeId)}>{use.label} · {use.field}<ArrowUpRight size={10} /></button>)}</div>}
            {!owner && !available.length && !consumers.length && <small>Open a component to expose its internal fields.</small>}
        </div>}
    </div>;
}
