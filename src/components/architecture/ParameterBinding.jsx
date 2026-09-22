import { cloneElement, createContext, useContext, useState } from 'react';
import { Plus, Unlink, X } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { getEntry } from '../../domain/library/index.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { resolveComponentContext } from '../../domain/componentContext.js';
import { resolveDimensionTuple } from '../../domain/expressions.js';
import { availableParameters, formatParameterValue, parameterFrames, parameterOrigin, parameterReferences, resolvedFieldValue } from '../../domain/parameterBindings.js';
import './parameter-binding.css';

const BindingContext = createContext(null);

export function ParameterBindingsProvider({ node, children }) {
    const graph = useArchitectureStore((s) => s.graph);
    const path = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);
    const context = useValidationContext();
    const report = useValidation();
    const level = resolveGraphLevel(graph, path, templates);
    const template = templates[node.templateId];
    const component = template ? resolveComponentContext(node, template, context.graph, report.shapes, context.scope) : null;
    const frames = parameterFrames(graph, path, templates);
    const parameters = availableParameters(frames);
    for (const frame of frames) {
        const name = frame.instance.iterationVariable || 'i';
        if (frame.instance.kind === 'repeat' && !parameters.some((parameter) => parameter.name === name)) parameters.push({ name, type: 'int', owner: 'Instance index' });
    }
    const suggestions = parameters.map((parameter) => ({ ...parameter, value: resolvedFieldValue(`$${parameter.name}`, context.scope) }));
    return <BindingContext.Provider value={{ node, frames, context, component, template, owner: templates[level.templateId], suggestions }}>{children}</BindingContext.Provider>;
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

/** References are edited directly; their origin is text, not another menu. */
export default function ParameterBinding({ param, value, onChange, children }) {
    const binding = useContext(BindingContext);
    if (!binding || !['int', 'float', 'bool', 'tuple'].includes(param.type) || param.derived) return children;
    const { frames, context, component, suggestions } = binding;
    const scope = param.name === 'count' ? context.scope : component?.scope ?? context.scope;
    const raw = value ?? param.default;
    const references = parameterReferences(raw).filter((name) => suggestions.some((parameter) => parameter.name === name) || String(raw).includes(`$${name}`));
    const resolved = param.type === 'tuple' ? resolveDimensionTuple(raw, scope) : resolvedFieldValue(raw, scope);
    const origins = [...new Set(references.map((name) => parameterOrigin(name, frames)?.label).filter(Boolean))];
    const canDetach = param.type === 'tuple' ? Array.isArray(resolved) && resolved.every((part) => typeof part === 'number' && Number.isFinite(part))
        : param.type === 'bool' ? typeof resolved === 'boolean' || ['true', 'false'].includes(resolved)
            : typeof resolved === 'number' && Number.isFinite(resolved);
    return <div className="df-field-binding">
        {cloneElement(children, { suggestions })}
        {(references.length > 0 || String(raw).includes('$')) && <div className={`df-parameter-source${canDetach ? '' : ' is-unresolved'}`}>
            <span className="df-parameter-link-state">{canDetach ? 'Linked' : 'Unresolved'}</span>
            {canDetach && <strong>{formatParameterValue(resolved)}</strong>}
            {origins.length > 0 && <span className="df-parameter-origin" title={origins.join(' · ')}>from {origins.join(' · ')}</span>}
            {canDetach && <button type="button" className="df-parameter-unlink" aria-label={`Unlink ${param.name}`} title={`Replace reference with ${formatParameterValue(resolved)}`} onClick={() => onChange(resolved)}><Unlink size={11} /></button>}
        </div>}
    </div>;
}

/** One component action replaces the menus previously attached to every field. */
export function ExposeParameter() {
    const binding = useContext(BindingContext);
    const exposeField = useTemplatesStore((s) => s.exposeField);
    const [open, setOpen] = useState(false);
    const [field, setField] = useState('');
    const [name, setName] = useState('');
    if (!binding?.owner) return null;
    const { node, template, owner } = binding;
    const specs = (node.kind === 'layer' ? getEntry(node.libraryKey)?.parameters ?? [] : template?.freeParameters ?? []).filter((spec) => !spec.derived && ['int', 'float', 'bool', 'tuple'].includes(spec.type));
    if (node.kind === 'repeat' && node.count !== 'auto') specs.unshift({ name: 'count', type: 'int', description: 'Number of instances' });
    if (!specs.length) return null;
    const selected = specs.find((spec) => spec.name === field) ?? specs[0];
    const draft = name.trim();
    const valid = /^[A-Za-z_][A-Za-z0-9_]*$/.test(draft) && !owner.freeParameters?.some((parameter) => parameter.name === draft);
    const values = node.kind === 'layer' ? node.parameters : node.kind === 'repeat' ? { ...node.bindings, count: node.count } : node.parameterValues;
    const submit = (event) => {
        event.preventDefault();
        if (valid && exposeField(owner.id, { nodeId: node.id, field: selected.name, name: draft, spec: selected, defaultValue: values?.[selected.name] ?? selected.default })) setOpen(false);
    };
    return <section className="df-expose-parameter">
        {!open ? <button type="button" className="df-expose-trigger" onClick={() => { setName(selected.name); setOpen(true); }}><Plus size={11} />Expose a field…</button> : <form onSubmit={submit}>
            <header><strong>Expose in {owner.name}</strong><button type="button" aria-label="Close expose field" onClick={() => setOpen(false)}><X size={12} /></button></header>
            <label>Field<select aria-label="Field to expose" value={selected.name} onChange={(event) => { setField(event.target.value); setName(event.target.value); }}>{specs.map((spec) => <option key={spec.name} value={spec.name}>{spec.name.replaceAll('_', ' ')}</option>)}</select></label>
            <label>Parameter name<input aria-label="Exposed parameter name" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. dropout" spellCheck={false} /></label>
            {!valid && draft && <small>Use an unused name with letters, numbers or underscores.</small>}
            <button className="df-expose-submit" type="submit" disabled={!valid}>Expose parameter</button>
        </form>}
    </section>;
}
