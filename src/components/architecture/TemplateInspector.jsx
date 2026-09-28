import { useState } from 'react';
import { ArrowUpRight, Globe, Plus, Trash2, Variable } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { globalParameters } from '../../domain/expressions.js';
import { globalParameterUses, parameterConsumers } from '../../domain/parameterBindings.js';
import VariableInput from './VariableInput.jsx';
import './template-inspector.css';

const NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** The value decides the type: numbers, booleans and tuples; formulas keep theirs. */
function inferType(value, current) {
    const text = String(value ?? '').trim();
    if (/^(true|false)$/i.test(text)) return 'bool';
    if (/^[-+]?\d+$/.test(text)) return 'int';
    if (/^[-+]?(\d+\.\d*|\.\d+)(e[-+]?\d+)?$/i.test(text) || /^[-+]?\d+e[-+]?\d+$/i.test(text)) return 'float';
    if (/^[([].*[)\]]$/.test(text)) return 'tuple';
    return current ?? 'int';
}

function ParameterRow({ parameter, uses, suggestions, onChange, onRemove }) {
    const [open, setOpen] = useState(false);
    const value = Array.isArray(parameter.default) ? parameter.default.join(', ') : parameter.default ?? '';
    return <li className="df-params-item">
        {/* df-param-row lets the $ suggestions open across the whole row. */}
        <div className="df-params-row df-param-row">
            <code title={parameter.description || undefined}>${parameter.name}</code>
            <VariableInput className="df-params-value" aria-label={`${parameter.name} value`} spellCheck={false} autoComplete="off"
                placeholder="—" value={value} suggestions={suggestions.filter((suggestion) => suggestion.name !== parameter.name)}
                onChange={(event) => {
                    const next = event.target.value;
                    onChange({ default: next === '' ? null : next, type: inferType(next, parameter.type) });
                }} />
            <button type="button" className="df-params-uses" aria-expanded={open} disabled={!uses.length} onClick={() => setOpen(!open)}
                title={uses.length ? 'Show where it is used' : 'Not used yet: type it as $name in a field'}>
                {uses.length ? `${uses.length} ${uses.length === 1 ? 'use' : 'uses'}` : 'unused'}
            </button>
            <button type="button" className="df-params-remove" onClick={onRemove}
                aria-label={`Remove parameter ${parameter.name}`}
                title={uses.length ? `Remove ${parameter.name}: the ${uses.length} ${uses.length === 1 ? 'field that uses it keeps' : 'fields that use it keep'} its value${value !== '' ? ` (${value})` : ''}` : `Remove ${parameter.name}`}>
                <Trash2 size={12} aria-hidden="true" />
            </button>
        </div>
        {open && <div className="df-params-consumers">
            {uses.map((use) => use.onOpen
                ? <button type="button" key={use.key} onClick={use.onOpen}>{use.text}<ArrowUpRight size={10} aria-hidden="true" /></button>
                : <span key={use.key}>{use.text}</span>)}
        </div>}
    </li>;
}

/** One row per variable: name, value, where it is used; plus a row to add one. */
function ParameterList({ parameters, derived = [], suggestions, usesOf, onChange, onRemove, onAdd, reserved, empty }) {
    const [draft, setDraft] = useState('');
    const name = draft.trim().replace(/^\$/, '');
    const duplicate = reserved.includes(name);
    const invalid = !!name && !NAME.test(name);
    const add = (event) => {
        event.preventDefault();
        if (!name || invalid || duplicate) return;
        onAdd(name);
        setDraft('');
    };
    return <>
        {parameters.length > 0
            ? <ul className="df-params-list">{parameters.map((parameter) => <ParameterRow key={parameter.name} parameter={parameter} uses={usesOf(parameter.name)}
                suggestions={suggestions} onChange={(patch) => onChange(parameter.name, patch)} onRemove={() => onRemove(parameter.name)} />)}</ul>
            : <p className="df-params-empty">{empty}</p>}
        {derived.length > 0 && <section className="df-params-derived" aria-label="Parameters from inputs">
            <h3>From inputs</h3>
            {derived.map((parameter) => <div key={parameter.name} className="df-params-row is-derived">
                <code>{parameter.name}</code><span>{parameter.source}</span>
            </div>)}
        </section>}
        <form className="df-params-add" onSubmit={add}>
            <span aria-hidden="true">$</span>
            <input aria-label="New parameter name" placeholder="new_parameter" autoComplete="off" spellCheck={false}
                value={draft} onChange={(event) => setDraft(event.target.value)} aria-invalid={invalid || duplicate || undefined} />
            <button type="submit" disabled={!name || invalid || duplicate} aria-label="Add parameter" title="Add parameter"><Plus size={13} /></button>
        </form>
        {(invalid || duplicate) && <p className="df-params-error" role="alert">{duplicate ? 'This name already exists.' : 'Letters, digits and _; start with a letter.'}</p>}
    </>;
}

function TemplateParameters({ template, globals, onOpenConsumer }) {
    const patchTemplate = useTemplatesStore((s) => s.patchTemplate);
    const addFreeParameter = useTemplatesStore((s) => s.addFreeParameter);
    const patchFreeParameter = useTemplatesStore((s) => s.patchFreeParameter);
    const removeFreeParameter = useTemplatesStore((s) => s.removeFreeParameter);
    const parameters = template.freeParameters ?? [];
    const own = parameters.filter((parameter) => !parameter.derived);
    const suggestions = [
        ...globals.filter((parameter) => !parameters.some((local) => local.name === parameter.name)).map((parameter) => ({ name: parameter.name, value: parameter.default, owner: 'Global' })),
        ...own.map((parameter) => ({ name: parameter.name, value: parameter.default })),
    ];
    const usesOf = (name) => parameterConsumers(template, name).map((use) => use.parameter
        ? { key: `${use.label}-default`, text: `$${use.label} default` }
        : { key: `${use.edgeId ?? use.nodeId}-${use.field}`, text: `${use.label} · ${use.field}`, onOpen: onOpenConsumer ? () => onOpenConsumer(use.nodeId) : undefined });
    return <div className="df-params nodrag nopan nowheel">
        <header className="df-params-header">
            <Variable size={14} aria-hidden="true" />
            <input aria-label="Component name" title="Component name" value={template.name} onChange={(event) => patchTemplate(template.id, { name: event.target.value })} />
        </header>
        <p className="df-params-note">Shared defaults</p>
        <ParameterList parameters={own} suggestions={suggestions} usesOf={usesOf}
            derived={parameters.filter((parameter) => parameter.derived).map((parameter) => ({ name: parameter.name,
                source: `${template.graph.nodes[parameter.derived.inputId]?.label ?? 'input'} · ${parameter.derived.axis === -1 ? 'last dim' : `axis ${parameter.derived.axis}`}` }))}
            onChange={(name, patch) => patchFreeParameter(template.id, name, patch)}
            onRemove={(name) => removeFreeParameter(template.id, name)}
            onAdd={(name) => addFreeParameter(template.id, { name, type: 'int', description: '', default: null })}
            reserved={parameters.map((parameter) => parameter.name)}
            empty={<>No parameters yet. Add one below, then type <code>$name</code> in a field.</>} />
    </div>;
}

/** Parameters of the component selected, or of the level open in the canvas. */
export default function TemplateInspector({ templateId }) {
    const graph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const selectedNodeIds = useArchitectureStore((s) => s.selectedNodeIds);
    const openLevel = useArchitectureStore((s) => s.openLevel);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const templates = useTemplatesStore((s) => s.templates);
    const level = resolveGraphLevel(graph, currentPath, templates, inspectedIterations);
    const template = templates[templateId ?? level.templateId];

    if (!template) return null;
    const selected = selectedNodeIds.map((id) => level.graph.nodes[id]).find((node) => node && resolveGraphLevel(graph, [...currentPath, node.id], templates, inspectedIterations).templateId === template.id);
    const path = level.templateId === template.id ? currentPath : selected ? [...currentPath, selected.id] : null;
    const openConsumer = path ? (id) => { openLevel(path); selectOnly(id); setExpanded(id); } : null;
    return <TemplateParameters key={template.id} template={template} globals={globalParameters(graph)} onOpenConsumer={openConsumer} />;
}

/** Model-wide variables, always available: any field can use them as $name. */
export function GlobalParameters() {
    const graph = useArchitectureStore((s) => s.graph);
    const templates = useTemplatesStore((s) => s.templates);
    const addGlobalParameter = useArchitectureStore((s) => s.addGlobalParameter);
    const patchGlobalParameter = useArchitectureStore((s) => s.patchGlobalParameter);
    const removeGlobalParameter = useArchitectureStore((s) => s.removeGlobalParameter);
    const openLevel = useArchitectureStore((s) => s.openLevel);
    const selectOnly = useArchitectureStore((s) => s.selectOnly);
    const setExpanded = useArchitectureStore((s) => s.setExpanded);
    const parameters = globalParameters(graph);
    const usesOf = (name) => globalParameterUses(graph, templates, name).map((use) => use.parameter
        ? { key: `${use.label}-default`, text: `$${use.label} default` }
        : use.root
            ? { key: `root-${use.edgeId ?? use.nodeId}-${use.field}`, text: `${use.label} · ${use.field}`, onOpen: () => { openLevel([]); selectOnly(use.nodeId); setExpanded(use.nodeId); } }
            : { key: `${use.template}-${use.edgeId ?? use.nodeId}-${use.field}`, text: `${use.template} › ${use.label} · ${use.field}` });
    return <div className="df-params nodrag nopan nowheel">
        <header className="df-params-header">
            <Globe size={14} aria-hidden="true" />
            <strong>Global parameters</strong>
        </header>
        <p className="df-params-note">Available everywhere as <code>$name</code></p>
        <ParameterList parameters={parameters} suggestions={parameters.map((parameter) => ({ name: parameter.name, value: parameter.default }))} usesOf={usesOf}
            onChange={(name, patch) => patchGlobalParameter(name, patch)}
            onRemove={(name) => removeGlobalParameter(name)}
            onAdd={(name) => addGlobalParameter({ name, type: 'int', default: null })}
            reserved={parameters.map((parameter) => parameter.name)}
            empty={<>No global parameters yet, e.g. <code>d_model</code> or <code>n_layers</code>.</>} />
    </div>;
}
