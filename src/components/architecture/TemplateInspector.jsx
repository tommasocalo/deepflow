import { useId, useState } from 'react';
import { ArrowUpRight, ChevronDown, Eye, EyeOff, Plus, Trash2, Variable } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { parameterConsumers } from '../../domain/parameterBindings.js';
import './template-inspector.css';

const PARAM_TYPES = ['int', 'float', 'string', 'bool', 'tuple', 'enum'];

function TemplateParameters({ template, embedded, initiallyOpen, onOpenConsumer }) {
    const patchTemplate = useTemplatesStore((s) => s.patchTemplate);
    const addFreeParameter = useTemplatesStore((s) => s.addFreeParameter);
    const patchFreeParameter = useTemplatesStore((s) => s.patchFreeParameter);
    const removeFreeParameter = useTemplatesStore((s) => s.removeFreeParameter);
    const [draftName, setDraftName] = useState('');
    const hintId = useId();

    const free = (template.freeParameters ?? []).filter((parameter) => !parameter.derived);
    const derived = (template.freeParameters ?? []).filter((parameter) => parameter.derived);
    const name = draftName.trim();
    const validName = /^[A-Za-z_][A-Za-z0-9_]*$/.test(name);
    const duplicate = (template.freeParameters ?? []).some((parameter) => parameter.name === name);
    const canAdd = validName && !duplicate;
    const nameHint = duplicate
        ? 'A parameter with this name already exists.'
        : name && !validName
            ? 'Use letters, numbers and underscores; start with a letter or underscore.'
            : 'Reference this parameter in a block as $name.';

    const handleAdd = (event) => {
        event.preventDefault();
        if (!canAdd) return;
        addFreeParameter(template.id, { name, type: 'int', description: '', default: null });
        setDraftName('');
    };

    return (
        <details open={initiallyOpen || undefined} className={`df-template-inspector nodrag nopan nowheel ${embedded ? 'is-embedded' : 'is-floating'}`}>
            <summary className="df-template-summary">
                <Variable size={15} aria-hidden="true" />
                <span>Template parameters</span>
                <span className="df-template-count" aria-label={`${free.length} parameters`}>{free.length}</span>
                <ChevronDown className="df-template-chevron" size={14} aria-hidden="true" />
            </summary>

            <div className="df-template-content">
                <p className="df-template-note">Edits apply to all instances.</p>
                <label className="df-template-field">
                    <span>Template name</span>
                    <input
                        autoFocus={initiallyOpen}
                        value={template.name}
                        onChange={(event) => patchTemplate(template.id, { name: event.target.value })}
                    />
                </label>
                <label className="df-template-field">
                    <span>Description</span>
                    <textarea
                        rows={2}
                        placeholder="What does this template do?"
                        value={template.description ?? ''}
                        onChange={(event) => patchTemplate(template.id, { description: event.target.value })}
                    />
                </label>

                <div className="df-template-parameters">
                    <h3>Exposed parameters</h3>
                    {free.length === 0 && (
                        <p className="df-template-note">Use <code>$name</code> in block parameters.</p>
                    )}
                    {free.map((parameter) => {
                        const uses = parameterConsumers(template, parameter.name);
                        return (
                        <section key={parameter.name} className="df-template-parameter" aria-label={`Parameter ${parameter.name}`}>
                            <div className="df-template-parameter-heading">
                                <code>${parameter.name}</code>
                                <button type="button" className="df-field-visibility" aria-label={`${parameter.name} in ${template.name} settings`}
                                    aria-pressed={!parameter.hidden} title={`${parameter.hidden ? 'Show' : 'Hide'} ${parameter.name} in component settings`}
                                    onClick={() => patchFreeParameter(template.id, parameter.name, { hidden: !parameter.hidden })}>
                                    {parameter.hidden ? <EyeOff size={12} /> : <Eye size={12} />}
                                </button>
                                <button
                                    type="button"
                                    className="df-template-remove"
                                    onClick={() => removeFreeParameter(template.id, parameter.name)}
                                    disabled={uses.length > 0}
                                    aria-label={`Remove parameter ${parameter.name}`}
                                    title={uses.length ? 'Unlink the fields using this parameter before removing it' : `Remove ${parameter.name}`}
                                >
                                    <Trash2 size={13} aria-hidden="true" />
                                </button>
                            </div>
                            <label className="df-template-field">
                                <span>Type</span>
                                <select
                                    aria-label={`${parameter.name} type`}
                                    value={parameter.type}
                                    onChange={(event) => patchFreeParameter(template.id, parameter.name, { type: event.target.value })}
                                >
                                    {PARAM_TYPES.map((type) => <option key={type} value={type}>{type}</option>)}
                                </select>
                            </label>
                            <label className="df-template-field">
                                <span>Default value</span>
                                <input
                                    aria-label={`${parameter.name} default value`}
                                    placeholder="Optional"
                                    value={parameter.default ?? ''}
                                    onChange={(event) => patchFreeParameter(template.id, parameter.name, { default: event.target.value || null })}
                                />
                            </label>
                            <label className="df-template-field">
                                <span>Description</span>
                                <input
                                    aria-label={`${parameter.name} description`}
                                    placeholder="What does this control?"
                                    value={parameter.description ?? ''}
                                    onChange={(event) => patchFreeParameter(template.id, parameter.name, { description: event.target.value })}
                                />
                            </label>
                            <details className="df-template-uses">
                                <summary>Used by {uses.length} {uses.length === 1 ? 'field' : 'fields'}</summary>
                                {uses.map((use) => use.parameter ? <span key={`${use.label}-${use.field}`}>{use.label} · default</span> : <button type="button" key={`${use.nodeId}-${use.field}`} disabled={!onOpenConsumer} onClick={() => onOpenConsumer?.(use.nodeId)}>{use.label} · {use.field}<ArrowUpRight size={11} /></button>)}
                                {!uses.length && <span>Link an internal field to ${parameter.name}.</span>}
                            </details>
                        </section>
                    ); })}
                    {derived.length > 0 && <section className="df-template-derived"><h3>From inputs</h3>{derived.map((parameter) => <p key={parameter.name}><strong>{parameter.name}</strong><span>{template.graph.nodes[parameter.derived.inputId]?.label ?? 'Input'} · {parameter.derived.axis === -1 ? 'last dimension' : `axis ${parameter.derived.axis}`}</span></p>)}</section>}
                </div>

                <form className="df-template-add" onSubmit={handleAdd}>
                    <label className="df-template-field">
                        <span>New parameter</span>
                        <input
                            placeholder="e.g. hidden_size"
                            autoComplete="off"
                            spellCheck={false}
                            value={draftName}
                            onChange={(event) => setDraftName(event.target.value)}
                            aria-invalid={name ? !canAdd : undefined}
                            aria-describedby={hintId}
                        />
                    </label>
                    <p id={hintId} className="df-template-note">{nameHint}</p>
                    <button type="submit" className="df-template-add-button" disabled={!canAdd}>
                        <Plus size={14} aria-hidden="true" /> Add parameter
                    </button>
                </form>
            </div>
        </details>
    );
}

/** Parameters of the shared template currently open in the canvas. */
export default function TemplateInspector({ embedded = false, templateId, initiallyOpen = false }) {
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
    return <TemplateParameters key={template.id} template={template} embedded={embedded} initiallyOpen={initiallyOpen} onOpenConsumer={openConsumer} />;
}
