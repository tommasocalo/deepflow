import React, { useEffect, useRef } from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronDown, ChevronRight, Repeat, X } from 'lucide-react';
import { PYTORCH_BLOCKS, getEntry } from '../../domain/library/index.js';
import { createVariantPatch, getNodeFamily } from '../../domain/library/nodeFamilies.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import ParameterField from './ParameterField.jsx';
import OperationIcon from './OperationIcon.jsx';
import { useInspectorPlacement } from './useInspectorPlacement.js';
import { compactConnectionWarnings, compactParameterError, partitionParameterErrors } from './inspectorIssues.js';
import './nodes.css';
import './inspector.css';

const SHAPE_PRIORITY = [
    'in_channels', 'out_channels', 'in_features', 'out_features', 'embed_dim', 'num_heads',
    'd_model', 'nhead', 'hidden_size', 'input_size', 'num_features', 'num_embeddings', 'embedding_dim',
    'max_len', 'normalized_shape', 'kernel_size', 'p', 'shape', 'name', 'start_dim', 'end_dim', 'dim',
];
const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);

const FIELD_LABELS = {
    in_features: 'Input', out_features: 'Output', in_channels: 'In channels', out_channels: 'Out channels',
    embed_dim: 'Width', embedding_dim: 'Width', num_heads: 'Heads', d_model: 'Width', nhead: 'Heads',
    input_size: 'Input', hidden_size: 'Hidden', num_features: 'Features', num_embeddings: 'Vocabulary',
    max_len: 'Context', normalized_shape: 'Shape', kernel_size: 'Kernel', p: 'Dropout',
    start_dim: 'Start dim', end_dim: 'End dim', dim: 'Axis', iterationVariable: 'Index',
    split_size_or_sections: 'Sections', padding_idx: 'Padding index',
};

// A few width bands keep numeric settings small without resizing for every keystroke.
function inspectorWidth(node) {
    if (node.kind !== 'layer') return 200;
    const entry = getEntry(node.libraryKey);
    const primary = (entry?.parameters ?? []).filter((p) => p.required || SHAPE_PRIORITY.includes(p.name));
    const longest = Math.max(0, ...primary.map((p) => {
        const value = node.parameters[p.name];
        return String(Array.isArray(value) ? value.join(', ') : value ?? '').length;
    }));
    const contentWidth = longest <= 7 ? 168 : longest <= 11 ? 200 : 224;
    const variantWidth = node.libraryKey === 'deepflow.nn.PositionalEmbedding' ? 200
        : getNodeFamily(node.libraryKey)?.key === 'attention' ? 200 : 168;
    return Math.max(contentWidth, variantWidth);
}

function InspectorField({ param, value, label, ...props }) {
    const text = Array.isArray(value) ? value.join(', ') : String(value ?? '');
    const wide = param.name === 'shape' || (text.length > 13 && param.type !== 'enum' && param.type !== 'bool');
    return (
        <div className={`df-inspector-field${wide ? ' is-wide' : ''}`} data-parameter={param.name}>
            <ParameterField
                param={{ ...param, description: `${param.name}${param.description ? ` — ${param.description}` : ''}` }}
                value={value}
                label={label ?? FIELD_LABELS[param.name] ?? param.name.replaceAll('_', ' ')}
                ariaLabel={param.name}
                {...props}
            />
        </div>
    );
}

function ParamGroup({ params, values, fieldErrors, onPatch, collapsed = false }) {
    const detailsRef = useRef(null);
    const hasErrors = params.some((p) => fieldErrors[p.name]?.length);
    useEffect(() => {
        if (hasErrors && detailsRef.current) detailsRef.current.open = true;
    }, [hasErrors]);
    if (!params.length) return null;
    const fields = <div className="df-inspector-fields">{params.map((p) => (
        <InspectorField
            key={p.name}
            param={p}
            value={values[p.name]}
            onChange={(v) => onPatch({ [p.name]: v })}
            errors={(fieldErrors[p.name] ?? []).map((message) => compactParameterError(p.name, message))}
        />
    ))}</div>;
    return collapsed ? (
        <details ref={detailsRef} className="df-inspector-group df-inspector-options">
            <summary>Options</summary>
            {fields}
        </details>
    ) : fields;
}

function Issues({ errors, warnings }) {
    if (!errors.length && !warnings.length) return null;
    return (
        <div className="df-inspector-issues">
            {errors.map((e, i) => <div key={`e${i}`} className="df-inspector-issue is-error"><AlertCircle size={11} strokeWidth={3} /><span>{e}</span></div>)}
            {compactConnectionWarnings(warnings).map((w, i) => <div key={`w${i}`} className="df-inspector-issue is-warning"><AlertTriangle size={11} strokeWidth={3} /><span>{w}</span></div>)}
        </div>
    );
}

function LayerBody({ id, node, fieldErrors }) {
    const entry = getEntry(node.libraryKey);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const patchLayerParameters = useArchitectureStore((s) => s.patchLayerParameters);
    const params = entry?.parameters ?? [];
    const primary = params.filter((p) => p.required || SHAPE_PRIORITY.includes(p.name));
    const options = params.filter((p) => !p.required && !SHAPE_PRIORITY.includes(p.name));
    const family = getNodeFamily(node.libraryKey);
    const moduleOptions = family?.variants ?? (entry && entry.category !== 'IO'
        ? PYTORCH_BLOCKS.filter((c) => c.category === entry.category && !TEMPLATE_ONLY.has(c.key) && !getNodeFamily(c.key))
        : []);
    const selectedModule = moduleOptions.find((option) => option.key === node.libraryKey);
    const switchModule = (libraryKey) => {
        const patch = createVariantPatch(node, libraryKey);
        if (patch) patchNode(id, patch);
    };
    const onPatch = (patch) => patchLayerParameters(id, patch);
    return (
        <>
            {moduleOptions.length > 1 && (
                <label className="df-inspector-module">
                    <span>Type</span>
                    <span className="df-inspector-select">
                        <select aria-label={family ? "Operation type" : "Module"} title={selectedModule?.name} value={node.libraryKey} onChange={(e) => switchModule(e.target.value)}>
                            {moduleOptions.map((o) => <option key={o.key} value={o.key}>{o.name}</option>)}
                        </select>
                        <ChevronDown size={11} strokeWidth={3} aria-hidden="true" />
                    </span>
                </label>
            )}
            {params.length ? (
                <>
                    <ParamGroup params={primary} values={node.parameters} fieldErrors={fieldErrors} onPatch={onPatch} />
                    <ParamGroup params={options} values={node.parameters} fieldErrors={fieldErrors} onPatch={onPatch} collapsed />
                </>
            ) : <p className="df-inspector-note">No parameters.</p>}
        </>
    );
}

function Bindings({ freeParams, values, onChange, hint }) {
    if (!freeParams.length) return <p className="df-inspector-note">No configurable parameters.</p>;
    return (
        <section className="df-inspector-group">
            <div className="df-inspector-fields">
                {freeParams.map((p) => (
                    <InspectorField
                        key={p.name}
                        param={p}
                        label={`$${p.name}`}
                        value={values[p.name]}
                        placeholder={p.default != null ? String(p.default) : 'Default'}
                        onChange={(v) => onChange(p.name, v)}
                    />
                ))}
            </div>
            {hint && <p className="df-inspector-note">{hint}</p>}
        </section>
    );
}

function SupernodeBody({ id, node }) {
    const template = useTemplatesStore((s) => s.templates[node.templateId]);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const convertInstanceToRepeat = useArchitectureStore((s) => s.convertInstanceToRepeat);
    const values = node.parameterValues ?? {};
    return (
        <>
            {!template && <p className="df-inspector-note">Template missing — re-create or delete this node.</p>}
            <div className="df-inspector-actions">
                <button type="button" disabled={!template} onClick={() => enterSupernode(id)}><ChevronRight size={12} strokeWidth={3} /> Open</button>
                <button type="button" disabled={!template} onClick={() => convertInstanceToRepeat(id, 2)} title="Repeat this template N times"><Repeat size={12} strokeWidth={3} /> Repeat</button>
            </div>
            <Bindings
                freeParams={template?.freeParameters ?? []}
                values={values}
                onChange={(name, v) => patchNode(id, { parameterValues: { ...values, [name]: v } })}
            />
        </>
    );
}

function RepeatBody({ id, node }) {
    const template = useTemplatesStore((s) => s.templates[node.templateId]);
    const patchRepeatNode = useArchitectureStore((s) => s.patchRepeatNode);
    const patchRepeatBinding = useArchitectureStore((s) => s.patchRepeatBinding);
    const enterSupernode = useArchitectureStore((s) => s.enterSupernode);
    const convertRepeatToInstance = useArchitectureStore((s) => s.convertRepeatToInstance);
    const iterVar = node.iterationVariable || 'i';
    return (
        <>
            {!template && <p className="df-inspector-note">Template missing — re-create or delete this node.</p>}
            <div className="df-inspector-actions">
                <button type="button" disabled={!template} onClick={() => enterSupernode(id)}><ChevronRight size={12} strokeWidth={3} /> Open template</button>
                <button type="button" disabled={!template} onClick={() => convertRepeatToInstance(id)} title="Back to a single instance"><Boxes size={12} strokeWidth={3} /> Single</button>
            </div>
            <section className="df-inspector-group">
                <div className="df-inspector-fields">
                    <InspectorField
                        param={{ name: 'count', type: 'int', description: 'Number of iterations' }}
                        label="count"
                        value={node.count}
                        onChange={(v) => { const n = Number(v); if (Number.isInteger(n) && n > 0) patchRepeatNode(id, { count: n }); }}
                    />
                    <InspectorField
                        param={{ name: 'iterationVariable', type: 'string', description: 'Name bound to the iteration index in formulas' }}
                        label="index"
                        value={iterVar}
                        onChange={(v) => patchRepeatNode(id, { iterationVariable: (v || 'i').trim() || 'i' })}
                    />
                </div>
            </section>
            <Bindings
                freeParams={template?.freeParameters ?? []}
                values={node.bindings ?? {}}
                onChange={(name, v) => patchRepeatBinding(id, name, v)}
                hint={`Formulas may use ${iterVar}, e.g. 64 * 2 ** ${iterVar}.`}
            />
        </>
    );
}

/**
 * Floating settings card with a shared operation icon, paired primary fields,
 * and secondary options in a disclosure.
 */
export default function NodeInspector() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const clearExpanded = useArchitectureStore((s) => s.clearExpanded);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const patchRepeatNode = useArchitectureStore((s) => s.patchRepeatNode);
    const validation = useValidation();
    const panelRef = useInspectorPlacement(expandedNodeId);

    const graph = resolveGraphLevel(rootGraph, currentPath, templates).graph;
    const node = expandedNodeId ? graph?.nodes?.[expandedNodeId] : null;
    if (!node) return null;

    const id = node.id;
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const { fields: fieldErrors, remaining: nodeErrors } = node.kind === 'layer'
        ? partitionParameterErrors(getEntry(node.libraryKey)?.parameters ?? [], errors)
        : { fields: {}, remaining: errors };
    const ContainerIcon = node.kind === 'repeat' ? Repeat : Boxes;
    const rename = (label) => (node.kind === 'repeat' ? patchRepeatNode(id, { label }) : patchNode(id, { label }));

    return (
        <aside ref={panelRef} className="df-inspector df-panel-subtle nodrag nopan nowheel" aria-label={`${node.label} inspector`} style={{ '--df-inspector-width': `${inspectorWidth(node)}px` }} onKeyDown={(e) => e.stopPropagation()}>
            <header className="df-inspector-header">
                {node.kind === 'layer'
                    ? <OperationIcon libraryKey={node.libraryKey} size={24} />
                    : <ContainerIcon size={22} strokeWidth={2.5} aria-hidden="true" />}
                <input aria-label="Block name" title="Block name" value={node.label} onChange={(e) => rename(e.target.value)} />
                <button type="button" className="df-inspector-close" onClick={clearExpanded} aria-label="Close inspector"><X size={13} strokeWidth={3} /></button>
            </header>
            <div className="df-inspector-body">
                {node.kind === 'layer' && <LayerBody id={id} node={node} fieldErrors={fieldErrors} />}
                {node.kind === 'supernode' && <SupernodeBody id={id} node={node} />}
                {node.kind === 'repeat' && <RepeatBody id={id} node={node} />}
                <Issues errors={nodeErrors} warnings={warnings} />
            </div>
        </aside>
    );
}
