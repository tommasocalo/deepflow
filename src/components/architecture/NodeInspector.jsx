import React from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronRight, Layers3, Repeat, X } from 'lucide-react';
import { PYTORCH_BLOCKS, getEntry } from '../../domain/library/index.js';
import { createVariantPatch, getNodeFamily } from '../../domain/library/nodeFamilies.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { formatShape } from '../../validation/index.js';
import { styleFor } from './categoryStyle.js';
import ParameterField from './ParameterField.jsx';
import { useInspectorPlacement } from './useInspectorPlacement.js';
import './nodes.css';
import './inspector.css';

const SHAPE_PRIORITY = [
    'in_channels', 'out_channels', 'in_features', 'out_features', 'embed_dim', 'num_heads',
    'd_model', 'nhead', 'hidden_size', 'input_size', 'num_features', 'num_embeddings', 'embedding_dim',
    'max_len', 'normalized_shape', 'kernel_size', 'p', 'shape', 'name',
];
const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);

function InspectorField({ param, value, ...props }) {
    const text = Array.isArray(value) ? value.join(', ') : String(value ?? '');
    const wide = param.name === 'shape' || (text.length > 14 && param.type !== 'enum' && param.type !== 'bool');
    return (
        <div className={`df-inspector-field${wide ? ' is-wide' : ''}`}>
            <ParameterField param={param} value={value} {...props} />
        </div>
    );
}

function ParamGroup({ title, params, values, errors, onPatch, collapsed = false }) {
    if (!params.length) return null;
    const fields = params.map((p) => (
        <InspectorField
            key={p.name}
            param={p}
            value={values[p.name]}
            onChange={(v) => onPatch({ [p.name]: v })}
            error={errors.some((e) => e.toLowerCase().includes(p.name.toLowerCase()))}
        />
    ));
    if (collapsed) {
        return (
            <details className="df-inspector-group">
                <summary>{title} <span className="df-inspector-count">{params.length}</span></summary>
                <div className="df-inspector-fields">{fields}</div>
            </details>
        );
    }
    return (
        <section className="df-inspector-group">
            <h3>{title}</h3>
            <div className="df-inspector-fields">{fields}</div>
        </section>
    );
}

function Issues({ errors, warnings }) {
    if (!errors.length && !warnings.length) return null;
    return (
        <div className="df-inspector-issues">
            {errors.map((e, i) => <div key={`e${i}`} className="df-inspector-issue is-error"><AlertCircle size={11} strokeWidth={3} /><span>{e}</span></div>)}
            {warnings.map((w, i) => <div key={`w${i}`} className="df-inspector-issue is-warning"><AlertTriangle size={11} strokeWidth={3} /><span>{w}</span></div>)}
        </div>
    );
}

function LayerBody({ id, node, errors }) {
    const entry = getEntry(node.libraryKey);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const patchLayerParameters = useArchitectureStore((s) => s.patchLayerParameters);
    const params = entry?.parameters ?? [];
    const required = params.filter((p) => p.required);
    const shape = params.filter((p) => !p.required && SHAPE_PRIORITY.includes(p.name));
    const options = params.filter((p) => !p.required && !SHAPE_PRIORITY.includes(p.name));
    const family = getNodeFamily(node.libraryKey);
    const moduleOptions = family?.variants ?? (entry && entry.category !== 'IO'
        ? PYTORCH_BLOCKS.filter((c) => c.category === entry.category && !TEMPLATE_ONLY.has(c.key))
        : []);
    const switchModule = (libraryKey) => {
        const patch = createVariantPatch(node, libraryKey);
        if (patch) patchNode(id, patch);
    };
    const onPatch = (patch) => patchLayerParameters(id, patch);
    return (
        <>
            {moduleOptions.length > 1 && (
                <label className="df-inspector-module">
                    <span>{family ? 'Variant' : 'Module'}</span>
                    <select aria-label={family ? "Operation type" : "Module"} value={node.libraryKey} onChange={(e) => switchModule(e.target.value)} className="df-input">
                        {moduleOptions.map((o) => <option key={o.key} value={o.key}>{o.name}</option>)}
                    </select>
                </label>
            )}
            {params.length ? (
                <>
                    <ParamGroup title="Required" params={required} values={node.parameters} errors={errors} onPatch={onPatch} />
                    <ParamGroup title={required.length ? 'Shape' : 'Parameters'} params={shape} values={node.parameters} errors={errors} onPatch={onPatch} />
                    <ParamGroup title="Options" params={options} values={node.parameters} errors={errors} onPatch={onPatch} collapsed={required.length > 0 || shape.length > 0} />
                </>
            ) : <p className="df-inspector-note">No parameters.</p>}
        </>
    );
}

function Bindings({ freeParams, values, onChange, hint }) {
    if (!freeParams.length) return <p className="df-inspector-note">No configurable parameters.</p>;
    return (
        <section className="df-inspector-group">
            <h3>Parameters</h3>
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
            <p className="df-inspector-note">{template ? (template.description || `Instance of “${template.name}”.`) : 'Template missing — re-create or delete this node.'}</p>
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
            <p className="df-inspector-note">{template ? (template.description || `Repeats “${template.name}”.`) : 'Template missing — re-create or delete this node.'}</p>
            <div className="df-inspector-actions">
                <button type="button" disabled={!template} onClick={() => enterSupernode(id)}><ChevronRight size={12} strokeWidth={3} /> Open template</button>
                <button type="button" disabled={!template} onClick={() => convertRepeatToInstance(id)} title="Back to a single instance"><Boxes size={12} strokeWidth={3} /> Single</button>
            </div>
            <section className="df-inspector-group">
                <h3>Repetition</h3>
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
 * Floating panel with everything that used to live inside an expanded node:
 * name, module, parameters, bindings, issues, output activation.
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
    const outShape = validation.shapes[id];
    const entry = node.kind === 'layer' ? getEntry(node.libraryKey) : null;
    const accent = node.kind === 'layer' ? styleFor(entry?.category ?? 'IO').accent : '#f59e0b';
    const kind = node.kind === 'layer' ? (entry?.category ?? 'block') : node.kind === 'repeat' ? 'repeat' : 'template';
    const Icon = node.kind === 'repeat' ? Repeat : node.kind === 'supernode' ? Boxes : Layers3;
    const rename = (label) => (node.kind === 'repeat' ? patchRepeatNode(id, { label }) : patchNode(id, { label }));

    return (
        <aside ref={panelRef} className="df-inspector df-panel-subtle nodrag nopan nowheel" aria-label={`${node.label} inspector`} onKeyDown={(e) => e.stopPropagation()}>
            <header className="df-inspector-header">
                <span className="df-inspector-accent" style={{ background: accent }} />
                <div className="df-inspector-heading">
                    <div className="df-inspector-kind"><Icon size={12} strokeWidth={2.5} aria-hidden="true" /><span>{kind}</span></div>
                    <input aria-label="Block name" value={node.label} onChange={(e) => rename(e.target.value)} />
                </div>
                <button type="button" className="df-inspector-close" onClick={clearExpanded} aria-label="Close inspector"><X size={13} strokeWidth={3} /></button>
            </header>
            <div className="df-inspector-body">
                {node.kind === 'layer' && <LayerBody id={id} node={node} errors={errors} />}
                {node.kind === 'supernode' && <SupernodeBody id={id} node={node} />}
                {node.kind === 'repeat' && <RepeatBody id={id} node={node} />}
                <Issues errors={errors} warnings={warnings} />
            </div>
            <footer className="df-inspector-footer">
                <span>Output shape</span>
                <code>{outShape ? formatShape(outShape) : '—'}</code>
            </footer>
        </aside>
    );
}
