import { useComponentTemplate } from '../../hooks/useComponentTemplate.js';
import React, { useEffect, useRef } from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronDown, Repeat, X } from 'lucide-react';
import { PYTORCH_BLOCKS, getEntry } from '../../domain/library/index.js';
import { createVariantPatch, getNodeFamily, isOperationKey } from '../../domain/library/nodeFamilies.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { resolveComponentContext } from '../../domain/componentContext.js';
import { STANDARD_COMPONENTS } from '../../domain/components/index.js';
import ParameterField from './ParameterField.jsx';
import ParameterBinding, { DerivedParameter, ExposeParameter, ParameterBindingsProvider } from './ParameterBinding.jsx';
import OperationIcon from './OperationIcon.jsx';
import { useInspectorPlacement } from './useInspectorPlacement.js';
import { compactConnectionWarnings, compactParameterError, partitionParameterErrors } from './inspectorIssues.js';
import './nodes.css';
import './inspector.css';

const SHAPE_PRIORITY = [
    'in_channels', 'out_channels', 'in_features', 'out_features', 'embed_dim', 'num_heads',
    'd_model', 'nhead', 'hidden_size', 'input_size', 'num_features', 'num_embeddings', 'embedding_dim',
    'max_len', 'normalized_shape', 'kernel_size', 'p', 'shape', 'name', 'start_dim', 'end_dim', 'dim', 'dim0', 'dim1', 'chunks', 'count', 'factor', 'requires_grad', 'value', 'exponent', 'start', 'end', 'step',
];
const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);

const FIELD_LABELS = {
    in_features: 'Input', out_features: 'Output', in_channels: 'In channels', out_channels: 'Out channels',
    embed_dim: 'Width', embedding_dim: 'Width', num_heads: 'Heads', d_model: 'Width', nhead: 'Heads',
    input_size: 'Input', hidden_size: 'Hidden', num_features: 'Features', num_embeddings: 'Vocabulary',
    max_len: 'Context', normalized_shape: 'Shape', kernel_size: 'Kernel', p: 'Dropout',
    start_dim: 'Start dim', end_dim: 'End dim', dim: 'Axis', iterationVariable: 'Index',
    split_size_or_sections: 'Sections', padding_idx: 'Padding index',
    requires_grad: 'Trainable', keepdim: 'Keep axis', enabled: 'Enabled',
    bias: 'Bias',
    H: 'Heads', D: 'Width', d_head: 'Head width', n_heads: 'Heads', dropout: 'Dropout',
    shape: 'Shape', factor: 'Factor', value: 'Value', exponent: 'Exponent', chunks: 'Parts', name: 'Name',
};

// Width follows the field schema, never the number of characters being typed.
function inspectorWidth(node, template) {
    const parameters = node.kind === 'layer' ? getEntry(node.libraryKey)?.parameters ?? [] : template?.freeParameters ?? [];
    return parameters.some((parameter) => parameter.type === 'tuple' && (parameter.name === 'shape' || parameter.editor === 'dimensions')) ? 328 : 248;
}

function InspectorField({ param, value, label, fullRow = false, ...props }) {
    const wide = ['tuple', 'string'].includes(param.type) || param.name === 'factor';
    return (
        <div className={`df-inspector-field${wide ? ' is-wide' : ''}${fullRow ? ' is-alone' : ''}`} data-parameter={param.name}>
            <ParameterBinding param={param} value={value} onChange={props.onChange}><ParameterField
                param={{ ...param, description: `${param.name}${param.description ? ` — ${param.description}` : ''}` }}
                value={value}
                label={label ?? FIELD_LABELS[param.name] ?? param.name.replaceAll('_', ' ')}
                ariaLabel={param.name}
                compactBoolean={param.name === 'bias'}
                {...props}
            /></ParameterBinding>
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
            fullRow={params.length === 1}
            value={values[p.name]}
            onChange={(v) => onPatch({ [p.name]: v })}
            errors={(fieldErrors[p.name] ?? []).map((message) => compactParameterError(p.name, message))}
        />
    ))}</div>;
    const inlineBias = params.length === 1 && params[0].name === 'bias';
    return collapsed && !inlineBias ? (
        <details ref={detailsRef} className="df-inspector-group df-inspector-options">
            <summary>Options</summary>
            {fields}
        </details>
    ) : inlineBias && collapsed ? <section className="df-inspector-group">{fields}</section> : fields;
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
                        <select aria-label={family || isOperationKey(node.libraryKey) ? "Operation type" : "Module"} title={selectedModule?.name} value={node.libraryKey} onChange={(e) => switchModule(e.target.value)}>
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
                {freeParams.filter((p) => !p.derived).map((p) => (
                    <InspectorField
                        key={p.name}
                        param={p}
                        fullRow={freeParams.filter((parameter) => !parameter.derived).length === 1}
                        label={FIELD_LABELS[p.name] ?? p.name.replaceAll('_', ' ')}
                        value={values[p.name]}
                        placeholder={p.default != null ? String(p.default) : 'Default'}
                        onChange={(v) => onChange(p.name, v)}
                    />
                ))}
            </div>
            {freeParams.some((p) => p.derived) && <div className="df-derived-parameters">{freeParams.filter((p) => p.derived).map((p) => <DerivedParameter key={p.name} param={p} />)}</div>}
            {hint && <p className="df-inspector-note">{hint}</p>}
        </section>
    );
}

function SupernodeBody({ id, node }) {
    const template = useComponentTemplate(node);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const values = node.parameterValues ?? {};
    return (
        <>
            {!template && <p className="df-inspector-note">Template missing — re-create or delete this node.</p>}
            <Bindings
                freeParams={template?.freeParameters ?? []}
                values={values}
                onChange={(name, v) => patchNode(id, { parameterValues: { ...values, [name]: v } })}
            />
        </>
    );
}

function RepeatBody({ id, node }) {
    const template = useComponentTemplate(node);
    const patchRepeatNode = useArchitectureStore((s) => s.patchRepeatNode);
    const patchRepeatBinding = useArchitectureStore((s) => s.patchRepeatBinding);
    const context = useValidationContext();
    const report = useValidation();
    const { count, error } = resolveComponentContext(node, template, context.graph, report.shapes, context.scope);
    const mode = node.mode ?? 'sequential';
    const iterVar = node.iterationVariable || 'i';
    return (
        <>
            {!template && <p className="df-inspector-note">Template missing — re-create or delete this node.</p>}
            <section className="df-inspector-group">
                <p className="df-inspector-note">{mode === 'parallel' ? 'One input slice per instance. Output is a tensor collection: connect Concat, Add or Stack to combine it.' : 'Each block feeds the next. Output from the last block.'}</p>
                {node.count === 'auto' ? <div className="df-repeat-auto">
                    <div className="df-derived-parameter"><span>Instances</span><strong>{count ?? 'Unresolved'}</strong><small>Auto · from input collections</small></div>
                    <button type="button" onClick={() => patchRepeatNode(id, { count: count ?? 2 })}>Set manually</button>
                    {error && <p className="df-param-errors">{error}</p>}
                </div> : <><div className="df-inspector-fields">
                    <InspectorField
                        param={{ name: 'count', type: 'int', description: 'Number of instances, or a template parameter such as $H; each instance has its own weights' }}
                        label={mode === 'parallel' ? 'Instances' : 'Steps'}
                        fullRow
                        value={node.count}
                        onChange={(v) => patchRepeatNode(id, { count: v })}
                        errors={error ? [error] : []}
                    />
                </div>
                {mode === 'parallel' && <button type="button" className="df-repeat-infer" onClick={() => patchRepeatNode(id, { count: 'auto' })}>Use input count</button>}</>}
            </section>
            <Bindings
                freeParams={template?.freeParameters ?? []}
                values={node.bindings ?? {}}
                onChange={(name, v) => patchRepeatBinding(id, name, v)}
            />
            <details className="df-inspector-group df-inspector-options">
                <summary>Parameter formulas</summary>
                <div className="df-inspector-fields">
                    <InspectorField
                        param={{ name: 'iterationVariable', type: 'string', description: 'Zero-based index available in parameter formulas; instance 1 uses index 0' }}
                        label="Index variable"
                        value={iterVar}
                        onChange={(v) => patchRepeatNode(id, { iterationVariable: (v || 'i').trim() || 'i' })}
                    />
                </div>
                <p className="df-inspector-note"><code>{iterVar} = 0…{count ? count - 1 : '?'}</code>. For example, <code>64 * 2 ** {iterVar}</code>.</p>
            </details>
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
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const templates = useTemplatesStore((s) => s.templates);
    const expandedNodeId = useArchitectureStore((s) => s.expandedNodeId);
    const clearExpanded = useArchitectureStore((s) => s.clearExpanded);
    const patchNode = useArchitectureStore((s) => s.patchNode);
    const patchRepeatNode = useArchitectureStore((s) => s.patchRepeatNode);
    const validation = useValidation();
    const panelRef = useInspectorPlacement(expandedNodeId);

    const graph = resolveGraphLevel(rootGraph, currentPath, templates, inspectedIterations).graph;
    const node = expandedNodeId ? graph?.nodes?.[expandedNodeId] : null;
    const componentTemplate = useComponentTemplate(node);
    if (!node) return null;

    const id = node.id;
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const issueParameters = node.kind === 'layer' ? getEntry(node.libraryKey)?.parameters ?? []
        : node.kind === 'repeat' ? [{ name: 'count' }] : [];
    const { fields: fieldErrors, remaining: nodeErrors } = partitionParameterErrors(issueParameters, errors);
    const ContainerIcon = node.kind === 'repeat' ? Repeat : Boxes;
    const standard = STANDARD_COMPONENTS.find((component) => component.key === (componentTemplate?.standardKey ?? componentTemplate?.componentStyleKey));
    const operation = node.kind === 'layer' && isOperationKey(node.libraryKey) && getEntry(node.libraryKey)?.category !== 'Parameters';
    const rename = (label) => (node.kind === 'repeat' ? patchRepeatNode(id, { label }) : patchNode(id, { label }));

    return (
        <aside ref={panelRef} className="df-inspector df-panel-subtle nodrag nopan nowheel" aria-label={`${node.label} inspector`} style={{ '--df-inspector-width': `${inspectorWidth(node, componentTemplate)}px` }} onKeyDown={(e) => e.stopPropagation()}>
            <header className="df-inspector-header">
                {node.kind === 'layer'
                    ? <OperationIcon libraryKey={node.libraryKey} size={24} />
                    : standard ? <OperationIcon libraryKey={standard.libraryKey} size={24} /> : <ContainerIcon size={22} strokeWidth={2.5} aria-hidden="true" />}
                {operation ? <strong className="df-inspector-title" title={node.label}>Operation</strong> : <input aria-label="Block name" title="Block name" value={node.label} onChange={(e) => rename(e.target.value)} />}
                <button type="button" className="df-inspector-close" onClick={clearExpanded} aria-label="Close inspector"><X size={13} strokeWidth={3} /></button>
            </header>
            <ParameterBindingsProvider key={node.id} node={node}><div className="df-inspector-body">
                {node.kind === 'layer' && <LayerBody id={id} node={node} fieldErrors={fieldErrors} />}
                {node.kind === 'supernode' && <SupernodeBody id={id} node={node} />}
                {node.kind === 'repeat' && <RepeatBody id={id} node={node} />}
                <Issues errors={nodeErrors} warnings={warnings} />
                <ExposeParameter />
            </div></ParameterBindingsProvider>
        </aside>
    );
}
