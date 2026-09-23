import { useComponentTemplate } from '../../hooks/useComponentTemplate.js';
import React from 'react';
import { AlertCircle, AlertTriangle, Boxes, ChevronDown, Repeat, X } from 'lucide-react';
import { PYTORCH_BLOCKS, getEntry } from '../../domain/library/index.js';
import { createVariantPatch, getNodeFamily, isOperationKey } from '../../domain/library/nodeFamilies.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation, useValidationContext } from '../../hooks/useValidation.js';
import { resolveComponentContext } from '../../domain/componentContext.js';
import { resolveParameterValue } from '../../domain/expressions.js';
import { STANDARD_COMPONENTS } from '../../domain/components/index.js';
import ParameterField from './ParameterField.jsx';
import ParameterBinding, { DerivedParameter, ParameterBindingsProvider } from './ParameterBinding.jsx';
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
    init: 'Distribution', scale: 'Scale', disabled_init: 'When disabled',
    elementwise_affine: 'Affine', affine: 'Affine', eps: 'Epsilon', momentum: 'Momentum', stride: 'Stride', padding: 'Padding',
    dilation: 'Dilation', groups: 'Groups', output_padding: 'Output padding', output_size: 'Output size', num_layers: 'Layers',
    bidirectional: 'Bidirectional', dim_feedforward: 'Feed-forward', norm_first: 'Norm first', activation: 'Activation',
    negative_slope: 'Slope', dtype: 'Data type', batch_first: 'Batch first', inplace: 'In place', ceil_mode: 'Ceil mode',
    count_include_pad: 'Count padding', approximate: 'Approximation',
};

// Implementation details rather than architecture: shown only when their
// effective value already differs from the default, so nothing set is hidden.
const ADVANCED_FIELDS = new Set(['requires_grad', 'enabled', 'padding_idx', 'inplace', 'batch_first', 'ceil_mode', 'count_include_pad', 'eps', 'momentum', 'approximate']);
const SCALE_FREE_INIT = new Set(['kaiming_uniform', 'zeros', 'ones']);

function effectiveValue(param, values, scope) {
    const raw = values?.[param.name];
    return raw === undefined || raw === null || raw === '' ? param.default ?? null : resolveParameterValue(raw, scope) ?? null;
}

/** Fields worth showing, architecture first and toggles last. */
function shownParameters(params, values, scope) {
    const differs = (param) => String(effectiveValue(param, values, scope)) !== String(param.default ?? null);
    const enabled = params.find((param) => param.name === 'enabled');
    const init = params.find((param) => param.name === 'init');
    const shown = params.filter((param) => {
        if (param.name === 'disabled_init') return !!enabled && differs(enabled);
        if (param.name === 'scale' && init && SCALE_FREE_INIT.has(effectiveValue(init, values, scope))) return false;
        return !ADVANCED_FIELDS.has(param.name) || differs(param);
    });
    const primary = (param) => param.required || SHAPE_PRIORITY.includes(param.name);
    const last = (param) => param.name === 'disabled_init';
    return [
        ...shown.filter((param) => param.type !== 'bool' && primary(param) && !last(param)),
        ...shown.filter((param) => param.type !== 'bool' && !primary(param) && !last(param)),
        ...shown.filter((param) => param.type === 'bool'),
        ...shown.filter(last),
    ];
}

function InspectorField({ param, value, label, ...props }) {
    // Only a tensor shape needs its own line; everything else sits beside its label.
    const wide = param.type === 'tuple' && (param.name === 'shape' || param.editor === 'dimensions');
    return (
        <div className={`df-inspector-field${wide ? ' is-wide' : ''}`} data-parameter={param.name}>
            <ParameterBinding param={param} value={value}><ParameterField
                param={{ ...param, description: `${param.name}${param.description ? ` — ${param.description}` : ''}` }}
                value={value}
                label={label ?? FIELD_LABELS[param.name] ?? param.name.replaceAll('_', ' ')}
                ariaLabel={param.name}
                compactBoolean={param.type === 'bool'}
                {...props}
            /></ParameterBinding>
        </div>
    );
}

function ParamGroup({ params, values, fieldErrors, onPatch, labels = {}, leading = null }) {
    if (!params.length && !leading) return null;
    return <div className="df-inspector-fields">{leading}{params.map((p) => (
        <InspectorField
            key={p.name}
            param={p}
            label={labels[p.name]}
            value={values[p.name]}
            onChange={(v) => onPatch({ [p.name]: v })}
            errors={(fieldErrors[p.name] ?? []).map((message) => compactParameterError(p.name, message))}
        />
    ))}</div>;
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
    const scope = useValidationContext()?.scope ?? {};
    const params = shownParameters(entry?.parameters ?? [], node.parameters, scope);
    const distribution = node.parameters.init ?? 'normal';
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
    const typeRow = moduleOptions.length > 1 && (
        <label className="df-param-row df-inspector-type">
            <span className="df-param-label-line"><span className="df-param-label">Type</span></span>
            <span className="df-param-control df-inspector-select">
                <select aria-label={family || isOperationKey(node.libraryKey) ? "Operation type" : "Module"} title={selectedModule?.name} value={node.libraryKey} onChange={(e) => switchModule(e.target.value)}>
                    {moduleOptions.map((o) => <option key={o.key} value={o.key}>{o.name}</option>)}
                </select>
                <ChevronDown size={11} strokeWidth={3} aria-hidden="true" />
            </span>
        </label>
    );
    return params.length || typeRow ? (
        <ParamGroup params={params} values={node.parameters} fieldErrors={fieldErrors} onPatch={onPatch} leading={typeRow}
            labels={{ scale: distribution === 'uniform' ? 'Bound (±)' : distribution === 'normal' ? 'Std dev' : 'Scale' }} />
    ) : <p className="df-inspector-note">No parameters.</p>;
}

function Bindings({ freeParams, values, onChange, hint, leading = null }) {
    const scope = useValidationContext()?.scope ?? {};
    if (!freeParams.length && !leading) return <p className="df-inspector-note">No configurable parameters.</p>;
    const shown = shownParameters(freeParams.filter((p) => !p.derived && !p.hidden), values, scope);
    return (
        <section className="df-inspector-group">
            <div className="df-inspector-fields">
                {leading}
                {shown.map((p) => (
                    <InspectorField
                        key={p.name}
                        param={p}
                        label={FIELD_LABELS[p.name] ?? p.name.replaceAll('_', ' ')}
                        value={values[p.name]}
                        placeholder={p.default != null ? String(p.default) : 'Default'}
                        onChange={(v) => onChange(p.name, v)}
                    />
                ))}
                {freeParams.filter((p) => p.derived).map((p) => <DerivedParameter key={p.name} param={p} />)}
            </div>
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
    const countLabel = mode === 'parallel' ? 'Instances' : 'Steps';
    const countRow = node.count === 'auto' ? <>
        <div className="df-param-row df-derived-row" data-parameter="count">
            <div className="df-param-label-line"><span className="df-param-label">{countLabel}</span></div>
            <span className="df-param-control" title="Counted from the collections this repeat receives">
                <strong>{count ?? '?'}</strong><small>from inputs</small>
                <button type="button" className="df-row-action" onClick={() => patchRepeatNode(id, { count: count ?? 2 })}>Set</button>
            </span>
        </div>
        {error && <p className="df-param-errors df-row-note">{error}</p>}
    </> : <>
        <InspectorField
            param={{ name: 'count', type: 'int', description: 'Number of instances, or a parameter such as $n_layers; each has its own weights. Formulas can use i: 0 for the first, 1 for the second…' }}
            label={countLabel}
            value={node.count}
            onChange={(v) => patchRepeatNode(id, { count: v })}
            errors={error ? [error] : []}
        />
        {mode === 'parallel' && <button type="button" className="df-row-action df-row-note" onClick={() => patchRepeatNode(id, { count: 'auto' })}>Count from inputs</button>}
    </>;
    return (
        <>
            {!template && <p className="df-inspector-note">Template missing — re-create or delete this node.</p>}
            <p className="df-inspector-note">{mode === 'parallel' ? 'Parallel · outputs form a collection' : 'Sequential · each step feeds the next'}</p>
            <Bindings
                freeParams={template?.freeParameters ?? []}
                values={node.bindings ?? {}}
                onChange={(name, v) => patchRepeatBinding(id, name, v)}
                leading={countRow}
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
        <aside ref={panelRef} className="df-inspector df-panel-subtle nodrag nopan nowheel" aria-label={`${node.label} inspector`} onKeyDown={(e) => e.stopPropagation()}>
            <header className="df-inspector-header">
                {node.kind === 'layer'
                    ? <OperationIcon libraryKey={node.libraryKey} size={24} />
                    : standard ? <OperationIcon libraryKey={standard.libraryKey} size={24} /> : <ContainerIcon size={22} strokeWidth={2.5} aria-hidden="true" />}
                {operation ? <strong className="df-inspector-title" title={node.label}>Operation</strong> : <input aria-label="Block name" title="Block name" size={Math.max(6, node.label.length + 1)} value={node.label} onChange={(e) => rename(e.target.value)} />}
                <button type="button" className="df-inspector-close" onClick={clearExpanded} aria-label="Close inspector"><X size={13} strokeWidth={3} /></button>
            </header>
            <ParameterBindingsProvider key={node.id} node={node}><div className="df-inspector-body">
                {node.kind === 'layer' && <LayerBody id={id} node={node} fieldErrors={fieldErrors} />}
                {node.kind === 'supernode' && <SupernodeBody id={id} node={node} />}
                {node.kind === 'repeat' && <RepeatBody id={id} node={node} />}
                <Issues errors={nodeErrors} warnings={warnings} />
            </div></ParameterBindingsProvider>
        </aside>
    );
}
