import { memo, useId } from 'react';
import { ArrowDownToLine, ArrowUpFromLine, Box, Braces, GitMerge, Layers, Maximize, SlidersHorizontal, Sparkles, Waves } from 'lucide-react';
import { PYTORCH_BLOCKS, defaultParameterValues, getEntry } from '../../domain/library/index.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import { styleFor } from './categoryStyle.js';
import ParameterField from './ParameterField.jsx';
import NodeFrame from './NodeFrame.jsx';
import './nodes.css';

const SHAPE_PRIORITY = [
    'in_channels', 'out_channels', 'in_features', 'out_features',
    'embed_dim', 'num_heads', 'd_model', 'nhead', 'hidden_size', 'input_size',
    'num_features', 'kernel_size', 'p', 'shape', 'name',
];
const TEMPLATE_ONLY = new Set(['deepflow.template.Input', 'deepflow.template.Output']);
const INPUTS = new Set(['deepflow.io.Input', 'deepflow.template.Input']);
const OUTPUTS = new Set(['deepflow.io.Output', 'deepflow.template.Output']);
const CATEGORY_ICONS = {
    Convolution: Layers, Pooling: Maximize, Linear: Box, Reshape: Braces,
    Attention: Sparkles, Activation: Waves, Normalization: SlidersHorizontal,
    Combine: GitMerge, 'Element-wise': GitMerge,
};

function summaryFields(entry, parameters) {
    if (!entry) return [];
    const ordered = [...entry.parameters].sort((a, b) => {
        const ai = SHAPE_PRIORITY.indexOf(a.name);
        const bi = SHAPE_PRIORITY.indexOf(b.name);
        return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
    });
    // Show missing required values so an unfinished block is visible at a glance.
    return ordered.filter((param) => param.required || parameters[param.name] != null)
        .slice(0, 2)
        .map((param) => ({ name: param.name, value: parameters[param.name] }));
}

function ParamGroup({ title, params, parameters, errors, onPatch, collapsed = false }) {
    if (!params.length) return null;
    const fields = params.map((param) => (
        <ParameterField
            key={param.name}
            param={param}
            value={parameters[param.name]}
            onChange={(value) => onPatch({ [param.name]: value })}
            error={errors.some((message) => message.toLowerCase().includes(param.name.toLowerCase()))}
        />
    ));
    if (collapsed) {
        return (
            <details className="df-param-group df-param-options">
                <summary>{title}<span>{params.length}</span></summary>
                <div className="df-param-fields">{fields}</div>
            </details>
        );
    }
    return (
        <section className="df-param-group">
            <h4 className="df-node-section-title">{title}<span>{params.length}</span></h4>
            <div className="df-param-fields">{fields}</div>
        </section>
    );
}

const LayerNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const entry = getEntry(node.libraryKey);
    const removeNode = useArchitectureStore((state) => state.removeNode);
    const patchNode = useArchitectureStore((state) => state.patchNode);
    const patchLayerParameters = useArchitectureStore((state) => state.patchLayerParameters);
    const expandedNodeId = useArchitectureStore((state) => state.expandedNodeId);
    const toggleExpanded = useArchitectureStore((state) => state.toggleExpanded);
    const validation = useValidation();
    const errors = validation.errors[id] ?? [];
    const warnings = validation.warnings[id] ?? [];
    const style = styleFor(entry?.category ?? 'IO');
    const moduleId = useId();
    const showExpanded = expandedNodeId === id;
    const moduleOptions = entry && entry.category !== 'IO'
        ? PYTORCH_BLOCKS.filter((candidate) => candidate.category === entry.category && !TEMPLATE_ONLY.has(candidate.key))
        : [];
    const params = entry?.parameters ?? [];
    const requiredParams = params.filter((param) => param.required);
    const shapeParams = params.filter((param) => !param.required && SHAPE_PRIORITY.includes(param.name));
    const optionParams = params.filter((param) => !param.required && !SHAPE_PRIORITY.includes(param.name));
    const summary = summaryFields(entry, node.parameters);
    const Icon = INPUTS.has(node.libraryKey) ? ArrowDownToLine : OUTPUTS.has(node.libraryKey) ? ArrowUpFromLine : CATEGORY_ICONS[entry?.category] ?? Box;

    const switchModule = (libraryKey) => {
        const nextEntry = getEntry(libraryKey);
        if (!nextEntry || nextEntry.key === node.libraryKey) return;
        const nextParameters = defaultParameterValues(nextEntry);
        for (const param of nextEntry.parameters) {
            if (Object.hasOwn(node.parameters, param.name)) nextParameters[param.name] = node.parameters[param.name];
        }
        patchNode(id, {
            libraryKey: nextEntry.key,
            label: node.label === entry?.name ? nextEntry.name : node.label,
            parameters: nextParameters,
        });
    };

    return (
        <NodeFrame
            label={node.label}
            category={entry?.category === 'IO' ? 'Input / output' : entry?.category ?? 'Unknown block'}
            icon={Icon}
            accent={style.accent}
            tint={style.tint}
            selected={selected}
            expanded={showExpanded}
            onToggle={() => toggleExpanded(id)}
            onRename={(label) => patchNode(id, { label })}
            onRemove={() => removeNode(id)}
            errors={errors}
            warnings={warnings}
            outputShape={validation.shapes[id]}
            hasInput={!INPUTS.has(node.libraryKey)}
            hasOutput={!OUTPUTS.has(node.libraryKey)}
            summary={summary.length > 0 ? (
                <div className="df-node-summary">
                    {summary.map((field) => (
                        <div key={field.name} className="df-node-summary-row">
                            <span>{field.name}</span>
                            <code className={field.value == null ? 'is-unset' : ''} title={String(field.value ?? 'Not set')}>
                                {field.value == null ? 'Not set' : String(field.value)}
                            </code>
                        </div>
                    ))}
                </div>
            ) : <p className="df-node-empty-summary">{entry?.description ?? node.libraryKey}</p>}
        >
            {entry?.description && <p className="df-node-description">{entry.description}</p>}
            {moduleOptions.length > 1 && (
                <div className="df-param-row df-node-module">
                    <label className="df-param-label" htmlFor={moduleId}>Module</label>
                    <select id={moduleId} value={node.libraryKey} onChange={(event) => switchModule(event.target.value)} className="df-param-input nodrag">
                        {moduleOptions.map((option) => <option key={option.key} value={option.key}>{option.name}</option>)}
                    </select>
                </div>
            )}
            {params.length ? (
                <>
                    <ParamGroup title="Required" params={requiredParams} parameters={node.parameters} errors={errors} onPatch={(patch) => patchLayerParameters(id, patch)} />
                    <ParamGroup title={requiredParams.length ? 'Dimensions' : 'Parameters'} params={shapeParams} parameters={node.parameters} errors={errors} onPatch={(patch) => patchLayerParameters(id, patch)} />
                    <ParamGroup title="Options" params={optionParams} parameters={node.parameters} errors={errors} onPatch={(patch) => patchLayerParameters(id, patch)} collapsed={requiredParams.length > 0 || shapeParams.length > 0} />
                </>
            ) : <p className="df-node-hint">This block has no parameters.</p>}
        </NodeFrame>
    );
};

export default memo(LayerNodeView);
