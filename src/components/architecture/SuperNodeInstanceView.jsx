import { memo } from 'react';
import { Boxes, ArrowUpRight, Repeat } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import ParameterField from './ParameterField.jsx';
import NodeFrame from './NodeFrame.jsx';

const SuperNodeInstanceView = ({ id, data, selected }) => {
    const node = data.node;
    const template = useTemplatesStore((state) => state.templates[node.templateId]);
    const removeNode = useArchitectureStore((state) => state.removeNode);
    const patchNode = useArchitectureStore((state) => state.patchNode);
    const enterSupernode = useArchitectureStore((state) => state.enterSupernode);
    const convertInstanceToRepeat = useArchitectureStore((state) => state.convertInstanceToRepeat);
    const expandedNodeId = useArchitectureStore((state) => state.expandedNodeId);
    const toggleExpanded = useArchitectureStore((state) => state.toggleExpanded);
    const validation = useValidation();
    const innerCount = template ? Object.keys(template.graph.nodes).length : 0;
    const bindings = node.parameterValues ?? {};
    const freeParams = template?.freeParameters ?? [];
    const summaryParams = freeParams.filter((param) => bindings[param.name] != null).slice(0, 2);
    const errors = validation.errors[id] ?? [];

    return (
        <NodeFrame
            label={node.label}
            category="Template"
            icon={Boxes}
            selected={selected}
            expanded={expandedNodeId === id}
            onToggle={() => toggleExpanded(id)}
            onRename={(label) => patchNode(id, { label })}
            onRemove={() => removeNode(id)}
            onOpen={template ? () => enterSupernode(id) : undefined}
            subtitle={template ? `${innerCount} blocks · ${freeParams.length} parameters` : 'Template missing'}
            errors={errors}
            warnings={validation.warnings[id] ?? []}
            outputShape={validation.shapes[id]}
            summary={summaryParams.length ? (
                <div className="df-node-summary">
                    {summaryParams.map((param) => (
                        <div key={param.name} className="df-node-summary-row">
                            <span>${param.name}</span><code title={String(bindings[param.name])}>{String(bindings[param.name])}</code>
                        </div>
                    ))}
                </div>
            ) : null}
        >
            <p className="df-node-description">{template?.name ?? 'Missing template'}<span className="df-node-hint-line">Values below apply to this instance.</span></p>
            <div className="df-node-actions">
                <button type="button" disabled={!template} className="df-node-action is-primary" onClick={() => enterSupernode(id)}>
                    <ArrowUpRight size={14} /> Open template
                </button>
                <button type="button" disabled={!template} className="df-node-action" onClick={() => convertInstanceToRepeat(id, 2)} title="Repeat this template multiple times">
                    <Repeat size={14} /> Repeat
                </button>
            </div>
            {freeParams.length ? (
                <section className="df-param-group">
                    <h4 className="df-node-section-title">Instance parameters<span>{freeParams.length}</span></h4>
                    <div className="df-param-fields">
                        {freeParams.map((param) => (
                            <ParameterField key={param.name} param={param} label={`$${param.name}`} value={bindings[param.name]}
                                error={errors.some((message) => message.includes(param.name))}
                                onChange={(value) => patchNode(id, { parameterValues: { ...bindings, [param.name]: value } })} />
                        ))}
                    </div>
                </section>
            ) : <p className="df-node-hint">Open the template to expose reusable parameters.</p>}
        </NodeFrame>
    );
};

export default memo(SuperNodeInstanceView);
