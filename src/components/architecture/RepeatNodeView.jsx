import { memo, useId } from 'react';
import { ArrowUpRight, Repeat, RotateCcw } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useValidation } from '../../hooks/useValidation.js';
import ParameterField from './ParameterField.jsx';
import NodeFrame from './NodeFrame.jsx';

const RepeatNodeView = ({ id, data, selected }) => {
    const node = data.node;
    const template = useTemplatesStore((state) => state.templates[node.templateId]);
    const removeNode = useArchitectureStore((state) => state.removeNode);
    const enterSupernode = useArchitectureStore((state) => state.enterSupernode);
    const patchRepeatNode = useArchitectureStore((state) => state.patchRepeatNode);
    const patchRepeatBinding = useArchitectureStore((state) => state.patchRepeatBinding);
    const convertRepeatToInstance = useArchitectureStore((state) => state.convertRepeatToInstance);
    const expandedNodeId = useArchitectureStore((state) => state.expandedNodeId);
    const toggleExpanded = useArchitectureStore((state) => state.toggleExpanded);
    const validation = useValidation();
    const countId = useId();
    const free = template?.freeParameters ?? [];
    const iterVar = node.iterationVariable || 'i';
    const errors = validation.errors[id] ?? [];
    const summaryParams = free.filter((param) => node.bindings?.[param.name] != null).slice(0, 2);

    return (
        <NodeFrame
            label={node.label}
            category="Repeat"
            icon={Repeat}
            badge={`× ${node.count}`}
            repeated
            selected={selected}
            expanded={expandedNodeId === id}
            onToggle={() => toggleExpanded(id)}
            onRename={(label) => patchRepeatNode(id, { label })}
            onRemove={() => removeNode(id)}
            onOpen={template ? () => enterSupernode(id) : undefined}
            subtitle={template ? `${template.name} · ${free.length} parameters` : 'Template missing'}
            errors={errors}
            warnings={validation.warnings[id] ?? []}
            outputShape={validation.shapes[id]}
            summary={summaryParams.length ? (
                <div className="df-node-summary">
                    {summaryParams.map((param) => (
                        <div key={param.name} className="df-node-summary-row">
                            <span>${param.name}</span><code title={String(node.bindings[param.name])}>{String(node.bindings[param.name])}</code>
                        </div>
                    ))}
                </div>
            ) : null}
        >
            <div className="df-node-actions">
                <button type="button" disabled={!template} className="df-node-action is-primary" onClick={() => enterSupernode(id)}>
                    <ArrowUpRight size={14} /> Open template
                </button>
                <button type="button" className="df-node-action" onClick={() => convertRepeatToInstance(id)} title="Convert back to a single template instance">
                    <RotateCcw size={14} /> Make single
                </button>
            </div>
            <section className="df-param-group">
                <h4 className="df-node-section-title">Repetition</h4>
                <div className="df-param-fields">
                    <div className="df-param-row">
                        <label htmlFor={countId} className="df-param-label">Iterations</label>
                        <input
                            id={countId} type="number" min="1" step="1" value={node.count} className="nodrag df-param-input"
                            onChange={(event) => patchRepeatNode(id, { count: Math.max(1, Number.parseInt(event.target.value, 10) || 1) })}
                        />
                    </div>
                    <ParameterField param={{ name: 'iterationVariable', type: 'string', description: 'Variable used in per-iteration expressions.' }}
                        label="Index variable" value={iterVar} onChange={(value) => patchRepeatNode(id, { iterationVariable: value || 'i' })} />
                </div>
                <p className="df-node-hint">Index <code>{iterVar}</code> runs from 0 to {Math.max(0, Number(node.count) - 1)}.</p>
            </section>
            {free.length ? (
                <section className="df-param-group">
                    <h4 className="df-node-section-title">Parameter expressions<span>{free.length}</span></h4>
                    <div className="df-param-fields">
                        {free.map((param) => (
                            <ParameterField key={param.name} param={{ ...param, type: 'string' }} label={`$${param.name}`}
                                value={node.bindings?.[param.name]} placeholder={`64 * 2 ** ${iterVar}`}
                                error={errors.some((message) => message.includes(param.name))}
                                onChange={(value) => patchRepeatBinding(id, param.name, value)} />
                        ))}
                    </div>
                    <p className="df-node-hint">Use a fixed value or an expression with <code>{iterVar}</code> to vary each iteration.</p>
                </section>
            ) : <p className="df-node-hint">Open the template to expose parameters for each iteration.</p>}
        </NodeFrame>
    );
};

export default memo(RepeatNodeView);
