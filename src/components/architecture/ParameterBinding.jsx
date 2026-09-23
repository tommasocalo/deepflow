import { useComponentTemplate } from '../../hooks/useComponentTemplate.js';
import { cloneElement, createContext, useContext } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { resolveGraphLevel } from '../../domain/architecture/index.js';
import { resolveLevelContext, useValidationContext } from '../../hooks/useValidation.js';
import { asFreeParamRef } from '../../domain/expressions.js';
import { exactParameterValue, isApproximateDisplay } from '../../domain/parameterDisplay.js';
import { availableParameters, exposedFieldParameter, fieldAlias, formatParameterValue, parameterFrames, resolvedFieldValue } from '../../domain/parameterBindings.js';
import './parameter-binding.css';

const BindingContext = createContext(null);

export function ParameterBindingsProvider({ node, children }) {
    const graph = useArchitectureStore((s) => s.graph);
    const path = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const context = useValidationContext();
    const level = resolveGraphLevel(graph, path, templates, inspectedIterations);
    const template = useComponentTemplate(node);
    const child = template ? resolveLevelContext(graph, [...path, node.id], templates, inspectedIterations) : null;
    const component = child ? { scope: child.scope, inputs: child.externalInputs, count: child.repeat?.count ?? 1, index: child.repeat?.index ?? 0 } : null;
    const frames = parameterFrames(graph, path, templates, inspectedIterations);
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

/** References already identify the variable; the eye controls the parent UI.
 * A field shown through the eye keeps displaying its own value: edits write the
 * shared value behind the alias, never the alias name. */
export default function ParameterBinding({ param, value, children }) {
    const binding = useContext(BindingContext);
    const toggleFieldExposure = useTemplatesStore((s) => s.toggleFieldExposure);
    const patchFreeParameter = useTemplatesStore((s) => s.patchFreeParameter);
    if (!binding || !['int', 'float', 'bool', 'tuple'].includes(param.type) || param.derived) return children;
    const { node, owner, frames, context, component } = binding;
    const scope = param.name === 'count' ? context.scope : component?.scope ?? context.scope;
    const label = children.props.label ?? param.name.replaceAll('_', ' ');
    const alias = owner ? fieldAlias(owner, node.id, param.name) : null;
    const raw = alias ? alias.default : value === undefined ? param.default : value;
    const exposed = alias ?? (owner ? exposedFieldParameter(owner, node.id, param.name) : null);
    const shown = !!exposed && !exposed.hidden;
    // A field cannot reference the alias that stands for itself.
    const suggestions = alias ? binding.suggestions.filter((suggestion) => suggestion.name !== alias.name) : binding.suggestions;
    // An empty field that inherited its default stays visually empty (grey default).
    const inherits = alias?.exposes.inherited && alias.default === (param.default ?? null);
    const aliasProps = alias ? {
        value: inherits ? undefined : alias.default,
        onChange: (next) => patchFreeParameter(owner.id, alias.name, { default: next === null || next === '' ? param.default ?? null : next }),
    } : {};
    const parentLabel = frames.at(-1)?.instance.label ?? owner?.name;
    const VisibilityIcon = shown ? Eye : EyeOff;
    const action = owner && <button type="button" className="df-field-visibility nodrag"
        aria-label={`${label} in ${parentLabel} settings`} aria-pressed={shown}
        title={`${shown ? 'Hide' : 'Show'} ${label} in ${parentLabel} settings`}
        onMouseDown={(event) => event.stopPropagation()}
        onClick={(event) => {
            event.stopPropagation();
            toggleFieldExposure(owner.id, { nodeId: node.id, field: param.name, spec: param, label,
                defaultValue: raw, reservedNames: binding.suggestions.map((suggestion) => suggestion.name) });
        }}><VisibilityIcon size={12} aria-hidden="true" /></button>;
    const calculated = ['int', 'float'].includes(param.type) && typeof raw === 'string' && raw.trim()
        && !asFreeParamRef(raw.trim()) && !Number.isFinite(Number(raw));
    const resolved = calculated ? resolvedFieldValue(raw, scope) : null;
    const showResult = typeof resolved === 'number' && Number.isFinite(resolved);
    return <div className="df-field-binding">
        {cloneElement(children, { suggestions, labelAction: action, ...aliasProps })}
        {showResult && <div className="df-parameter-result">
            <span>Result: </span><strong title={`Exact value: ${exactParameterValue(resolved)}`}>{isApproximateDisplay(resolved) ? '≈ ' : ''}{formatParameterValue(resolved)}</strong>
        </div>}
    </div>;
}
