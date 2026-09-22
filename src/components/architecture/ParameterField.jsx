import { useId } from 'react';
import './parameter-shape.css';

const NUMBER_LITERAL = /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$/;

function shapeDimensions(value) {
    if (Array.isArray(value)) return value;
    if (value === null || value === undefined || value === '') return [''];
    const text = String(value).trim();
    // Preserve arithmetic parentheses within each dimension. Legacy tuple
    // strings have a single outer pair around the comma-separated dimensions.
    const wrapped = (text.startsWith('(') && text.endsWith(')')) || (text.startsWith('[') && text.endsWith(']'));
    let depth = 0;
    const wholeWrapper = wrapped && ![...text.slice(0, -1)].some((character) => {
        if (character === '(' || character === '[') depth++;
        if (character === ')' || character === ']') depth--;
        return depth === 0;
    });
    const tuple = wholeWrapper ? text.slice(1, -1) : text;
    return tuple.split(',').map((part) => part.trim()).filter((part, index, parts) => part !== '' || index !== parts.length - 1);
}

function ShapeField({ value, onChange, common, className, name }) {
    const dimensions = shapeDimensions(value);
    const update = (index, next) => onChange(dimensions.map((dimension, at) => at === index ? next : dimension));
    return <div className="df-param-shape" role="group" aria-label={`${name} dimensions`}>
        {dimensions.map((dimension, index) => <span className="df-shape-dimension" key={index}>
            {index > 0 && <span className="df-shape-times" aria-hidden="true">×</span>}
            <span className="df-shape-cell">
                <input {...common} id={index === 0 ? common.id : `${common.id}-${index}`}
                    aria-label={`${name} dimension ${index + 1}`} className={`${className}${typeof dimension === 'string' && dimension !== '' && !NUMBER_LITERAL.test(dimension) ? ' is-expression' : ''}`}
                    style={{ '--shape-dimension-width': `${Math.min(19, Math.max(3, String(dimension).length + 1))}ch` }}
                    type="text" spellCheck={false} autoComplete="off" value={dimension ?? ''} placeholder="Size"
                    title={`Dimension ${index + 1}. Accepts a number, parameter reference, or expression.`}
                    onChange={(event) => update(index, event.target.value)}
                    onBlur={(event) => {
                        const raw = event.currentTarget.value;
                        if (NUMBER_LITERAL.test(raw) && Number.isFinite(Number(raw)) && dimension !== Number(raw)) update(index, Number(raw));
                    }} />
                <button type="button" className="nodrag df-shape-remove" aria-label={`Remove ${name} dimension ${index + 1}`}
                    disabled={dimensions.length < 2} title="Remove dimension"
                    onMouseDown={(event) => event.stopPropagation()}
                    onClick={(event) => { event.stopPropagation(); onChange(dimensions.filter((_, at) => at !== index)); }}>×</button>
            </span>
        </span>)}
        <button type="button" className="nodrag df-shape-add" aria-label={`Add ${name} dimension`} title="Add dimension"
            onMouseDown={(event) => event.stopPropagation()}
            onClick={(event) => { event.stopPropagation(); onChange([...dimensions, 1]); }}>+</button>
    </div>;
}

const ParameterField = ({ param, value, onChange, error, errors = [], label, ariaLabel, placeholder, compactBoolean = false }) => {
    const id = useId();
    const hasError = !!error || errors.length > 0;
    const errorId = `${id}-errors`;
    const display = value === null || value === undefined ? '' : value;
    const numeric = param.type === 'int' || param.type === 'float';
    const booleanExpression = param.type === 'bool' && typeof value === 'string' && !['true', 'false', ''].includes(value);
    const expression = booleanExpression || (numeric && typeof value === 'string' && value !== '' && !NUMBER_LITERAL.test(value));
    const booleanValue = value ?? param.default ?? false;
    const checked = booleanValue === true || booleanValue === 'true';
    const inlineBoolean = compactBoolean && param.type === 'bool' && !booleanExpression;
    const textValue = Array.isArray(display) ? display.join(', ') : String(display);
    const className = `df-param-input nodrag${hasError ? ' is-invalid' : ''}${expression ? ' is-expression' : ''}`;
    const baseWidth = param.type === 'int' ? 7 : param.type === 'float' ? 8 : param.type === 'tuple' ? 9 : 12;
    const widthCh = numeric && typeof display === 'string' ? Math.min(16, Math.max(baseWidth, display.length + 1)) : baseWidth;
    const common = {
        style: { '--parameter-width': `${widthCh}ch` },
        id,
        'aria-invalid': hasError,
        'aria-describedby': errors.length ? errorId : undefined,
        'aria-label': ariaLabel,
        className,
        onClick: (event) => event.stopPropagation(),
        onMouseDown: (event) => event.stopPropagation(),
        onKeyDown: (event) => event.stopPropagation(),
    };

    return (
        <div className={`df-param-row${hasError ? ' has-error' : ''}${inlineBoolean ? ' df-param-row--toggle' : ''}`}>
            <label htmlFor={id} title={param.description} className="df-param-label">
                {label ?? param.name}
                {param.required && <span className="df-param-required" title="Required">*</span>}
            </label>
            {param.name === 'shape' && param.type === 'tuple' ? (
                <ShapeField value={value ?? param.default} onChange={onChange} common={common} className={className} name={ariaLabel ?? param.name} />
            ) : inlineBoolean ? (
                <input {...common} className="df-param-checkbox nodrag" type="checkbox" checked={checked}
                    onChange={(event) => onChange(event.target.checked)} />
            ) : param.type === 'bool' && !booleanExpression ? (
                <label className="df-param-bool" htmlFor={id}>
                    <input
                        {...common}
                        className="df-param-checkbox nodrag"
                        type="checkbox"
                        checked={checked}
                        onChange={(event) => onChange(event.target.checked)}
                    />
                    <span>{value === null || value === undefined ? 'Default' : checked ? 'True' : 'False'}</span>
                </label>
            ) : param.type === 'enum' ? (
                <select {...common} value={display} onChange={(event) => onChange(event.target.value || null)}>
                    <option value="">Default</option>
                    {param.options?.map((option) => <option key={option} value={option}>{option}</option>)}
                </select>
            ) : param.type === 'tuple' && textValue.length > 22 ? (
                <textarea {...common} rows={2} className={`${className} df-param-tuple`} value={textValue}
                    spellCheck={false} placeholder={placeholder ?? 'Dimensions'}
                    onChange={(event) => onChange(event.target.value || null)} />
            ) : (
                <input
                    {...common}
                    type="text"
                    spellCheck={false}
                    autoComplete="off"
                    value={Array.isArray(display) ? display.join(', ') : display}
                    placeholder={placeholder ?? (param.default != null ? String(param.default) : param.required ? 'Required' : 'Default')}
                    title={numeric ? `${param.description ?? param.name}. Accepts a number or a template expression, e.g. $width.` : param.description}
                    onChange={(event) => {
                        // Keep the exact text while editing: normalizing `1.0` here
                        // would discard the decimal before the user can type `1.05`.
                        const raw = event.target.value;
                        onChange(raw === '' ? null : raw);
                    }}
                    onBlur={(event) => {
                        const raw = event.currentTarget.value;
                        if (!numeric || !NUMBER_LITERAL.test(raw)) return;
                        const number = Number(raw);
                        if (Number.isFinite(number) && value !== number) onChange(number);
                    }}
                />
            )}
            {errors.length > 0 && <div id={errorId} className="df-param-errors">{errors.map((message, index) => <span key={index}>{message}</span>)}</div>}
        </div>
    );
};

export default ParameterField;
