import { useId } from 'react';

const NUMBER_LITERAL = /^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?$/;

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
            {inlineBoolean ? (
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
