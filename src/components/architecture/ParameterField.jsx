import React from 'react';

const ParameterField = ({ param, value, onChange, error }) => {
    const id = `pf-${param.name}`;
    const display = value === null || value === undefined ? '' : value;
    const labelClass = `text-[10px] font-mono whitespace-nowrap ${error ? 'text-red-700' : 'text-text-secondary'}`;
    const inputBase = `border ${error ? 'border-red-600' : 'border-black/35'} rounded-md px-1.5 py-0.5 text-[11px] bg-white/95 font-mono focus:outline-none focus:ring-1 focus:ring-black/25 focus:border-black`;

    if (param.type === 'bool') {
        return (
            <label className="inline-flex items-center gap-1.5 h-7 px-1.5 rounded-md hover:bg-white/45 cursor-pointer">
                <input
                    id={id}
                    type="checkbox"
                    checked={!!value}
                    onChange={(e) => onChange(e.target.checked)}
                    onClick={(e) => e.stopPropagation()}
                    className="w-3.5 h-3.5 border-2 border-black accent-black nodrag"
                />
                <span className={labelClass} title={param.description}>{param.name}</span>
            </label>
        );
    }

    if (param.type === 'enum') {
        return (
            <label className="inline-flex items-center gap-1.5 h-7 px-1.5 rounded-md hover:bg-white/45">
                <span className={labelClass} title={param.description}>{param.name}</span>
                <select
                    id={id}
                    value={display ?? ''}
                    onChange={(e) => onChange(e.target.value || null)}
                    onClick={(e) => e.stopPropagation()}
                    className={`${inputBase} nodrag`}
                    style={{ maxWidth: '12ch' }}
                >
                    <option value="">default</option>
                    {param.options?.map((opt) => (
                        <option key={opt} value={opt}>{opt}</option>
                    ))}
                </select>
            </label>
        );
    }

    const widthCh =
        param.type === 'int'    ? 5
        : param.type === 'float' ? 7
        : param.type === 'tuple' ? 8
        : 12;

    return (
        <label className="inline-flex items-center gap-1.5 h-7 px-1.5 rounded-md hover:bg-white/45">
            <span className={labelClass} title={param.description}>
                {param.name}
                {param.required && <span className="text-pink-600">*</span>}
            </span>
            <input
                id={id}
                type={param.type === 'int' || param.type === 'float' ? 'number' : 'text'}
                step={param.type === 'float' ? 'any' : undefined}
                value={display}
                placeholder={param.default !== undefined && param.default !== null ? String(param.default) : ''}
                onChange={(e) => {
                    const raw = e.target.value;
                    if (raw === '') return onChange(null);
                    if (param.type === 'int') return onChange(Number.parseInt(raw, 10));
                    if (param.type === 'float') return onChange(Number.parseFloat(raw));
                    onChange(raw);
                }}
                onClick={(e) => e.stopPropagation()}
                className={`${inputBase} nodrag text-right`}
                style={{ width: `${widthCh}ch` }}
            />
        </label>
    );
};

export default ParameterField;
