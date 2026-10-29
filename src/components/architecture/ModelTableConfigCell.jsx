import { useRef, useState } from 'react';
import VariableInput from './VariableInput.jsx';
import { parseCellValue, precheckCellValue } from '../../domain/modelTable/writeTargets.js';
import { emptyAllowed, fieldScope, initialText } from './modelTableEdits.js';

const widthOf = (text, placeholder) => `${Math.min(22, Math.max(4, String(text || placeholder || '').length + 1))}ch`;
const LONG_TEXT = 20;

function shownText(field, cells) {
    return cells === 'symbols' ? field.symbols : { text: field.values.text, tone: field.values.varies ? 'varies' : 'value' };
}

/** `p=0.1`, `k=3` like a PyTorch repr; symbols such as × and + attach directly. */
function Prefix({ prefix }) {
    if (!prefix) return null;
    return <span className="df-mt-prefix">{prefix}{/^[A-Za-z_]/.test(prefix) ? '=' : ''}</span>;
}

/** A field as text: expressions in the expression colour, inherited defaults muted. */
function FieldText({ field, cells, info, errors, onClick }) {
    const shown = shownText(field, cells);
    const error = errors?.[0];
    const title = [
        `${field.name}${cells === 'symbols' && field.values.text !== shown.text ? ` = ${field.values.text}` : ''}${cells === 'values' && field.symbols.text !== shown.text ? ` · written ${field.symbols.text}` : ''}`,
        shown.text.length > LONG_TEXT ? shown.text : null, field.values.title, error, info?.hint,
    ].filter(Boolean).join('\n');
    // Long formulas are cut here; the title keeps them whole.
    const text = shown.text.length > LONG_TEXT ? `${shown.text.slice(0, LONG_TEXT - 2)}…` : shown.text;
    return <span className={`df-mt-field is-${shown.tone}${info?.editable ? ' is-editable' : ''}${error ? ' is-invalid' : ''}`}
        data-field={field.name} title={title} onClick={onClick}>
        <Prefix prefix={field.prefix} />{text}
    </span>;
}

function Separator({ part }) {
    return part.kind === 'note'
        ? <span className="df-mt-note" title={part.title}>{part.text}</span>
        : <span className="df-mt-separator" aria-hidden="true">{part.text}</span>;
}

/** Shown while a cell is open: every editable field becomes an input. */
function ConfigEditor({ row, cells, targets, context, suggestions, focusField, seed, onDone, onMessage }) {
    const editable = row.config.filter((part) => part.kind === 'field' && targets[part.name]?.editable);
    const firstName = editable.some((part) => part.name === focusField) ? focusField : editable[0]?.name;
    const [drafts, setDrafts] = useState(() => Object.fromEntries(editable.map((part) => {
        const initial = initialText(targets[part.name], context);
        return [part.name, part.name === firstName && seed !== undefined && seed !== null ? seed : initial];
    })));
    const [errors, setErrors] = useState({});
    const [initial] = useState(() => Object.fromEntries(editable.map((part) => [part.name, initialText(targets[part.name], context)])));
    const done = useRef(false);
    const editorRef = useRef(null);
    // VariableInput keeps its own ref, so inputs are found by their field name.
    const focusInput = (name) => [...(editorRef.current?.querySelectorAll('[data-field-input]') ?? [])]
        .find((element) => element.dataset.fieldInput === name)?.focus();

    const finish = (edits, options) => {
        if (done.current) return;
        done.current = true;
        onDone(edits, options);
    };
    /** Returns the first problem, or null after handing the edits over. */
    const commit = (options) => {
        const edits = [];
        const found = {};
        for (const part of editable) {
            const text = drafts[part.name] ?? '';
            if (text === initial[part.name]) continue;
            const info = targets[part.name];
            const value = parseCellValue(text, part.spec);
            const error = precheckCellValue(value, { spec: part.spec, scope: fieldScope(row, part, context.templates),
                literalOnly: info.target.literalOnly, source: part.source, required: emptyAllowed(info) ? undefined : true });
            if (error) found[part.name] = error;
            else edits.push({ field: part, value, cells });
        }
        const invalid = editable.find((part) => found[part.name]);
        if (invalid) {
            setErrors(found);
            return `${invalid.name}: ${found[invalid.name]}`;
        }
        finish(edits, options);
        return null;
    };
    const commitOrExplain = () => {
        const problem = commit();
        if (!problem) return;
        // Focus first: focusing announces the field, and the problem must stay visible.
        focusInput(editable.find((part) => problem.startsWith(`${part.name}:`))?.name);
        onMessage(problem);
    };
    const cancel = () => finish(null);
    const move = (index, step) => {
        const next = editable[index + step];
        if (next) focusInput(next.name);
        else commitOrExplain();
    };
    const onKeyDown = (event, index) => {
        // While $ completion is open, Enter, Escape and arrows belong to the list.
        if (event.currentTarget.getAttribute('aria-expanded') === 'true' && ['Enter', 'Escape', 'ArrowUp', 'ArrowDown'].includes(event.key)) {
            event.stopPropagation();
            return;
        }
        if (event.key === 'Enter') { event.preventDefault(); commitOrExplain(); }
        else if (event.key === 'Escape') { event.preventDefault(); cancel(); }
        else if (event.key === 'Tab') { event.preventDefault(); move(index, event.shiftKey ? -1 : 1); }
        event.stopPropagation();
    };
    // Say where the value goes before anything is written.
    const announce = (part) => onMessage(`${part.name}: ${targets[part.name].hint} · Enter apply · Esc cancel`);
    const update = (name, text) => {
        setDrafts((current) => ({ ...current, [name]: text }));
        if (errors[name]) setErrors((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== name)));
    };

    return <span ref={editorRef} className="df-mt-editor nokey" role="group" aria-label={`Edit ${row.label} settings`}
        onBlur={(event) => {
            if (done.current || editorRef.current?.contains(event.relatedTarget)) return;
            // Leaving the cell keeps a valid edit and drops an invalid one;
            // focus stays wherever the user moved it.
            const problem = commit({ refocus: false });
            if (problem) {
                finish(null, { refocus: false });
                onMessage(`Edit cancelled · ${problem}`);
            }
        }}>
        {row.config.map((part, partIndex) => {
            if (part.kind !== 'field') return <Separator key={partIndex} part={part} />;
            const info = targets[part.name];
            const index = editable.indexOf(part);
            if (index < 0) {
                const shown = shownText(part, cells);
                return <span key={part.name} role="textbox" aria-readonly="true" aria-label={`${row.label} ${part.name}`} title={info?.hint ?? ''}
                    className={`df-mt-field is-${shown.tone} is-readonly`}><Prefix prefix={part.prefix} />{shown.text}</span>;
            }
            const text = drafts[part.name] ?? '';
            const placeholder = part.values.text;
            const common = {
                'data-field-input': part.name,
                'aria-label': `${row.label} ${part.name}`,
                'aria-invalid': !!errors[part.name],
                title: errors[part.name] ?? info.hint,
                autoFocus: part.name === firstName,
                className: `df-mt-input${errors[part.name] ? ' is-invalid' : ''}`,
                onKeyDown: (event) => onKeyDown(event, index),
                onClick: (event) => event.stopPropagation(),
                onDoubleClick: (event) => event.stopPropagation(),
            };
            if (part.spec?.type === 'enum') {
                return <span key={part.name} className="df-mt-input-wrap"><Prefix prefix={part.prefix} />
                    <select {...common} value={text} onFocus={() => announce(part)} onChange={(event) => update(part.name, event.target.value)}>
                        {!part.spec.options.includes(text) && <option value={text}>{text || 'Default'}</option>}
                        {part.spec.options.map((option) => <option key={option} value={option}>{option}</option>)}
                    </select>
                </span>;
            }
            return <span key={part.name} className="df-mt-input-wrap"><Prefix prefix={part.prefix} />
                <VariableInput {...common} type="text" spellCheck={false} autoComplete="off" floatingSuggestions
                    suggestions={info.target.literalOnly ? [] : suggestions} value={text} placeholder={placeholder}
                    style={{ '--parameter-width': widthOf(text, placeholder) }}
                    onFocus={(event) => {
                        announce(part);
                        if (seed === undefined || seed === null || part.name !== firstName) event.currentTarget.select?.();
                    }}
                    onChange={(event) => update(part.name, event.target.value)} />
            </span>;
        })}
    </span>;
}

/** The defining settings of a row, as written (Symbols) or resolved (Values). */
export default function ModelTableConfigCell({ row, cells, targets, context, fieldErrors, editing, suggestions, onStartEdit, onDone, onMessage }) {
    if (editing) {
        return <ConfigEditor row={row} cells={cells} targets={targets} context={context} suggestions={suggestions}
            focusField={editing.fieldName} seed={editing.seed} onDone={onDone} onMessage={onMessage} />;
    }
    return <span className="df-mt-config">
        {row.config.map((part, index) => part.kind === 'field'
            ? <FieldText key={part.name} field={part} cells={cells} info={targets[part.name]} errors={fieldErrors[part.name]}
                onClick={() => { if (targets[part.name]?.editable) onStartEdit(part.name); }} />
            : <Separator key={index} part={part} />)}
    </span>;
}
