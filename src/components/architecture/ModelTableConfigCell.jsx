import { useId, useRef, useState } from 'react';
import VariableInput from './VariableInput.jsx';
import { parseCellValue, precheckCellValue, targetSpec } from '../../domain/modelTable/writeTargets.js';
import { fitConfigParts } from '../../domain/modelTable/fitText.js';
import { drawnParts, shownFieldText as shownText } from '../../domain/modelTable/configFields.js';
import { emptyAllowed, fieldScope, initialText } from './modelTableEdits.js';

const widthOf = (text, placeholder) => `${Math.min(22, Math.max(4, String(text || placeholder || '').length + 1))}ch`;


/** `p=0.1`, `k=3` like a PyTorch repr; symbols such as × and + attach directly. */
function Prefix({ prefix }) {
    if (!prefix) return null;
    return <span className="df-mt-prefix">{prefix}{/^[A-Za-z_]/.test(prefix) ? '=' : ''}</span>;
}

/** A field as text: expressions in the expression colour, inherited defaults muted. */
function FieldText({ field, cells, text, info, errors, more, onClick }) {
    const shown = shownText(field, cells);
    const error = errors?.[0];
    const title = [
        `${field.name}${cells === 'symbols' && field.values.text !== shown.text ? ` = ${field.values.text}` : ''}${cells === 'values' && field.symbols.text !== shown.text ? ` · written ${field.symbols.text}` : ''}`,
        text !== shown.text ? shown.text : null, field.values.title, field.unknownReason ? `?: ${field.unknownReason}` : null, error, info?.hint, more ? `Also: ${more}` : null,
    ].filter(Boolean).join('\n');
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

/** Names a binding can also use: its repeat's index, then parameters declared before it. */
function bindingSuggestions(row, part, suggestions, templates) {
    if (part.source !== 'bindings' && part.source !== 'parameterValues') return suggestions;
    const extra = [];
    if (row.kind === 'repeat') extra.push({ name: row.iterationVariable ?? 'i', type: 'int', owner: `${row.label} instance index`, value: 0 });
    for (const parameter of templates[row.definitionIds[0]]?.freeParameters ?? []) {
        if (parameter.name === part.name) break;
        extra.push({ ...parameter, owner: `${row.label}, declared earlier`, value: row.instanceScope?.[parameter.name] ?? null });
    }
    const names = new Set(extra.map((item) => item.name));
    return [...suggestions.filter((item) => !names.has(item.name)), ...extra];
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
    // The focused field says where its value goes when that is not the field itself.
    const [focused, setFocused] = useState(firstName);
    const [initial] = useState(() => Object.fromEntries(editable.map((part) => [part.name, initialText(targets[part.name], context)])));
    const done = useRef(false);
    const editorRef = useRef(null);
    const errorId = useId();
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
            // Values mode writes the source it follows: check against that value's own spec.
            const spec = info.target.literalOnly ? targetSpec(info.target, { rootGraph: context.graph, templates: context.templates }, part.spec) : part.spec;
            const value = parseCellValue(text, spec);
            const error = precheckCellValue(value, { spec, scope: fieldScope(row, part, context.templates), libraryKey: row.libraryKey,
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
        // An input method is still composing: Enter picks its candidate.
        if (event.nativeEvent?.isComposing || event.keyCode === 229) {
            event.stopPropagation();
            return;
        }
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
    const announce = (part) => {
        setFocused(part.name);
        onMessage(`${part.name}: ${targets[part.name].hint} · Enter apply · Esc cancel`);
    };
    const update = (name, text) => {
        setDrafts((current) => ({ ...current, [name]: text }));
        if (errors[name]) setErrors((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== name)));
    };

    return <span ref={editorRef} className="df-mt-editor nokey" role="group" aria-label={`Edit ${row.label} settings`}
        onBlur={(event) => {
            if (done.current || editorRef.current?.contains(event.relatedTarget)) return;
            // Switching windows keeps the edit open; focus comes back to it.
            if (!event.relatedTarget && typeof document.hasFocus === 'function' && !document.hasFocus()) return;
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
                // A concealed default that cannot be edited here stays out of the editor too.
                if (part.concealed) return null;
                const shown = shownText(part, cells);
                return <span key={part.name} role="textbox" aria-readonly="true" aria-label={`${row.label} ${part.name}`} title={info?.hint ?? ''}
                    className={`df-mt-field is-${shown.tone} is-readonly`}><Prefix prefix={part.prefix} />{shown.text}</span>;
            }
            const text = drafts[part.name] ?? '';
            const placeholder = part.values.text;
            const error = errors[part.name];
            const tagId = info.tag ? `${errorId}-${part.name}-target` : null;
            const common = {
                'data-field-input': part.name,
                'aria-label': `${row.label} ${part.name}`,
                'aria-invalid': !!error,
                'aria-describedby': [error ? `${errorId}-${part.name}` : null, tagId].filter(Boolean).join(' ') || undefined,
                title: error ?? info.hint,
                autoFocus: part.name === firstName,
                className: `df-mt-input${error ? ' is-invalid' : ''}`,
                onKeyDown: (event) => onKeyDown(event, index),
                onClick: (event) => event.stopPropagation(),
                onDoubleClick: (event) => event.stopPropagation(),
            };
            const bubble = <>
                {error && <span id={`${errorId}-${part.name}`} className="df-mt-input-error">{error}</span>}
                {/* Values mode can write a global, a default or a container binding: say which, beside the field. */}
                {tagId && <span id={tagId} className={`df-mt-editor-target${focused === part.name ? '' : ' is-idle'}`}>writes {info.tag}</span>}
            </>;
            if (part.spec?.type === 'enum') {
                return <span key={part.name} className="df-mt-input-wrap"><Prefix prefix={part.prefix} />
                    <select {...common} value={text} onFocus={() => announce(part)} onChange={(event) => update(part.name, event.target.value)}>
                        {!part.spec.options.includes(text) && <option value={text}>{text || 'Default'}</option>}
                        {part.spec.options.map((option) => <option key={option} value={option}>{option}</option>)}
                    </select>
                    {bubble}
                </span>;
            }
            return <span key={part.name} className="df-mt-input-wrap"><Prefix prefix={part.prefix} />
                <VariableInput {...common} type="text" spellCheck={false} autoComplete="off" floatingSuggestions
                    suggestions={info.target.literalOnly ? [] : bindingSuggestions(row, part, suggestions ?? [], context.templates)} value={text} placeholder={placeholder}
                    style={{ '--parameter-width': widthOf(text, placeholder) }}
                    onFocus={(event) => {
                        announce(part);
                        if (seed === undefined || seed === null || part.name !== firstName) event.currentTarget.select?.();
                    }}
                    onChange={(event) => update(part.name, event.target.value)} />
                {bubble}
            </span>;
        })}
    </span>;
}

/** The defining settings of a row, as written (Symbols) or resolved (Values). */
export default function ModelTableConfigCell({ row, cells, targets, context, fieldErrors, editing, suggestions, budget, labelled = false, onStartEdit, onReadOnly, onDone, onMessage }) {
    if (editing) {
        return <ConfigEditor row={row} cells={cells} targets={targets} context={context} suggestions={suggestions}
            focusField={editing.fieldName} seed={editing.seed} onDone={onDone} onMessage={onMessage} />;
    }
    // Fit the cell without changing what it says: expressions are cut where the
    // ellipsis belongs to them, values stay whole, extra fields give way to "…".
    const parts = drawnParts(row);
    const fit = fitConfigParts(parts, {
        budget,
        display: (part) => shownText(part, cells).text,
        expression: (part) => shownText(part, cells).tone === 'expression',
    });
    const visible = fit.hiddenFrom === null ? parts : parts.slice(0, fit.hiddenFrom);
    const hidden = fit.hiddenFrom === null ? [] : parts.slice(fit.hiddenFrom).filter((part) => part.kind !== 'text');
    const hiddenText = hidden.map((part) => part.kind === 'field' ? `${part.name} ${shownText(part, cells).text}` : part.text).join(' · ');
    // A cut last field already ends in "…"; it names the hidden parts in its title instead.
    const lastCut = hidden.length > 0 && !!fit.texts[visible.length - 1]?.endsWith('…');
    // The cell's label reads the settings as a sentence; the drawn parts are hidden from screen readers.
    return <span className="df-mt-config" aria-hidden={labelled || undefined}>
        {visible.map((part, index) => part.kind === 'field'
            ? <FieldText key={part.name} field={part} cells={cells} text={fit.texts[index]} info={targets[part.name]} errors={fieldErrors[part.name]}
                more={lastCut && index === visible.length - 1 ? hiddenText : null}
                onClick={(event) => {
                    if (targets[part.name]?.editable) onStartEdit(part.name);
                    else if (targets[part.name]) {
                        // The row still becomes active; the reason replaces the usual focus line.
                        event.stopPropagation();
                        onReadOnly(`${part.name}: ${targets[part.name].hint}`);
                    }
                }} />
            : <Separator key={index} part={part} />)}
        {hidden.length > 0 && !lastCut && <span className="df-mt-more-fields" title={hiddenText}>…</span>}
    </span>;
}
