import { useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { completeVariableReference, formatVariableValue, variableReferenceAtCaret } from './variableCompletion';
import './variable-input.css';

/** Text input with scoped parameter completion. It remains a normal input for
 * selection, arithmetic, clipboard shortcuts, and native text editing. */
export default function VariableInput({ suggestions = [], multiline = false, onChange, onFocus, onBlur, onKeyDown, onSelect, ...props }) {
    const inputRef = useRef(null);
    const listId = useId();
    const [reference, setReference] = useState(null);
    const [focused, setFocused] = useState(false);
    const [highlighted, setHighlighted] = useState(0);
    const [dismissed, setDismissed] = useState(null);
    const [anchor, setAnchor] = useState(null);
    const referenceKey = reference && `${reference.start}:${reference.end}:${reference.prefix}`;
    const seen = new Set();
    const options = reference ? suggestions.filter((item) => {
        const name = String(item.name ?? '').replace(/^\$/, '');
        if (!name || seen.has(name) || !name.toLowerCase().startsWith(reference.prefix.toLowerCase())) return false;
        seen.add(name);
        return true;
    }) : [];
    const open = focused && reference && dismissed !== referenceKey && options.length > 0;
    const active = Math.min(highlighted, options.length - 1);

    const updateReference = (input, changed = false) => {
        const row = input.closest('.df-param-row');
        if (row) {
            const rowRect = row.getBoundingClientRect();
            const inspectorRect = row.closest('.df-inspector')?.getBoundingClientRect();
            const width = inspectorRect?.width ? Math.min(Math.max(184, rowRect.width), inspectorRect.width - 20) : Math.max(184, rowRect.width);
            const left = inspectorRect?.width ? Math.max(inspectorRect.left + 10 - rowRect.left, Math.min(0, inspectorRect.right - 10 - rowRect.left - width)) : 0;
            setAnchor({ host: row, width, left });
        }
        const next = variableReferenceAtCaret(input.value, input.selectionStart);
        const nextKey = next && `${next.start}:${next.end}:${next.prefix}`;
        if (changed || nextKey !== referenceKey) {
            setHighlighted(0);
            setDismissed(null);
        }
        setReference(next);
    };
    const choose = (option) => {
        const input = inputRef.current;
        if (!input || !reference) return;
        const next = completeVariableReference(input.value, reference, option.name);
        // Native input dispatch keeps the consumer's ordinary onChange contract
        // and React's controlled-input tracker in sync.
        const view = input.ownerDocument.defaultView;
        const prototype = multiline ? view.HTMLTextAreaElement.prototype : view.HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, next.value);
        input.setSelectionRange(next.caret, next.caret);
        input.dispatchEvent(new view.Event('input', { bubbles: true }));
        const completed = variableReferenceAtCaret(next.value, next.caret);
        setReference(completed);
        setDismissed(`${completed.start}:${completed.end}:${completed.prefix}`);
        input.focus();
    };
    const Input = multiline ? 'textarea' : 'input';
    const list = open && <span id={listId} role="listbox" aria-label="Available parameters"
        style={anchor ? { width: anchor.width, left: anchor.left } : undefined}
        className={`df-variable-suggestions nodrag${anchor ? ' is-portaled' : ''}`}>
        {options.map((option, index) => <span key={option.name} id={`${listId}-${index}`}
            role="option" aria-selected={index === active} className="df-variable-option"
            onMouseDown={(event) => { event.preventDefault(); event.stopPropagation(); }}
            onClick={(event) => { event.stopPropagation(); choose(option); }}>
            <span className="df-variable-option-main"><strong>${option.name.replace(/^\$/, '')}</strong><span>{formatVariableValue(option.value)}</span></span>
            {option.owner && <span className="df-variable-option-owner">{option.owner}</span>}
        </span>)}
    </span>;

    return <span className={`df-variable-field${open ? ' is-completing' : ''}`}>
        <Input {...props} ref={inputRef}
            role={suggestions.length ? 'combobox' : undefined}
            aria-autocomplete={suggestions.length ? 'list' : undefined}
            aria-expanded={suggestions.length ? !!open : undefined}
            aria-controls={open ? listId : undefined}
            aria-activedescendant={open ? `${listId}-${active}` : undefined}
            onChange={(event) => { updateReference(event.currentTarget, true); onChange?.(event); }}
            onFocus={(event) => { setFocused(true); updateReference(event.currentTarget); onFocus?.(event); }}
            onBlur={(event) => { setFocused(false); onBlur?.(event); }}
            onSelect={(event) => { updateReference(event.currentTarget); onSelect?.(event); }}
            onKeyDown={(event) => {
                onKeyDown?.(event);
                if (!open) return;
                if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                    event.preventDefault();
                    event.stopPropagation();
                    const direction = event.key === 'ArrowDown' ? 1 : -1;
                    setHighlighted((active + direction + options.length) % options.length);
                } else if (event.key === 'Enter') {
                    event.preventDefault();
                    event.stopPropagation();
                    choose(options[active]);
                } else if (event.key === 'Escape') {
                    event.preventDefault();
                    event.stopPropagation();
                    setDismissed(referenceKey);
                }
            }} />
        {anchor && list ? createPortal(list, anchor.host) : list}
    </span>;
}
