import { useId, useLayoutEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { completeVariableReference, formatVariableValue, variableReferenceAtCaret } from './variableCompletion';
import './variable-input.css';

/** Text input with scoped parameter completion. It remains a normal input for
 * selection, arithmetic, clipboard shortcuts, and native text editing. */
export default function VariableInput({ suggestions = [], multiline = false, floatingSuggestions = false, onChange, onFocus, onBlur, onKeyDown, onSelect, ...props }) {
    const inputRef = useRef(null);
    const listRef = useRef(null);
    const completedRef = useRef(null);
    const listId = useId();
    const [reference, setReference] = useState(null);
    const [focused, setFocused] = useState(false);
    const [highlighted, setHighlighted] = useState(0);
    const [dismissed, setDismissed] = useState(null);
    const [anchor, setAnchor] = useState(null);
    const [floatingStyle, setFloatingStyle] = useState(null);
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
    useLayoutEffect(() => {
        if (!floatingSuggestions || !open) return undefined;
        const input = inputRef.current;
        const list = listRef.current;
        if (!input || !list) return undefined;
        const position = () => {
            const rect = input.getBoundingClientRect();
            const viewportWidth = document.documentElement.clientWidth;
            const viewportHeight = document.documentElement.clientHeight;
            const gap = 4;
            const margin = 8;
            const width = Math.min(Math.max(200, rect.width), viewportWidth - margin * 2);
            const left = Math.max(margin, Math.min(rect.left, viewportWidth - width - margin));
            const desiredHeight = Math.min(180, list.scrollHeight + 2);
            const below = viewportHeight - rect.bottom - gap - margin;
            const above = rect.top - gap - margin;
            const placeAbove = below < desiredHeight && above > below;
            const available = Math.max(40, placeAbove ? above : below);
            const maxHeight = Math.min(180, available);
            const top = placeAbove ? Math.max(margin, rect.top - gap - Math.min(desiredHeight, maxHeight)) : rect.bottom + gap;
            setFloatingStyle({ top, left, width, maxHeight });
        };
        position();
        const observer = new ResizeObserver(position);
        observer.observe(input);
        observer.observe(list);
        window.addEventListener('scroll', position, true);
        window.addEventListener('resize', position);
        window.visualViewport?.addEventListener('resize', position);
        window.visualViewport?.addEventListener('scroll', position);
        return () => {
            observer.disconnect();
            window.removeEventListener('scroll', position, true);
            window.removeEventListener('resize', position);
            window.visualViewport?.removeEventListener('resize', position);
            window.visualViewport?.removeEventListener('scroll', position);
        };
    }, [floatingSuggestions, open, options.length, referenceKey]);
    useLayoutEffect(() => {
        if (!open) return;
        const option = document.getElementById(`${listId}-${active}`);
        if (!floatingSuggestions) {
            option?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' });
            return;
        }
        const list = listRef.current;
        if (!list || !option) return;
        const optionRect = option.getBoundingClientRect();
        const listRect = list.getBoundingClientRect();
        if (optionRect.bottom > listRect.bottom) list.scrollTop += optionRect.bottom - listRect.bottom;
        else if (optionRect.top < listRect.top) list.scrollTop -= listRect.top - optionRect.top;
    }, [open, active, listId, floatingSuggestions]);

    const updateReference = (input, changed = false) => {
        const row = input.closest('.df-param-row');
        if (row) setAnchor(row);
        const next = variableReferenceAtCaret(input.value, input.selectionStart);
        const nextKey = next && `${next.start}:${next.end}:${next.prefix}`;
        // Browsers dispatch a delayed selectionchange after completion sets the
        // caret. Keep that event from reopening the just-accepted suggestion.
        if (completedRef.current?.value === input.value && completedRef.current?.key === nextKey) {
            setDismissed(nextKey);
        } else if (changed || nextKey !== referenceKey) {
            completedRef.current = null;
            setHighlighted(0);
            setDismissed(null);
        }
        setReference(next);
    };
    const choose = (option) => {
        const input = inputRef.current;
        if (!input || !reference) return;
        const next = completeVariableReference(input.value, reference, option.name);
        const completed = variableReferenceAtCaret(next.value, next.caret);
        const completedKey = `${completed.start}:${completed.end}:${completed.prefix}`;
        completedRef.current = { value: next.value, key: completedKey };
        // Native input dispatch keeps the consumer's ordinary onChange contract
        // and React's controlled-input tracker in sync.
        const view = input.ownerDocument.defaultView;
        const prototype = multiline ? view.HTMLTextAreaElement.prototype : view.HTMLInputElement.prototype;
        Object.getOwnPropertyDescriptor(prototype, 'value').set.call(input, next.value);
        input.setSelectionRange(next.caret, next.caret);
        input.dispatchEvent(new view.Event('input', { bubbles: true }));
        setReference(completed);
        setDismissed(completedKey);
        input.focus();
    };
    const Input = multiline ? 'textarea' : 'input';
    const list = open && <span id={listId} ref={listRef} role="listbox" aria-label="Available parameters"
        style={floatingSuggestions ? floatingStyle ?? { visibility: 'hidden' } : undefined}
        className={`df-variable-suggestions nodrag${anchor && !floatingSuggestions ? ' is-portaled' : ''}${floatingSuggestions ? ' is-floating' : ''}`}>
        {options.map((option, index) => <span key={option.name} id={`${listId}-${index}`}
            role="option" aria-selected={index === active} className="df-variable-option"
            onMouseDown={(event) => { event.preventDefault(); event.stopPropagation(); }}
            onClick={(event) => { event.stopPropagation(); choose(option); }}>
            <span className="df-variable-option-main"><strong>${option.name.replace(/^\$/, '')}</strong><span>{option.value === null && option.nullable ? 'None' : formatVariableValue(option.value)}</span></span>
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
        {floatingSuggestions && list ? createPortal(list, document.body) : anchor && list ? createPortal(list, anchor) : list}
    </span>;
}
