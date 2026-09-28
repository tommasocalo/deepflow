import { useMemo, useRef, useState } from 'react';
import { useValidationContext } from '../../hooks/useValidation.js';
import { repeatRuleSuggestions, selectRepeatInstances } from '../../domain/architecture/repeatRule.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import './repeat-iteration.css';

/** "Attention heads" → head, "Transformer blocks" → block; otherwise instance/step. */
function instanceNoun(label, mode) {
    const word = String(label ?? '').trim().split(/\s+/).at(-1)?.toLowerCase() ?? '';
    if (/^[a-z]{3,}ies$/.test(word)) return `${word.slice(0, -3)}y`;
    if (/^[a-z]{3,}s$/.test(word) && !/ss$/.test(word)) return word.slice(0, -1);
    return mode === 'parallel' ? 'instance' : 'step';
}
const pluralOf = (noun) => (noun.endsWith('y') ? `${noun.slice(0, -1)}ies` : `${noun}s`);

/** Every instance when few; otherwise both ends and the neighbourhood of the current one. */
function stripIndices(count, index) {
    if (count <= 24) return Array.from({ length: count }, (_, i) => i);
    const picked = new Set([0, 1, 2, count - 3, count - 2, count - 1]);
    for (let i = index - 2; i <= index + 2; i++) if (i >= 0 && i < count) picked.add(i);
    return [...picked].sort((a, b) => a - b);
}

export default function RepeatIterationBar() {
    const context = useValidationContext();
    const graph = useArchitectureStore((s) => s.graph);
    const templates = useTemplatesStore((s) => s.templates);
    const inspectedIterations = useArchitectureStore((s) => s.inspectedIterations);
    const selections = useArchitectureStore((s) => s.repeatSelections);
    const editModes = useArchitectureStore((s) => s.repeatEditModes);
    const editError = useArchitectureStore((s) => s.repeatEditError);
    const setInspectedIteration = useArchitectureStore((s) => s.setInspectedIteration);
    const getStructureScope = useArchitectureStore((s) => s.getStructureScope);
    const setStructureScope = useArchitectureStore((s) => s.setStructureScope);
    const setRepeatSelection = useArchitectureStore((s) => s.setRepeatSelection);
    const setRepeatEditMode = useArchitectureStore((s) => s.setRepeatEditMode);
    const repeat = context?.repeat;
    const pathKey = JSON.stringify(repeat?.path ?? []);
    const selection = selections[pathKey];
    const [openPath, setOpenPath] = useState(null);
    const ruleOpen = openPath === pathKey;
    const setRuleOpen = (next) => setOpenPath((previous) => {
        const open = previous === pathKey;
        return (typeof next === 'function' ? next(open) : next) ? pathKey : null;
    });
    const [ruleState, setRuleState] = useState(null);
    const rule = ruleState?.pathKey === pathKey ? ruleState.rule : selection?.rule ?? '';
    const setRule = (next) => setRuleState({ pathKey, rule: next });
    const [focused, setFocused] = useState(false);
    const [caret, setCaret] = useState(0);
    const inputRef = useRef(null);
    const preview = useMemo(() => {
        if (!repeat || !rule.trim()) return { indices: [], error: null };
        try {
            return { ...selectRepeatInstances({ graph, path: repeat.path, templates, inspectedIterations,
                count: repeat.count, rule }), error: null };
        } catch (error) {
            return { indices: [], error: error.message };
        }
    }, [graph, inspectedIterations, repeat, rule, templates]);
    const suggestions = useMemo(() => repeat
        ? repeatRuleSuggestions({ graph, path: repeat.path, templates, inspectedIterations }) : [],
    [graph, inspectedIterations, repeat, templates]);
    if (!repeat) return null;

    const scope = getStructureScope(repeat.path);
    const node = context.levels?.at(-1)?.instance;
    const noun = instanceNoun(repeat.label, repeat.mode);
    const Noun = noun[0].toUpperCase() + noun.slice(1);
    const plural = pluralOf(noun);
    const number = repeat.index + 1;
    const own = (i) => !!node?.structureOverrides?.[String(i)]?.enabled;
    const select = (index) => setInspectedIteration(repeat.path, Math.max(0, Math.min(repeat.count - 1, index)));
    const indices = stripIndices(repeat.count, repeat.index);
    const selected = new Set(selection?.indices?.filter((index) => index < repeat.count) ?? []);
    const highlighted = ruleOpen && rule.trim() && !preview.error ? new Set(preview.indices) : selected;
    const mode = editModes[pathKey] ?? (scope.local ? 'current' : 'all');
    const token = rule.slice(0, caret).match(/(?:^|[^\w.])((?:local|global)\.[\w]*|[a-zA-Z_][\w]*)$/)?.[1] ?? '';
    const completions = focused && token ? [...new Set(suggestions)]
        .filter((name) => name.startsWith(token) && name !== token).slice(0, 6) : [];
    const complete = (name) => {
        const start = caret - token.length;
        const next = `${rule.slice(0, start)}${name}${rule.slice(caret)}`;
        setRule(next);
        setCaret(start + name.length);
        inputRef.current?.focus();
    };
    return (
        <div className={`df-iteration-bar${scope.local ? ' is-local' : ''}`}>
            <div className="df-iteration-counter">
                <button type="button" aria-label="Previous instance" disabled={repeat.index === 0} onClick={() => select(repeat.index - 1)}>‹</button>
                <label className="df-iteration-title">{Noun}
                    <input aria-label="Inspected instance number" type="number" min="1" max={Math.max(1, repeat.count)} value={number} onChange={(event) => select(Number(event.target.value) - 1)} />
                    <span>of {repeat.count}</span>
                </label>
                <button type="button" aria-label="Next instance" disabled={repeat.index >= repeat.count - 1} onClick={() => select(repeat.index + 1)}>›</button>
            </div>
            <div className="df-iteration-strip" role="group" aria-label={`${plural} of ${repeat.label}`}>
                {indices.map((i, at) => <span key={i} className="df-iteration-slot">
                    {at > 0 && i - indices[at - 1] > 1 && <span className="df-iteration-gap" aria-hidden="true">…</span>}
                    <button type="button" aria-label={`${Noun} ${i + 1}${highlighted.has(i) ? ', selected by rule' : ''}${own(i) ? ', own structure' : ''}`}
                        aria-current={i === repeat.index ? 'true' : undefined}
                        className={`${i === repeat.index ? 'is-current' : ''}${own(i) ? ' is-own' : ''}${highlighted.has(i) ? ' is-selected' : ''}`}
                        title={`${Noun} ${i + 1}${highlighted.has(i) ? ' · selected by rule' : ''}${own(i) ? ' · has its own structure' : ''}`} onClick={() => select(i)} />
                </span>)}
            </div>
            <div className="df-iteration-rule-anchor">
                <button type="button" className={`df-iteration-rule-trigger${selected.size ? ' has-selection' : ''}`} aria-expanded={ruleOpen}
                    onClick={() => setRuleOpen((open) => !open)}>Select by rule{selected.size ? ` · ${selected.size}` : ''}</button>
                {ruleOpen && <div className="df-iteration-rule-panel" role="group" aria-label="Select instances by rule">
                    <label htmlFor="df-repeat-rule">Select by rule</label>
                    <input id="df-repeat-rule" ref={inputRef} aria-label="Instance selection rule" value={rule}
                        placeholder="position % 2 == 0" spellCheck={false}
                        onChange={(event) => { setRule(event.target.value); setCaret(event.target.selectionStart ?? event.target.value.length); }}
                        onClick={(event) => setCaret(event.target.selectionStart ?? rule.length)}
                        onFocus={() => setFocused(true)} onBlur={() => setTimeout(() => setFocused(false), 120)}
                        onKeyDown={(event) => {
                            if (event.key === 'Enter' && !preview.error && preview.indices.length) {
                                setRepeatSelection(repeat.path, preview.indices, rule); setRuleOpen(false);
                            }
                        }} />
                    {completions.length > 0 && <div className="df-iteration-rule-completions" role="listbox" aria-label="Rule variables">
                        {completions.map((name) => <button key={name} type="button" role="option" aria-selected="false"
                            onMouseDown={(event) => event.preventDefault()} onClick={() => complete(name)}>{name}</button>)}
                    </div>}
                    <p className="df-iteration-rule-help">position starts at 1. Use numeric local.* or global.* values.</p>
                    <div className="df-iteration-rule-footer">
                        <strong className={preview.error ? 'is-error' : ''}>{preview.error ?? `${preview.indices.length} of ${repeat.count} in preview`}</strong>
                        <button type="button" disabled={!!preview.error || !preview.indices.length} onClick={() => {
                            setRepeatSelection(repeat.path, preview.indices, rule); setRuleOpen(false);
                        }}>Select</button>
                        {!!selection?.indices.length && <button type="button" onClick={() => { setRepeatSelection(repeat.path, [], ''); setRule(''); }}>Clear</button>}
                    </div>
                </div>}
            </div>
            <label className="df-iteration-apply">Parameter changes
                <select aria-label="Apply parameter changes to" value={mode} onChange={(event) => setRepeatEditMode(repeat.path, event.target.value)}>
                    <option value="current">Current</option>
                    <option value="selected" disabled={!selected.size}>Selected ({selected.size})</option>
                    <option value="all">All</option>
                </select>
            </label>
            <button type="button" role="switch" aria-checked={!scope.local} className="df-iteration-toggle"
                onClick={() => setStructureScope(scope.local ? 'shared' : 'local', repeat.path)}
                title={scope.local
                    ? `${noun} ${number} has its own structure. Switch to the shared definition; the private copy is saved.`
                    : `${noun} ${number} uses the shared structure. Switch to a private definition for this instance.`}>
                <span className="df-iteration-switch" aria-hidden="true" />Shared structure
            </button>
            {editError?.pathKey === pathKey && <span className="df-iteration-edit-error" role="alert">{editError.message}</span>}
        </div>
    );
}
