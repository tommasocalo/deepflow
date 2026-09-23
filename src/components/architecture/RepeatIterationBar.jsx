import { useValidationContext } from '../../hooks/useValidation.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import './repeat-iteration.css';

/** "Attention heads" → head, "Transformer blocks" → block; otherwise instance/step. */
function instanceNoun(label, mode) {
    const word = String(label ?? '').trim().split(/\s+/).at(-1)?.toLowerCase() ?? '';
    if (/^[a-z]{3,}ies$/.test(word)) return `${word.slice(0, -3)}y`;
    if (/^[a-z]{3,}s$/.test(word) && !/ss$/.test(word)) return word.slice(0, -1);
    return mode === 'parallel' ? 'instance' : 'step';
}

/**
 * Which occurrence of a repeat is on the canvas, and one toggle: off, it shares
 * the structure of the others and a change reaches all of them; on, it has its
 * own copy and changes stay local. Switching off keeps that copy for later.
 */
export default function RepeatIterationBar() {
    const context = useValidationContext();
    const setInspectedIteration = useArchitectureStore((s) => s.setInspectedIteration);
    const getStructureScope = useArchitectureStore((s) => s.getStructureScope);
    const setStructureScope = useArchitectureStore((s) => s.setStructureScope);
    const repeat = context?.repeat;
    if (!repeat) return null;
    const scope = getStructureScope(repeat.path);
    const noun = instanceNoun(repeat.label, repeat.mode);
    const number = repeat.index + 1;
    const select = (index) => setInspectedIteration(repeat.path, Math.max(0, Math.min(repeat.count - 1, index)));
    return (
        <div className={`df-iteration-bar${scope.local ? ' is-local' : ''}`}>
            <div className="df-iteration-counter">
                <strong>{noun[0].toUpperCase() + noun.slice(1)}</strong>
                <button type="button" aria-label="Previous instance" disabled={repeat.index === 0} onClick={() => select(repeat.index - 1)}>‹</button>
                <input aria-label="Inspected instance number" type="number" min="1" max={Math.max(1, repeat.count)} value={number} onChange={(event) => select(Number(event.target.value) - 1)} />
                <button type="button" aria-label="Next instance" disabled={repeat.index >= repeat.count - 1} onClick={() => select(repeat.index + 1)}>›</button>
                <span>of {repeat.count}</span>
            </div>
            <button type="button" role="switch" aria-checked={scope.local} className="df-iteration-toggle"
                onClick={() => setStructureScope(scope.local ? 'shared' : 'local', repeat.path)}
                title={scope.local
                    ? `On: changes affect ${noun} ${number} only. Turn off to share the structure again; this version is kept.`
                    : `Off: ${noun} ${number} shares its structure, so changes affect every ${noun} sharing it.${scope.hasSavedLocal ? ' A saved version of its own is available.' : ''}`}>
                <span className="df-iteration-switch" aria-hidden="true" />Own structure
            </button>
        </div>
    );
}
