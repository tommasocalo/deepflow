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
const plural = (noun) => (noun.endsWith('y') ? `${noun.slice(0, -1)}ies` : `${noun}s`);

/**
 * Which occurrence of a repeat is on the canvas, and which structure it uses.
 * Shared: the occurrence uses the structure common to the others, so a change
 * reaches all of them. Own: it gets its own copy and changes stay local.
 */
export default function RepeatIterationBar() {
    const context = useValidationContext();
    const setInspectedIteration = useArchitectureStore((s) => s.setInspectedIteration);
    const getStructureScope = useArchitectureStore((s) => s.getStructureScope);
    const setStructureScope = useArchitectureStore((s) => s.setStructureScope);
    const repeat = context?.repeat;
    if (!repeat) return null;
    const scope = getStructureScope(repeat.path);
    const node = context.levels?.at(-1)?.instance;
    const noun = instanceNoun(repeat.label, repeat.mode);
    const Noun = noun[0].toUpperCase() + noun.slice(1);
    const number = repeat.index + 1;
    const own = Object.entries(node?.structureOverrides ?? {}).filter(([index, override]) => override?.enabled && Number(index) < repeat.count).length;
    const sharing = repeat.count - own;
    const select = (index) => setInspectedIteration(repeat.path, Math.max(0, Math.min(repeat.count - 1, index)));
    const effect = scope.local
        ? `Changes apply to ${noun} ${number} only.`
        : own
            ? `Changes apply to the ${sharing} ${sharing === 1 ? noun : plural(noun)} sharing it; ${own} ${own === 1 ? 'has its' : 'have their'} own.`
            : `Changes apply to all ${repeat.count} ${repeat.count === 1 ? noun : plural(noun)}.`;
    return (
        <div className={`df-iteration-bar${scope.local ? ' is-local' : ''}`}>
            <div className="df-iteration-counter">
                <strong>{Noun}</strong>
                <button type="button" aria-label="Previous instance" disabled={repeat.index === 0} onClick={() => select(repeat.index - 1)}>‹</button>
                <input aria-label="Inspected instance number" type="number" min="1" max={Math.max(1, repeat.count)} value={number} onChange={(event) => select(Number(event.target.value) - 1)} />
                <button type="button" aria-label="Next instance" disabled={repeat.index >= repeat.count - 1} onClick={() => select(repeat.index + 1)}>›</button>
                <span>of {repeat.count}</span>
            </div>
            <div className="df-iteration-scope" role="group" aria-label={`Structure used by ${noun} ${number}`}>
                <button type="button" aria-pressed={!scope.local} onClick={() => setStructureScope('shared', repeat.path)}
                    title={`${Noun} ${number} uses the structure shared with the other ${plural(noun)}. Its own version, if any, is kept.`}>Shared structure</button>
                <button type="button" aria-pressed={scope.local} onClick={() => setStructureScope('local', repeat.path)}
                    title={scope.hasSavedLocal ? `Use the saved structure of ${noun} ${number} again.` : `Give ${noun} ${number} its own copy of the structure.`}>Own structure</button>
            </div>
            <span className="df-iteration-effect" role="status">
                {effect}{!scope.local && scope.hasSavedLocal ? ` Its own version is saved.` : ''}
            </span>
        </div>
    );
}
