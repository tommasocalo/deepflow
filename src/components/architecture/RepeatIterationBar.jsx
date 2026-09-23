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
const pluralOf = (noun) => (noun.endsWith('y') ? `${noun.slice(0, -1)}ies` : `${noun}s`);

/** Every instance when few; otherwise both ends and the neighbourhood of the current one. */
function stripIndices(count, index) {
    if (count <= 24) return Array.from({ length: count }, (_, i) => i);
    const picked = new Set([0, 1, 2, count - 3, count - 2, count - 1]);
    for (let i = index - 2; i <= index + 2; i++) if (i >= 0 && i < count) picked.add(i);
    return [...picked].sort((a, b) => a - b);
}

/**
 * Left: which instance is open, as a counter and a strip with one square per
 * instance (current in black, instances with their own structure in yellow).
 * Right: whether changes made here reach every instance or only this one.
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
    const plural = pluralOf(noun);
    const number = repeat.index + 1;
    const own = (i) => !!node?.structureOverrides?.[String(i)]?.enabled;
    const select = (index) => setInspectedIteration(repeat.path, Math.max(0, Math.min(repeat.count - 1, index)));
    const indices = stripIndices(repeat.count, repeat.index);
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
                    <button type="button" aria-label={`${Noun} ${i + 1}${own(i) ? ', own structure' : ''}`} aria-current={i === repeat.index ? 'true' : undefined}
                        className={`${i === repeat.index ? 'is-current' : ''}${own(i) ? ' is-own' : ''}`}
                        title={`${Noun} ${i + 1}${own(i) ? ' · has its own structure' : ''}`} onClick={() => select(i)} />
                </span>)}
            </div>
            <button type="button" role="switch" aria-checked={!scope.local} className="df-iteration-toggle"
                onClick={() => setStructureScope(scope.local ? 'shared' : 'local', repeat.path)}
                title={scope.local
                    ? `Off: changes stay in ${noun} ${number}, which has its own copy. Turn on to use the shared structure again; this copy is kept.`
                    : `On: ${noun} ${number} shares its structure, so changes reach every ${noun} sharing it. Turn off to change ${noun} ${number} alone.`}>
                <span className="df-iteration-switch" aria-hidden="true" />Apply changes to all {plural}
            </button>
        </div>
    );
}
