import { useValidationContext } from '../../hooks/useValidation.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import './repeat-iteration.css';

export default function RepeatIterationBar() {
    const context = useValidationContext();
    const setInspectedIteration = useArchitectureStore((s) => s.setInspectedIteration);
    const repeat = context?.repeat;
    if (!repeat) return null;
    const select = (index) => setInspectedIteration(repeat.path, Math.max(0, Math.min(repeat.count - 1, index)));
    return (
        <div className="df-iteration-bar">
            <strong>{repeat.mode === 'parallel' ? 'Parallel instance' : 'Sequence step'}</strong>
            <div className="df-iteration-counter">
                <button type="button" aria-label="Previous instance" disabled={repeat.index === 0} onClick={() => select(repeat.index - 1)}>‹</button>
                <input aria-label="Inspected instance number" type="number" min="1" max={Math.max(1, repeat.count)} value={repeat.index + 1} onChange={(event) => select(Number(event.target.value) - 1)} />
                <span>/ {repeat.count}</span>
                <button type="button" aria-label="Next instance" disabled={repeat.index >= repeat.count - 1} onClick={() => select(repeat.index + 1)}>›</button>
            </div>
        </div>
    );
}
