import { useValidationContext } from '../../hooks/useValidation.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';

export default function RepeatIterationBar() {
    const context = useValidationContext();
    const setInspectedIteration = useArchitectureStore((s) => s.setInspectedIteration);
    const repeat = context?.repeat;
    if (!repeat) return null;
    const select = (index) => setInspectedIteration(repeat.path, Math.max(0, Math.min(repeat.count - 1, index)));
    return (
        <div className="df-iteration-bar" style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '5px 12px', borderBottom: '1px solid #111', background: '#fff', fontSize: 11, flexWrap: 'wrap' }}>
            <strong>{repeat.mode === 'parallel' ? 'Parallel instance' : 'Sequence step'}</strong>
            <button type="button" aria-label="Previous instance" disabled={repeat.index === 0} onClick={() => select(repeat.index - 1)}>‹</button>
            <input aria-label="Inspected instance index" type="number" min="0" max={Math.max(0, repeat.count - 1)} value={repeat.index} onChange={(event) => select(Number(event.target.value))} style={{ width: 45, border: '1px solid #111', font: 'bold 11px monospace', padding: '2px 4px' }} />
            <button type="button" aria-label="Next instance" disabled={repeat.index >= repeat.count - 1} onClick={() => select(repeat.index + 1)}>›</button>
            <span>of {repeat.count} · index starts at 0</span>
            <span style={{ color: '#666' }}>Shared template · edits apply to every instance</span>
        </div>
    );
}
