import { Lightbulb, Wand2 } from 'lucide-react';
import { hasText, t } from '../lessons/engine/i18n.js';
import RichText from './RichText.jsx';

/**
 * The help ladder (docs/edu/SPEC.md §3.3): a nudge, then a specific hint, then Show me, which performs the step as
 * one undoable edit. Hints stay visible once opened; the level is saved with the step's progress. `now` is a text key
 * for the learner's current state (StepView: how to draw or delete a wire), shown at the top of the first hint.
 */
export default function HintLadder({ textPrefix, level = 0, passed = false, busy = false, onHint, facts, now = null }) {
    const label = level <= 0 ? t('hint.first') : level === 1 ? t('hint.next') : level === 2 ? t('hint.showMe') : t('hint.again');
    const hints = [1, 2].filter((index) => level >= index && hasText(`${textPrefix}.hint${index}`));
    const Icon = level >= 2 ? Wand2 : Lightbulb;
    return (
        <div className="edu-hints">
            {hints.length > 0 && (
                <ol className="edu-hint-list">
                    {hints.map((index) => (
                        <li key={index} className="edu-hint">
                            <span className="edu-hint-label">{t('hint.label', { level: index })}</span>
                            {/* The first hint also answers the state the learner is in now (a wire to draw or to delete). */}
                            {index === 1 && now && !passed && <p className="edu-hint-now"><strong>{t('hint.now')}</strong> {t(now)}</p>}
                            <RichText text={t(`${textPrefix}.hint${index}`)} params={facts} />
                        </li>
                    ))}
                </ol>
            )}
            {!passed && (
                <button type="button" className="df-button edu-hint-button" onClick={onHint} disabled={busy} aria-label={label}>
                    <Icon size={13} strokeWidth={2.5} aria-hidden="true" />
                    {label}
                </button>
            )}
        </div>
    );
}
