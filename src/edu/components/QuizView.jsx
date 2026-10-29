import { useEffect, useId, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { hasText, t } from '../lessons/engine/i18n.js';
import RichText from './RichText.jsx';

/**
 * One quiz or warm-up item per screen (docs/edu/SPEC.md §3.3). Choosing an option shows that option's explanation
 * at once; retries are unlimited and only the first answer counts in the log.
 */
export default function QuizView({ kind, items, index, records = {}, facts, onAnswer, onNext, onBack, onSkip }) {
    const item = items[index];
    const id = useId();
    const headingRef = useRef(null);
    const record = item ? records[item.id] : null;
    // The parent keys this view by item, so the choice starts from the saved answer of the item shown.
    const [chosen, setChosen] = useState(record?.last ?? null);

    useEffect(() => { headingRef.current?.focus(); }, [item?.id]);

    if (!item) return null;
    const correct = chosen !== null && chosen === item.correct;
    const last = index === items.length - 1;
    const misconception = chosen && !correct ? item.misconceptions?.[chosen] : null;
    const choose = (option) => {
        setChosen(option);
        onAnswer(item.id, option);
    };
    const nextLabel = kind === 'warmup' ? (last ? t('warmup.start') : t('quiz.next')) : (last ? t('quiz.finish') : t('quiz.next'));

    return (
        <section className="edu-quiz" aria-labelledby={`${id}-prompt`}>
            <p className="edu-subgoal">{t(kind === 'warmup' ? 'warmup.heading' : 'quiz.heading')} · {t('quiz.item', { index: index + 1, total: items.length })}</p>
            {kind === 'warmup' && index === 0 && <p className="edu-muted">{t('warmup.intro')}</p>}
            <h2 id={`${id}-prompt`} ref={headingRef} tabIndex={-1} className="edu-step-title"><RichText inline text={t(`${item.id}.prompt`)} params={facts} /></h2>
            <fieldset className="edu-options">
                <legend className="edu-sr-only">{t(`${item.id}.prompt`)}</legend>
                {item.options.map((option) => {
                    const state = chosen === option ? (option === item.correct ? ' is-correct' : ' is-wrong') : '';
                    return (
                        <label key={option} className={`edu-option${state}`}>
                            <input type="radio" name={`${id}-${item.id}`} value={option} checked={chosen === option} onChange={() => choose(option)} />
                            <RichText inline text={t(`${item.id}.opt.${option}`)} params={facts} />
                        </label>
                    );
                })}
            </fieldset>
            <div role="status" aria-live="polite" className="edu-quiz-feedback">
                {chosen && (
                    <div className={`edu-callout ${correct ? 'edu-callout-success' : 'edu-callout-misconception'}`}>
                        <strong>{t(correct ? 'quiz.correct' : 'quiz.incorrect')}. </strong>
                        <RichText inline text={t(`${item.id}.why.${chosen}`)} params={facts} />
                        {misconception && hasText(`mis.${misconception}`) && <> <RichText inline text={t(`mis.${misconception}`)} /></>}
                        {!correct && <span className="edu-muted"> {t('quiz.tryAgain')}</span>}
                    </div>
                )}
            </div>
            <footer className="edu-nav">
                <button type="button" className="df-button" onClick={onBack} disabled={!onBack}>
                    <ChevronLeft size={13} strokeWidth={2.5} aria-hidden="true" />{t('nav.back')}
                </button>
                {onSkip && <button type="button" className="edu-link-button" onClick={onSkip}>{t('warmup.skip')}</button>}
                <button type="button" className={`df-button${chosen ? ' df-button-primary' : ''}`} onClick={onNext} disabled={!chosen && kind !== 'warmup'}>
                    {nextLabel}<ChevronRight size={13} strokeWidth={2.5} aria-hidden="true" />
                </button>
            </footer>
        </section>
    );
}
