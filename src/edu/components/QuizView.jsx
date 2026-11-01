import { useEffect, useId, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import { hasText, t } from '../lessons/engine/i18n.js';
import { quizOrder } from '../lessons/engine/order.js';
import RichText from './RichText.jsx';

/**
 * One quiz or warm-up item per screen (docs/edu/SPEC.md §3.3). Choosing an option only selects it; **Check** commits
 * it and shows that option's explanation (review r1: radios that committed on change logged a wrong first answer for a
 * keyboard user arrowing past the distractors). Retries are unlimited and only the first checked answer counts in the
 * log. Options appear in the learner's seeded order (engine/order.js).
 */
export default function QuizView({ kind, items, index, records = {}, facts, onAnswer, onNext, onBack, onSkip }) {
    const item = items[index];
    const id = useId();
    const headingRef = useRef(null);
    const record = item ? records[item.id] : null;
    // The parent keys this view by item, so the choice starts from the saved answer of the item shown.
    const [chosen, setChosen] = useState(record?.last ?? null);
    const [checked, setChecked] = useState(record?.last ?? null);

    useEffect(() => { headingRef.current?.focus(); }, [item?.id]);

    if (!item) return null;
    const order = quizOrder(item);
    const correct = checked !== null && checked === item.correct;
    const last = index === items.length - 1;
    const misconception = checked && !correct ? item.misconceptions?.[checked] : null;
    const pending = !!chosen && chosen !== checked;
    const check = (event) => {
        event.preventDefault();
        if (!pending) return;
        setChecked(chosen);
        onAnswer(item.id, chosen);
    };
    const nextLabel = kind === 'warmup' ? (last ? t('warmup.start') : t('quiz.next')) : (last ? t('quiz.finish') : t('quiz.next'));
    const promptId = `${id}-prompt`;

    return (
        <section className="edu-quiz" aria-labelledby={promptId}>
            <p className="edu-subgoal">{t(kind === 'warmup' ? 'warmup.heading' : 'quiz.heading')} · {t('quiz.item', { index: index + 1, total: items.length })}</p>
            {kind === 'warmup' && index === 0 && <p className="edu-muted">{t('warmup.intro')}</p>}
            <h2 id={promptId} ref={headingRef} tabIndex={-1} className="edu-step-title"><RichText inline text={t(`${item.id}.prompt`)} params={facts} /></h2>
            <form onSubmit={check} noValidate>
                {/* The heading names the group: no second, hidden copy of the prompt for screen readers. */}
                <fieldset className="edu-options" aria-labelledby={promptId}>
                    {order.map((option) => {
                        const state = checked === option && chosen === option ? (option === item.correct ? ' is-correct' : ' is-wrong') : chosen === option ? ' is-chosen' : '';
                        return (
                            <label key={option} className={`edu-option${state}`} data-option={option}>
                                <input type="radio" name={`${id}-${item.id}`} value={option} checked={chosen === option} onChange={() => setChosen(option)} />
                                <RichText inline text={t(`${item.id}.opt.${option}`)} params={facts} />
                            </label>
                        );
                    })}
                </fieldset>
                <button type="submit" className={`df-button edu-submit${pending ? ' df-button-primary' : ''}`} disabled={!pending}>{t('quiz.check')}</button>
            </form>
            <div role="status" aria-live="polite" className="edu-quiz-feedback">
                {checked && (
                    <div className={`edu-callout ${correct ? 'edu-callout-success' : 'edu-callout-misconception'}`}>
                        <strong>{t(correct ? 'quiz.correct' : 'quiz.incorrect')}. </strong>
                        <RichText inline text={t(`${item.id}.why.${checked}`)} params={facts} />
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
                <button type="button" className={`df-button${checked ? ' df-button-primary' : ''}`} onClick={onNext} disabled={!checked && kind !== 'warmup'}>
                    {nextLabel}<ChevronRight size={13} strokeWidth={2.5} aria-hidden="true" />
                </button>
            </footer>
        </section>
    );
}
