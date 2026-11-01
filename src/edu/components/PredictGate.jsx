import { useEffect, useId, useRef, useState } from 'react';
import { hasText, t } from '../lessons/engine/i18n.js';
import { expectedValue, formatPredictionValue, gradePrediction, parsePredictionInput } from '../lessons/engine/predict.js';
import RichText from './RichText.jsx';

/**
 * A prediction gate (docs/edu/SPEC.md §3.3, P6 predict–observe–explain). The answer is committed before the tool
 * reveals the value and cannot be changed until Reset step. Predictions are never marked failed.
 *
 * An outcome prediction (`outcome`: the step's check needs a training run, engine/observe.js) is only locked on Submit;
 * its reveal waits until the run has been observed (`observed`) and then compares the answer with that run
 * (`observation.text`). Other predictions reveal the tool's value at once. Choice options are shown in `order`.
 */
export default function PredictGate({
    predict, textPrefix, answer, onSubmit, params, disabled = false, focusReveal = false, order = null,
    outcome = false, observed = false, observation = null,
}) {
    const id = useId();
    const [draft, setDraft] = useState('');
    const [error, setError] = useState('');
    const revealRef = useRef(null);
    const focusedRef = useRef(false);
    const optionText = (option) => t(`${textPrefix}.predict.opt.${option}`);
    // Lesson markup (option texts may hold `code`): rendered through RichText wherever it is shown.
    const display = (value) => (predict.kind === 'choice' ? optionText(value) : formatPredictionValue(predict, value));
    const prompt = <RichText inline text={t(`${textPrefix}.predict.prompt`)} params={params} />;
    const waiting = !!answer && outcome && !observed;

    // The answered gate is a new instance (StepView keys it by state). Focus its heading once when the answer was just
    // given in this step, not when the learner comes back to a step answered earlier.
    useEffect(() => {
        if (!answer || !focusReveal || focusedRef.current) return;
        focusedRef.current = true;
        revealRef.current?.focus();
    }, [answer, focusReveal]);

    const submit = (event) => {
        event.preventDefault();
        const value = parsePredictionInput(predict, draft);
        if (value === null) {
            setError(t(predict.kind === 'number' ? 'predict.invalidNumber' : predict.kind === 'shape' ? 'predict.invalidShape' : 'predict.chooseOne'));
            return;
        }
        setError('');
        onSubmit(predict.id, value);
    };

    if (waiting) {
        // Predict, then observe: nothing about the answer shows before the learner's own run.
        return (
            <section className="edu-predict is-locked" aria-labelledby={`${id}-reveal`}>
                <p className="edu-predict-prompt">{prompt}</p>
                <h3 id={`${id}-reveal`} ref={revealRef} tabIndex={-1} className="edu-reveal-heading">
                    <RichText inline text={t('predict.lockedIn', { value: display(answer.value) })} />
                </h3>
                <p className="edu-muted">{t('predict.waiting')}</p>
            </section>
        );
    }

    if (answer) {
        const grade = gradePrediction(predict, answer.value);
        const verdict = answer.source === 'show-me' ? t('predict.byShowMe', { value: display(answer.value) })
            : t(grade.verdict === 'exact' ? 'predict.exact' : grade.verdict === 'close' ? 'predict.close' : 'predict.notQuite');
        const fromRun = outcome && observation && !observation.tool;
        const shown = fromRun ? t('predict.yourRun', { value: observation.text })
            : t('predict.tool', { value: observation?.text ?? display(expectedValue(predict)) });
        return (
            <section className={`edu-predict is-revealed${grade.correct ? ' is-correct' : ''}`} aria-labelledby={`${id}-reveal`}>
                <p className="edu-predict-prompt">{prompt}</p>
                {/* An outcome reveal arrives with the run, not with a click: announce it. */}
                <div aria-live={outcome ? 'polite' : undefined}>
                    <h3 id={`${id}-reveal`} ref={revealRef} tabIndex={-1} className="edu-reveal-heading"><RichText inline text={verdict} /></h3>
                    <p className="edu-reveal-compare">
                        {answer.source !== 'show-me' && (
                            <>
                                <RichText inline text={t('predict.you', { value: display(answer.value) })} />
                                <span aria-hidden="true"> · </span>
                            </>
                        )}
                        <RichText inline text={shown} />
                    </p>
                </div>
                <RichText text={t(`${textPrefix}.predict.reveal`)} params={params} />
                {grade.misconception && hasText(`mis.${grade.misconception}`) && (
                    <p className="edu-callout edu-callout-misconception"><strong>{t('panel.misconception')}: </strong><RichText inline text={t(`mis.${grade.misconception}`)} /></p>
                )}
                <p className="edu-muted">{t('predict.locked')}</p>
            </section>
        );
    }

    const inputId = `${id}-input`;
    const errorId = `${id}-error`;
    const options = order ?? predict.options ?? [];
    return (
        <section className="edu-predict" aria-labelledby={`${id}-title`}>
            <h3 id={`${id}-title`} className="edu-section-label">{t('predict.heading')}</h3>
            <form onSubmit={submit} noValidate>
                {predict.kind === 'choice' ? (
                    <fieldset className="edu-options" aria-describedby={error ? errorId : undefined}>
                        <legend className="edu-predict-prompt">{prompt}</legend>
                        {options.map((option) => (
                            <label key={option} className={`edu-option${draft === option ? ' is-chosen' : ''}`} data-option={option}>
                                <input type="radio" name={`${id}-choice`} value={option} checked={draft === option} disabled={disabled}
                                    onChange={() => { setDraft(option); setError(''); }} />
                                <RichText inline text={optionText(option)} params={params} />
                            </label>
                        ))}
                    </fieldset>
                ) : (
                    <>
                        <label className="edu-predict-prompt" htmlFor={inputId}>{prompt}</label>
                        <input id={inputId} className="df-input edu-predict-input" value={draft} disabled={disabled}
                            inputMode={predict.kind === 'number' ? 'decimal' : 'text'} autoComplete="off" spellCheck={false}
                            placeholder={predict.kind === 'shape' ? 'B, …' : ''}
                            aria-describedby={[predict.kind === 'shape' ? `${id}-help` : null, error ? errorId : null].filter(Boolean).join(' ') || undefined}
                            aria-invalid={error ? true : undefined}
                            onChange={(event) => { setDraft(event.target.value); setError(''); }} />
                        {predict.kind === 'shape' && <p id={`${id}-help`} className="edu-muted">{t('predict.shapeHelp')}</p>}
                    </>
                )}
                {error && <p id={errorId} className="edu-field-error" role="alert">{error}</p>}
                <button type="submit" className="df-button df-button-primary edu-submit" disabled={disabled}>{t('predict.submit')}</button>
            </form>
        </section>
    );
}
