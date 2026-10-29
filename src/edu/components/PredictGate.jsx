import { useEffect, useId, useRef, useState } from 'react';
import { hasText, t } from '../lessons/engine/i18n.js';
import { expectedValue, formatPredictionValue, gradePrediction, parsePredictionInput } from '../lessons/engine/predict.js';
import RichText from './RichText.jsx';

/**
 * A prediction gate (docs/edu/SPEC.md §3.3, P6 predict–observe–explain). The answer is committed before the tool
 * reveals the value and cannot be changed until Reset step. Predictions are never marked failed.
 */
export default function PredictGate({ predict, textPrefix, answer, onSubmit, params, disabled = false, focusReveal = false }) {
    const id = useId();
    const [draft, setDraft] = useState('');
    const [error, setError] = useState('');
    const revealRef = useRef(null);
    const focusedRef = useRef(false);
    const optionText = (option) => t(`${textPrefix}.predict.opt.${option}`);
    const display = (value) => (predict.kind === 'choice' ? optionText(value) : formatPredictionValue(predict, value));

    // The answered gate is a new instance (StepView keys it by state). Focus its reveal once when the answer was just
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

    if (answer) {
        const grade = gradePrediction(predict, answer.value);
        const verdict = answer.source === 'show-me' ? t('predict.byShowMe', { value: display(answer.value) })
            : t(grade.verdict === 'exact' ? 'predict.exact' : grade.verdict === 'close' ? 'predict.close' : 'predict.notQuite');
        return (
            <section className={`edu-predict is-revealed${grade.correct ? ' is-correct' : ''}`} aria-labelledby={`${id}-reveal`}>
                <p className="edu-predict-prompt"><RichText inline text={t(`${textPrefix}.predict.prompt`)} params={params} /></p>
                <h3 id={`${id}-reveal`} ref={revealRef} tabIndex={-1} className="edu-reveal-heading">{verdict}</h3>
                <p className="edu-reveal-compare">
                    {answer.source !== 'show-me' && (
                        <>
                            <span>{t('predict.you', { value: display(answer.value) })}</span>
                            <span aria-hidden="true"> · </span>
                        </>
                    )}
                    <span>{t('predict.tool', { value: display(expectedValue(predict)) })}</span>
                </p>
                <RichText text={t(`${textPrefix}.predict.reveal`)} params={params} />
                {grade.misconception && hasText(`mis.${grade.misconception}`) && (
                    <p className="edu-callout edu-callout-misconception"><strong>{t('panel.misconception')}: </strong><RichText inline text={t(`mis.${grade.misconception}`)} /></p>
                )}
            </section>
        );
    }

    const inputId = `${id}-input`;
    const errorId = `${id}-error`;
    return (
        <section className="edu-predict" aria-labelledby={`${id}-title`}>
            <h3 id={`${id}-title`} className="edu-section-label">{t('predict.heading')}</h3>
            <form onSubmit={submit} noValidate>
                {predict.kind === 'choice' ? (
                    <fieldset className="edu-options" aria-describedby={error ? errorId : undefined}>
                        <legend className="edu-predict-prompt"><RichText inline text={t(`${textPrefix}.predict.prompt`)} params={params} /></legend>
                        {predict.options.map((option) => (
                            <label key={option} className={`edu-option${draft === option ? ' is-chosen' : ''}`}>
                                <input type="radio" name={`${id}-choice`} value={option} checked={draft === option} disabled={disabled}
                                    onChange={() => { setDraft(option); setError(''); }} />
                                <RichText inline text={optionText(option)} params={params} />
                            </label>
                        ))}
                    </fieldset>
                ) : (
                    <>
                        <label className="edu-predict-prompt" htmlFor={inputId}><RichText inline text={t(`${textPrefix}.predict.prompt`)} params={params} /></label>
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
