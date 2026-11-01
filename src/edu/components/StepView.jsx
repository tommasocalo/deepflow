import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Crosshair, MapPin, RotateCcw, Undo2, X } from 'lucide-react';
import { useArchitectureHistory } from '../../stores/architectureHistory.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import TrainerCard from '../trainer/components/TrainerCard.jsx';
import { focusNode, requestFit, requestTrainingFit } from '../lessons/engine/canvasBridge.js';
import { hasText, t } from '../lessons/engine/i18n.js';
import { revealValues } from '../lessons/engine/reveal.js';
import { isTargetDone, isVisible, resolveNodeTargetId, resolveTarget } from '../lessons/engine/targets.js';
import { useLessonSession } from '../lessons/engine/session.js';
import { useEduUiStore } from '../shell/uiStore.js';
import { useBridgeSource } from '../useBridgeSource.js';
import HintLadder from './HintLadder.jsx';
import PaletteChips from './PaletteChips.jsx';
import ParamCountReadout from './ParamCountReadout.jsx';
import PredictGate from './PredictGate.jsx';
import RichText from './RichText.jsx';
import SoftmaxTemperature from './SoftmaxTemperature.jsx';
import Spotlight from './Spotlight.jsx';

const TAB_OF_TARGET = [
    [/^(ribbon\.|graph\.checks|field:)/, 'architecture'],
    [/^data\./, 'data'],
    [/^training\./, 'training'],
];
const named = (refs, prefix) => refs.some((ref) => typeof ref === 'string' && ref.startsWith(prefix));

/**
 * "Where is it?": bring the step's next target on screen: its tab, the root level where node targets start (centred
 * on the canvas), the code panel, the Training canvas fitted beside its palette, or a list or panel scrolled to it.
 */
function revealTarget(step, options) {
    const ui = useEduUiStore.getState();
    const refs = step.target ?? [];
    const nodeRef = refs.find((ref) => typeof ref === 'object');
    if (step.tab) ui.setTab(step.tab);
    else {
        const first = refs.find((ref) => typeof ref === 'string' && TAB_OF_TARGET.some(([pattern]) => pattern.test(ref)));
        const tab = nodeRef ? 'architecture' : TAB_OF_TARGET.find(([pattern]) => pattern.test(first ?? ''))?.[1];
        if (tab) ui.setTab(tab);
    }
    // The notebook button and the Colab note sit on train.py; a tab target keeps the open tab so its ring shows.
    if (named(refs, 'code.notebook') || named(refs, 'code.colabNote')) ui.openCode('train');
    else if (named(refs, 'code.tab:')) ui.openCode(ui.codeTab);
    if (named(refs, 'training.')) setTimeout(requestTrainingFit, 120);
    if (nodeRef) {
        const store = useArchitectureStore.getState();
        if (store.currentPath.length) store.goToLevel(0);
        setTimeout(() => {
            const id = resolveNodeTargetId(nodeRef.node ?? nodeRef.nodeOpen);
            if (id) focusNode(id); else requestFit();
        }, 80);
    }
    // A target scrolled out of its list or panel (not a canvas node: the canvases above handle those).
    setTimeout(() => {
        const element = refs.filter((ref) => !isTargetDone(ref, globalThis.document, options))
            .map((ref) => resolveTarget(ref, globalThis.document, { rendered: true })).find(Boolean);
        if (element && !isVisible(element) && !element.closest('.react-flow')) scrollToElement(element);
    }, 160);
}

/**
 * Centre `element` in each scrolling box around it, innermost first. Unlike scrollIntoView this leaves
 * overflow:hidden boxes alone: they hold the page layout (the stage and the drawer), and scrolling them shifts it.
 */
function scrollToElement(element) {
    const view = element.ownerDocument.defaultView;
    for (let node = element.parentElement; node && node !== element.ownerDocument.body; node = node.parentElement) {
        const style = view.getComputedStyle(node);
        const vertical = /auto|scroll/.test(style.overflowY) && node.scrollHeight > node.clientHeight;
        const horizontal = /auto|scroll/.test(style.overflowX) && node.scrollWidth > node.clientWidth;
        if (!vertical && !horizontal) continue;
        const box = node.getBoundingClientRect();
        const rect = element.getBoundingClientRect();
        if (vertical) node.scrollTop += rect.top + rect.height / 2 - (box.top + box.height / 2);
        if (horizontal) node.scrollLeft += rect.left + rect.width / 2 - (box.left + box.width / 2);
    }
}

function statusText(result, busy) {
    if (busy) return t('panel.working');
    if (result?.issue) return t(result.issue.key, result.issue.params);
    if (result?.detail) return t(result.detail.key, result.detail.params);
    return '';
}

export default function StepView({ module, step, index, total, record, answers, runTags = {}, runtime, result, facts, isLast }) {
    const session = useLessonSession.getState();
    const id = useId();
    const headingRef = useRef(null);
    const [targetMissing, setTargetMissing] = useState(false);
    const [toastAt, setToastAt] = useState(null);
    const trainer = useBridgeSource('trainer');
    const canUndo = useArchitectureHistory((state) => state.canUndo);
    // Re-render on navigation: inside a block a nodeOpen target names, that target is done (see isTargetDone).
    useArchitectureStore((state) => state.currentPath);
    const prefix = `${module.id}.${step.id}`;
    const status = record?.status;
    const done = status === 'passed' || status === 'skipped';
    const passing = !!result?.pass;
    const answer = step.predict ? answers?.[step.predict.id] ?? null : null;
    const locked = step.locks?.gate === 'train' && !!step.predict && !answer;
    const busy = !!runtime?.busy;
    // Run tags belong to the module (M5 compares its Dropout run with the run tagged in an earlier step).
    const live = revealValues(step.reveal, { trainer, enteredAtSeq: runtime?.enteredAtSeq ?? 0, runTags });
    const params = { ...facts, ...live.params };
    const showMeAt = runtime?.showMe?.at ?? null;
    // Tabs visited in this step count as used targets, so the ring moves on to what comes next.
    const visitedTabs = runtime?.visitedTabs;
    const targetOptions = useMemo(() => ({ visitedTabs: visitedTabs ?? [] }), [visitedTabs]);

    useEffect(() => { headingRef.current?.focus({ preventScroll: false }); }, [module.id, step.id]);
    useEffect(() => {
        if (!showMeAt || !runtime?.showMe?.undoable) return undefined;
        const show = setTimeout(() => setToastAt(showMeAt), 0);
        const hide = setTimeout(() => setToastAt(null), 8000);
        return () => { clearTimeout(show); clearTimeout(hide); };
    }, [showMeAt, runtime?.showMe?.undoable]);

    const missingTimer = useRef(null);
    useEffect(() => () => clearTimeout(missingTimer.current), []);
    // A target can take a moment to appear (tab switch, canvas mount): only offer "Where is it?" after a second.
    const onResolve = (found) => {
        clearTimeout(missingTimer.current);
        if (found === null) missingTimer.current = setTimeout(() => setTargetMissing(true), 1200);
        else setTargetMissing(false);
    };
    const misconception = !passing && result?.misconception && hasText(`mis.${result.misconception}`) ? result.misconception : null;
    const embeds = step.embeds ?? [];
    const nextLabel = isLast ? (module.quiz?.length ? t('nav.toQuiz') : t('nav.next')) : t('nav.next');
    const canAdvance = passing || done;

    return (
        <article className="edu-step" aria-labelledby={`${id}-title`}>
            <p className="edu-subgoal">{t(`subgoal.${step.subgoal}`)}</p>
            <h2 id={`${id}-title`} ref={headingRef} tabIndex={-1} className="edu-step-title">{t(`${prefix}.title`)}</h2>
            <RichText className="edu-step-body" text={t(`${prefix}.body`)} params={params} />
            {hasText(`${prefix}.why`) && (
                <details className="edu-why">
                    <summary>{t('panel.why')}</summary>
                    <RichText text={t(`${prefix}.why`)} params={params} />
                </details>
            )}

            {step.palette?.length > 0 && !passing && <PaletteChips items={step.palette} moduleId={module.id} stepId={step.id} />}
            {step.predict && (
                <PredictGate key={`${prefix}-${answer ? 'answered' : 'open'}`} predict={step.predict} textPrefix={prefix} answer={answer} params={params}
                    disabled={busy} focusReveal={!!runtime?.revealShown} onSubmit={(predictId, value) => session.submitPrediction(predictId, value)} />
            )}
            {embeds.includes('trainer') && <div className="edu-embed edu-embed-trainer"><TrainerCard locked={locked} lockedReason={locked ? t('panel.trainLocked') : ''} /></div>}
            {embeds.includes('paramCount') && (!step.predict || answer ? <ParamCountReadout /> : <p className="edu-embed edu-muted">{t('paramCount.hidden')}</p>)}
            {embeds.includes('softmaxTemperature') && <SoftmaxTemperature />}
            {(step.buttons ?? []).map((button) => (
                <button key={button.id} type="button" className="df-button edu-step-button" disabled={busy} onClick={() => session.runButton(button.id)}>
                    {t(`${prefix}.button.${button.id}`)}
                </button>
            ))}

            <div className={`edu-status${passing || done ? ' is-done' : ''}`} role="status" aria-live="polite">
                {passing || done ? (
                    <>
                        <span className="edu-status-icon" aria-hidden="true">✓</span>
                        <span><strong>{t('panel.done')}</strong>{passing && <span className="edu-sr-only"> — {t('panel.stepComplete')}</span>}</span>
                    </>
                ) : (
                    <>
                        <span className="edu-status-icon" aria-hidden="true">○</span>
                        <span><strong>{t('panel.notYet')}: </strong><RichText inline text={statusText(result, busy)} /></span>
                    </>
                )}
            </div>
            {!passing && result?.issue?.nodeId && (
                <button type="button" className="edu-link-button edu-show-on-canvas" onClick={() => { useEduUiStore.getState().setTab('architecture'); setTimeout(() => focusNode(result.issue.nodeId), 60); }}>
                    <Crosshair size={12} strokeWidth={2.5} aria-hidden="true" />{t('panel.showOnCanvas')}
                </button>
            )}
            {done && !passing && !runtime?.passedHere && <p className="edu-muted">{t('panel.reviewing')}</p>}
            {(passing || done) && hasText(`${prefix}.done`) && <RichText className="edu-done" text={t(`${prefix}.done`)} params={params} />}
            {(passing || answer) && live.variant && hasText(`${prefix}.reveal.${live.variant}`) && (
                <RichText className="edu-callout edu-callout-info" text={t(`${prefix}.reveal.${live.variant}`)} params={params} />
            )}
            {misconception && (
                <p className="edu-callout edu-callout-misconception"><strong>{t('panel.misconception')}: </strong><RichText inline text={t(`mis.${misconception}`)} /></p>
            )}
            {runtime?.error && <p className="edu-callout edu-callout-error" role="alert">{t('panel.actionFailed', { error: runtime.error })}</p>}
            {runtime?.skippedAhead > 0 && !done && <p className="edu-callout edu-callout-info">{t('panel.skippedAhead')}</p>}
            {targetMissing && !passing && (step.target ?? []).some((ref) => !isTargetDone(ref, globalThis.document, targetOptions)) && (
                <button type="button" className="edu-link-button" onClick={() => revealTarget(step, targetOptions)}>
                    <MapPin size={12} strokeWidth={2.5} aria-hidden="true" />{t('panel.whereIsIt')}
                </button>
            )}
            {toastAt && toastAt === showMeAt && (
                <div className="edu-toast" role="status">
                    <span>{t('panel.showMeDone')}</span>
                    <button type="button" className="edu-link-button" disabled={!canUndo} onClick={() => { session.undoShowMe(); setToastAt(null); }}>
                        <Undo2 size={12} strokeWidth={2.5} aria-hidden="true" />{t('panel.undo')}
                    </button>
                    <button type="button" className="df-icon-button" aria-label={t('panel.dismiss')} onClick={() => setToastAt(null)}><X size={12} /></button>
                </div>
            )}

            <HintLadder textPrefix={prefix} level={record?.hintLevel ?? 0} passed={passing} busy={busy} facts={params} onHint={() => session.useHint()} />

            <footer className="edu-nav">
                <button type="button" className="df-button" onClick={() => session.back()} disabled={busy || index === 0} aria-label={t('nav.back')}>
                    <ChevronLeft size={13} strokeWidth={2.5} aria-hidden="true" />{t('nav.back')}
                </button>
                <button type="button" className="df-button" onClick={() => session.resetStep()} disabled={busy} aria-label={t('nav.reset')}>
                    <RotateCcw size={12} strokeWidth={2.5} aria-hidden="true" />{t('nav.reset')}
                </button>
                <span className="edu-nav-spacer" />
                {!canAdvance && <button type="button" className="edu-link-button" onClick={() => session.skip()} disabled={busy}>{t('nav.skip')}</button>}
                <button type="button" className={`df-button${canAdvance ? ' df-button-primary edu-hard' : ''}`} onClick={() => session.next()} disabled={busy || !canAdvance}>
                    {nextLabel}<ChevronRight size={13} strokeWidth={2.5} aria-hidden="true" />
                </button>
            </footer>
            <p className="edu-sr-only">{t('panel.step', { index: index + 1, total })}</p>
            <Spotlight targets={step.target ?? []} active={!passing && !done} onResolve={onResolve} options={targetOptions} />
        </article>
    );
}
