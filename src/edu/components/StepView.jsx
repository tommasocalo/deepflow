import { useEffect, useId, useMemo, useRef, useState } from 'react';
import { ChevronLeft, ChevronRight, Crosshair, MapPin, RotateCcw, Undo2, X } from 'lucide-react';
import { useArchitectureHistory } from '../../stores/architectureHistory.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useTrainingStore } from '../../stores/trainingStore.js';
import TrainerCard from '../trainer/components/TrainerCard.jsx';
import { focusNode, requestFit, requestTrainingFit, selectEdge } from '../lessons/engine/canvasBridge.js';
import { hasText, t } from '../lessons/engine/i18n.js';
import { logEvent } from '../lessons/engine/log.js';
import { isOutcomePrediction, observePrediction } from '../lessons/engine/observe.js';
import { predictionOrder } from '../lessons/engine/order.js';
import { revealValues } from '../lessons/engine/reveal.js';
import { isTargetDone, isVisible, reachedOpenTarget, resolveNodeTargetId, resolveTarget } from '../lessons/engine/targets.js';
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

/**
 * "Where is it?": bring the step's next target on screen: the first one not used yet (review r1: a block the step
 * asked to open counts as used, so the next target is the ribbon button). Its tab, the code panel, the Training canvas
 * fitted beside its palette, a node target centred on the canvas (inside the opened block when it is there, else at
 * the root, never leaving a block the step asked to open), or a list, panel or ribbon scrolled to it.
 */
function revealTarget(step, options) {
    const ui = useEduUiStore.getState();
    const refs = step.target ?? [];
    const pending = refs.filter((ref) => !isTargetDone(ref, globalThis.document, options));
    const next = pending[0];
    if (next === undefined) return;
    const nodeRef = typeof next === 'object' ? next : null;
    if (step.tab) ui.setTab(step.tab);
    else {
        const tab = nodeRef ? 'architecture' : TAB_OF_TARGET.find(([pattern]) => pattern.test(next))?.[1];
        if (tab) ui.setTab(tab);
    }
    // The notebook button and the Colab note sit on train.py; a tab target keeps the open tab so its ring shows.
    if (!nodeRef && (next.startsWith('code.notebook') || next.startsWith('code.colabNote'))) ui.openCode('train');
    else if (!nodeRef && next.startsWith('code.tab:')) ui.openCode(ui.codeTab);
    if (!nodeRef && next.startsWith('training.')) setTimeout(requestTrainingFit, 120);
    if (nodeRef) {
        const store = useArchitectureStore.getState();
        const match = nodeRef.node ?? nodeRef.nodeOpen;
        const here = !!resolveNodeTargetId(match);
        if (!here && store.currentPath.length && !reachedOpenTarget(refs)) store.goToLevel(0);
        setTimeout(() => {
            const id = resolveNodeTargetId(match);
            if (id) focusNode(id, { stay: useArchitectureStore.getState().currentPath.length > 0 }); else requestFit();
        }, 80);
    }
    // A target scrolled out of its list, panel or the ribbon (not a canvas node: the canvases above handle those).
    setTimeout(() => {
        const element = pending.map((ref) => resolveTarget(ref, globalThis.document, { rendered: true })).find(Boolean);
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
    // The check says what to connect; a block left without wires is named after it.
    return [result?.detail ? t(result.detail.key, result.detail.params) : '', result?.hint ? t(result.hint.key, result.hint.params) : ''].filter(Boolean).join(' ');
}

const WIRE_DELETES = new Set(['fb.extra-wire', 'fb.too-many-inputs', 'fb.too-many-inputs.ports', 'fb.port-twice']);

/** What Hint 1 adds for the state the learner is in: how to delete a wire, or how to draw one. */
function stateHint(result) {
    if (!result || result.pass) return null;
    if (result.issue && WIRE_DELETES.has(result.issue.key)) return 'hint.wireDelete';
    if (result.hint) return 'hint.wireAdd';
    return null;
}

const MISCONCEPTION_SETTLE_MS = 6000;

/**
 * The misconception to show: `diagnosed` once the workspace has not changed for MISCONCEPTION_SETTLE_MS, or at once
 * after a training run or a hint that came after the last edit (the learner signals they think they are done).
 */
function useSettledMisconception(diagnosed, { trainerSeq, hintLevel }) {
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    const training = useTrainingStore((state) => state.graph);
    const [shown, setShown] = useState(null);
    // Settled: the same workspace for a moment.
    useEffect(() => {
        if (!diagnosed) return undefined;
        const timer = setTimeout(() => setShown({ id: diagnosed, graph, templates, training }), MISCONCEPTION_SETTLE_MS);
        return () => clearTimeout(timer);
    }, [diagnosed, graph, templates, training]);
    // Signalled: a run or a hint since the last edit shows it at once.
    const signals = useRef(null);
    useEffect(() => {
        const previous = signals.current;
        signals.current = { trainerSeq, hintLevel };
        if (!previous || !diagnosed || (trainerSeq <= previous.trainerSeq && hintLevel <= previous.hintLevel)) return undefined;
        const timer = setTimeout(() => setShown({ id: diagnosed, graph, templates, training }), 0);
        return () => clearTimeout(timer);
    }, [trainerSeq, hintLevel, diagnosed, graph, templates, training]);
    // Any edit hides it again until the new state settles.
    return shown && shown.id === diagnosed && shown.graph === graph && shown.templates === templates && shown.training === training ? diagnosed : null;
}

/**
 * The study log (review r1, RQ2): which misconception callouts a learner saw and which status messages, each once per
 * visit to the step.
 */
function useStudyLog(moduleId, stepId, { misconception, status }) {
    const seen = useRef(new Set());
    const statusKey = status?.key ?? null;
    useEffect(() => { seen.current = new Set(); }, [moduleId, stepId]);
    useEffect(() => {
        if (!misconception || seen.current.has(`m:${misconception}`)) return;
        seen.current.add(`m:${misconception}`);
        logEvent('misconception-shown', { module: moduleId, step: stepId, data: { id: misconception } });
    }, [misconception, moduleId, stepId]);
    useEffect(() => {
        if (!statusKey || seen.current.has(`s:${statusKey}`)) return;
        seen.current.add(`s:${statusKey}`);
        logEvent('status-shown', { module: moduleId, step: stepId, data: { key: statusKey } });
    }, [statusKey, moduleId, stepId]);
}

/**
 * Bring `element` into the panel's scrolling area, above the sticky dock: its top when `top` (a reveal to read from
 * its heading), else its bottom (the done sentence under the trainer card). Only the panel body scrolls.
 */
function revealInPanel(element, { top = false } = {}) {
    const body = element?.closest?.('.edu-panel-body');
    if (!body?.getBoundingClientRect) return;
    const box = body.getBoundingClientRect();
    // The dock, or in the phone drawer only its navigation row, covers the bottom of the body.
    const dock = body.querySelector('.edu-step-dock')?.getBoundingClientRect().height || body.querySelector('.edu-step-dock > .edu-nav')?.getBoundingClientRect().height || 0;
    const rect = element.getBoundingClientRect();
    const visibleTop = box.top + 8;
    const visibleBottom = box.bottom - dock - 8;
    if (top || rect.top < visibleTop) {
        if (rect.top < visibleTop || rect.top > visibleBottom) body.scrollTop += rect.top - visibleTop;
    } else if (rect.bottom > visibleBottom) {
        body.scrollTop += Math.min(rect.bottom - visibleBottom, rect.top - visibleTop);
    }
}

/**
 * After a run: bring the trainer card's result (its confusion matrix, else its curve and boundary thumbnail) and the
 * reveal's "Your run shows" line below it into view together (review r2: the reveal grew above the card and pushed
 * the result behind the dock). When both do not fit, the line is kept in view with as much of the result as fits.
 */
function revealRunInPanel(result, line) {
    const body = line?.closest?.('.edu-panel-body');
    if (!body?.getBoundingClientRect) return;
    const box = body.getBoundingClientRect();
    const dock = body.querySelector('.edu-step-dock')?.getBoundingClientRect().height || body.querySelector('.edu-step-dock > .edu-nav')?.getBoundingClientRect().height || 0;
    const visibleTop = box.top + 8;
    const visibleBottom = box.bottom - dock - 8;
    const top = (result ?? line).getBoundingClientRect().top;
    const bottom = line.getBoundingClientRect().bottom;
    body.scrollTop += bottom - top <= visibleBottom - visibleTop ? top - visibleTop : bottom - visibleBottom;
}

/** "Show on canvas" for the problem the status names: the wire to delete, or the block. */
function showProblem(problem) {
    useEduUiStore.getState().setTab('architecture');
    setTimeout(() => {
        if (!problem.edgeId || !selectEdge(problem.edgeId)) focusNode(problem.nodeId);
    }, 60);
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
    // Predict, observe, explain: an outcome prediction is revealed with the run that passed the step, read from it.
    const outcome = isOutcomePrediction(step);
    const observed = outcome && (passing || done);
    const observation = observed && answer ? observePrediction(step, { trainer, enteredAtSeq: runtime?.enteredAtSeq ?? 0, runId: result?.runId }, prefix) : null;
    // Only a Show me that passed the step offers its toast (review r2: it toasted a change that left the step failing).
    const showMe = runtime?.showMe ?? null;
    const showMeAt = showMe?.passed ? showMe.at : null;
    // Tabs visited in this step count as used targets, so the ring moves on to what comes next; so does the palette
    // once its blocks are on the canvas.
    const visitedTabs = runtime?.visitedTabs;
    const entryNodeIds = runtime?.entryNodeIds ?? null;
    const targetOptions = useMemo(() => ({ visitedTabs: visitedTabs ?? [], palette: step.palette ?? null, entryNodeIds }), [visitedTabs, step.palette, entryNodeIds]);

    useEffect(() => { headingRef.current?.focus({ preventScroll: false }); }, [module.id, step.id]);
    useEffect(() => {
        if (!showMeAt || !showMe?.undoable) return undefined;
        const show = setTimeout(() => setToastAt(showMeAt), 0);
        const hide = setTimeout(() => setToastAt(null), 8000);
        return () => { clearTimeout(show); clearTimeout(hide); };
    }, [showMeAt, showMe?.undoable]);

    // The study log records what the learner's run showed next to what they predicted (once per visit).
    const loggedReveal = useRef(false);
    const observedOption = observation?.option ?? observation?.value ?? null;
    useEffect(() => {
        if (!outcome || !observation || !answer || loggedReveal.current || !passing) return;
        loggedReveal.current = true;
        logEvent('predict-observed', { module: module.id, step: step.id, data: { id: step.predict.id, said: answer.value, observed: observedOption, text: observation.text } });
    }, [outcome, observation, observedOption, answer, passing, module.id, step.id, step.predict?.id]);

    // On a pass, show what it says: the reveal of an outcome prediction, otherwise the done sentence.
    const doneRef = useRef(null);
    const wasPassing = useRef(passing);
    useEffect(() => {
        if (passing && !wasPassing.current) {
            const article = doneRef.current?.closest('.edu-step');
            const reveal = outcome ? article?.querySelector('.edu-predict.is-revealed') : null;
            const card = article?.querySelector('.edu-embed-trainer');
            const align = () => {
                if (reveal && card) revealRunInPanel(card.querySelector('.edu-tr-confusion') ?? card.querySelector('.edu-tr-card-body'), reveal.querySelector('.edu-reveal-compare') ?? reveal);
                else revealInPanel(reveal ?? doneRef.current, { top: !!reveal });
            };
            // Once now, and again when the card has drawn its result (its boundary and run list arrive a little later).
            const timers = [setTimeout(align, 60), setTimeout(align, 450)];
            wasPassing.current = passing;
            return () => timers.forEach(clearTimeout);
        }
        wasPassing.current = passing;
        return undefined;
    }, [passing, outcome]);

    const missingTimer = useRef(null);
    useEffect(() => () => clearTimeout(missingTimer.current), []);
    // A target can take a moment to appear (tab switch, canvas mount): only offer "Where is it?" after a second.
    const onResolve = (found) => {
        clearTimeout(missingTimer.current);
        if (found === null) missingTimer.current = setTimeout(() => setTargetMissing(true), 1200);
        else setTargetMissing(false);
    };
    const diagnosed = !passing && result?.misconception && hasText(`mis.${result.misconception}`) ? result.misconception : null;
    // A misconception callout waits until the learner seems finished (review r1: it fired after only Input was set):
    // the model has stayed the same for a moment, or they trained or asked for a hint since their last edit. A
    // complete wrong model (`firm`, review r2: the Sigmoid trap) is named at once, in the status line instead.
    const settled = useSettledMisconception(diagnosed, { trainerSeq: trainer?.seq ?? 0, hintLevel: record?.hintLevel ?? 0 });
    const firm = !!diagnosed && !!result?.firm && !done;
    const misconception = firm ? null : settled;
    // A pass kept from earlier whose model has changed since: the status says what holds now (review r2).
    const drift = done && !passing && !!result?.drift;
    const problem = result?.issue ?? result?.hint ?? null;
    // Once every target is used (the palette's blocks are on the canvas), the ring moves to the block the status names.
    const targets = step.target ?? [];
    const targetsUsed = targets.length > 0 && targets.every((ref) => isTargetDone(ref, globalThis.document, targetOptions));
    const spotlightTargets = targetsUsed && problem?.nodeId ? [{ node: { id: problem.nodeId } }] : targets;
    useStudyLog(module.id, step.id, { misconception: firm ? diagnosed : misconception, status: passing || (done && !drift) ? null : result?.issue ?? result?.detail ?? null });
    const embeds = step.embeds ?? [];
    // An outcome prediction is answered above the trainer card and revealed below it, next to the run's result.
    const revealBelow = outcome && observed && !!answer && embeds.includes('trainer');
    const gate = step.predict && (
        <PredictGate key={`${prefix}-${answer ? 'answered' : 'open'}`} predict={step.predict} textPrefix={prefix} answer={answer} params={params}
            disabled={busy} focusReveal={!!runtime?.revealShown && !revealBelow} onSubmit={(predictId, value) => session.submitPrediction(predictId, value)}
            outcome={outcome} observed={observed} observation={observation}
            order={step.predict.kind === 'choice' ? predictionOrder(step.predict, prefix) : null} />
    );
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

            {step.palette?.length > 0 && !passing && <PaletteChips items={step.palette} moduleId={module.id} stepId={step.id} entryNodeIds={entryNodeIds} />}
            {step.predict && !revealBelow && gate}
            {embeds.includes('trainer') && <div className="edu-embed edu-embed-trainer"><TrainerCard locked={locked} lockedReason={locked ? t('panel.trainLocked') : ''} /></div>}
            {revealBelow && gate}
            {embeds.includes('paramCount') && (!step.predict || answer ? <ParamCountReadout /> : <p className="edu-embed edu-muted">{t('paramCount.hidden')}</p>)}
            {embeds.includes('softmaxTemperature') && <SoftmaxTemperature />}
            {(step.buttons ?? []).map((button) => (
                <button key={button.id} type="button" className="df-button edu-step-button" disabled={busy} onClick={() => session.runButton(button.id)}>
                    {t(`${prefix}.button.${button.id}`)}
                </button>
            ))}

            {done && !passing && !drift && !runtime?.passedHere && <p className="edu-muted">{t('panel.reviewing')}</p>}
            {/* The done sentence states facts about the model: it goes when the model no longer has them. */}
            <div ref={doneRef}>{(passing || (done && !drift)) && hasText(`${prefix}.done`) && <RichText className="edu-done" text={t(`${prefix}.done`)} params={params} />}</div>
            {(passing || answer) && live.variant && hasText(`${prefix}.reveal.${live.variant}`) && (
                <RichText className="edu-callout edu-callout-info" text={t(`${prefix}.reveal.${live.variant}`)} params={params} />
            )}
            {misconception && (
                <p className="edu-callout edu-callout-misconception"><strong>{t('panel.misconception')}: </strong><RichText inline text={t(`mis.${misconception}`)} /></p>
            )}
            {runtime?.error && <p className="edu-callout edu-callout-error" role="alert">{t('panel.actionFailed', { error: runtime.error })}</p>}
            {showMe && !showMe.passed && !runtime?.error && !passing && <p className="edu-callout edu-callout-error" role="alert">{t('panel.showMeIncomplete')}</p>}
            {runtime?.skippedAhead > 0 && !done && <p className="edu-muted">{t('panel.skippedAhead')}</p>}

            <HintLadder textPrefix={prefix} level={record?.hintLevel ?? 0} passed={passing} busy={busy} facts={params} onHint={() => session.useHint()}
                now={stateHint(result)} />

            {/* The status and the way on stay at the bottom of the panel however long the step is (review r1). */}
            <div className="edu-step-dock">
                <div className={`edu-status${passing || done ? ' is-done' : ''}${drift ? ' is-drift' : ''}${firm ? ' is-misconception' : ''}`} role="status" aria-live="polite">
                    {drift ? (
                        <>
                            <span className="edu-status-icon" aria-hidden="true">✓</span>
                            <span>
                                <strong>{t(status === 'skipped' ? 'panel.skippedEarlier' : 'panel.passedEarlier')}</strong>
                                {' · '}{t('panel.now')}: <RichText inline text={statusText(result, busy)} />
                            </span>
                        </>
                    ) : passing || done ? (
                        <>
                            <span className="edu-status-icon" aria-hidden="true">✓</span>
                            <span><strong>{t('panel.done')}</strong>{passing && <span className="edu-sr-only"> — {t('panel.stepComplete')}</span>}</span>
                        </>
                    ) : firm ? (
                        <>
                            <span className="edu-status-icon" aria-hidden="true">○</span>
                            <span><strong>{t('panel.misconception')}: </strong><RichText inline text={t(`mis.${diagnosed}`)} /></span>
                        </>
                    ) : (
                        <>
                            <span className="edu-status-icon" aria-hidden="true">○</span>
                            <span><strong>{t('panel.notYet')}: </strong><RichText inline text={statusText(result, busy)} /></span>
                        </>
                    )}
                </div>
                {((!passing && problem?.nodeId) || (targetMissing && !passing && (step.target ?? []).some((ref) => !isTargetDone(ref, globalThis.document, targetOptions)))) && (
                    <p className="edu-dock-links">
                        {!passing && problem?.nodeId && (
                            <button type="button" className="edu-link-button edu-show-on-canvas" onClick={() => showProblem(problem)}>
                                <Crosshair size={12} strokeWidth={2.5} aria-hidden="true" />{t(problem.edgeId ? 'panel.showWire' : 'panel.showOnCanvas')}
                            </button>
                        )}
                        {targetMissing && !passing && (step.target ?? []).some((ref) => !isTargetDone(ref, globalThis.document, targetOptions)) && (
                            <button type="button" className="edu-link-button" onClick={() => revealTarget(step, targetOptions)}>
                                <MapPin size={12} strokeWidth={2.5} aria-hidden="true" />{t('panel.whereIsIt')}
                            </button>
                        )}
                    </p>
                )}
                {toastAt && toastAt === showMeAt && (
                    <div className="edu-toast" role="status">
                        <span>{t(showMe?.restarted ? 'panel.showMeRestarted' : 'panel.showMeDone')}</span>
                        <button type="button" className="edu-link-button" disabled={!canUndo} onClick={() => { session.undoShowMe(); setToastAt(null); }}>
                            <Undo2 size={12} strokeWidth={2.5} aria-hidden="true" />{t('panel.undo')}
                        </button>
                        <button type="button" className="df-icon-button" aria-label={t('panel.dismiss')} onClick={() => setToastAt(null)}><X size={12} /></button>
                    </div>
                )}
                <footer className="edu-nav">
                    <button type="button" className="df-button" onClick={() => session.back()} disabled={busy || index === 0} aria-label={t('nav.back')}>
                        <ChevronLeft size={13} strokeWidth={2.5} aria-hidden="true" />{t('nav.back')}
                    </button>
                    <button type="button" className="df-button" onClick={() => session.resetStep()} disabled={busy} aria-label={t('nav.reset')}>
                        <RotateCcw size={12} strokeWidth={2.5} aria-hidden="true" /><span className="edu-nav-label">{t('nav.reset')}</span>
                    </button>
                    <span className="edu-nav-spacer" />
                    {!canAdvance && <button type="button" className="edu-link-button" onClick={() => session.skip()} disabled={busy}>{t('nav.skip')}</button>}
                    <button type="button" className={`df-button${canAdvance ? ' df-button-primary edu-hard' : ''}`} onClick={() => session.next()} disabled={busy || !canAdvance}>
                        {nextLabel}<ChevronRight size={13} strokeWidth={2.5} aria-hidden="true" />
                    </button>
                </footer>
            </div>
            <p className="edu-sr-only">{t('panel.step', { index: index + 1, total })}</p>
            <Spotlight targets={spotlightTargets} active={!passing && !done} onResolve={onResolve} options={targetOptions} />
        </article>
    );
}
