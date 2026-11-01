import { useEffect, useRef, useState } from 'react';
import { ChevronUp, Minus, PanelRightOpen, Redo2, Undo2, X } from 'lucide-react';
import { useArchitectureHistory } from '../../stores/architectureHistory.js';
import { factValues } from '../lessons/engine/context.js';
import { t } from '../lessons/engine/i18n.js';
import { isModuleComplete } from '../lessons/engine/progress.js';
import { getModule, nextModule } from '../lessons/engine/registry.js';
import { useLessonSession } from '../lessons/engine/session.js';
import { useEduUiStore } from '../shell/uiStore.js';
import QuizView from './QuizView.jsx';
import RecapView from './RecapView.jsx';
import StepView from './StepView.jsx';
import { requestLearnHeadingFocus } from './focus.js';
import { moduleTitle } from './format.js';
import { CHECK_DELAY, useMediaQuery, useModules, useStepEvaluation } from './hooks.js';
import './lesson-panel.css';

const NARROW = '(max-width: 899px)';
const DRAWER_MIN = 120;
const DRAWER_MAX = 0.85;

const clampDrawer = (height) => Math.round(Math.min(window.innerHeight * DRAWER_MAX, Math.max(DRAWER_MIN, height)));

function segmentStatus(record, step, current) {
    const status = record?.steps?.[step.id]?.status;
    if (current) return 'current';
    return status === 'passed' || status === 'skipped' || status === 'visited' ? status : 'todo';
}

/**
 * The step segments are one tab stop (review r1: eight stops on every step): Tab reaches the current step's segment,
 * the arrow keys, Home and End move between them, Enter or Space opens one.
 */
function ProgressSegments({ module, record, stepIndex, phase, onGo }) {
    const buttons = useRef([]);
    const stop = phase === 'steps' ? stepIndex : 0;
    const onKeyDown = (event) => {
        const index = buttons.current.indexOf(event.target);
        if (index < 0) return;
        const last = module.steps.length - 1;
        const next = { ArrowRight: index + 1, ArrowDown: index + 1, ArrowLeft: index - 1, ArrowUp: index - 1, Home: 0, End: last }[event.key];
        if (next === undefined) return;
        event.preventDefault();
        buttons.current[Math.max(0, Math.min(last, next))]?.focus();
    };
    return (
        <ol className="edu-segments" aria-label={t('panel.progress')} onKeyDown={onKeyDown}>
            {module.steps.map((step, index) => {
                const current = phase === 'steps' && index === stepIndex;
                const status = segmentStatus(record, step, current);
                const label = t('panel.segment', { index: index + 1, title: t(`${module.id}.${step.id}.title`), status: t(`panel.segment.${status}`) });
                return (
                    <li key={step.id}>
                        <button type="button" ref={(element) => { buttons.current[index] = element; }} tabIndex={index === stop ? 0 : -1}
                            className={`edu-segment is-${status}`} aria-label={label} title={`${t(`subgoal.${step.subgoal}`)} · ${t(`${module.id}.${step.id}.title`)}`}
                            aria-current={current ? 'step' : undefined} onClick={() => onGo(index)} />
                    </li>
                );
            })}
        </ol>
    );
}

/**
 * Undo and Redo for the architecture, always in view in a lesson (review r1: the ribbon's History group is cut off in
 * the narrower canvas column). They act on the same history as the ribbon and Ctrl+Z.
 */
function HistoryButtons() {
    const canUndo = useArchitectureHistory((state) => state.canUndo);
    const canRedo = useArchitectureHistory((state) => state.canRedo);
    const undoLabel = useArchitectureHistory((state) => state.undoLabel);
    const redoLabel = useArchitectureHistory((state) => state.redoLabel);
    const history = () => useArchitectureHistory.getState();
    return (
        <span className="edu-history" role="group" aria-label={t('panel.history')}>
            <button type="button" className="df-icon-button" onClick={() => history().undo()} disabled={!canUndo} aria-label={t('panel.undoEdit')}
                aria-keyshortcuts="Control+Z Meta+Z" title={canUndo ? t('panel.undoTitle', { label: undoLabel }) : t('panel.undoEdit')}>
                <Undo2 size={15} aria-hidden="true" />
            </button>
            <button type="button" className="df-icon-button" onClick={() => history().redo()} disabled={!canRedo} aria-label={t('panel.redoEdit')}
                aria-keyshortcuts="Control+Shift+Z Meta+Shift+Z" title={canRedo ? t('panel.redoTitle', { label: redoLabel }) : t('panel.redoEdit')}>
                <Redo2 size={15} aria-hidden="true" />
            </button>
        </span>
    );
}

/**
 * The lesson panel (docs/edu/SPEC.md §3.3), docked beside the tab content: 360 px wide at ≥ 1200 px, 320 px
 * between 900 and 1199 px, a bottom drawer below 900 px. It never covers the canvas: the stage shrinks instead.
 */
export default function LessonPanel({ checkDelay = CHECK_DELAY }) {
    useModules();
    const moduleId = useLessonSession((state) => state.moduleId);
    const phase = useLessonSession((state) => state.phase);
    const stepIndex = useLessonSession((state) => state.stepIndex);
    const quizIndex = useLessonSession((state) => state.quizIndex);
    const warmupIndex = useLessonSession((state) => state.warmupIndex);
    const collapsed = useLessonSession((state) => state.collapsed);
    const runtime = useLessonSession((state) => state.runtime);
    const progress = useLessonSession((state) => state.progress);
    const drawerOpen = useEduUiStore((state) => state.drawerOpen);
    // Undo and Redo where the learner edits the document: the architecture and the training graph (review r2).
    const onDocument = useEduUiStore((state) => state.activeTab === 'architecture' || state.activeTab === 'training');
    const narrow = useMediaQuery(NARROW);
    const panelRef = useRef(null);
    // Below 900 px the drawer's handle toggles it and, dragged (or with the arrow keys), resizes it.
    const [drawerHeight, setDrawerHeight] = useState(null);
    const dragRef = useRef(null);
    const draggedRef = useRef(false);
    const module = getModule(moduleId);
    const step = module?.steps?.[stepIndex] ?? null;
    const result = useStepEvaluation(`${moduleId}:${phase}:${stepIndex}`, { enabled: !!module && phase === 'steps' && !!step, delay: checkDelay });
    const session = useLessonSession.getState();
    // A lesson left inside a module that has been updated since reloads that module's start (session initialState).
    const restart = useLessonSession((state) => state.restart);
    useEffect(() => { if (restart) useLessonSession.getState().startModule(restart); }, [restart]);

    if (!module) {
        return <aside className="edu-lesson-panel" aria-label={t('panel.label')}><p className="edu-muted">{t('panel.noModule')}</p></aside>;
    }
    const record = progress.modules?.[module.id] ?? null;
    const facts = factValues(module);
    const total = module.steps.length;
    const passing = !!result?.pass;
    const isCollapsed = narrow ? !drawerOpen : collapsed;
    const toggle = () => (narrow ? useEduUiStore.getState().setDrawerOpen(!drawerOpen) : session.setCollapsed(!collapsed));
    const exit = () => {
        requestLearnHeadingFocus();
        session.exitLesson({ tab: 'learn' });
    };
    const resizeDrawer = (height) => {
        if (!drawerOpen) useEduUiStore.getState().setDrawerOpen(true);
        setDrawerHeight(clampDrawer(height));
    };
    const handleEvents = {
        onPointerDown: (event) => {
            if (event.button !== 0) return;
            dragRef.current = { y: event.clientY, start: panelRef.current?.getBoundingClientRect().height ?? 0, moved: false };
            event.currentTarget.setPointerCapture?.(event.pointerId);
        },
        onPointerMove: (event) => {
            const drag = dragRef.current;
            if (!drag) return;
            const delta = drag.y - event.clientY;
            if (!drag.moved && Math.abs(delta) < 6) return;
            drag.moved = true;
            resizeDrawer(drag.start + delta);
        },
        onPointerUp: (event) => {
            draggedRef.current = !!dragRef.current?.moved;
            dragRef.current = null;
            event.currentTarget.releasePointerCapture?.(event.pointerId);
        },
        onPointerCancel: () => { dragRef.current = null; },
        onClick: () => {
            if (draggedRef.current) { draggedRef.current = false; return; }
            toggle();
        },
        onKeyDown: (event) => {
            if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return;
            event.preventDefault();
            const current = panelRef.current?.getBoundingClientRect().height ?? window.innerHeight * 0.45;
            resizeDrawer(current + (event.key === 'ArrowUp' ? 1 : -1) * window.innerHeight * 0.1);
        },
    };

    if (isCollapsed && !narrow) {
        return (
            <aside className="edu-lesson-panel is-collapsed" aria-label={t('panel.label')}>
                <button type="button" className="df-icon-button" onClick={toggle} aria-label={t('panel.expand')} title={t('panel.expand')}>
                    <PanelRightOpen size={15} aria-hidden="true" />
                </button>
                <span className="edu-rail-step" aria-hidden="true">{phase === 'steps' ? `${stepIndex + 1}/${total}` : phase === 'quiz' ? 'Q' : phase === 'recap' ? '★' : 'W'}</span>
                <span className={`edu-rail-status${passing ? ' is-done' : ''}`} aria-hidden="true">{passing ? '✓' : '○'}</span>
                <span className="edu-sr-only" role="status">{passing ? t('panel.stepComplete') : t('panel.step', { index: stepIndex + 1, total })}</span>
            </aside>
        );
    }

    let body = null;
    if (phase === 'warmup') {
        body = <QuizView key={`warmup-${warmupIndex}`} kind="warmup" items={module.warmup ?? []} index={warmupIndex} records={record?.warmup ?? {}} facts={facts}
            onAnswer={(id, option) => session.answerQuiz(id, option)} onNext={() => session.nextWarmup()}
            onBack={warmupIndex > 0 ? () => session.back() : null} onSkip={() => session.skipWarmup()} />;
    } else if (phase === 'quiz') {
        body = <QuizView key={`quiz-${quizIndex}`} kind="quiz" items={module.quiz ?? []} index={quizIndex} records={record?.quiz ?? {}} facts={facts}
            onAnswer={(id, option) => session.answerQuiz(id, option)} onNext={() => session.nextQuiz()} onBack={() => session.back()} />;
    } else if (phase === 'recap') {
        const next = nextModule(module.id);
        body = <RecapView module={module} next={next} facts={facts} complete={isModuleComplete(progress, module)}
            onNext={() => session.startModule(next.id)} onLearn={exit} />;
    } else if (step) {
        body = <StepView key={`${module.id}-${step.id}`} module={module} step={step} index={stepIndex} total={total} record={record?.steps?.[step.id] ?? null}
            answers={record?.answers ?? {}} runTags={record?.runTags ?? {}} runtime={runtime} result={result} facts={facts} isLast={stepIndex === total - 1}
            updated={!!record?.migratedFrom && !Object.values(record.steps ?? {}).some((item) => ['passed', 'skipped'].includes(item.status))} />;
    }

    const countText = phase === 'steps' ? t('panel.step', { index: stepIndex + 1, total })
        : t(phase === 'warmup' ? 'warmup.heading' : phase === 'quiz' ? 'quiz.heading' : 'recap.heading');
    const shortCount = phase === 'steps' ? `${stepIndex + 1}/${total}${passing ? ' ✓' : ''}` : countText;
    const drawerStyle = narrow && !isCollapsed && drawerHeight ? { height: drawerHeight, maxHeight: 'none' } : undefined;

    return (
        <aside ref={panelRef} className={`edu-lesson-panel${narrow ? ' is-drawer' : ''}${isCollapsed ? ' is-closed' : ''}`} aria-label={t('panel.label')}
            style={drawerStyle}>
            {narrow && (
                <div className="edu-drawer-bar">
                    <button type="button" className="edu-drawer-handle" aria-expanded={!isCollapsed} title={t('panel.drawerHint')} {...handleEvents}>
                        <span className="edu-drawer-grip" aria-hidden="true" />
                        <span className="edu-drawer-title">
                            {isCollapsed ? t('panel.drawerClosed', { count: shortCount, title: t(`${module.id}.title`) }) : `${t('panel.drawer')} · ${moduleTitle(module)}`}
                        </span>
                        <ChevronUp size={14} aria-hidden="true" className="edu-drawer-chevron" />
                    </button>
                    {onDocument && <HistoryButtons />}
                    <button type="button" className="df-icon-button" onClick={exit} aria-label={t('panel.exit')} title={t('panel.exit')}>
                        <X size={15} aria-hidden="true" />
                    </button>
                </div>
            )}
            {!isCollapsed && (
                <>
                    {!narrow && (
                        <header className="edu-panel-header">
                            <p className="edu-panel-module">{moduleTitle(module)}</p>
                            {onDocument && <HistoryButtons />}
                            <button type="button" className="df-icon-button" onClick={toggle} aria-label={t('panel.collapse')} title={t('panel.collapse')}>
                                <Minus size={15} aria-hidden="true" />
                            </button>
                            <button type="button" className="df-icon-button" onClick={exit} aria-label={t('panel.exit')} title={t('panel.exit')}>
                                <X size={15} aria-hidden="true" />
                            </button>
                        </header>
                    )}
                    <div className="edu-panel-progress">
                        <p className="edu-panel-count">{countText}</p>
                        <ProgressSegments module={module} record={record} stepIndex={stepIndex} phase={phase} onGo={(index) => session.goToStep(index)} />
                    </div>
                    <div className="edu-panel-body">{body}</div>
                </>
            )}
        </aside>
    );
}
