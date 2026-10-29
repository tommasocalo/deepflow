import { useRef } from 'react';
import { ChevronUp, Minus, PanelRightOpen, X } from 'lucide-react';
import { factValues } from '../lessons/engine/context.js';
import { t } from '../lessons/engine/i18n.js';
import { isModuleComplete } from '../lessons/engine/progress.js';
import { getModule, nextModule } from '../lessons/engine/registry.js';
import { useLessonSession } from '../lessons/engine/session.js';
import { useEduUiStore } from '../shell/uiStore.js';
import QuizView from './QuizView.jsx';
import RecapView from './RecapView.jsx';
import StepView from './StepView.jsx';
import { moduleTitle } from './format.js';
import { CHECK_DELAY, useMediaQuery, useModules, useStepEvaluation } from './hooks.js';
import './lesson-panel.css';

const NARROW = '(max-width: 899px)';

function segmentStatus(record, step, current) {
    const status = record?.steps?.[step.id]?.status;
    if (current) return 'current';
    return status === 'passed' || status === 'skipped' || status === 'visited' ? status : 'todo';
}

function ProgressSegments({ module, record, stepIndex, phase, onGo }) {
    return (
        <ol className="edu-segments" aria-label={t('panel.progress')}>
            {module.steps.map((step, index) => {
                const current = phase === 'steps' && index === stepIndex;
                const status = segmentStatus(record, step, current);
                const label = t('panel.segment', { index: index + 1, title: t(`${module.id}.${step.id}.title`), status: t(`panel.segment.${status}`) });
                return (
                    <li key={step.id}>
                        <button type="button" className={`edu-segment is-${status}`} aria-label={label} title={`${t(`subgoal.${step.subgoal}`)} · ${t(`${module.id}.${step.id}.title`)}`}
                            aria-current={current ? 'step' : undefined} onClick={() => onGo(index)} />
                    </li>
                );
            })}
        </ol>
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
    const narrow = useMediaQuery(NARROW);
    const panelRef = useRef(null);
    const module = getModule(moduleId);
    const step = module?.steps?.[stepIndex] ?? null;
    const result = useStepEvaluation(`${moduleId}:${phase}:${stepIndex}`, { enabled: !!module && phase === 'steps' && !!step, delay: checkDelay });
    const session = useLessonSession.getState();

    if (!module) {
        return <aside className="edu-lesson-panel" aria-label={t('panel.label')}><p className="edu-muted">{t('panel.noModule')}</p></aside>;
    }
    const record = progress.modules?.[module.id] ?? null;
    const facts = factValues(module);
    const total = module.steps.length;
    const passing = !!result?.pass;
    const isCollapsed = narrow ? !drawerOpen : collapsed;
    const toggle = () => (narrow ? useEduUiStore.getState().setDrawerOpen(!drawerOpen) : session.setCollapsed(!collapsed));

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
        body = <QuizView kind="warmup" items={module.warmup ?? []} index={warmupIndex} records={record?.warmup ?? {}} facts={facts}
            onAnswer={(id, option) => session.answerQuiz(id, option)} onNext={() => session.nextWarmup()}
            onBack={warmupIndex > 0 ? () => session.back() : null} onSkip={() => session.skipWarmup()} />;
    } else if (phase === 'quiz') {
        body = <QuizView kind="quiz" items={module.quiz ?? []} index={quizIndex} records={record?.quiz ?? {}} facts={facts}
            onAnswer={(id, option) => session.answerQuiz(id, option)} onNext={() => session.nextQuiz()} onBack={() => session.back()} />;
    } else if (phase === 'recap') {
        const next = nextModule(module.id);
        body = <RecapView module={module} next={next} facts={facts} complete={isModuleComplete(progress, module)}
            onNext={() => session.startModule(next.id)} onLearn={() => session.exitLesson({ tab: 'learn' })} />;
    } else if (step) {
        body = <StepView key={`${module.id}-${step.id}`} module={module} step={step} index={stepIndex} total={total} record={record?.steps?.[step.id] ?? null}
            answers={record?.answers ?? {}} runtime={runtime} result={result} facts={facts} isLast={stepIndex === total - 1} />;
    }

    return (
        <aside ref={panelRef} className={`edu-lesson-panel${narrow ? ' is-drawer' : ''}${isCollapsed ? ' is-closed' : ''}`} aria-label={t('panel.label')}>
            {narrow && (
                <button type="button" className="edu-drawer-handle" onClick={toggle} aria-expanded={!isCollapsed} aria-label={t('panel.drawer')}>
                    <span className="edu-drawer-grip" aria-hidden="true" />
                    <span>{t('panel.drawer')} · {moduleTitle(module)}</span>
                    <ChevronUp size={14} aria-hidden="true" className="edu-drawer-chevron" />
                </button>
            )}
            {!isCollapsed && (
                <>
                    <header className="edu-panel-header">
                        <p className="edu-panel-module">{moduleTitle(module)}</p>
                        {!narrow && (
                            <button type="button" className="df-icon-button" onClick={toggle} aria-label={t('panel.collapse')} title={t('panel.collapse')}>
                                <Minus size={15} aria-hidden="true" />
                            </button>
                        )}
                        <button type="button" className="df-icon-button" onClick={() => session.exitLesson({ tab: 'learn' })} aria-label={t('panel.exit')} title={t('panel.exit')}>
                            <X size={15} aria-hidden="true" />
                        </button>
                    </header>
                    <div className="edu-panel-progress">
                        <p className="edu-panel-count">{phase === 'steps' ? t('panel.step', { index: stepIndex + 1, total }) : t(phase === 'warmup' ? 'warmup.heading' : phase === 'quiz' ? 'quiz.heading' : 'recap.heading')}</p>
                        <ProgressSegments module={module} record={record} stepIndex={stepIndex} phase={phase} onGo={(index) => session.goToStep(index)} />
                    </div>
                    <div className="edu-panel-body">{body}</div>
                </>
            )}
        </aside>
    );
}
