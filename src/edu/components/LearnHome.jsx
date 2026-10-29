import { useEffect, useRef } from 'react';
import { Download, Play, RotateCcw, SquareDashedMousePointer } from 'lucide-react';
import { t } from '../lessons/engine/i18n.js';
import { learningLogExport } from '../lessons/engine/log.js';
import { moduleStatus } from '../lessons/engine/progress.js';
import { useLessonSession } from '../lessons/engine/session.js';
import ModuleCard from './ModuleCard.jsx';
import { useModules, useStorageDegraded } from './hooks.js';
import './learn.css';

function download(name, text) {
    const blob = new Blob([text], { type: 'application/json' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}

/** Which module "Continue" opens, and its label. */
function continueTarget(modules, progress, session) {
    if (!modules.length) return null;
    const last = modules.find((module) => module.id === progress.lastModuleId);
    const title = (module) => t(`${module.id}.title`);
    if (!last) return { module: modules[0], label: t('learn.start', { number: modules[0].order, title: title(modules[0]) }) };
    const summary = moduleStatus(progress, last);
    if (summary.status === 'completed') {
        const next = modules.find((module) => moduleStatus(progress, module).status !== 'completed');
        if (next && next !== last) {
            const started = moduleStatus(progress, next).status === 'in-progress';
            return { module: next, label: started ? continueLabel(next, moduleStatus(progress, next), session) : t('learn.start', { number: next.order, title: title(next) }) };
        }
        return { module: last, label: t('learn.allDone', { number: last.order, title: title(last) }) };
    }
    return { module: last, label: continueLabel(last, summary, session) };
}

function continueLabel(module, summary, session) {
    const title = t(`${module.id}.title`);
    if (session.moduleId === module.id && session.phase === 'quiz') return t('learn.continueQuiz', { number: module.order, title });
    if (session.moduleId === module.id && session.phase === 'recap') return t('learn.continueRecap', { number: module.order, title });
    return t('learn.continue', { number: module.order, title, step: summary.currentIndex + 1, total: summary.total });
}

let shownBefore = false;

/** The Learn home (docs/edu/SPEC.md §3.2): the curriculum, progress, Continue and the way into the sandbox. */
export default function LearnHome() {
    const headingRef = useRef(null);
    // Coming back from a lesson (Exit, Back to Learn) unmounts the control that had focus: continue from the heading.
    useEffect(() => {
        const active = document.activeElement;
        if (shownBefore && (!active || active === document.body)) headingRef.current?.focus({ preventScroll: true });
        shownBefore = true;
    }, []);
    const modules = useModules();
    const progress = useLessonSession((state) => state.progress);
    const moduleId = useLessonSession((state) => state.moduleId);
    const phase = useLessonSession((state) => state.phase);
    const busy = useLessonSession((state) => state.runtime.busy);
    const degraded = useStorageDegraded();
    const session = useLessonSession.getState();
    const target = continueTarget(modules, progress, { moduleId, phase });

    const open = (module) => { if (!busy) session.startModule(module.id); };
    const reset = () => {
        if (window.confirm(t('learn.resetConfirm'))) session.resetProgress();
    };
    const exportLog = () => download('deepflow-edu-log.json', JSON.stringify(learningLogExport(progress), null, 2));

    return (
        <div className="edu-learn" data-edu-learn="">
            <div className="edu-learn-inner">
                <header className="edu-learn-header">
                    <h1 ref={headingRef} tabIndex={-1}>{t('learn.title')}</h1>
                    <p className="edu-learn-subtitle">{modules.length ? t(modules.length === 1 ? 'learn.subtitleOne' : 'learn.subtitle', { count: modules.length }) : t('learn.empty')}</p>
                    <div className="edu-learn-actions">
                        {target && (
                            <button type="button" className="df-button df-button-primary edu-hard edu-continue" onClick={() => open(target.module)} disabled={busy}>
                                <Play size={13} strokeWidth={2.5} aria-hidden="true" />{target.label}
                            </button>
                        )}
                        <button type="button" className="df-button" onClick={() => session.openSandbox()} disabled={busy}>
                            <SquareDashedMousePointer size={13} strokeWidth={2.5} aria-hidden="true" />{t('learn.sandbox')}
                        </button>
                    </div>
                    {degraded && <p className="edu-callout edu-callout-warning" role="status">{t('learn.degraded')}</p>}
                </header>

                {modules.length > 0 ? (
                    <div className="edu-module-grid" role="list" aria-label={t('learn.modules')}>
                        {modules.map((module) => {
                            const after = (module.recommendedAfter ?? []).map((id) => modules.find((item) => item.id === id))
                                .find((item) => item && moduleStatus(progress, item).status !== 'completed');
                            return (
                                <div role="listitem" key={module.id}>
                                    <ModuleCard module={module} summary={moduleStatus(progress, module)} recommendedAfter={after?.order ?? null} onOpen={open} />
                                </div>
                            );
                        })}
                    </div>
                ) : (
                    <p className="edu-learn-empty" role="status">{t('learn.noModules')}</p>
                )}

                <footer className="edu-learn-footer">
                    <span>{t('learn.saved')}</span>
                    <button type="button" className="edu-link-button" onClick={reset} disabled={busy}><RotateCcw size={12} strokeWidth={2.5} aria-hidden="true" />{t('learn.reset')}</button>
                    <button type="button" className="edu-link-button" onClick={exportLog}><Download size={12} strokeWidth={2.5} aria-hidden="true" />{t('learn.download')}</button>
                </footer>
            </div>
        </div>
    );
}
