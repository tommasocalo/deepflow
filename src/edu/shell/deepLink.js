import { getModule } from '../lessons/engine/registry.js';
import { useLessonSession } from '../lessons/engine/session.js';
import { useEduUiStore } from './uiStore.js';

/**
 * Deep links (docs/edu/SPEC.md §6.12): `#/learn` opens Learn, `#/learn/m3` starts or resumes module 3 and
 * `#/learn/m3/s4` opens step 4 (earlier unseen steps are marked visited, not passed). While a lesson runs, the
 * hash follows the current step through history.replaceState.
 */
export function parseLearnHash(hash) {
    const match = /^#\/learn(?:\/([\w-]+)(?:\/([\w-]+))?)?\/?$/.exec(hash ?? '');
    if (!match) return null;
    return { moduleId: match[1] ?? null, stepId: match[2] ?? null };
}

export function learnHash(moduleId, stepId) {
    if (!moduleId) return '#/learn';
    return stepId ? `#/learn/${moduleId}/${stepId}` : `#/learn/${moduleId}`;
}

/** Apply a hash; returns a promise that settles when the lesson (if any) has opened. */
export async function applyDeepLink(hash) {
    const link = parseLearnHash(hash);
    if (!link) return false;
    if (!link.moduleId) {
        useEduUiStore.getState().setTab('learn');
        return true;
    }
    const module = getModule(link.moduleId);
    if (!module) {
        useEduUiStore.getState().setTab('learn');
        return false;
    }
    const session = useLessonSession.getState();
    const stepId = link.stepId && module.steps.some((step) => step.id === link.stepId) ? link.stepId : undefined;
    const current = session.mode === 'lesson' && session.moduleId === module.id ? session.currentStep()?.id : null;
    if (current && (!stepId || stepId === current) && session.phase === 'steps') return true;
    await session.startModule(module.id, stepId ? { stepId } : {});
    return true;
}

function currentHash(state) {
    if (state.mode !== 'lesson' || !state.moduleId) return null;
    if (state.phase !== 'steps') return learnHash(state.moduleId);
    return learnHash(state.moduleId, getModule(state.moduleId)?.steps?.[state.stepIndex]?.id);
}

/** Write the hash for the current position; outside a lesson, a stale #/learn/… hash is removed. */
export function syncHash(state = useLessonSession.getState()) {
    if (typeof window === 'undefined' || !window.history?.replaceState) return;
    const next = currentHash(state);
    const { pathname, search, hash } = window.location;
    if (next && hash !== next) window.history.replaceState(window.history.state, '', `${pathname}${search}${next}`);
    else if (!next && /^#\/learn\/./.test(hash)) window.history.replaceState(window.history.state, '', `${pathname}${search}`);
}

/** Apply the startup hash, follow later hash changes and keep the hash in sync. Returns a cleanup function. */
export function installDeepLinks() {
    if (typeof window === 'undefined') return () => {};
    let applying = applyDeepLink(window.location.hash);
    const onHashChange = () => { applying = applyDeepLink(window.location.hash); };
    window.addEventListener('hashchange', onHashChange);
    const unsubscribe = useLessonSession.subscribe((state, previous) => {
        if (state.mode !== previous.mode || state.moduleId !== previous.moduleId || state.stepIndex !== previous.stepIndex || state.phase !== previous.phase) {
            Promise.resolve(applying).finally(() => syncHash(useLessonSession.getState()));
        }
    });
    Promise.resolve(applying).finally(() => syncHash(useLessonSession.getState()));
    return () => {
        window.removeEventListener('hashchange', onHashChange);
        unsubscribe();
    };
}
