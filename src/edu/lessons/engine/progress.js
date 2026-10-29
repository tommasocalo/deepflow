import { CURRICULUM_VERSION } from '../../config.js';

/**
 * Lesson progress (docs/edu/SPEC.md §6.8.2) as pure reducers: every function returns a new progress object and
 * never mutates its input. The session store persists the result under deepflow-edu:progress:v1.
 */
export const PROGRESS_VERSION = 1;
export const STEP_STATUSES = Object.freeze(['visited', 'passed', 'skipped']);

export function emptyProgress() {
    return { version: PROGRESS_VERSION, curriculumVersion: CURRICULUM_VERSION, lastModuleId: null, modules: {} };
}

export function emptyModuleRecord(moduleVersion = 1) {
    return { moduleVersion, status: 'not-started', currentStepId: null, steps: {}, answers: {}, quiz: {}, warmup: {}, runTags: {} };
}

function moduleRecord(progress, moduleId) {
    return progress?.modules?.[moduleId] ?? emptyModuleRecord();
}

function withModule(progress, moduleId, update) {
    const base = progress ?? emptyProgress();
    const current = moduleRecord(base, moduleId);
    return { ...base, modules: { ...base.modules, [moduleId]: update(current) } };
}

function withStep(record, stepId, update) {
    const step = record.steps?.[stepId] ?? { status: 'visited', hintLevel: 0 };
    return { ...record, steps: { ...record.steps, [stepId]: update(step) } };
}

export function startModule(progress, moduleId, moduleVersion = 1) {
    const next = withModule(progress, moduleId, (record) => ({
        ...record,
        moduleVersion: record.moduleVersion ?? moduleVersion,
        status: record.status === 'completed' ? 'completed' : 'in-progress',
    }));
    return { ...next, lastModuleId: moduleId };
}

/** First entry marks the step visited; a passed or skipped step keeps its status. */
export function enterStep(progress, moduleId, stepId, now = Date.now()) {
    return withModule(progress, moduleId, (record) => ({
        ...withStep(record, stepId, (step) => ({ ...step, enteredAt: step.enteredAt ?? now })),
        status: record.status === 'completed' ? 'completed' : 'in-progress',
        currentStepId: stepId,
    }));
}

/** Mark steps visited without entering them (deep links skip ahead). */
export function visitSteps(progress, moduleId, stepIds) {
    return withModule(progress, moduleId, (record) => {
        let next = record;
        for (const stepId of stepIds) {
            if (!next.steps?.[stepId]) next = withStep(next, stepId, (step) => ({ ...step, status: 'visited' }));
        }
        return next;
    });
}

export function passStep(progress, moduleId, stepId, now = Date.now()) {
    return withModule(progress, moduleId, (record) => withStep(record, stepId, (step) => (
        step.status === 'passed' ? step : { ...step, status: 'passed', passedAt: now }
    )));
}

export function skipStep(progress, moduleId, stepId, now = Date.now()) {
    return withModule(progress, moduleId, (record) => withStep(record, stepId, (step) => (
        step.status === 'passed' ? step : { ...step, status: 'skipped', skippedAt: now }
    )));
}

/** Hint levels only rise: 1 = first hint, 2 = second hint, 3 = Show me. */
export function raiseHint(progress, moduleId, stepId, level) {
    return withModule(progress, moduleId, (record) => withStep(record, stepId, (step) => ({
        ...step, hintLevel: Math.max(step.hintLevel ?? 0, Math.min(3, level)),
    })));
}

export function recordAnswer(progress, moduleId, id, value, correct, { now = Date.now(), source = 'learner' } = {}) {
    return withModule(progress, moduleId, (record) => ({
        ...record,
        answers: { ...record.answers, [id]: { value, correct: !!correct, at: now, ...(source !== 'learner' ? { source } : {}) } },
    }));
}

export function clearAnswer(progress, moduleId, id) {
    return withModule(progress, moduleId, (record) => {
        if (!record.answers?.[id]) return record;
        const { [id]: _removed, ...answers } = record.answers;
        return { ...record, answers };
    });
}

/** The first answer and its correctness are kept; later attempts only count. */
function recordChoice(bucket) {
    return (progress, moduleId, id, option, correct) => withModule(progress, moduleId, (record) => {
        const previous = record[bucket]?.[id];
        const entry = previous
            ? { ...previous, attempts: (previous.attempts ?? 1) + 1, last: option, solved: previous.solved || !!correct }
            : { first: option, correct: !!correct, attempts: 1, last: option, solved: !!correct };
        return { ...record, [bucket]: { ...record[bucket], [id]: entry } };
    });
}
export const recordQuiz = recordChoice('quiz');
export const recordWarmup = recordChoice('warmup');

export function skipWarmup(progress, moduleId) {
    return withModule(progress, moduleId, (record) => ({ ...record, warmupSkipped: true }));
}

export function tagRun(progress, moduleId, tag, metrics) {
    return withModule(progress, moduleId, (record) => ({ ...record, runTags: { ...record.runTags, [tag]: metrics } }));
}

export function completeModule(progress, moduleId, now = Date.now()) {
    return withModule(progress, moduleId, (record) => ({ ...record, status: 'completed', completedAt: record.completedAt ?? now }));
}

/** Every step passed or skipped and every quiz item answered. */
export function isModuleComplete(progress, module) {
    const record = progress?.modules?.[module.id];
    if (!record) return false;
    const steps = module.steps ?? [];
    const quiz = module.quiz ?? [];
    return steps.every((step) => ['passed', 'skipped'].includes(record.steps?.[step.id]?.status))
        && quiz.every((item) => record.quiz?.[item.id]);
}

/** Learn-home summary of one module. */
export function moduleStatus(progress, module) {
    const record = progress?.modules?.[module.id] ?? null;
    const steps = module.steps ?? [];
    const done = steps.filter((step) => ['passed', 'skipped'].includes(record?.steps?.[step.id]?.status)).length;
    const status = record?.status ?? 'not-started';
    const currentIndex = Math.max(0, steps.findIndex((step) => step.id === record?.currentStepId));
    const firstOpen = steps.findIndex((step) => !['passed', 'skipped'].includes(record?.steps?.[step.id]?.status));
    return { status, done, total: steps.length, currentStepId: record?.currentStepId ?? null, currentIndex, firstOpenIndex: firstOpen, record };
}

/**
 * Bring stored progress up to date with the installed modules. When a module's version changed, step records
 * whose ids still exist are kept, the others dropped, and `currentStepId` is dropped if its step is gone.
 */
export function migrate(progress, modules = []) {
    if (!progress || typeof progress !== 'object' || progress.version !== PROGRESS_VERSION || typeof progress.modules !== 'object') {
        return emptyProgress();
    }
    const next = { ...emptyProgress(), ...progress, modules: { ...progress.modules } };
    for (const module of modules) {
        const record = next.modules[module.id];
        if (!record) continue;
        if ((record.moduleVersion ?? 1) === (module.version ?? 1)) continue;
        const ids = new Set((module.steps ?? []).map((step) => step.id));
        const steps = Object.fromEntries(Object.entries(record.steps ?? {}).filter(([id]) => ids.has(id)));
        const completed = record.status === 'completed' && isModuleComplete({ modules: { [module.id]: { ...record, steps } } }, module);
        next.modules[module.id] = {
            ...emptyModuleRecord(module.version ?? 1),
            ...record,
            moduleVersion: module.version ?? 1,
            steps,
            currentStepId: ids.has(record.currentStepId) ? record.currentStepId : null,
            status: completed ? 'completed' : Object.keys(steps).length ? 'in-progress' : record.status === 'not-started' ? 'not-started' : 'in-progress',
        };
    }
    if (next.lastModuleId && !modules.some((module) => module.id === next.lastModuleId) && modules.length) next.lastModuleId = null;
    return next;
}
