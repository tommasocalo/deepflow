import { CURRICULUM_VERSION } from '../../config.js';

/**
 * Lesson progress (docs/edu/SPEC.md §6.8.2) as pure reducers: every function returns a new progress object and
 * never mutates its input. The session store persists the result under deepflow-edu:progress:v1.
 */
export const PROGRESS_VERSION = 1;
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

/**
 * Steps a jump or Skip filled in: the lesson applied their Show me edits so later steps open on the right model. They
 * keep their status (visited, not passed) and are never filled in twice.
 */
export function fillSteps(progress, moduleId, stepIds, now = Date.now()) {
    return withModule(progress, moduleId, (record) => {
        let next = record;
        for (const stepId of stepIds) next = withStep(next, stepId, (step) => ({ ...step, filledIn: step.filledIn ?? now }));
        return next;
    });
}

export function passStep(progress, moduleId, stepId, now = Date.now()) {
    return withModule(progress, moduleId, (record) => withStep(record, stepId, (step) => (
        step.status === 'passed' ? step : { ...step, status: 'passed', passedAt: now }
    )));
}

/** Reset step: a passed or skipped step is open again (its entry time and hint level stay). */
export function reopenStep(progress, moduleId, stepId) {
    return withModule(progress, moduleId, (record) => {
        const step = record.steps?.[stepId];
        if (!step || !['passed', 'skipped'].includes(step.status)) return record;
        const { passedAt: _passed, skippedAt: _skipped, ...rest } = step;
        return { ...record, steps: { ...record.steps, [stepId]: { ...rest, status: 'visited' } } };
    });
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

const readable = (progress) => !!progress && typeof progress === 'object' && progress.version === PROGRESS_VERSION && typeof progress.modules === 'object';

/** The modules whose stored records were made for another version of the module (see migrate). */
export function updatedModules(progress, modules = []) {
    if (!readable(progress)) return [];
    return modules.filter((module) => progress.modules[module.id] && (progress.modules[module.id].moduleVersion ?? 1) !== (module.version ?? 1));
}

/**
 * Bring stored progress up to date with the installed modules. A new module version may renumber its steps (review
 * r2: M5 version 2 put "The final score" at s5 and moved Dropout to s6), so the old step records, current step and
 * completion no longer say which step they belong to: they are dropped, and the module's steps start again. What
 * names its question rather than its place is kept while its id exists: prediction answers, quiz and warm-up
 * results, and run tags. `migratedFrom` keeps the old version.
 */
export function migrate(progress, modules = []) {
    if (!readable(progress)) return emptyProgress();
    const next = { ...emptyProgress(), ...progress, modules: { ...progress.modules } };
    const keep = (entries, ids) => Object.fromEntries(Object.entries(entries ?? {}).filter(([id]) => ids.has(id)));
    for (const module of updatedModules(progress, modules)) {
        const record = next.modules[module.id];
        const steps = module.steps ?? [];
        const answers = keep(record.answers, new Set(steps.map((step) => step.predict?.id).filter(Boolean)));
        const quiz = keep(record.quiz, new Set((module.quiz ?? []).map((item) => item.id)));
        const warmup = keep(record.warmup, new Set((module.warmup ?? []).map((item) => item.id)));
        const runTags = keep(record.runTags, new Set(steps.map((step) => step.tagRun).filter(Boolean)));
        const started = (record.status ?? 'not-started') !== 'not-started' || [answers, quiz, warmup].some((entries) => Object.keys(entries).length);
        next.modules[module.id] = {
            ...emptyModuleRecord(module.version ?? 1),
            answers, quiz, warmup, runTags,
            ...(record.warmupSkipped ? { warmupSkipped: true } : {}),
            status: started ? 'in-progress' : 'not-started',
            migratedFrom: record.moduleVersion ?? 1,
        };
    }
    if (next.lastModuleId && !modules.some((module) => module.id === next.lastModuleId) && modules.length) next.lastModuleId = null;
    return next;
}
