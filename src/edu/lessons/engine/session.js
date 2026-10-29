import { create } from 'zustand';
import { useArchitectureHistory } from '../../../stores/architectureHistory.js';
import { callSource, onSourceRegistered, readSource, registerSource, subscribeSource } from '../../bridge.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { runActions } from './actions.js';
import { requestFit } from './canvasBridge.js';
import { completedEpochs, evaluateStep, finalMetrics, trainerRuns } from './checks.js';
import { buildCheckContext, factValues, subscribeCheckContext } from './context.js';
import { logEvent } from './log.js';
import { gradePrediction } from './predict.js';
import * as P from './progress.js';
import { getModule, getModules, stepIndexById } from './registry.js';
import { eduKey, eduKeys, readJSON, removeKey, writeJSON } from './storage.js';
import {
    captureWorkspace, clearStepEntry, loadFixtureIntoWorkspace, loadSlot, loadStepEntry, restoreWorkspace, runStoryboard, saveSlot,
    saveStepEntry, slotKey,
} from './workspace.js';

/**
 * The lesson session (docs/edu/SPEC.md §6.7): sandbox or lesson mode, the position in the current module, and
 * lesson progress. Actions are async because a step's setup and Show me can load fixtures and train.
 */
export const SESSION_KEY = eduKey('session:v1');
export const PROGRESS_KEY = eduKey('progress:v1');
const PHASES = ['warmup', 'steps', 'quiz', 'recap'];
const PERSISTED = ['mode', 'moduleId', 'phase', 'stepIndex', 'quizIndex', 'warmupIndex', 'collapsed'];
const freshRuntime = (overrides = {}) => ({ enteredAtSeq: 0, visitedTabs: [], revealShown: false, busy: false, error: null, showMe: null, ...overrides });

function trainerSeq() {
    return readSource('trainer')?.seq ?? 0;
}

function loadSession() {
    const saved = readJSON(SESSION_KEY, null) ?? {};
    const session = {
        mode: saved.mode === 'lesson' ? 'lesson' : 'sandbox',
        moduleId: typeof saved.moduleId === 'string' ? saved.moduleId : null,
        phase: PHASES.includes(saved.phase) ? saved.phase : 'steps',
        stepIndex: Number.isInteger(saved.stepIndex) && saved.stepIndex >= 0 ? saved.stepIndex : 0,
        quizIndex: Number.isInteger(saved.quizIndex) && saved.quizIndex >= 0 ? saved.quizIndex : 0,
        warmupIndex: Number.isInteger(saved.warmupIndex) && saved.warmupIndex >= 0 ? saved.warmupIndex : 0,
        collapsed: !!saved.collapsed,
    };
    const module = session.moduleId ? getModule(session.moduleId) : null;
    if (session.mode === 'lesson' && !module) return { ...session, mode: 'sandbox', moduleId: null };
    if (module) session.stepIndex = Math.min(session.stepIndex, Math.max(0, module.steps.length - 1));
    return session;
}

function initialState() {
    const session = loadSession();
    const progress = P.migrate(readJSON(PROGRESS_KEY, null), getModules());
    return { ...session, progress, headless: false, runtime: freshRuntime({ visitedTabs: [useEduUiStore.getState().activeTab] }) };
}

export const useLessonSession = create((set, get) => {
    const module = () => getModule(get().moduleId);
    const step = () => module()?.steps?.[get().stepIndex] ?? null;
    const record = () => get().progress.modules?.[get().moduleId] ?? null;
    const updateProgress = (reducer) => set((state) => ({ progress: reducer(state.progress) }));
    const updateRuntime = (patch) => set((state) => ({ runtime: { ...state.runtime, ...patch } }));
    const log = (type, extra = {}) => logEvent(type, { module: get().moduleId, ...extra });
    const actionContext = (label) => ({
        module: module(),
        headless: get().headless,
        label,
        answer: (id, value) => get().submitPrediction(id, value, { source: 'show-me' }),
    });
    const fitSoon = () => { if (!get().headless) setTimeout(requestFit, 60); };

    /** Leave the learner on a tab where the panel is visible. */
    const showStepTab = (target) => {
        const ui = useEduUiStore.getState();
        if (target?.tab) ui.setTab(target.tab);
        else if (ui.activeTab === 'learn') ui.setTab('architecture');
    };

    async function enterStep(index, { runSetup = true } = {}) {
        const current = module();
        const target = current?.steps?.[index];
        if (!target) return false;
        const firstEntry = !record()?.steps?.[target.id];
        set({ phase: 'steps', stepIndex: index, runtime: freshRuntime({ busy: firstEntry && runSetup && !!target.setup?.length }) });
        if (firstEntry && runSetup && target.setup?.length) {
            const result = await runActions(target.setup, actionContext('step setup'));
            if (!result.ok) updateRuntime({ error: result.error });
        }
        showStepTab(target);
        const restored = !firstEntry && get().restoringEntry ? loadStepEntry(current.id, target.id) : null;
        if (!restored) saveStepEntry(current.id, target.id, captureWorkspace());
        updateProgress((progress) => P.enterStep(progress, current.id, target.id));
        updateRuntime({ enteredAtSeq: trainerSeq(), visitedTabs: [useEduUiStore.getState().activeTab], revealShown: false, busy: false });
        set({ restoringEntry: false });
        log('step-enter', { step: target.id });
        return true;
    }

    async function advance() {
        const current = module();
        if (!current) return;
        const index = get().stepIndex;
        if (index < current.steps.length - 1) {
            await enterStep(index + 1);
            return;
        }
        if (current.quiz?.length) set({ phase: 'quiz', quizIndex: 0, runtime: freshRuntime() });
        else get().finishModule();
    }

    return {
        ...initialState(),
        restoringEntry: true,

        setHeadless: (headless) => set({ headless: !!headless }),
        setCollapsed: (collapsed) => set({ collapsed: !!collapsed }),
        currentModule: module,
        currentStep: step,

        /** The step part of the CheckContext. */
        stepInfo: () => {
            const moduleRecord = record();
            const answers = { ...(moduleRecord?.warmup ?? {}), ...(moduleRecord?.quiz ?? {}), ...(moduleRecord?.answers ?? {}) };
            const tags = Object.fromEntries(Object.entries(moduleRecord?.runTags ?? {}).map(([tag, value]) => [tag, value?.runId ?? null]));
            const runtime = get().runtime;
            return { answers, enteredAtSeq: runtime.enteredAtSeq, visitedTabs: runtime.visitedTabs, tags };
        },
        /** Evaluate the current step against the live stores. */
        evaluate: () => {
            const current = module();
            const target = step();
            if (!current || !target) return { pass: false, detail: { key: 'panel.empty', params: {} } };
            return evaluateStep(target, buildCheckContext({ module: current, step: get().stepInfo(), facts: factValues(current) }));
        },
        trainLocked: () => {
            const target = step();
            if (get().mode !== 'lesson' || get().phase !== 'steps' || target?.locks?.gate !== 'train') return false;
            return !!target.predict && !record()?.answers?.[target.predict.id];
        },

        startModule: async (id, { stepId } = {}) => {
            const target = getModule(id);
            if (!target) return false;
            const state = get();
            if (state.runtime.busy) return false;
            const sameModule = state.mode === 'lesson' && state.moduleId === id;
            if (!sameModule) {
                saveSlot(state.mode === 'lesson' && state.moduleId ? state.moduleId : 'sandbox', captureWorkspace());
                set({ mode: 'lesson', moduleId: id, runtime: freshRuntime({ busy: true }) });
                const slot = loadSlot(id);
                try {
                    if (slot) restoreWorkspace(slot);
                    else await loadFixtureIntoWorkspace(target, target.start, { runAfter: async (after) => {
                        const result = await runActions(after, actionContext('fixture'));
                        if (!result.ok) throw new Error(result.error);
                    } });
                } catch (error) {
                    console.warn('[edu session] could not load the module workspace', error);
                    updateRuntime({ error: error.message });
                }
                fitSoon();
            }
            updateProgress((progress) => P.startModule(progress, id, target.version ?? 1));
            log('module-start');
            const summary = P.moduleStatus(get().progress, target);
            let index = stepId ? stepIndexById(target, stepId) : -1;
            if (index >= 0) {
                const skipped = target.steps.slice(0, index).map((item) => item.id);
                updateProgress((progress) => P.visitSteps(progress, id, skipped));
            } else if (summary.currentStepId) {
                index = Math.max(0, stepIndexById(target, summary.currentStepId));
            } else {
                index = summary.firstOpenIndex >= 0 ? summary.firstOpenIndex : 0;
            }
            const moduleRecord = get().progress.modules[id];
            const warmupPending = !stepId && (target.warmup?.length ?? 0) > 0 && !moduleRecord?.warmupSkipped
                && !Object.keys(moduleRecord?.warmup ?? {}).length && !Object.keys(moduleRecord?.steps ?? {}).length;
            set({ stepIndex: index, quizIndex: 0, warmupIndex: 0, restoringEntry: sameModule && get().restoringEntry });
            if (warmupPending) {
                set({ phase: 'warmup', runtime: freshRuntime() });
                showStepTab(null);
                return true;
            }
            await enterStep(index);
            return true;
        },

        goToStep: async (index) => {
            if (get().runtime.busy || !module()?.steps?.[index]) return false;
            return enterStep(index);
        },

        next: async () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return false;
            const status = record()?.steps?.[target.id]?.status;
            const result = get().evaluate();
            if (!result.pass && !['passed', 'skipped'].includes(status)) return false;
            if (result.pass && status !== 'passed') {
                updateProgress((progress) => P.passStep(progress, current.id, target.id));
                log('step-pass', { step: target.id });
                if (target.tagRun && result.runId) get().tagRun(target.tagRun, result.runId);
            }
            await advance();
            return true;
        },

        skip: async () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return false;
            updateProgress((progress) => P.skipStep(progress, current.id, target.id));
            log('step-skip', { step: target.id });
            await advance();
            return true;
        },

        back: async () => {
            const state = get();
            if (state.runtime.busy) return false;
            if (state.phase === 'quiz' && state.quizIndex > 0) { set({ quizIndex: state.quizIndex - 1 }); return true; }
            if (state.phase === 'quiz' || state.phase === 'recap') return enterStep(Math.max(0, (module()?.steps.length ?? 1) - 1), { runSetup: false });
            if (state.phase === 'warmup') { if (state.warmupIndex > 0) set({ warmupIndex: state.warmupIndex - 1 }); return true; }
            if (state.stepIndex === 0) return false;
            return enterStep(state.stepIndex - 1, { runSetup: false });
        },

        resetStep: () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return false;
            const snapshot = loadStepEntry(current.id, target.id);
            if (snapshot) restoreWorkspace(snapshot);
            if (target.predict) updateProgress((progress) => P.clearAnswer(progress, current.id, target.predict.id));
            updateRuntime({ enteredAtSeq: trainerSeq(), visitedTabs: [useEduUiStore.getState().activeTab], revealShown: false, error: null, showMe: null });
            log('reset-step', { step: target.id });
            fitSoon();
            return !!snapshot;
        },

        useHint: async () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return 0;
            const level = record()?.steps?.[target.id]?.hintLevel ?? 0;
            if (level >= 2) {
                await get().showMe();
                return 3;
            }
            updateProgress((progress) => P.raiseHint(progress, current.id, target.id, level + 1));
            log('hint', { step: target.id, data: { level: level + 1 } });
            return level + 1;
        },

        showMe: async () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return { ok: false };
            const undoBefore = useArchitectureHistory.getState().undoCount;
            updateRuntime({ busy: true, error: null });
            const result = await runActions(target.solution ?? [], actionContext('show me'));
            updateProgress((progress) => P.raiseHint(progress, current.id, target.id, 3));
            log('show-me', { step: target.id, data: { ok: result.ok } });
            const undoable = useArchitectureHistory.getState().undoCount > undoBefore;
            updateRuntime({ busy: false, error: result.ok ? null : result.error, showMe: { at: Date.now(), undoable } });
            fitSoon();
            return result;
        },

        /** A step's extra action button, e.g. M7 s7 "Load GPT-2 small". */
        runButton: async (buttonId) => {
            const current = module();
            const target = step();
            const button = target?.buttons?.find((item) => item.id === buttonId);
            if (!current || !button || get().runtime.busy) return { ok: false };
            updateRuntime({ busy: true, error: null });
            const result = await runActions(button.actions ?? [], actionContext(`button ${buttonId}`));
            log('step-button', { step: target.id, data: { id: buttonId, ok: result.ok } });
            updateRuntime({ busy: false, error: result.ok ? null : result.error });
            fitSoon();
            return result;
        },

        submitPrediction: (id, value, { source = 'learner' } = {}) => {
            const current = module();
            if (!current) return null;
            const owner = current.steps.find((item) => item.predict?.id === id);
            if (!owner) return null;
            if (record()?.answers?.[id]) return record().answers[id];
            const grade = gradePrediction(owner.predict, value);
            updateProgress((progress) => P.recordAnswer(progress, current.id, id, value, grade.correct, { source }));
            log('predict', { step: owner.id, data: { id, value, correct: grade.correct, source } });
            if (owner === step()) updateRuntime({ revealShown: true });
            return get().progress.modules[current.id].answers[id];
        },

        answerQuiz: (id, optionId) => {
            const current = module();
            if (!current) return null;
            const warmup = current.warmup?.find((item) => item.id === id);
            const item = warmup ?? current.quiz?.find((candidate) => candidate.id === id);
            if (!item || !item.options.includes(optionId)) return null;
            const correct = optionId === item.correct;
            const attempt = ((warmup ? record()?.warmup : record()?.quiz)?.[id]?.attempts ?? 0) + 1;
            updateProgress((progress) => (warmup ? P.recordWarmup : P.recordQuiz)(progress, current.id, id, optionId, correct));
            log(warmup ? 'warmup' : 'quiz', { data: { id, option: optionId, correct, attempt } });
            return { correct };
        },

        nextQuiz: () => {
            const current = module();
            if (!current) return;
            if (get().quizIndex < (current.quiz?.length ?? 0) - 1) set({ quizIndex: get().quizIndex + 1 });
            else get().finishModule();
        },

        nextWarmup: async () => {
            const current = module();
            if (!current) return;
            if (get().warmupIndex < (current.warmup?.length ?? 0) - 1) { set({ warmupIndex: get().warmupIndex + 1 }); return; }
            await enterStep(get().stepIndex);
        },

        skipWarmup: async () => {
            const current = module();
            if (!current) return;
            updateProgress((progress) => P.skipWarmup(progress, current.id));
            await enterStep(get().stepIndex);
        },

        finishModule: () => {
            const current = module();
            if (!current) return;
            if (P.isModuleComplete(get().progress, current) && record()?.status !== 'completed') {
                updateProgress((progress) => P.completeModule(progress, current.id));
                log('module-complete');
            }
            set({ phase: 'recap', runtime: freshRuntime() });
        },

        /** Remember a run's final metrics under a tag (M5 compares the Dropout run with the tagged overfit run). */
        tagRun: (tag, runId) => {
            const current = module();
            const run = trainerRuns(readSource('trainer')).find((item) => item.id === runId);
            if (!current || !run) return;
            const final = finalMetrics(run) ?? {};
            updateProgress((progress) => P.tagRun(progress, current.id, tag, {
                runId, epochs: completedEpochs(run), valLoss: final.valLoss ?? null, trainLoss: final.trainLoss ?? null,
                valAccuracy: final.valAccuracy ?? null, trainAccuracy: final.trainAccuracy ?? null, at: Date.now(),
            }));
            try { callSource('trainer', 'tagRun', runId, tag); } catch { /* tagging in the trainer is optional */ }
        },

        /** Save the lesson, return to the sandbox workspace and open `tab` (Learn by default). */
        exitLesson: ({ tab = 'learn' } = {}) => {
            const state = get();
            if (state.runtime.busy) return false;
            if (state.mode === 'lesson' && state.moduleId) {
                saveSlot(state.moduleId, captureWorkspace());
                const sandbox = loadSlot('sandbox');
                if (sandbox) restoreWorkspace(sandbox);
                else runStoryboard('gpt2');
                fitSoon();
            }
            set({ mode: 'sandbox', runtime: freshRuntime() });
            useEduUiStore.getState().setTab(tab);
            return true;
        },

        openSandbox: () => get().exitLesson({ tab: 'architecture' }),

        /** Learn home → Reset progress. The sandbox slot is kept. */
        resetProgress: () => {
            if (get().mode === 'lesson') get().exitLesson({ tab: 'learn' });
            for (const key of eduKeys()) {
                if (key.startsWith(eduKey('slot:')) && key !== slotKey('sandbox')) removeKey(key);
            }
            clearStepEntry();
            set({ progress: P.emptyProgress(), moduleId: null, phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0, runtime: freshRuntime() });
        },

        /** Re-read session and progress from storage (tests and module-registry swaps). */
        hydrate: () => set({ ...initialState(), restoringEntry: true }),
    };
});


// ---- persistence and live wiring ---------------------------------------------------------------------------------

const persistedSlice = (state) => Object.fromEntries(PERSISTED.map((key) => [key, state[key]]));
useLessonSession.subscribe((state, previous) => {
    if (PERSISTED.some((key) => state[key] !== previous[key])) writeJSON(SESSION_KEY, persistedSlice(state));
    if (state.progress !== previous.progress) writeJSON(PROGRESS_KEY, state.progress);
});

/** Visiting a tab counts for `{ visited }` checks; opening the code panel is logged. */
useEduUiStore.subscribe((state, previous) => {
    const session = useLessonSession.getState();
    if (state.activeTab !== previous.activeTab && session.mode === 'lesson' && !session.runtime.visitedTabs.includes(state.activeTab)) {
        useLessonSession.setState({ runtime: { ...session.runtime, visitedTabs: [...session.runtime.visitedTabs, state.activeTab] } });
    }
    if (state.codeOpen && (!previous.codeOpen || state.codeTab !== previous.codeTab) && session.mode === 'lesson') {
        logEvent('code-open', { module: session.moduleId, step: session.currentStep()?.id, data: { tab: state.codeTab } });
    }
});

/** Log each finished training run once (§6.8.4). */
const loggedRuns = new Set();
let unsubscribeTrainer = null;
function watchTrainer() {
    unsubscribeTrainer?.();
    unsubscribeTrainer = subscribeSource('trainer', () => {
        const session = useLessonSession.getState();
        for (const run of trainerRuns(readSource('trainer'))) {
            if (loggedRuns.has(run.id) || !['completed', 'stopped'].includes(run.status)) continue;
            loggedRuns.add(run.id);
            if (session.mode !== 'lesson') continue;
            const final = finalMetrics(run) ?? {};
            logEvent('run', { module: session.moduleId, step: session.currentStep()?.id, data: {
                dataset: run.datasetId ?? null, epochs: completedEpochs(run), finalValAccuracy: final.valAccuracy ?? null, finalValLoss: final.valLoss ?? null,
            } });
        }
    });
}
watchTrainer();
onSourceRegistered((name) => { if (name === 'trainer') watchTrainer(); });

/**
 * Read-only lesson state for other stages: the trainer UI can disable Train while a prediction gate is open,
 * wherever the Train button lives.
 */
const lessonListeners = new Set();
useLessonSession.subscribe(() => { for (const listener of lessonListeners) listener(); });
registerSource('lesson', {
    read: () => {
        const state = useLessonSession.getState();
        const target = state.currentStep();
        return {
            mode: state.mode, moduleId: state.moduleId, phase: state.phase, stepId: target?.id ?? null,
            trainLocked: state.trainLocked(), lockedReason: state.trainLocked() ? 'panel.trainLocked' : '',
        };
    },
    subscribe: (listener) => { lessonListeners.add(listener); return () => lessonListeners.delete(listener); },
});

/** subscribeCheckContext plus the session itself (answers, visited tabs, step changes). */
export function subscribeLessonContext(listener) {
    const unsubscribeContext = subscribeCheckContext(listener);
    const unsubscribeSession = useLessonSession.subscribe((state, previous) => {
        if (state.progress !== previous.progress || state.runtime !== previous.runtime || state.stepIndex !== previous.stepIndex
            || state.moduleId !== previous.moduleId || state.phase !== previous.phase) listener('session');
    });
    return () => { unsubscribeContext(); unsubscribeSession(); };
}
