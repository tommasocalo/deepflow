import { create } from 'zustand';
import { useArchitectureHistory } from '../../../stores/architectureHistory.js';
import { callSource, onSourceRegistered, readSource, registerSource, subscribeSource } from '../../bridge.js';
import { useEduUiStore } from '../../shell/uiStore.js';
import { runActions } from './actions.js';
import { requestFit } from './canvasBridge.js';
import { completedEpochs, createCheckContext, evaluateCheck, evaluateStep, finalMetrics, trainerRuns, workspaceCheck } from './checks.js';
import { buildCheckContext, factValues, readUiState, subscribeCheckContext } from './context.js';
import { hasText } from './i18n.js';
import { clearLog, logEvent } from './log.js';
import { isOutcomePrediction } from './observe.js';
import { learnerSeed, predictionOrder, quizOrder } from './order.js';
import { paletteUsed } from './palette.js';
import { gradePrediction } from './predict.js';
import * as P from './progress.js';
import { getModule, getModules, stepIndexById } from './registry.js';
import { eduKey, eduKeys, readJSON, removeKey, writeJSON } from './storage.js';
import {
    captureWorkspace, clearStepEntry, deleteSlot, loadFixtureIntoWorkspace, loadSlot, loadStepEntry, restoreSettings, restoreWorkspace,
    runStoryboard, saveSlot, saveStepEntry, slotKey, STEP_ENTRY_KEY,
} from './workspace.js';

/**
 * The lesson session (docs/edu/SPEC.md §6.7): sandbox or lesson mode, the position in the current module, and
 * lesson progress. Actions are async because a step's setup and Show me can load fixtures and train.
 */
export const SESSION_KEY = eduKey('session:v1');
export const PROGRESS_KEY = eduKey('progress:v1');
const PHASES = ['warmup', 'steps', 'quiz', 'recap'];
const PERSISTED = ['mode', 'moduleId', 'phase', 'stepIndex', 'quizIndex', 'warmupIndex', 'collapsed'];
const freshRuntime = (overrides = {}) => ({
    enteredAtSeq: 0, visitedTabs: [], revealShown: false, busy: false, error: null, showMe: null, skippedAhead: 0, passedHere: false,
    entryNodeIds: null, entryMisconception: null, ...overrides,
});
/** Checks about the model's structure: their message cannot name a block that is in the wrong place. */
const STRUCTURE_PREDICATES = new Set(['chain', 'edge', 'has', 'component']);
/** Show me actions that train: they wait until the model is right, so a wrong attempt is never trained. */
const TRAINING_ACTIONS = new Set(['train', 'evaluateInit']);

/** Show me actions that change the workspace. Answers, training runs and navigation are never replayed for a learner. */
const WORKSPACE_ACTIONS = new Set([
    'loadFixture', 'loadStoryboard', 'useDataset', 'setDatasetOptions', 'setParam', 'setGlobal', 'insertBetween', 'insertAfter',
    'buildChain', 'connect', 'disconnect', 'remove', 'group', 'repeat', 'setOptimizer', 'setLoss', 'setTrainer',
]);
const VIEW_FLAGS = ['showDimensions', 'showGradientFlow'];

/** The workspace edits of a step's Show me: what a step leaves behind for the steps after it (docs/edu/SPEC.md §3.3). */
export function workspaceEdits(actions = []) {
    return actions.flatMap((action) => {
        if (WORKSPACE_ACTIONS.has(action?.do)) return [action];
        if (action?.do !== 'setUi') return [];
        // The view flags belong to the workspace snapshot; the tab and the code panel are navigation.
        const flags = Object.fromEntries(VIEW_FLAGS.filter((flag) => action[flag] !== undefined).map((flag) => [flag, action[flag]]));
        return Object.keys(flags).length ? [{ do: 'setUi', ...flags }] : [];
    });
}

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

/**
 * The stored session and progress, brought up to date. A module installed in a new version starts its steps again
 * (P.migrate): its saved workspace and step entry belong to its old steps and are dropped, and a lesson left inside it
 * reopens at its first step and reloads its start (`restart`, which the lesson panel or a deep link acts on).
 */
/** The stored entry of the step on screen, read once after a reload (see entryInfo). */
let storedEntry = { key: null, value: null };

function initialState() {
    storedEntry = { key: null, value: null };
    const session = loadSession();
    const stored = readJSON(PROGRESS_KEY, null);
    const updated = P.updatedModules(stored, getModules());
    const progress = P.migrate(stored, getModules());
    for (const module of updated) {
        deleteSlot(module.id);
        if (readJSON(STEP_ENTRY_KEY, null)?.moduleId === module.id) clearStepEntry();
        logEvent('module-updated', { module: module.id, data: { from: stored.modules[module.id].moduleVersion ?? 1, to: module.version ?? 1 } });
    }
    if (updated.length) writeJSON(PROGRESS_KEY, progress);
    const restart = session.mode === 'lesson' && updated.some((module) => module.id === session.moduleId) ? session.moduleId : null;
    if (restart) Object.assign(session, { phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0 });
    return { ...session, progress, restart, headless: false, runtime: freshRuntime({ visitedTabs: [useEduUiStore.getState().activeTab] }) };
}

export const useLessonSession = create((set, get) => {
    const module = () => getModule(get().moduleId);
    const step = () => module()?.steps?.[get().stepIndex] ?? null;
    const record = () => get().progress.modules?.[get().moduleId] ?? null;
    const updateProgress = (reducer) => set((state) => ({ progress: reducer(state.progress) }));
    const updateRuntime = (patch) => set((state) => ({ runtime: { ...state.runtime, ...patch } }));
    const log = (type, extra = {}) => logEvent(type, { module: get().moduleId, ...extra });
    // What the learner was shown in this visit to a step, logged once each (review r2: the status log had doubles).
    const shownThisVisit = new Set();
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

    /**
     * `resume` re-enters the step the learner left (Continue, a reload, a deep link to the current step): its entry
     * snapshot is kept, so Reset step still returns to the state the step started from. Moving to another step
     * captures a new snapshot.
     */
    async function enterStep(index, { runSetup = true, resume = false } = {}) {
        const current = module();
        const target = current?.steps?.[index];
        if (!target) return false;
        // A step filled in by a jump already ran its setup; one only marked visited by an older deep link did not.
        const previous = record()?.steps?.[target.id];
        const firstEntry = !previous?.enteredAt && !previous?.filledIn;
        set({ phase: 'steps', stepIndex: index, runtime: freshRuntime({ busy: firstEntry && runSetup && !!target.setup?.length }) });
        shownThisVisit.clear();
        if (firstEntry && runSetup && target.setup?.length) {
            const result = await runActions(target.setup, actionContext('step setup'));
            if (!result.ok) updateRuntime({ error: result.error });
            // A setup can load a new model (M2 s7's XOR start): show all of it, Output included (review r1).
            fitSoon();
        }
        showStepTab(target);
        const kept = !firstEntry && resume ? loadStepEntry(current.id, target.id) : null;
        const entry = kept ?? saveStepEntry(current.id, target.id, captureWorkspace());
        updateProgress((progress) => P.enterStep(progress, current.id, target.id));
        updateRuntime({
            enteredAtSeq: trainerSeq(), visitedTabs: [useEduUiStore.getState().activeTab], revealShown: false, busy: false,
            entryNodeIds: Object.keys(entry?.graph?.nodes ?? {}), entryMisconception: diagnoseSnapshot(current, target, entry),
        });
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

    /** The misconception a step diagnoses in a snapshot (its entry): one it starts with is no mistake of the learner's. */
    function diagnoseSnapshot(current, target, snapshot) {
        if (!snapshot?.graph || !(target.diagnose?.length || target.kind === 'task')) return null;
        const ctx = createCheckContext({
            graph: snapshot.graph, templates: snapshot.templates, training: snapshot.training, ui: readUiState(), data: readSource('data'),
            trainer: readSource('trainer'), step: get().stepInfo(), facts: factValues(current), module: current,
        });
        const result = evaluateStep(target, ctx);
        return result.pass ? null : result.misconception ?? null;
    }

    /**
     * The step's entry: the root node ids and the misconception it started with. Entering a step records them in the
     * runtime; after a reload the runtime is fresh, so they come from the stored step entry instead (read once).
     */
    function entryOf(current, target) {
        const runtime = get().runtime;
        if (runtime.entryNodeIds) return { nodeIds: runtime.entryNodeIds, misconception: runtime.entryMisconception };
        const key = `${current.id}:${target.id}`;
        if (storedEntry.key !== key) {
            const snapshot = loadStepEntry(current.id, target.id);
            const value = snapshot ? { nodeIds: Object.keys(snapshot.graph?.nodes ?? {}), misconception: diagnoseSnapshot(current, target, snapshot) } : null;
            storedEntry = { key, value: value ?? { nodeIds: null, misconception: null } };
        }
        return storedEntry.value;
    }

    /** Whether the workspace part of a step's check holds now (engine/checks.js workspaceCheck). */
    function workspaceReady(item) {
        const part = workspaceCheck(item?.check);
        if (!part) return true;
        const current = module();
        return evaluateCheck(part, buildCheckContext({ module: current, step: get().stepInfo(), facts: factValues(current) })).pass;
    }

    /**
     * Run a step's Show me edits on the learner's attempt, reusing the blocks they added in it (§6.5). When that does
     * not give the model the step asks for (review r2: a Sigmoid before the Output, a second ReLU beside the first),
     * the step starts over from its entry snapshot instead: the model (with its training graph) and its edits form one
     * undo step, so Undo brings the attempt back; the data binding and trainer settings return to the entry's.
     * @returns {Promise<{ ok: boolean, error?: string, restarted: boolean }>}
     */
    async function completeAttempt(item, edits, label) {
        const current = module();
        const entry = loadStepEntry(current.id, item.id);
        const history = () => useArchitectureHistory.getState();
        const undoBefore = history().undoCount;
        const first = await runActions(edits, { ...actionContext(label), entryNodeIds: entry ? Object.keys(entry.graph?.nodes ?? {}) : null });
        if (!entry || (first.ok && workspaceReady(item))) return { ...first, restarted: false };
        while (history().undoCount > undoBefore && history().undo()) { /* back to the learner's attempt */ }
        restoreSettings(entry);
        const again = await runActions(edits, { ...actionContext(label), startFrom: entry });
        return { ...again, restarted: true };
    }

    /**
     * Leave the steps `from … to − 1` behind as if the learner had done them (Skip, a progress segment, a deep link):
     * each step that was never passed or skipped runs its setup (when it was never entered) and the workspace edits of
     * its Show me, so the next step opens on the model the earlier ones build. The step the learner is on completes
     * their own attempt, as its Show me does. Answers and training runs are not replayed.
     * @returns {Promise<{ ok: boolean, error?: string, changed: number }>} `changed`: steps that changed the workspace
     */
    async function fastForward(from, to) {
        const current = module();
        const filled = [];
        let changed = 0;
        const done = (extra = {}) => {
            if (filled.length) updateProgress((progress) => P.fillSteps(progress, current.id, filled));
            return { ok: true, changed, ...extra };
        };
        for (let index = Math.max(0, from); index < Math.min(to, current?.steps?.length ?? 0); index++) {
            const item = current.steps[index];
            const stepRecord = record()?.steps?.[item.id];
            // Passed, skipped and filled-in steps already left their changes in the workspace.
            if (['passed', 'skipped'].includes(stepRecord?.status) || stepRecord?.filledIn) continue;
            const entered = !!stepRecord?.enteredAt;
            const setup = entered ? [] : item.setup ?? [];
            const edits = workspaceEdits(item.solution);
            if (setup.length) {
                const result = await runActions(setup, actionContext('step setup'));
                if (!result.ok) return done({ ok: false, error: result.error });
            }
            if (edits.length) {
                // The step the learner is on completes their attempt, or starts over when it cannot (completeAttempt).
                const result = entered ? await completeAttempt(item, edits, `skip ${item.id}`) : await runActions(edits, actionContext(`skip ${item.id}`));
                if (!result.ok) return done({ ok: false, error: result.error });
            }
            filled.push(item.id);
            if (setup.length || edits.length) changed += 1;
        }
        return done();
    }

    /** Move forward from step `from` to step `to` > `from`, filling in the steps in between (see fastForward). */
    async function jumpForward(from, to, { resume = false } = {}) {
        const current = module();
        updateRuntime({ busy: true, error: null });
        const result = await fastForward(from, to);
        const passedOver = current.steps.slice(Math.max(0, from), to).map((item) => item.id);
        updateProgress((progress) => P.visitSteps(progress, current.id, passedOver));
        if (to > from) log('skip-ahead', { step: current.steps[to].id, data: { from: current.steps[Math.max(0, from)]?.id ?? null, changed: result.changed, ok: result.ok } });
        await enterStep(to, { resume });
        updateRuntime({ error: result.ok ? null : result.error, skippedAhead: result.changed });
        fitSoon();
        return true;
    }

    /** Where a jump starts: the step the learner is on (or about to start after the warm-up); past the end otherwise. */
    const jumpOrigin = () => (['steps', 'warmup'].includes(get().phase) ? get().stepIndex : module()?.steps?.length ?? 0);

    return {
        ...initialState(),

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
        /** The current step's entry (root node ids, starting misconception), also right after a reload. */
        entryInfo: () => {
            const current = module();
            const target = step();
            return current && target && get().phase === 'steps' ? entryOf(current, target) : { nodeIds: null, misconception: null };
        },
        /** Evaluate the current step against the live stores. */
        evaluate: () => {
            const current = module();
            const target = step();
            if (!current || !target) return { pass: false, detail: { key: 'panel.empty', params: {} } };
            const ctx = buildCheckContext({ module: current, step: get().stepInfo(), facts: factValues(current) });
            const result = evaluateStep(target, ctx);
            if (result.pass) return result;
            // `firm`: a finished wrong model, which the panel names at once (review r2: the Sigmoid trap read "After
            // Linear, connect an Output" for 6 s while Sigmoid was the problem). Finished: valid, nothing left without
            // wires, the palette's blocks all placed, and a misconception the step did not start with; and the check
            // fails on the model's structure, which its own message cannot explain. A half-done model (M1 s5 with
            // only Input set) keeps the check's message and the 6 s wait.
            const entry = result.misconception ? entryOf(current, target) : null;
            const firm = !!result.misconception && result.misconception !== entry.misconception
                && STRUCTURE_PREDICATES.has(result.predicate) && !!ctx.report?.ok && !ctx.issues.length && !ctx.hints.length
                && (!target.palette?.length || paletteUsed(target.palette, ctx.graph, ctx.templates, entry.nodeIds));
            // `drift`: a step passed (or skipped) earlier whose model no longer holds: the panel keeps the pass but
            // says what changed (review r2: M1 s5 still said "15 parameters" after Output went back to 1).
            const kept = ['passed', 'skipped'].includes(record()?.steps?.[target.id]?.status);
            const part = kept ? workspaceCheck(target.check) : null;
            const drift = !!part && !evaluateCheck(part, ctx).pass;
            return firm || drift ? { ...result, ...(firm ? { firm } : {}), ...(drift ? { drift } : {}) } : result;
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
            // A module updated since the learner's last visit starts from its start fixture (initialState).
            const restart = state.restart === id;
            const sameModule = state.mode === 'lesson' && state.moduleId === id && !restart;
            // Continue returns to the quiz, or to the recap of an unfinished module, when the learner left it there.
            // Review of a completed module starts again at the first step.
            const completed = state.progress.modules?.[id]?.status === 'completed';
            const resumePhase = !stepId && state.moduleId === id
                && (state.phase === 'quiz' || (state.phase === 'recap' && !completed)) ? state.phase : null;
            // The step the workspace belongs to, where a deep link further on starts filling in (see fastForward).
            let origin = sameModule ? jumpOrigin() : 0;
            if (!sameModule) {
                // The workspace on screen is kept in its slot, unless it is an updated module's old one.
                if (!restart) saveSlot(state.mode === 'lesson' && state.moduleId ? state.moduleId : 'sandbox', captureWorkspace());
                // The code panel shows the outgoing workspace's code: close it with that workspace.
                useEduUiStore.getState().closeCode();
                set({ mode: 'lesson', moduleId: id, restart: null, runtime: freshRuntime({ busy: true }) });
                const slot = loadSlot(id);
                try {
                    if (slot) {
                        restoreWorkspace(slot);
                        const left = P.moduleStatus(state.progress, target);
                        origin = left.currentStepId ? Math.max(0, stepIndexById(target, left.currentStepId)) : Math.max(0, left.firstOpenIndex);
                    } else {
                        await loadFixtureIntoWorkspace(target, target.start, { runAfter: async (after) => {
                            const result = await runActions(after, actionContext('fixture'));
                            if (!result.ok) throw new Error(result.error);
                        } });
                    }
                } catch (error) {
                    console.warn('[edu session] could not load the module workspace', error);
                    updateRuntime({ error: error.message });
                }
                fitSoon();
            }
            updateProgress((progress) => P.startModule(progress, id, target.version ?? 1));
            log('module-start', { data: { learnerSeed: learnerSeed() } });
            if (resumePhase) {
                set({ phase: resumePhase, quizIndex: Math.min(state.quizIndex, Math.max(0, (target.quiz?.length ?? 1) - 1)), runtime: freshRuntime() });
                showStepTab(null);
                return true;
            }
            const summary = P.moduleStatus(get().progress, target);
            let index = stepId ? stepIndexById(target, stepId) : -1;
            if (index >= 0) {
                // A deep link further on fills in the steps it passes over; one back, or to this step, only opens it.
                if (index > origin) {
                    set({ phase: 'steps', stepIndex: origin, quizIndex: 0, warmupIndex: 0 });
                    return jumpForward(origin, index);
                }
                const skipped = target.steps.slice(0, index).map((item) => item.id);
                updateProgress((progress) => P.visitSteps(progress, id, skipped));
            } else if (summary.status === 'completed') {
                // Review starts at the first step; finished steps open read-only (setup is not re-run).
                index = 0;
            } else if (summary.currentStepId) {
                index = Math.max(0, stepIndexById(target, summary.currentStepId));
            } else {
                index = summary.firstOpenIndex >= 0 ? summary.firstOpenIndex : 0;
            }
            const moduleRecord = get().progress.modules[id];
            const resume = target.steps[index]?.id === moduleRecord?.currentStepId;
            const warmupPending = !stepId && (target.warmup?.length ?? 0) > 0 && !moduleRecord?.warmupSkipped
                && !Object.keys(moduleRecord?.warmup ?? {}).length && !Object.keys(moduleRecord?.steps ?? {}).length;
            set({ stepIndex: index, quizIndex: 0, warmupIndex: 0 });
            if (warmupPending) {
                set({ phase: 'warmup', runtime: freshRuntime() });
                showStepTab(null);
                return true;
            }
            await enterStep(index, { resume });
            return true;
        },

        /**
         * A progress segment: a later step fills in the steps before it (see fastForward); an earlier one opens
         * read-only, as Back does; the current step stays as it is.
         */
        goToStep: async (index) => {
            if (get().runtime.busy || !module()?.steps?.[index]) return false;
            const origin = jumpOrigin();
            if (index > origin || (index === origin && get().phase !== 'steps')) return jumpForward(origin, index);
            if (index === origin) return true;
            return enterStep(index, { runSetup: false });
        },

        /** Latch a passing check: the step stays done while the learner explores (it is saved as passed at once). */
        notePass: (result = get().evaluate()) => {
            const current = module();
            const target = step();
            if (!current || !target || !result?.pass || get().mode !== 'lesson' || get().phase !== 'steps') return false;
            if (record()?.steps?.[target.id]?.status === 'passed') return false;
            updateProgress((progress) => P.passStep(progress, current.id, target.id));
            updateRuntime({ passedHere: true });
            log('step-pass', { step: target.id });
            if (target.tagRun && result.runId) get().tagRun(target.tagRun, result.runId);
            return true;
        },

        next: async () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return false;
            const status = record()?.steps?.[target.id]?.status;
            const result = get().evaluate();
            if (!result.pass && !['passed', 'skipped'].includes(status)) return false;
            get().notePass(result);
            await advance();
            return true;
        },

        /** Skip fills the step in (its Show me edits, without the answer or a run) so the next step can be done. */
        skip: async () => {
            const current = module();
            const target = step();
            if (!current || !target || get().runtime.busy) return false;
            const index = get().stepIndex;
            updateRuntime({ busy: true, error: null });
            const filled = await fastForward(index, index + 1);
            updateProgress((progress) => P.skipStep(progress, current.id, target.id));
            log('step-skip', { step: target.id, data: { filled: filled.changed > 0, ok: filled.ok } });
            await advance();
            if (!filled.ok) updateRuntime({ error: filled.error });
            fitSoon();
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
            // Reset starts the step over: a pass it reached is open again.
            updateProgress((progress) => P.reopenStep(progress, current.id, target.id));
            updateRuntime({ enteredAtSeq: trainerSeq(), visitedTabs: [useEduUiStore.getState().activeTab], revealShown: false, error: null, showMe: null, passedHere: false });
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
            updateRuntime({ busy: true, error: null, showMe: null });
            // The edits first, on the learner's attempt or from the step's start (completeAttempt); training after them.
            const solution = target.solution ?? [];
            const cut = solution.findIndex((action) => TRAINING_ACTIONS.has(action?.do));
            let result = await completeAttempt(target, cut < 0 ? solution : solution.slice(0, cut), 'show me');
            const { restarted } = result;
            if (result.ok && cut >= 0) result = await runActions(solution.slice(cut), actionContext('show me'));
            updateProgress((progress) => P.raiseHint(progress, current.id, target.id, 3));
            // Show me only claims a change that passed the step (review r2); otherwise the panel says it is not done.
            const passed = !!get().evaluate().pass;
            log('show-me', { step: target.id, data: { ok: result.ok, restarted, passed } });
            const undoable = useArchitectureHistory.getState().undoCount > undoBefore;
            updateRuntime({ busy: false, error: result.ok ? null : result.error, showMe: { at: Date.now(), undoable, restarted, passed } });
            fitSoon();
            return { ok: result.ok, error: result.error, restarted, passed };
        },

        /** The Show me toast's Undo: revert the change and, when Show me passed the step just now, open it again. */
        undoShowMe: () => {
            const current = module();
            const target = step();
            const history = useArchitectureHistory.getState();
            if (!history.canUndo) return false;
            history.undo();
            if (current && target && get().runtime.passedHere) updateProgress((progress) => P.reopenStep(progress, current.id, target.id));
            updateRuntime({ passedHere: false, showMe: null });
            log('undo-show-me', { step: target?.id });
            return true;
        },

        /**
         * Log something the panel showed in this visit to the step (a status message, a misconception callout) once,
         * however often the panel renders it. `key` tells shown things apart. Returns whether it was logged.
         */
        logShown: (type, key, extra = {}) => {
            const id = `${get().moduleId}|${get().stepIndex}|${type}|${key}`;
            if (shownThisVisit.has(id)) return false;
            shownThisVisit.add(id);
            log(type, extra);
            return true;
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
            // The options as this learner saw them, so position effects can be studied (engine/order.js).
            const shown = owner.predict.kind === 'choice' ? { shown: predictionOrder(owner.predict, `${current.id}.${owner.id}`) } : {};
            const misconception = grade.misconception ?? null;
            log('predict', { step: owner.id, data: { id, value, correct: grade.correct, source, misconception, ...shown } });
            // A prediction about the tool reveals at once, with its misconception callout; one about a run waits for it.
            if (misconception && hasText(`mis.${misconception}`) && !isOutcomePrediction(owner)) {
                get().logShown('misconception-shown', `predict:${misconception}`, { step: owner.id, data: { id: misconception, source: 'predict', item: id } });
            }
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
            const misconception = correct ? null : item.misconceptions?.[optionId] ?? null;
            log(warmup ? 'warmup' : 'quiz', { data: { id, option: optionId, correct, attempt, misconception, shown: quizOrder(item) } });
            // Every Check shows that option's explanation, with its misconception.
            if (misconception && hasText(`mis.${misconception}`)) log('misconception-shown', { data: { id: misconception, source: warmup ? 'warmup' : 'quiz', item: id } });
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
            useEduUiStore.getState().closeCode();
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
            clearLog();
            set({ progress: P.emptyProgress(), moduleId: null, phase: 'steps', stepIndex: 0, quizIndex: 0, warmupIndex: 0, runtime: freshRuntime() });
        },

        /** Re-read session and progress from storage (tests and module-registry swaps). */
        hydrate: () => set(initialState()),
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
 * The Data tab's Use this dataset in the current step: a step whose own Show me or setup leaves the output size to
 * the learner (`useDataset` with `setOutput: false`, docs/edu/SPEC.md §3.4) makes the button do the same.
 */
function datasetUseOf(state, step) {
    if (state.mode !== 'lesson' || state.phase !== 'steps' || !step) return null;
    const action = [...(step.solution ?? []), ...(step.setup ?? [])].find((item) => item?.do === 'useDataset');
    return action?.setOutput === false ? { setOutput: false } : null;
}

/**
 * Read-only lesson state for other stages: the trainer UI can disable Train while a prediction gate is open,
 * wherever the Train button lives, and the Data tab follows the step's dataset options.
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
            datasetUse: datasetUseOf(state, target),
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
