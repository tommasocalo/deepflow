/**
 * useTrainerStore (docs/edu/SPEC.md §6.10.1, §6.10.7, §6.10.8): run settings, the current run, the last runs and
 * the preflight. Registered as bridge source 'trainer' and workspace slice 'trainer'.
 *
 * start() captures the live graph, templates, training graph, data binding and settings, runs the preflight, prepares
 * the dataset and trains. In the browser the run is sliced into pieces of at most SLICE_MS that yield to the event
 * loop, and the store publishes at most 10 updates per second. `start({ sync: true })` trains to the end without
 * yielding (tests and the headless Show me). The computation never depends on where it pauses, so the same inputs
 * give the same history either way.
 */
import { create } from 'zustand';
import { exposeEduHandle, registerSlice, registerSource } from '../bridge.js';
import { eduKey, readJSON, writeJSON } from '../lessons/engine/storage.js';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import { useTrainingStore } from '../../stores/trainingStore.js';
import { prepareDataset } from '../data/prepare.js';
import { useEduDataStore } from '../data/store.js';
import { computeBoundary } from './boundary.js';
import { chanceLoss } from './metrics.js';
import { runPreflight } from './preflight.js';
import { createRun, drain, finishRun, runSteps } from './run.js';

export const TRAINER_KEY = eduKey('trainer:v1');
export const DEFAULT_SETTINGS = Object.freeze({ epochs: 50, batchSize: 32, seed: 1 });
export const SETTING_LIMITS = Object.freeze({
    epochs: { min: 1, max: 1000 },
    batchSize: { min: 1, max: 1024 },
    seed: { min: 0, max: 9999 },
});
/** Completed and stopped runs kept for comparison (most recent last). */
export const MAX_RUNS = 8;
export const SLICE_MS = 12;
export const PUSH_MS = 100;
export const BOUNDARY_MS = 100;
export const PREFLIGHT_DELAY = 200;

const now = () => (globalThis.performance?.now ? globalThis.performance.now() : Date.now());
const runError = (error, code = 'run-failed') => ({ code: error?.code ?? code, message: error?.message ?? String(error) });

/** Whole numbers inside SETTING_LIMITS; anything unreadable keeps the previous value (or the default). */
export function normalizeSettings(patch = {}, base = DEFAULT_SETTINGS) {
    const settings = { ...DEFAULT_SETTINGS, ...base };
    for (const [name, limits] of Object.entries(SETTING_LIMITS)) {
        if (patch?.[name] === undefined || patch[name] === null || patch[name] === '') continue;
        const value = Math.round(Number(patch[name]));
        if (!Number.isFinite(value)) continue;
        settings[name] = Math.min(limits.max, Math.max(limits.min, value));
    }
    return settings;
}

const EMPTY_PREFLIGHT = Object.freeze({
    ok: false, errors: [], warnings: [], objective: null, inputShape: null, outputShape: null, paramCount: null,
    trainableCount: null, datasetId: null, at: 0,
});

/** The live inputs of a run or a preflight. */
function captureInputs(settings) {
    return {
        graph: useArchitectureStore.getState().graph,
        templates: useTemplatesStore.getState().templates,
        training: useTrainingStore.getState().graph,
        binding: useEduDataStore.getState().current(),
        settings: { ...settings },
    };
}

function computePreflight(inputs) {
    const result = runPreflight({
        graph: inputs.graph, templates: inputs.templates, training: inputs.training,
        dataset: inputs.binding.spec, settings: inputs.settings,
    });
    // The compiled program stays out of the store; start() compiles again from the captured graph.
    const { program: _program, ...summary } = result;
    return { result, summary: { ...summary, datasetId: inputs.binding.spec?.id ?? null, at: Date.now() } };
}

function newRecord(inputs, seq, initOnly) {
    const { binding, settings } = inputs;
    const spec = binding.spec;
    const classes = spec?.target?.kind === 'classes';
    return {
        id: `run-${seq}-${Date.now().toString(36)}`,
        seq,
        tag: null,
        kind: initOnly ? 'init' : 'train',
        initOnly,
        status: 'running',
        error: null,
        startedAt: Date.now(),
        finishedAt: null,
        datasetId: spec?.id ?? null,
        datasetName: spec?.name ?? null,
        dataOptions: { ...binding.options },
        split: { ...binding.split },
        dataSeed: binding.dataSeed,
        settings: { ...settings },
        optimizer: null,
        loss: null,
        defaultsUsed: [],
        graph: inputs.graph,
        templates: inputs.templates,
        training: inputs.training,
        paramCount: null,
        trainableCount: null,
        history: [],
        epoch: 0,
        best: null,
        confusion: null,
        classNames: classes ? [...spec.target.classes] : null,
        numClasses: classes ? spec.target.size : null,
        chance: null,
        boundary: null,
    };
}

/** A copy of the record with the run's latest metrics (new arrays, so React sees the change). */
function snapshotRecord(record, run, extra = {}) {
    return {
        ...record,
        history: run.history.map((entry) => ({ ...entry })),
        epoch: run.epoch,
        best: run.best ? { ...run.best } : null,
        confusion: run.confusion ? run.confusion.map((row) => [...row]) : record.confusion,
        boundary: run.boundary ?? record.boundary,
        ...extra,
    };
}

/**
 * Yield to the event loop: MessageChannel avoids setTimeout's 4 ms clamp in browsers. Node (tests, also under jsdom)
 * uses setTimeout, because an open MessagePort would keep the process alive.
 */
const schedule = (() => {
    const browser = typeof globalThis.document !== 'undefined' && !globalThis.process?.versions?.node;
    if (browser && typeof globalThis.MessageChannel === 'function') {
        const channel = new globalThis.MessageChannel();
        const queue = [];
        channel.port1.onmessage = () => { queue.shift()?.(); };
        return (callback) => { queue.push(callback); channel.port2.postMessage(0); };
    }
    return (callback) => setTimeout(callback, 0);
})();

const savedSettings = normalizeSettings(readJSON(TRAINER_KEY, null) ?? {});

/** The active job: { record, run, dataset, iterator, stopRequested, aborted, sliced, resolve, promise, … }. */
let active = null;
let preflightTimer = null;

export const useTrainerStore = create((set, get) => {
    function finalize(job, status, { error = null, aborted = false } = {}) {
        if (active !== job) return job.final ?? null;
        active = null;
        const { run } = job;
        if (run && !run.done && status === 'stopped' && !aborted) drain(finishRun(run));
        const record = run
            ? snapshotRecord(job.record, run, { status, error, finishedAt: Date.now() })
            : { ...job.record, status, error, finishedAt: Date.now() };
        job.final = record;
        if (aborted) set({ status: 'idle' });
        else if (status === 'failed') set({ status: 'error', current: record });
        else if (!record.history.length) set({ status: 'idle', current: record });
        else set((state) => ({ status: 'idle', current: record, runs: [...state.runs, record].slice(-MAX_RUNS) }));
        job.resolve(record);
        return record;
    }

    function publish(job) {
        const time = now();
        if (time - job.lastPush < PUSH_MS) return;
        job.lastPush = time;
        const { run, dataset } = job;
        // The boundary is redrawn at most every BOUNDARY_MS, and never takes more than about a ninth of the time
        // (a 2-128-128-3 network needs ~80 ms for the 64 × 64 grid; a 2-8-1 network well under 1 ms).
        if (dataset.featureSize === 2 && run.history.length && time >= job.nextBoundaryAt && job.boundaryEpoch !== run.epoch) {
            const started = now();
            run.boundary = computeBoundary(run.model, dataset, run.objective.loss.type);
            job.boundaryEpoch = run.epoch;
            job.nextBoundaryAt = now() + Math.max(BOUNDARY_MS, 8 * (now() - started));
        }
        job.record = snapshotRecord(job.record, run);
        set({ current: job.record });
    }

    function slice(job) {
        if (active !== job) return;
        try {
            const started = now();
            while (!job.stopRequested) {
                if (job.iterator.next().done) {
                    finalize(job, 'completed');
                    return;
                }
                if (now() - started >= SLICE_MS) break;
            }
            if (job.stopRequested) {
                finalize(job, 'stopped');
                return;
            }
            publish(job);
        } catch (error) {
            finalize(job, 'failed', { error: runError(error) });
            return;
        }
        schedule(() => slice(job));
    }

    async function launch({ sync = false, initOnly = false } = {}) {
        if (active) {
            const previous = active;
            previous.stopRequested = true;
            if (previous.sliced) await previous.promise;
            else finalize(previous, 'stopped');
        }
        const inputs = captureInputs(get().settings);
        const seq = get().seq + 1;
        let resolve;
        const promise = new Promise((done) => { resolve = done; });
        const job = { record: newRecord(inputs, seq, initOnly), run: null, dataset: null, iterator: null, stopRequested: false,
            sliced: false, resolve, promise, lastPush: 0, nextBoundaryAt: 0, boundaryEpoch: -1, final: null };
        active = job;
        set({ seq, status: 'running', current: job.record });

        const { result, summary } = computePreflight(inputs);
        set({ preflight: summary });
        const objective = result.objective;
        job.record = {
            ...job.record,
            optimizer: objective ? { type: objective.optimizer.type, hyperparameters: { ...objective.optimizer.hyperparameters } } : null,
            loss: objective ? { type: objective.loss.type, hyperparameters: { ...objective.loss.hyperparameters } } : null,
            defaultsUsed: objective ? ['optimizer', 'loss'].filter((name) => objective[name].isDefault) : [],
            paramCount: result.paramCount,
            trainableCount: result.trainableCount,
            chance: objective ? chanceLoss(objective.loss.type, job.record.numClasses) : null,
        };
        if (!result.ok) {
            const first = result.errors[0] ?? { code: 'invalid-graph', message: 'This model cannot be trained.' };
            return finalize(job, 'failed', { error: { code: first.code, message: first.message, nodeId: first.nodeId ?? null } });
        }

        let dataset;
        try {
            const { spec, options, split, dataSeed } = inputs.binding;
            dataset = await prepareDataset(spec, options, split, dataSeed);
        } catch (error) {
            return finalize(job, 'failed', { error: runError(error, 'dataset-load') });
        }
        if (active !== job) return promise;
        if (job.stopRequested) return finalize(job, 'stopped');
        try {
            job.run = createRun({ program: result.program, dataset, objective, settings: inputs.settings });
            job.dataset = dataset;
            job.iterator = runSteps(job.run, { initOnly });
            if (sync) {
                drain(job.iterator);
                return finalize(job, 'completed');
            }
        } catch (error) {
            return finalize(job, 'failed', { error: runError(error) });
        }
        job.sliced = true;
        schedule(() => slice(job));
        return promise;
    }

    return {
        settings: savedSettings,
        status: 'idle',
        seq: 0,
        current: null,
        runs: [],
        preflight: EMPTY_PREFLIGHT,

        setSettings: (patch = {}) => {
            const previous = get().settings;
            const settings = normalizeSettings(patch, previous);
            if (Object.keys(settings).every((key) => settings[key] === previous[key])) return previous;
            set({ settings });
            return settings;
        },
        start: ({ sync = false } = {}) => launch({ sync, initOnly: false }),
        evaluateInit: ({ sync = false } = {}) => launch({ sync, initOnly: true }),
        stop: () => {
            const job = active;
            if (!job) return false;
            job.stopRequested = true;
            if (job.sliced) set({ status: 'stopping' });
            else finalize(job, 'stopped');
            return true;
        },
        tagRun: (runId, tag) => {
            const update = (record) => (record?.id === runId ? { ...record, tag } : record);
            set((state) => ({ runs: state.runs.map(update), current: update(state.current) }));
        },
        clearRuns: () => set((state) => (active ? { runs: [] } : { runs: [], current: null, status: state.status === 'error' ? 'idle' : state.status })),
        /** Recompute the preflight now (the store also does it PREFLIGHT_DELAY ms after any relevant change). */
        refreshPreflight: () => {
            clearTimeout(preflightTimer);
            preflightTimer = null;
            const { summary } = computePreflight(captureInputs(get().settings));
            set({ preflight: summary });
            return summary;
        },
        /** Drop the active run without recording it (workspace restore). */
        abort: () => {
            if (active) finalize(active, 'stopped', { aborted: true });
        },
    };
});

// ---- persistence, preflight refresh, bridge -----------------------------------------------------------------------

useTrainerStore.subscribe((state, previous) => {
    if (state.settings !== previous.settings) writeJSON(TRAINER_KEY, state.settings);
});

function queuePreflight() {
    clearTimeout(preflightTimer);
    preflightTimer = setTimeout(() => {
        preflightTimer = null;
        useTrainerStore.getState().refreshPreflight();
    }, PREFLIGHT_DELAY);
    // A pending refresh must never keep a Node process (tests) alive.
    preflightTimer?.unref?.();
}

useArchitectureStore.subscribe((state, previous) => { if (state.graph !== previous.graph) queuePreflight(); });
useTemplatesStore.subscribe((state, previous) => { if (state.templates !== previous.templates) queuePreflight(); });
useTrainingStore.subscribe((state, previous) => { if (state.graph !== previous.graph) queuePreflight(); });
useEduDataStore.subscribe((state, previous) => {
    if (state.datasetId !== previous.datasetId || state.options !== previous.options || state.split !== previous.split) queuePreflight();
});
useTrainerStore.subscribe((state, previous) => { if (state.settings.batchSize !== previous.settings.batchSize) queuePreflight(); });
queuePreflight();

/** What the lesson engine and codegen read through the bridge. */
export function readTrainer() {
    const { status, seq, settings, current, runs, preflight } = useTrainerStore.getState();
    return { status, seq, settings, current, runs, preflight };
}

registerSource('trainer', {
    read: readTrainer,
    subscribe: (listener) => useTrainerStore.subscribe(() => listener()),
    actions: {
        start: (options) => useTrainerStore.getState().start(options),
        stop: () => useTrainerStore.getState().stop(),
        evaluateInit: (options) => useTrainerStore.getState().evaluateInit(options),
        setSettings: (patch) => useTrainerStore.getState().setSettings(patch),
        tagRun: (runId, tag) => useTrainerStore.getState().tagRun(runId, tag),
        clearRuns: () => useTrainerStore.getState().clearRuns(),
    },
});

registerSlice('trainer', {
    capture: () => ({ settings: { ...useTrainerStore.getState().settings } }),
    restore: (value) => {
        useTrainerStore.getState().abort();
        useTrainerStore.setState({ settings: normalizeSettings(value?.settings ?? {}), runs: [], current: null, status: 'idle' });
        queuePreflight();
    },
});

exposeEduHandle('trainer', useTrainerStore);
