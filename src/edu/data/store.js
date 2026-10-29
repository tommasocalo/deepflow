/**
 * useEduDataStore (docs/edu/SPEC.md §6.9.2): which dataset the workspace is bound to, the last-used options per
 * dataset and the split. Persisted as the live binding of the active workspace (deepflow-edu:data:v1), registered as
 * bridge source 'data' and workspace slice 'data'.
 */
import { create } from 'zustand';
import { exposeEduHandle, registerSlice, registerSource } from '../bridge.js';
import { eduKey, readJSON, writeJSON } from '../lessons/engine/storage.js';
import { dataSeedOf, DATASETS, datasetSummary, getDataset, resolveOptions, resolveSplit } from './registry.js';
import { applyDataset } from './useDataset.js';

export const DATA_KEY = eduKey('data:v1');
export const DEFAULT_SPLIT = Object.freeze({ val: 0.2, test: 0 });

/**
 * Accept a slice or persisted value: { datasetId, options, split }. `options` is a map by dataset id; a flat
 * options object is read as the options of `datasetId`.
 */
export function normalizeBinding(value = {}) {
    const datasetId = getDataset(value?.datasetId) ? value.datasetId : null;
    const raw = value?.options && typeof value.options === 'object' ? value.options : {};
    const flat = datasetId && Object.values(raw).some((entry) => typeof entry !== 'object' || entry === null);
    const map = flat ? { [datasetId]: raw } : raw;
    const options = {};
    for (const [id, entry] of Object.entries(map)) {
        const spec = getDataset(id);
        if (spec && entry && typeof entry === 'object') options[id] = resolveOptions(spec, entry);
    }
    return { datasetId, options, split: resolveSplit({ ...DEFAULT_SPLIT, ...(value?.split ?? {}) }) };
}

const initial = normalizeBinding(readJSON(DATA_KEY, {}) ?? {});

export const useEduDataStore = create((set, get) => ({
    ...initial,
    /** The dataset shown in the Data tab (not part of the binding). */
    viewId: initial.datasetId ?? DATASETS[0]?.id ?? null,
    /** The result of the last Use this dataset: { datasetId, changed, warnings, at }. */
    lastUse: null,

    setDataset: (id) => {
        if (id !== null && !getDataset(id)) return false;
        set((state) => ({ datasetId: id, viewId: id ?? state.viewId }));
        return true;
    },
    /** Merge options for one dataset (default: the bound one, else the viewed one); values are clamped. */
    setOptions: (patch = {}, id = get().datasetId ?? get().viewId) => {
        const spec = getDataset(id);
        if (!spec) return false;
        set((state) => ({ options: { ...state.options, [id]: resolveOptions(spec, { ...state.options[id], ...patch }) } }));
        return true;
    },
    setSplit: (patch = {}) => set((state) => ({ split: resolveSplit({ ...state.split, ...patch }) })),
    setView: (id) => { if (getDataset(id)) set({ viewId: id }); },
    optionsFor: (id) => resolveOptions(getDataset(id), get().options[id] ?? {}),
    /** The bound dataset: { spec, options, split, dataSeed } (spec null when unbound). */
    current: () => {
        const { datasetId, split } = get();
        const spec = getDataset(datasetId);
        const options = spec ? get().optionsFor(datasetId) : {};
        return { spec, options, split: { ...split }, dataSeed: dataSeedOf(options) };
    },
    applyDataset: (id, options) => applyDataset(id, options),
}));

// The live binding survives reloads; DeepFlow's own keys hold the active graphs, this key holds their dataset.
useEduDataStore.subscribe((state, previous) => {
    if (state.datasetId !== previous.datasetId || state.options !== previous.options || state.split !== previous.split) {
        writeJSON(DATA_KEY, { datasetId: state.datasetId, options: state.options, split: state.split });
    }
});

/** What the lesson engine and codegen read through the bridge. */
export function readDataBinding() {
    const { spec, options, split, dataSeed } = useEduDataStore.getState().current();
    return { datasetId: spec?.id ?? null, spec: datasetSummary(spec), options, split, dataSeed };
}

registerSource('data', {
    read: readDataBinding,
    subscribe: (listener) => useEduDataStore.subscribe(() => listener()),
    actions: {
        useDataset: (id, options) => applyDataset(id, options),
        setDataset: (id) => useEduDataStore.getState().setDataset(id),
        setOptions: (patch, id) => useEduDataStore.getState().setOptions(patch, id),
        setSplit: (patch) => useEduDataStore.getState().setSplit(patch),
    },
});

registerSlice('data', {
    capture: () => {
        const { datasetId, options, split } = useEduDataStore.getState();
        return { datasetId, options: structuredClone(options), split: { ...split } };
    },
    restore: (value) => {
        const binding = normalizeBinding(value ?? {});
        useEduDataStore.setState((state) => ({ ...binding, viewId: binding.datasetId ?? state.viewId, lastUse: null }));
    },
});

exposeEduHandle('data', useEduDataStore);
exposeEduHandle('registry', { DATASETS, getDataset });
