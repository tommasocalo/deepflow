import { useEffect, useMemo, useRef, useState } from 'react';
import { effectiveObjective } from '../../objective.js';
import { OPTIMIZER_DEFAULTS } from '../../../domain/training/index.js';
import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { useTemplatesStore } from '../../../stores/templatesStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';
import { dataSeedOf, getDataset, resolveOptions } from '../../data/registry.js';
import { useEduDataStore } from '../../data/store.js';
import { staleReason } from '../describe.js';

/** The width of an element in CSS pixels (ResizeObserver), or `fallback` before it is measured. */
export function useMeasuredWidth(fallback = 280) {
    const ref = useRef(null);
    const [width, setWidth] = useState(fallback);
    useEffect(() => {
        const element = ref.current;
        if (!element) return undefined;
        const measure = () => {
            const next = Math.round(element.getBoundingClientRect?.().width ?? 0);
            if (next > 0) setWidth((current) => (Math.abs(current - next) >= 1 ? next : current));
        };
        measure();
        if (typeof globalThis.ResizeObserver !== 'function') return undefined;
        const observer = new globalThis.ResizeObserver(measure);
        observer.observe(element);
        return () => observer.disconnect();
    }, []);
    return [ref, width];
}

/** The training graph's effective optimizer and loss (docs/edu/SPEC.md §6.10.5), live. */
export function useObjective() {
    const training = useTrainingStore((state) => state.graph);
    const graph = useArchitectureStore((state) => state.graph);
    return useMemo(() => effectiveObjective(training, graph), [training, graph]);
}

/**
 * Write the optimizer type or learning rate into the training graph, which stays the single source of truth. With
 * no optimizer node yet, one is added (Adam) and wired from the loss, as the lesson actions do.
 */
export function patchOptimizer({ type, learningRate }) {
    const store = useTrainingStore.getState();
    const graph = useArchitectureStore.getState().graph;
    let { optimizer, loss } = effectiveObjective(store.graph, graph);
    if (!optimizer) {
        const id = store.addOptimizer(type ?? 'Adam', { x: 60, y: 100 });
        if (loss) store.connect({ source: loss.id, target: id });
        optimizer = useTrainingStore.getState().graph.nodes[id];
    }
    if (type && type !== optimizer.type) {
        const learning_rate = Number(learningRate ?? optimizer.hyperparameters?.learning_rate);
        const defaultLabel = [optimizer.type, `${optimizer.type} optimizer`].includes(optimizer.label);
        store.patchNode(optimizer.id, {
            type,
            label: defaultLabel ? `${type} optimizer` : optimizer.label,
            hyperparameters: { ...(OPTIMIZER_DEFAULTS[type] ?? {}), ...(Number.isFinite(learning_rate) ? { learning_rate } : {}) },
        });
        return;
    }
    if (learningRate !== undefined) store.patchHyperparameters(optimizer.id, { learning_rate: learningRate });
}

/**
 * Why the shown run no longer matches the live model or dataset ('model', 'dataset', 'both' or null; see staleReason),
 * kept up to date with the architecture, the templates and the Data tab's binding.
 */
export function useStaleReason(record) {
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    const datasetId = useEduDataStore((state) => state.datasetId);
    const optionMap = useEduDataStore((state) => state.options);
    const split = useEduDataStore((state) => state.split);
    return useMemo(() => {
        const spec = getDataset(datasetId);
        const options = spec ? resolveOptions(spec, optionMap?.[datasetId] ?? {}) : {};
        return staleReason(record, { graph, templates, binding: { spec, options, split, dataSeed: dataSeedOf(options) } });
    }, [record, graph, templates, datasetId, optionMap, split]);
}
