import { useEffect, useMemo, useRef, useState } from 'react';
import { onSourceRegistered, readSource, subscribeSource } from '../../bridge.js';
import { effectiveObjective } from '../../objective.js';
import { OPTIMIZER_DEFAULTS } from '../../../domain/training/index.js';
import { useArchitectureStore } from '../../../stores/architectureStore.js';
import { useTrainingStore } from '../../../stores/trainingStore.js';

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

/** A bridge source's read(), refreshed on its notifications (and once it is registered, if that happens later). */
export function useSourceValue(name) {
    const [value, setValue] = useState(() => readSource(name));
    useEffect(() => {
        let unsubscribe = subscribeSource(name, () => setValue(readSource(name)));
        const stop = onSourceRegistered((registered) => {
            if (registered !== name) return;
            unsubscribe();
            unsubscribe = subscribeSource(name, () => setValue(readSource(name)));
            setValue(readSource(name));
        });
        return () => { unsubscribe(); stop(); };
    }, [name]);
    return value;
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
