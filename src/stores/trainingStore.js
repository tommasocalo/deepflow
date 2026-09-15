import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
    addTrainingEdge,
    addTrainingNode,
    createLossNode,
    createOptimizerNode,
    createPhaseNode,
    createTrainingEdge,
    emptyTrainingGraph,
    moveTrainingNode,
    patchTrainingHyperparameters,
    patchTrainingNode,
    removeTrainingEdge,
    removeTrainingNode,
} from '../domain/training/index.js';

const STORAGE_KEY = 'deepflow:training-graph';

export const useTrainingStore = create(
    persist(
        (set, get) => ({
            graph: emptyTrainingGraph(),
            /** @type {string[]} */
            selectedNodeIds: [],
            /** @type {string | null} */
            expandedNodeId: null,

            addOptimizer: (type, position) => {
                const node = createOptimizerNode({ type, position });
                set({ graph: addTrainingNode(get().graph, node) });
                return node.id;
            },

            addLoss: (type, position, architectureOutputId = null) => {
                const node = createLossNode({ type, position, architectureOutputId });
                set({ graph: addTrainingNode(get().graph, node) });
                return node.id;
            },

            addPhase: (position, opts = {}) => {
                const node = createPhaseNode({ position, ...opts });
                set({ graph: addTrainingNode(get().graph, node) });
                return node.id;
            },

            removeNode: (id) => set({ graph: removeTrainingNode(get().graph, id) }),

            patchNode: (id, patch) => set({ graph: patchTrainingNode(get().graph, id, patch) }),

            patchHyperparameters: (id, hp) =>
                set({ graph: patchTrainingHyperparameters(get().graph, id, hp) }),

            moveNode: (id, position) => set({ graph: moveTrainingNode(get().graph, id, position) }),

            connect: ({ source, target, sourceHandle, targetHandle }) => {
                if (!source || !target || source === target) return;
                const edge = createTrainingEdge({ source, target, sourceHandle, targetHandle });
                set({ graph: addTrainingEdge(get().graph, edge) });
            },

            removeEdge: (edgeId) => set({ graph: removeTrainingEdge(get().graph, edgeId) }),

            setNodeSelected: (id, selected) =>
                set((s) => {
                    const next = new Set(s.selectedNodeIds);
                    if (selected) next.add(id);
                    else next.delete(id);
                    return { selectedNodeIds: [...next] };
                }),

            clearSelection: () => set({ selectedNodeIds: [] }),

            selectOnly: (id) => set({ selectedNodeIds: id ? [id] : [] }),

            toggleExpanded: (id) =>
                set((s) => ({ expandedNodeId: s.expandedNodeId === id ? null : id })),

            setExpanded: (id) => set({ expandedNodeId: id }),

            clearExpanded: () => set({ expandedNodeId: null }),

            resetGraph: () =>
                set({ graph: emptyTrainingGraph(), selectedNodeIds: [] }),
        }),
        { name: STORAGE_KEY, partialize: (state) => ({ graph: state.graph }) },
    ),
);
