import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import {
    addEdge,
    addNode,
    createEdge,
    createLayerNode,
    createRepeatNode,
    createSuperNodeInstance,
    createTemplate,
    emptyGraph,
    moveNode,
    patchEdge,
    patchLayerParameters,
    patchNode,
    removeEdge,
    removeNode,
} from '../domain/architecture/index.js';
import { getEntry } from '../domain/library/index.js';
import { useTemplatesStore } from './templatesStore.js';

const STORAGE_KEY = 'deepflow:architecture-graph';

const initialGraph = () => {
    const inputEntry = getEntry('deepflow.io.Input');
    const outputEntry = getEntry('deepflow.io.Output');
    const input = createLayerNode({
        libraryKey: 'deepflow.io.Input',
        position: { x: 200, y: 80 },
        entry: inputEntry,
    });
    const output = createLayerNode({
        libraryKey: 'deepflow.io.Output',
        position: { x: 200, y: 360 },
        entry: outputEntry,
    });
    let g = emptyGraph();
    g = addNode(g, input);
    g = addNode(g, output);
    return g;
};

/**
 * Walk the current path and return the graph being viewed plus enough
 * context to mutate it. Mutators receive a (newGraph) callback that knows
 * whether to set this store's graph or call into the templates store.
 */
function resolveLevel(state) {
    const path = state.currentPath;
    if (path.length === 0) {
        return {
            isRoot: true,
            templateId: null,
            label: 'main',
        };
    }
    // Walk the path. At each step, the supernode lives in the previous
    // level's graph and points at a template — the template's graph is the
    // next level.
    const tplStore = useTemplatesStore.getState();
    let parentGraph = state.graph;
    let tplId = null;
    let label = 'main';
    for (const sid of path) {
        const node = parentGraph.nodes[sid];
        if (!node || node.kind !== 'supernode') {
            return { isRoot: true, templateId: null, label: 'main' };
        }
        const tpl = tplStore.getTemplate(node.templateId);
        if (!tpl) return { isRoot: true, templateId: null, label: 'main' };
        tplId = node.templateId;
        label = node.label;
        parentGraph = tpl.graph;
    }
    return { isRoot: false, templateId: tplId, label };
}

/**
 * Read the graph at the current view level without mutating anything.
 */
function getViewGraph(state) {
    const level = resolveLevel(state);
    if (level.isRoot) return state.graph;
    const tpl = useTemplatesStore.getState().getTemplate(level.templateId);
    return tpl?.graph ?? emptyGraph();
}

export const useArchitectureStore = create(
    persist(
        (set, get) => {
            const updateCurrentGraph = (transform) => {
                const state = get();
                const level = resolveLevel(state);
                if (level.isRoot) {
                    set({ graph: transform(state.graph) });
                } else {
                    const tplStore = useTemplatesStore.getState();
                    const tpl = tplStore.getTemplate(level.templateId);
                    if (!tpl) return;
                    tplStore.patchTemplate(level.templateId, { graph: transform(tpl.graph) });
                }
            };

            return {
                graph: initialGraph(),
                /** @type {string[]} ids of currently-selected nodes in the active view */
                selectedNodeIds: [],
                /** @type {string | null} id of the currently expanded node, or null */
                expandedNodeId: null,
                /** @type {string[]} ids of supernode instances entered, root → leaf */
                currentPath: [],
                /** @type {string[]} library keys of recently-added blocks (most recent first) */
                recentLibraryKeys: [],

                /** @returns {import('../domain/architecture/types.js').ArchitectureGraph} */
                getViewGraph: () => getViewGraph(get()),
                getLevel: () => resolveLevel(get()),

                addLayerByLibraryKey: (libraryKey, position) => {
                    const entry = getEntry(libraryKey);
                    if (!entry) {
                        console.warn(`[architectureStore] unknown library key: ${libraryKey}`);
                        return null;
                    }
                    const node = createLayerNode({ libraryKey, position, entry });
                    updateCurrentGraph((g) => addNode(g, node));
                    set((s) => {
                        const next = [libraryKey, ...s.recentLibraryKeys.filter((k) => k !== libraryKey)];
                        return { recentLibraryKeys: next.slice(0, 6) };
                    });
                    return node.id;
                },

                addSuperNodeInstance: (templateId, position, label) => {
                    const node = createSuperNodeInstance({ templateId, position, label });
                    updateCurrentGraph((g) => addNode(g, node));
                    return node.id;
                },

                addRepeatNode: (templateId, position, opts = {}) => {
                    const node = createRepeatNode({ templateId, position, ...opts });
                    updateCurrentGraph((g) => addNode(g, node));
                    return node.id;
                },

                /**
                 * Promote an existing SuperNodeInstance into a RepeatNode in
                 * place, preserving its template, position, label, and
                 * (optionally) its bound parameter values as the formula for
                 * iteration 0.
                 */
                convertInstanceToRepeat: (instanceId, count = 2) => {
                    const view = getViewGraph(get());
                    const node = view.nodes[instanceId];
                    if (!node || node.kind !== 'supernode') return null;
                    const repeat = createRepeatNode({
                        id: node.id,
                        templateId: node.templateId,
                        position: node.position,
                        label: node.label,
                        count,
                        iterationVariable: 'i',
                        bindings: { ...(node.parameterValues ?? {}) },
                    });
                    updateCurrentGraph((g) => ({
                        ...g,
                        nodes: { ...g.nodes, [node.id]: repeat },
                    }));
                    return repeat.id;
                },

                /**
                 * Demote a RepeatNode back to a SuperNodeInstance. Used when
                 * the user wants a single instance instead of a repeat.
                 */
                convertRepeatToInstance: (repeatId) => {
                    const view = getViewGraph(get());
                    const node = view.nodes[repeatId];
                    if (!node || node.kind !== 'repeat') return null;
                    const instance = createSuperNodeInstance({
                        id: node.id,
                        templateId: node.templateId,
                        position: node.position,
                        label: node.label,
                        parameterValues: { ...(node.bindings ?? {}) },
                    });
                    updateCurrentGraph((g) => ({
                        ...g,
                        nodes: { ...g.nodes, [node.id]: instance },
                    }));
                    return instance.id;
                },

                patchRepeatNode: (id, patch) =>
                    updateCurrentGraph((g) => {
                        const existing = g.nodes[id];
                        if (!existing || existing.kind !== 'repeat') return g;
                        return { ...g, nodes: { ...g.nodes, [id]: { ...existing, ...patch } } };
                    }),

                patchRepeatBinding: (id, paramName, value) =>
                    updateCurrentGraph((g) => {
                        const existing = g.nodes[id];
                        if (!existing || existing.kind !== 'repeat') return g;
                        return {
                            ...g,
                            nodes: {
                                ...g.nodes,
                                [id]: {
                                    ...existing,
                                    bindings: { ...(existing.bindings ?? {}), [paramName]: value },
                                },
                            },
                        };
                    }),

                removeNode: (id) => updateCurrentGraph((g) => removeNode(g, id)),

                patchNode: (id, patch) => updateCurrentGraph((g) => patchNode(g, id, patch)),

                patchLayerParameters: (id, parameterPatch) =>
                    updateCurrentGraph((g) => patchLayerParameters(g, id, parameterPatch)),

                moveNode: (id, position) => updateCurrentGraph((g) => moveNode(g, id, position)),

                connect: ({ source, target, sourceHandle, targetHandle }) => {
                    if (!source || !target || source === target) return null;
                    const edge = createEdge({ source, target, sourceHandle, targetHandle });
                    updateCurrentGraph((g) => addEdge(g, edge));
                    return edge.id;
                },

                removeEdge: (edgeId) => updateCurrentGraph((g) => removeEdge(g, edgeId)),

                patchEdge: (edgeId, patch) => updateCurrentGraph((g) => patchEdge(g, edgeId, patch)),

                setEdgeGradientOp: (edgeId, gradientOp, gradientOpLambda) =>
                    updateCurrentGraph((g) =>
                        patchEdge(g, edgeId, {
                            gradientOp,
                            ...(gradientOpLambda !== undefined ? { gradientOpLambda } : {}),
                        }),
                    ),

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

                resetCurrentGraph: () => {
                    const level = resolveLevel(get());
                    if (level.isRoot) {
                        set({ graph: initialGraph(), selectedNodeIds: [], currentPath: [] });
                    } else {
                        useTemplatesStore.getState().patchTemplate(level.templateId, { graph: emptyGraph() });
                    }
                },

                // ---- navigation ----

                enterSupernode: (supernodeId) => {
                    const view = getViewGraph(get());
                    const node = view.nodes[supernodeId];
                    if (!node || node.kind !== 'supernode') return;
                    set({
                        currentPath: [...get().currentPath, supernodeId],
                        selectedNodeIds: [],
                    });
                },

                exitSupernode: () => {
                    const path = get().currentPath;
                    if (path.length === 0) return;
                    set({ currentPath: path.slice(0, -1), selectedNodeIds: [] });
                },

                goToLevel: (depth) =>
                    set({ currentPath: get().currentPath.slice(0, depth), selectedNodeIds: [] }),

                // ---- grouping ----

                /**
                 * Create a new template containing the selected nodes plus the
                 * edges among them, and replace those nodes in the current view
                 * with a single SuperNodeInstance.
                 *
                 * Phase 3 simplification: requires the selection to be
                 * connected to the rest of the graph through at most one
                 * external source endpoint and one external target endpoint.
                 * Returns the new template id, or null if the selection
                 * can't be grouped cleanly.
                 *
                 * @param {string[]} nodeIds
                 * @param {string} [name]
                 */
                groupSelection: (nodeIds, name = 'Supernode') => {
                    if (!nodeIds || nodeIds.length === 0) return null;
                    const state = get();
                    const view = getViewGraph(state);
                    const idSet = new Set(nodeIds);

                    /** @type {import('../domain/architecture/types.js').ArchEdge[]} */
                    const internalEdges = [];
                    /** @type {import('../domain/architecture/types.js').ArchEdge[]} */
                    const incomingEdges = [];
                    /** @type {import('../domain/architecture/types.js').ArchEdge[]} */
                    const outgoingEdges = [];
                    for (const e of view.edges) {
                        const sIn = idSet.has(e.source);
                        const tIn = idSet.has(e.target);
                        if (sIn && tIn) internalEdges.push(e);
                        else if (!sIn && tIn) incomingEdges.push(e);
                        else if (sIn && !tIn) outgoingEdges.push(e);
                    }

                    const externalSources = new Set(incomingEdges.map((e) => e.source));
                    const externalTargets = new Set(outgoingEdges.map((e) => e.target));
                    if (externalSources.size > 1 || externalTargets.size > 1) {
                        console.warn(
                            '[architectureStore] groupSelection: selection has multiple external boundaries; phase 3 supports 1-in / 1-out only.',
                        );
                        return null;
                    }

                    // Build the template's internal subgraph.
                    const internalNodes = {};
                    let minX = Infinity, minY = Infinity;
                    for (const id of nodeIds) {
                        const n = view.nodes[id];
                        if (!n) continue;
                        internalNodes[id] = n;
                        minX = Math.min(minX, n.position.x);
                        minY = Math.min(minY, n.position.y);
                    }
                    if (!Number.isFinite(minX)) { minX = 0; minY = 0; }

                    // Re-origin the internal positions so the template canvas
                    // starts near the top-left.
                    for (const id of Object.keys(internalNodes)) {
                        const n = internalNodes[id];
                        internalNodes[id] = {
                            ...n,
                            position: { x: n.position.x - minX + 200, y: n.position.y - minY + 160 },
                        };
                    }

                    // Add boundary blocks.
                    const tplInputEntry = getEntry('deepflow.template.Input');
                    const tplOutputEntry = getEntry('deepflow.template.Output');
                    const tplInput = createLayerNode({
                        libraryKey: 'deepflow.template.Input',
                        position: { x: 60, y: 40 },
                        entry: tplInputEntry,
                    });
                    const tplOutput = createLayerNode({
                        libraryKey: 'deepflow.template.Output',
                        position: { x: 60, y: 480 },
                        entry: tplOutputEntry,
                    });

                    /** @type {import('../domain/architecture/types.js').ArchEdge[]} */
                    const newInternalEdges = [...internalEdges];
                    // Wire boundary input → original incoming targets
                    for (const e of incomingEdges) {
                        newInternalEdges.push(createEdge({ source: tplInput.id, target: e.target }));
                    }
                    // Wire original outgoing sources → boundary output
                    for (const e of outgoingEdges) {
                        newInternalEdges.push(createEdge({ source: e.source, target: tplOutput.id }));
                    }

                    const templateGraph = {
                        nodes: { ...internalNodes, [tplInput.id]: tplInput, [tplOutput.id]: tplOutput },
                        edges: newInternalEdges,
                    };
                    const tpl = createTemplate({ name, description: '', graph: templateGraph });
                    useTemplatesStore.getState().createTemplate({
                        id: tpl.id,
                        name: tpl.name,
                        description: tpl.description,
                        graph: tpl.graph,
                    });

                    // Build the replacement instance in the current view.
                    const instance = createSuperNodeInstance({
                        templateId: tpl.id,
                        position: { x: minX, y: minY },
                        label: name,
                    });

                    updateCurrentGraph((g) => {
                        // remove the grouped nodes (also drops their edges via removeNode)
                        let next = g;
                        for (const id of nodeIds) next = removeNode(next, id);
                        // add the instance node
                        next = addNode(next, instance);
                        // re-route the external edges to/from the instance
                        for (const src of externalSources) {
                            next = addEdge(next, createEdge({ source: src, target: instance.id }));
                        }
                        for (const tgt of externalTargets) {
                            next = addEdge(next, createEdge({ source: instance.id, target: tgt }));
                        }
                        return next;
                    });

                    return tpl.id;
                },
            };
        },
        {
            name: STORAGE_KEY,
            partialize: (state) => ({ graph: state.graph, currentPath: state.currentPath }),
        },
    ),
);
