import { createContext, createElement, useContext } from 'react';
import { create, useStore } from 'zustand';
import { createStore } from 'zustand/vanilla';
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
    isValidConnection,
    moveNode,
    moveNodes,
    patchEdge,
    patchLayerParameters,
    patchNode,
    reconnectEdge,
    removeEdge,
    removeNode,
    resolveGraphLevel,
} from '../domain/architecture/index.js';
import { getEntry } from '../domain/library/index.js';
import { isValidLevelPath, nearestOpenAncestor, reconcileOpenLevels, sameLevelPath } from '../domain/architecture/levelNavigation.js';
import { containsTemplate } from '../domain/architecture/templateReferences.js';
import { copyGraphSelection, pasteGraphSelection } from '../domain/architecture/clipboard.js';
import { changeStructureScope, effectiveTemplateId, structureOverrideKey } from '../domain/architecture/instanceStructure.js';
import { instanceAsRepeat, repeatAsSingle, singleConversionIssue } from '../domain/architecture/repeatStructure.js';
import { componentBindingsForLayer, ensureStandardTemplate, getStandardComponentForLayer, STANDARD_COMPONENTS } from '../domain/components/index.js';
import { useTemplatesStore } from './templatesStore.js';
import { architectureDocumentReplaced, configureArchitectureHistory, isRestoringArchitectureHistory, withArchitectureHistory } from './architectureHistory.js';

const STORAGE_KEY = 'deepflow:architecture-graph';
const HISTORY_ACTIONS = {
    addLayerByLibraryKey: 'Add operation', addSuperNodeInstance: 'Add component', addStandardComponent: 'Add component',
    convertLayerToComponent: 'Open component definition', cloneComponent: 'Make independent component', cloneCurrentComponent: 'Make independent component',
    addRepeatNode: 'Add repeat', convertInstanceToRepeat: 'Repeat component', convertRepeatToInstance: 'Remove repeat',
    patchRepeatNode: 'Change repeat', patchRepeatBinding: 'Change repeat parameter',
    deleteElements: 'Delete selection', removeNode: 'Delete node', patchNode: 'Change node', patchLayerParameters: 'Change node settings',
    moveNode: 'Move node', moveNodes: 'Move nodes', connect: 'Connect nodes', reconnectEdge: 'Reconnect nodes',
    removeEdge: 'Delete connection', patchEdge: 'Change connection', setEdgeGradientOp: 'Change gradient flow',
    resetCurrentGraph: 'Reset graph', groupSelection: 'Group nodes',
    pasteSelection: 'Paste nodes', setStructureScope: 'Change structure scope',
};

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
    return resolveGraphLevel(state.graph, state.currentPath, useTemplatesStore.getState().templates, state.inspectedIterations);
}

/**
 * Read the graph at the current view level without mutating anything.
 */
function getViewGraph(state) {
    return resolveLevel(state).graph;
}

/** Only return changed navigation fields: moving nodes must not cause extra writes. */
function navigationPatch(state, graph = state.graph) {
    const next = reconcileOpenLevels(graph, useTemplatesStore.getState().templates, state.openPaths, state.currentPath, state.inspectedIterations);
    const patch = {};
    if (next.openPaths !== state.openPaths) patch.openPaths = next.openPaths;
    if (next.currentPath !== state.currentPath) {
        patch.currentPath = next.currentPath;
        patch.selectedNodeIds = [];
        patch.expandedNodeId = null;
    }
    return patch;
}

/** The same actions run against the main workspace or an independent pane. */
const createArchitectureState = ({ graph = initialGraph(), path = [], isActive = () => true } = {}) => (set, get) => {
    const storeId = Symbol('architecture-pane');
    const updateCurrentGraph = (transform) => {
        if (!isActive()) return;
        const state = get();
        const level = resolveLevel(state);
        if (level.isRoot) {
            const graph = transform(state.graph);
            if (graph !== state.graph) set({ graph, ...navigationPatch(state, graph) });
        } else {
            const tplStore = useTemplatesStore.getState();
            const tpl = tplStore.getTemplate(level.templateId);
            if (!tpl) return;
            const graph = transform(tpl.graph);
            if (graph !== tpl.graph) tplStore.patchTemplate(level.templateId, { graph });
        }
    };

    const matchesContext = (expected) => {
        if (!isActive()) return false;
        if (!expected) return true;
        const state = get();
        return (!expected.storeId || expected.storeId === storeId) &&
            expected.levelKey === JSON.stringify(state.currentPath) &&
            (!expected.graph || expected.graph === getViewGraph(state));
    };

    const canInsertTemplate = (templateId) => {
        if (!isActive()) return false;
        const level = resolveLevel(get());
        // A drag can start in another pane, whose palette has a different
        // editing context. Validate the receiving template before insertion.
        return level.isRoot || !containsTemplate(templateId, level.templateId, useTemplatesStore.getState().templates);
    };

    const cloneComponentAtPath = (path, name) => {
        if (!isActive() || !path.length) return null;
        const state = get();
        const templates = useTemplatesStore.getState();
        if (!isValidLevelPath(state.graph, templates.templates, path, state.inspectedIterations)) return null;
        const top = state.graph.nodes[path[0]];
        // Fork the containing definition too: rebinding a nested component
        // must not mutate the parent used by another root instance.
        const copiedTop = templates.cloneTemplate(top.templateId, path.length === 1 ? name : undefined);
        if (!copiedTop) return null;
        const graph = patchNode(state.graph, top.id, { templateId: copiedTop });
        const clonedLevel = resolveGraphLevel(graph, path, useTemplatesStore.getState().templates, state.inspectedIterations);
        if (path.length > 1 && name?.trim()) templates.patchTemplate(clonedLevel.templateId, { name: name.trim() });
        set({ graph, ...navigationPatch(state, graph) });
        return clonedLevel.templateId;
    };

    return withArchitectureHistory({
        graph,
        /** @type {string[]} ids of currently-selected nodes in the active view */
        selectedNodeIds: [],
        /** @type {string | null} id of the currently expanded node, or null */
        expandedNodeId: null,
        /** @type {string[]} ids of supernode or repeat instances entered, root → leaf */
        currentPath: path,
        /** @type {string[][]} open non-root levels, in insertion order */
        openPaths: path.length ? [path] : [],
        /** @type {string[]} library keys of recently-added blocks (most recent first) */
        recentLibraryKeys: [],
        /** Per-pane previews of a repeat's shared body, keyed by its absolute level path. */
        inspectedIterations: {},
        showDimensions: true,
        setShowDimensions: (value) => set({ showDimensions: Boolean(value) }),
        showGradientFlow: false,
        setShowGradientFlow: (value) => set({ showGradientFlow: Boolean(value) }),

        /** @returns {import('../domain/architecture/types.js').ArchitectureGraph} */
        getViewGraph: () => getViewGraph(get()),
        getLevel: () => resolveLevel(get()),
        getMutationContext: () => ({ storeId, levelKey: JSON.stringify(get().currentPath), graph: getViewGraph(get()) }),

        getStructureScope: (path = get().currentPath) => {
            const state = get();
            const templates = useTemplatesStore.getState().templates;
            const level = resolveGraphLevel(state.graph, path, templates, state.inspectedIterations);
            if (!path.length || level.isRoot) return { available: false, local: false, label: '', index: null, hasSavedLocal: false };
            const override = level.instance.structureOverrides?.[structureOverrideKey(level.instance, level.index ?? 0)];
            return { available: true, local: !!override?.enabled, label: level.label, index: level.index,
                hasSavedLocal: !!templates[override?.templateId] };
        },

        setStructureScope: (scope, path = get().currentPath) => {
            if (!isActive() || !['shared', 'local'].includes(scope)) return false;
            const state = get();
            const current = state.getStructureScope(path);
            if (!current.available) return false;
            if (current.local === (scope === 'local')) return true;
            const result = changeStructureScope(state.graph, useTemplatesStore.getState().templates, path, state.inspectedIterations, scope === 'local');
            if (!result) return false;
            useTemplatesStore.setState({ templates: result.templates });
            set({ graph: result.graph, ...sharedGraphPatch(get(), result.graph), expandedNodeId: null });
            return true;
        },

        copySelection: () => copyGraphSelection(getViewGraph(get()), get().selectedNodeIds),

        pasteSelection: (selection, offset, expected) => {
            if (!matchesContext(expected)) return { error: 'The destination changed. Paste again in the current graph.' };
            const level = resolveLevel(get());
            const result = pasteGraphSelection(level.graph, selection, { offset, templateId: level.templateId, templates: useTemplatesStore.getState().templates });
            if (result.error) return result;
            // Shared definitions stay shared; private versions belong to the copy.
            for (const id of result.nodeIds) {
                const node = result.graph.nodes[id];
                if (!node.structureOverrides) continue;
                const overrides = Object.fromEntries(Object.entries(node.structureOverrides).map(([key, override]) => {
                    const source = useTemplatesStore.getState().getTemplate(override.templateId);
                    const templateId = useTemplatesStore.getState().cloneTemplate(override.templateId, source?.name);
                    return [key, { ...override, templateId: templateId ?? override.templateId }];
                }));
                result.graph = patchNode(result.graph, id, { structureOverrides: overrides });
            }
            updateCurrentGraph(() => result.graph);
            set({ selectedNodeIds: result.nodeIds, expandedNodeId: null });
            return { nodeIds: result.nodeIds };
        },

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
            if (!canInsertTemplate(templateId)) return null;
            const node = createSuperNodeInstance({ templateId, position, label });
            updateCurrentGraph((g) => addNode(g, node));
            return node.id;
        },

        addStandardComponent: (key, position) => {
            if (!isActive()) return null;
            const component = STANDARD_COMPONENTS.find((item) => item.key === key);
            if (!component) return null;
            const templateId = ensureStandardTemplate(key, useTemplatesStore);
            return get().addSuperNodeInstance(templateId, position, component.label);
        },

        convertLayerToComponent: (id) => {
            if (!isActive()) return null;
            const view = getViewGraph(get());
            const node = view.nodes[id];
            const component = getStandardComponentForLayer(node);
            if (!component || view.edges.filter((edge) => edge.target === id).length > 1) return null;
            const templateId = ensureStandardTemplate(component.key, useTemplatesStore);
            if (!canInsertTemplate(templateId)) return null;
            const converted = { ...node, kind: 'supernode', templateId, parameterValues: componentBindingsForLayer(node) };
            delete converted.libraryKey;
            delete converted.parameters;
            const inputId = Object.values(useTemplatesStore.getState().getTemplate(templateId).graph.nodes).find((item) => item.libraryKey === 'deepflow.template.Input')?.id;
            updateCurrentGraph((graph) => ({ ...graph, nodes: { ...graph.nodes, [id]: converted },
                edges: graph.edges.some((edge) => edge.target === id && edge.targetHandle)
                    ? graph.edges.map((edge) => edge.target === id && edge.targetHandle ? { ...edge, targetHandle: inputId } : edge) : graph.edges }));
            return id;
        },

        cloneComponent: (id, name) => cloneComponentAtPath([...get().currentPath, id], name),
        cloneCurrentComponent: (name) => cloneComponentAtPath(get().currentPath, name),

        addRepeatNode: (templateId, position, opts = {}) => {
            const node = createRepeatNode({ templateId, position, ...opts });
            if (!canInsertTemplate(node.templateId)) return null;
            updateCurrentGraph((g) => addNode(g, node));
            return node.id;
        },

        /** Preserve repeat configuration when a component is toggled to Single. */
        convertInstanceToRepeat: (instanceId, count, mode) => {
            const node = getViewGraph(get()).nodes[instanceId];
            if (!node || node.kind !== 'supernode') return null;
            const repeat = instanceAsRepeat(node, count, mode);
            updateCurrentGraph((graph) => ({ ...graph, nodes: { ...graph.nodes, [node.id]: repeat } }));
            return repeat.id;
        },

        getSingleConversionIssue: (id) => singleConversionIssue(getViewGraph(get()).nodes[id], useTemplatesStore.getState().templates),

        /** Single uses iteration 0; unchanged formulas return when repeated again. */
        convertRepeatToInstance: (repeatId) => {
            const node = getViewGraph(get()).nodes[repeatId];
            const templates = useTemplatesStore.getState().templates;
            if (singleConversionIssue(node, templates)) return null;
            const instance = repeatAsSingle(node, templates[effectiveTemplateId(node, 0)]);
            updateCurrentGraph((graph) => ({ ...graph, nodes: { ...graph.nodes, [node.id]: instance } }));
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

        deleteElements: ({ nodeIds = [], edgeIds = [] }, expected) => {
            if (!matchesContext(expected)) return false;
            updateCurrentGraph((graph) => {
                let next = graph;
                for (const id of nodeIds) next = removeNode(next, id);
                for (const id of edgeIds) next = removeEdge(next, id);
                return next;
            });
            const view = getViewGraph(get());
            set((state) => ({
                selectedNodeIds: state.selectedNodeIds.filter((id) => view.nodes[id]),
                expandedNodeId: view.nodes[state.expandedNodeId] ? state.expandedNodeId : null,
            }));
            return true;
        },

        removeNode: (id) => {
            updateCurrentGraph((g) => removeNode(g, id));
            set((state) => ({
                selectedNodeIds: state.selectedNodeIds.filter((nodeId) => nodeId !== id),
                expandedNodeId: state.expandedNodeId === id ? null : state.expandedNodeId,
            }));
        },

        patchNode: (id, patch) => updateCurrentGraph((g) => patchNode(g, id, patch)),

        patchLayerParameters: (id, parameterPatch) =>
            updateCurrentGraph((g) => patchLayerParameters(g, id, parameterPatch)),

        moveNode: (id, position) => updateCurrentGraph((g) => moveNode(g, id, position)),

        moveNodes: (moves, expected) => {
            // Delayed events belong to one pane, level, and graph revision.
            if (!matchesContext(expected)) return false;
            updateCurrentGraph((g) => moveNodes(g, moves));
            return true;
        },

        connect: (connection, expected) => {
            if (!matchesContext(expected) || !isValidConnection(getViewGraph(get()), connection)) return null;
            const edge = createEdge(connection);
            updateCurrentGraph((g) => addEdge(g, edge));
            return edge.id;
        },

        reconnectEdge: (edgeId, connection, expected) => {
            if (!matchesContext(expected)) return false;
            const graph = getViewGraph(get());
            if (!graph.edges.some((edge) => edge.id === edgeId) || !isValidConnection(graph, connection, edgeId)) return false;
            updateCurrentGraph((g) => reconnectEdge(g, edgeId, connection));
            return true;
        },

        removeEdge: (edgeId, expected) => {
            if (!matchesContext(expected)) return false;
            const graph = getViewGraph(get());
            if (!graph.edges.some((edge) => edge.id === edgeId)) return false;
            updateCurrentGraph((g) => removeEdge(g, edgeId));
            return true;
        },

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
            if (!isActive()) return;
            const level = resolveLevel(get());
            if (level.isRoot) {
                set({ graph: initialGraph(), selectedNodeIds: [], expandedNodeId: null, currentPath: [], openPaths: [], inspectedIterations: {} });
            } else {
                useTemplatesStore.getState().patchTemplate(level.templateId, { graph: emptyGraph() });
                set({ selectedNodeIds: [], expandedNodeId: null });
            }
        },

        // ---- navigation ----

        openLevel: (path) => {
            const state = get();
            if (!isValidLevelPath(state.graph, useTemplatesStore.getState().templates, path, state.inspectedIterations)) return false;
            const currentPath = [...path];
            const openPaths = currentPath.length && !state.openPaths.some((open) => sameLevelPath(open, currentPath))
                ? [...state.openPaths, currentPath] : state.openPaths;
            if (sameLevelPath(state.currentPath, currentPath) && openPaths === state.openPaths) return true;
            set({ currentPath, openPaths, selectedNodeIds: [], expandedNodeId: null });
            return true;
        },

        closeLevel: (path) => {
            const state = get();
            const openPaths = state.openPaths.filter((open) => !sameLevelPath(open, path));
            if (openPaths.length === state.openPaths.length) return false;
            if (sameLevelPath(state.currentPath, path)) {
                set({ openPaths, currentPath: nearestOpenAncestor(path, openPaths), selectedNodeIds: [], expandedNodeId: null });
            } else {
                set({ openPaths });
            }
            return true;
        },

        enterSupernode: (supernodeId) => get().openLevel([...get().currentPath, supernodeId]),

        setInspectedIteration: (path, index) => {
            if (!Array.isArray(path) || !path.length || !Number.isInteger(index) || index < 0) return false;
            const state = get();
            if (!isValidLevelPath(state.graph, useTemplatesStore.getState().templates, path, state.inspectedIterations)) return false;
            const parent = resolveGraphLevel(state.graph, path.slice(0, -1), useTemplatesStore.getState().templates, state.inspectedIterations);
            if (parent.graph.nodes[path.at(-1)]?.kind !== 'repeat') return false;
            const key = JSON.stringify(path);
            if (state.inspectedIterations[key] !== index) {
                const inspectedIterations = { ...state.inspectedIterations, [key]: index };
                set({ inspectedIterations, ...sharedGraphPatch({ ...state, inspectedIterations }) });
            }
            return true;
        },

        enterRepeatIteration: (repeatId, index) => {
            const path = [...get().currentPath, repeatId];
            if (!get().setInspectedIteration(path, index)) return false;
            return get().openLevel(path);
        },

        exitSupernode: () => {
            const path = get().currentPath;
            if (path.length) get().openLevel(path.slice(0, -1));
        },

        goToLevel: (depth) => get().openLevel(get().currentPath.slice(0, Math.max(0, depth))),

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
            if (!isActive() || !nodeIds || nodeIds.length === 0) return null;
            const state = get();
            const view = getViewGraph(state);
            const idSet = new Set(nodeIds);
            if (nodeIds.some((id) => !view.nodes[id])) return null;

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

            // A standalone layer or disconnected chain still needs usable
            // template boundaries when wrapped from the creation toolbar.
            const entryIds = nodeIds.filter((id) => !internalEdges.some((edge) => edge.target === id));
            const exitIds = nodeIds.filter((id) => !internalEdges.some((edge) => edge.source === id));
            if ((!incomingEdges.length && entryIds.length !== 1) || (!outgoingEdges.length && exitIds.length !== 1)) return null;

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
                newInternalEdges.push(createEdge({ source: tplInput.id, target: e.target, targetHandle: e.targetHandle }));
            }
            // Wire original outgoing sources → boundary output
            for (const e of outgoingEdges) {
                newInternalEdges.push(createEdge({ source: e.source, sourceHandle: e.sourceHandle, target: tplOutput.id }));
            }
            if (!incomingEdges.length) newInternalEdges.push(createEdge({ source: tplInput.id, target: entryIds[0] }));
            if (!outgoingEdges.length) newInternalEdges.push(createEdge({ source: exitIds[0], target: tplOutput.id }));

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
    }, HISTORY_ACTIONS);
};

const architectureStore = create(
    persist(
        createArchitectureState(),
        {
            name: STORAGE_KEY,
            partialize: (state) => ({ graph: state.graph, currentPath: state.currentPath, openPaths: state.openPaths, inspectedIterations: state.inspectedIterations }),
            merge: (persisted, current) => {
                const saved = persisted ?? {};
                const graph = saved.graph ?? current.graph;
                const currentPath = Array.isArray(saved.currentPath) ? saved.currentPath : [];
                // Earlier workspaces only saved a breadcrumb. Keep its visible
                // ancestors as tabs when restoring that older format.
                const savedPaths = saved.openPaths ?? currentPath.map((_, index) => currentPath.slice(0, index + 1));
                const openPaths = [...(Array.isArray(savedPaths) ? savedPaths : []), currentPath];
                const navigation = reconcileOpenLevels(graph, useTemplatesStore.getState().templates, openPaths, currentPath, saved.inspectedIterations);
                return { ...current, ...saved, graph, ...navigation };
            },
        },
    ),
);

/** Reconcile view state without disturbing valid selections in other panes. */
function sharedGraphPatch(state, graph = state.graph) {
    const patch = navigationPatch(state, graph);
    const view = getViewGraph({ ...state, ...patch, graph });
    const selected = patch.selectedNodeIds ?? state.selectedNodeIds;
    const selectedNodeIds = selected.filter((id) => view.nodes[id]);
    if (selectedNodeIds.length !== selected.length) patch.selectedNodeIds = selectedNodeIds;
    const expanded = patch.expandedNodeId ?? state.expandedNodeId;
    if (expanded && !view.nodes[expanded]) patch.expandedNodeId = null;
    return patch;
}

function subscribeTemplateNavigation(store) {
    return useTemplatesStore.subscribe((state, previous) => {
        if (state.templates === previous.templates || isRestoringArchitectureHistory()) return;
        const architecture = store.getState();
        if (!architecture.openPaths.length && !architecture.currentPath.length) return;
        const patch = sharedGraphPatch(architecture);
        if (Object.keys(patch).length) store.setState(patch);
    });
}

// Restore document state before reconciling navigation: a cloned template may
// disappear together with the instance that refers to it in the same undo.
const documentPanes = new Set([architectureStore]);
configureArchitectureHistory({
    capture: () => ({ graph: architectureStore.getState().graph, templates: useTemplatesStore.getState().templates }),
    restore: ({ graph, templates }) => {
        useTemplatesStore.setState({ templates });
        for (const pane of documentPanes) {
            pane.setState({ graph, ...sharedGraphPatch(pane.getState(), graph) });
        }
    },
});
const unsubscribeGraphHistory = architectureStore.subscribe((state, previous) => {
    if (state.graph !== previous.graph) architectureDocumentReplaced();
});
const unsubscribeTemplateHistory = useTemplatesStore.subscribe((state, previous) => {
    if (state.templates !== previous.templates) architectureDocumentReplaced();
});

// Every pane resolves the same templates, while keeping its own open levels.
const unsubscribeTemplateNavigation = subscribeTemplateNavigation(architectureStore);
if (import.meta.hot) import.meta.hot.dispose(() => {
    unsubscribeTemplateNavigation();
    unsubscribeGraphHistory();
    unsubscribeTemplateHistory();
});

const ArchitectureStoreContext = createContext(null);

export function ArchitectureStoreProvider({ store, children }) {
    return createElement(ArchitectureStoreContext.Provider, { value: store }, children);
}

export function useArchitectureStoreApi() {
    return useContext(ArchitectureStoreContext) ?? architectureStore;
}

export function useArchitectureStore(selector) {
    return useStore(useArchitectureStoreApi(), selector);
}

// Existing import/export, persistence, and test callers still address the main
// workspace. Components read their nearest provider through the hook above.
Object.assign(useArchitectureStore, architectureStore);

/**
 * A non-persisted editor view. Only root graph references synchronize with the
 * main workspace; navigation, selection, and expanded cards stay local.
 */
export function createArchitecturePaneStore({ path = [] } = {}) {
    const graph = architectureStore.getState().graph;
    const inspectedIterations = { ...architectureStore.getState().inspectedIterations };
    const currentPath = isValidLevelPath(graph, useTemplatesStore.getState().templates, path, inspectedIterations) ? [...path] : [];
    let active = true;
    const pane = createStore(createArchitectureState({ graph, path: currentPath, isActive: () => active }));
    pane.setState({ inspectedIterations });
    documentPanes.add(pane);
    const unsubscribeMain = architectureStore.subscribe((state, previous) => {
        if (state.graph === previous.graph || pane.getState().graph === state.graph) return;
        const local = pane.getState();
        pane.setState({ graph: state.graph, ...sharedGraphPatch(local, state.graph) });
    });
    const unsubscribePane = pane.subscribe((state, previous) => {
        if (state.graph === previous.graph || architectureStore.getState().graph === state.graph) return;
        const main = architectureStore.getState();
        architectureStore.setState({ graph: state.graph, ...sharedGraphPatch(main, state.graph) });
    });
    const unsubscribeTemplates = subscribeTemplateNavigation(pane);
    pane.dispose = () => {
        active = false;
        documentPanes.delete(pane);
        unsubscribeMain();
        unsubscribePane();
        unsubscribeTemplates();
    };
    return pane;
}
