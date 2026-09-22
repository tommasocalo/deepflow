import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { withArchitectureHistory } from './architectureHistory.js';
import { exposeTemplateField } from '../domain/parameterBindings.js';
import {
    addEdge,
    addNode,
    createTemplate,
    moveNode,
    newDomainId,
    patchLayerParameters,
    patchNode,
    removeEdge,
    removeNode,
} from '../domain/architecture/index.js';

const STORAGE_KEY = 'deepflow:templates';
const HISTORY_ACTIONS = {
    createTemplate: 'Create template', cloneTemplate: 'Copy template', patchTemplate: 'Edit template', removeTemplate: 'Delete template',
    addNodeToTemplate: 'Add template node', removeNodeFromTemplate: 'Delete template node', patchTemplateNode: 'Change template node',
    patchTemplateLayerParameters: 'Change template settings', moveTemplateNode: 'Move template node',
    addEdgeToTemplate: 'Connect template nodes', removeEdgeFromTemplate: 'Delete template connection',
    addFreeParameter: 'Add template parameter', patchFreeParameter: 'Change template parameter', removeFreeParameter: 'Delete template parameter',
    exposeField: 'Expose component parameter',
};

/**
 * Templates store. Each template is a named, reusable subgraph that can be
 * instantiated as a SuperNodeInstance in any architecture or other template.
 * Editing a template propagates to every instance.
 */
export const useTemplatesStore = create(
    persist(
        (set, get) => withArchitectureHistory({
            /** @type {Object<string, import('../domain/architecture/types.js').ArchitectureTemplate>} */
            templates: {},

            createTemplate: ({ name, description, graph, freeParameters, id, ...metadata }) => {
                const tpl = { ...metadata, ...createTemplate({ name, description, graph, freeParameters, id }) };
                set((state) => ({ templates: { ...state.templates, [tpl.id]: tpl } }));
                return tpl.id;
            },

            getTemplate: (id) => get().templates[id],

            /** Copy a reusable definition and all nested definitions without retaining standard identity. */
            cloneTemplate: (id, name) => {
                const originals = get().templates;
                if (!originals[id]) return null;
                const copies = {};
                const copiedIds = new Map();
                const copy = (sourceId) => {
                    if (copiedIds.has(sourceId)) return copiedIds.get(sourceId);
                    const original = originals[sourceId];
                    if (!original) return sourceId;
                    const nextId = newDomainId();
                    copiedIds.set(sourceId, nextId);
                    const cloned = structuredClone(original);
                    cloned.componentStyleKey = original.componentStyleKey ?? original.standardKey;
                    delete cloned.standardKey;
                    cloned.id = nextId;
                    cloned.name = sourceId === id && name?.trim() ? name.trim() : `${original.name} copy`;
                    cloned.sourceTemplateId = sourceId;
                    for (const node of Object.values(cloned.graph.nodes)) {
                        if (['supernode', 'repeat'].includes(node.kind)) {
                            node.templateId = copy(node.templateId);
                            for (const override of Object.values(node.structureOverrides ?? {})) {
                                if (override?.templateId) override.templateId = copy(override.templateId);
                            }
                        }
                    }
                    copies[nextId] = cloned;
                    return nextId;
                };
                const copiedId = copy(id);
                set((state) => ({ templates: { ...state.templates, ...copies } }));
                return copiedId;
            },

            patchTemplate: (id, patch) =>
                set((state) =>
                    state.templates[id]
                        ? { templates: { ...state.templates, [id]: { ...state.templates[id], ...patch } } }
                        : state,
                ),

            removeTemplate: (id) =>
                set((state) => {
                    const { [id]: _gone, ...rest } = state.templates;
                    return { templates: rest };
                }),

            // ---- internal-graph operations ----

            addNodeToTemplate: (templateId, node) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, graph: addNode(tpl.graph, node) } },
                    };
                }),

            removeNodeFromTemplate: (templateId, nodeId) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, graph: removeNode(tpl.graph, nodeId) } },
                    };
                }),

            patchTemplateNode: (templateId, nodeId, patch) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, graph: patchNode(tpl.graph, nodeId, patch) } },
                    };
                }),

            patchTemplateLayerParameters: (templateId, nodeId, parameterPatch) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: {
                            ...state.templates,
                            [templateId]: { ...tpl, graph: patchLayerParameters(tpl.graph, nodeId, parameterPatch) },
                        },
                    };
                }),

            moveTemplateNode: (templateId, nodeId, position) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, graph: moveNode(tpl.graph, nodeId, position) } },
                    };
                }),

            addEdgeToTemplate: (templateId, edge) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, graph: addEdge(tpl.graph, edge) } },
                    };
                }),

            removeEdgeFromTemplate: (templateId, edgeId) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, graph: removeEdge(tpl.graph, edgeId) } },
                    };
                }),

            // ---- free parameters ----

            exposeField: (templateId, options) => {
                const template = exposeTemplateField(get().templates[templateId], options);
                if (!template) return false;
                set((state) => ({ templates: { ...state.templates, [templateId]: template } }));
                return true;
            },

            addFreeParameter: (templateId, paramSpec) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    const free = tpl.freeParameters ?? [];
                    if (free.some((p) => p.name === paramSpec.name)) return state;
                    return {
                        templates: {
                            ...state.templates,
                            [templateId]: { ...tpl, freeParameters: [...free, paramSpec] },
                        },
                    };
                }),

            patchFreeParameter: (templateId, name, patch) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    const free = (tpl.freeParameters ?? []).map((p) =>
                        p.name === name ? { ...p, ...patch } : p,
                    );
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, freeParameters: free } },
                    };
                }),

            removeFreeParameter: (templateId, name) =>
                set((state) => {
                    const tpl = state.templates[templateId];
                    if (!tpl) return state;
                    const free = (tpl.freeParameters ?? []).filter((p) => p.name !== name);
                    return {
                        templates: { ...state.templates, [templateId]: { ...tpl, freeParameters: free } },
                    };
                }),
        }, HISTORY_ACTIONS),
        { name: STORAGE_KEY },
    ),
);
