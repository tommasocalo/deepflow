import { useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { validate } from '../validation/index.js';

const EMPTY_REPORT = Object.freeze({
    topo: [],
    cyclic: [],
    shapes: {},
    errors: {},
    warnings: {},
    globalErrors: [],
    ok: true,
});

let lastKey = null;
let lastReport = EMPTY_REPORT;

/**
 * Resolve the graph at the current view level (walks currentPath through
 * the templates store), then validate it with the templates map plumbed in
 * so SuperNodeInstance shape inference can recurse through templates.
 *
 * @returns {import('../validation/validator.js').ValidationReport}
 */
export function useValidation() {
    const rootGraph = useArchitectureStore((s) => s.graph);
    const currentPath = useArchitectureStore((s) => s.currentPath);
    const templates = useTemplatesStore((s) => s.templates);

    const { viewGraph, scope } = useMemo(() => {
        if (!rootGraph || typeof rootGraph !== 'object' || !rootGraph.nodes) {
            return { viewGraph: null, scope: {} };
        }
        if (!currentPath || currentPath.length === 0) {
            return { viewGraph: rootGraph, scope: {} };
        }
        let parentGraph = rootGraph;
        let tpl = null;
        for (const sid of currentPath) {
            const node = parentGraph.nodes?.[sid];
            if (!node || node.kind !== 'supernode') {
                return { viewGraph: rootGraph, scope: {} };
            }
            tpl = templates?.[node.templateId];
            if (!tpl || !tpl.graph) {
                return { viewGraph: rootGraph, scope: {} };
            }
            parentGraph = tpl.graph;
        }
        const innerScope = {};
        for (const fp of tpl?.freeParameters ?? []) {
            if (fp.default !== null && fp.default !== undefined && fp.default !== '') {
                const n = Number(fp.default);
                innerScope[fp.name] = Number.isFinite(n) ? n : fp.default;
            }
        }
        return { viewGraph: parentGraph, scope: innerScope };
    }, [rootGraph, currentPath, templates]);

    if (!viewGraph) return EMPTY_REPORT;

    const key = `${viewGraph === rootGraph ? 'root' : 'tpl'}:${currentPath.join('/')}:${stamp(templates)}:${stamp(viewGraph)}`;

    if (key !== lastKey) {
        lastKey = key;
        try {
            lastReport = validate(viewGraph, { templates, scope });
        } catch (err) {
            console.error('[useValidation] validate() threw', err);
            lastReport = EMPTY_REPORT;
        }
    }
    return lastReport ?? EMPTY_REPORT;
}

// Cheap stable-ish hash so we re-validate when nested structures change
// even if their outer reference happens to be preserved.
function stamp(obj) {
    try {
        const s = JSON.stringify(obj);
        let h = 2166136261;
        for (let i = 0; i < s.length; i++) {
            h ^= s.charCodeAt(i);
            h = Math.imul(h, 16777619);
        }
        return (h >>> 0).toString(16);
    } catch {
        return '0';
    }
}
