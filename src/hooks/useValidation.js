import { useMemo } from 'react';
import { useArchitectureStore } from '../stores/architectureStore.js';
import { useTemplatesStore } from '../stores/templatesStore.js';
import { validate } from '../validation/index.js';
import { resolveGraphLevel } from '../domain/architecture/index.js';

const EMPTY_REPORT = Object.freeze({
    topo: [],
    cyclic: [],
    shapes: {},
    errors: {},
    warnings: {},
    globalErrors: [],
    ok: true,
});

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
        const level = resolveGraphLevel(rootGraph, currentPath, templates);
        const tpl = templates[level.templateId];
        const innerScope = {};
        for (const fp of tpl?.freeParameters ?? []) {
            if (fp.default !== null && fp.default !== undefined && fp.default !== '') {
                const n = Number(fp.default);
                innerScope[fp.name] = Number.isFinite(n) ? n : fp.default;
            }
        }
        return { viewGraph: level.graph, scope: innerScope };
    }, [rootGraph, currentPath, templates]);

    return useMemo(() => {
        if (!viewGraph) return EMPTY_REPORT;
        try {
            return validate(viewGraph, { templates, scope });
        } catch (err) {
            console.error('[useValidation] validate() threw', err);
            return {
                ...EMPTY_REPORT,
                ok: false,
                globalErrors: ['Validation could not complete. Review the graph and try again.'],
            };
        }
    }, [viewGraph, templates, scope]);
}
