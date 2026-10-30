import { countParametersByNode } from '../../../domain/optimization/parameterCount.js';
import { prepareDataset } from '../../data/prepare.js';
import { getDataset } from '../../data/registry.js';
import { matchNodes } from '../engine/nodes.js';

/**
 * Fact helpers (docs/edu/SPEC.md §6.3.6). A module fact is `{ value, at?, compute?, source? }`:
 * - `value` is what the lesson text shows through `{name}`;
 * - `compute(helpers)` recomputes it with the app's own counters and validator, on the workspace state reached after
 *   the Show me of step `at` ('start' = the start fixture, before any step). The content test
 *   (scripts/edu-content.test.mjs) walks every module and asserts `compute` returns `value`;
 * - `source: 'simulation' | 'literature'` marks numbers no test can recompute (see `note`).
 *
 * `helpers` is built from a CheckContext of that state; `fact(moduleId, name)` reads another verified fact.
 */
export function createFactHelpers(ctx, { facts = {} } = {}) {
    const options = { templates: ctx.templates, scope: ctx.scope, graph: ctx.graph, report: ctx.report, order: ctx.order };
    const find = (match) => {
        const node = matchNodes(ctx.graph, match, options)[0];
        if (!node) throw new Error(`No node matches ${JSON.stringify(match)}`);
        return node;
    };
    let byNode = null;
    return {
        ctx,
        /** countArchitectureParameters total; throws if the count is incomplete. */
        total: () => {
            if (!ctx.paramCount?.complete) throw new Error(`Incomplete parameter count: ${(ctx.paramCount?.unsupported ?? []).join(', ')}`);
            return ctx.paramCount.total;
        },
        /** countParametersByNode for one root node. */
        params: (match) => {
            byNode ??= countParametersByNode(ctx.graph, ctx.templates);
            const entry = byNode[find(match).id];
            if (!entry?.complete) throw new Error(`Incomplete parameter count for ${JSON.stringify(match)}`);
            return entry.total;
        },
        /** The inferred output shape of a root node, batch axis as null. */
        shape: (match) => ctx.report.shapes?.[find(match).id] ?? null,
        /** A root global parameter, resolved. */
        global: (name) => ctx.scope?.[name],
        /** The bound dataset (bridge source 'data'). */
        data: () => ctx.data,
        /** The bound dataset's full DatasetSpec from the registry (sizes, classes, normalisation). */
        dataset: () => {
            const spec = getDataset(ctx.data?.datasetId);
            if (!spec) throw new Error('No dataset is bound');
            return spec;
        },
        /** The bound dataset after the split and scaling of §5.3 (async): `(await prepared()).train.n`. */
        prepared: async () => {
            const binding = ctx.data;
            const spec = getDataset(binding?.datasetId);
            if (!spec) throw new Error('No dataset is bound');
            return prepareDataset(spec, binding.options ?? {}, binding.split, binding.dataSeed ?? binding.options?.seed ?? 1);
        },
        /** The trainer source: settings and runs. */
        trainer: () => ctx.trainer,
        /** Another fact's verified value: fact('m3', 'mlpParams'). */
        fact: (moduleId, name) => {
            const value = facts[moduleId]?.[name];
            if (value === undefined) throw new Error(`Unknown fact ${moduleId}.${name}`);
            return value;
        },
        /** Natural log rounded to two decimals, as text ('2.30'), the way lessons print losses. */
        lnText: (value) => Math.log(value).toFixed(2),
    };
}
