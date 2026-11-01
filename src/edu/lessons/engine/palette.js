import { cachedValidate } from '../../../domain/architecture/levelContext.js';
import { withGlobalParameters } from '../../../domain/expressions.js';
import { matchNodes } from './nodes.js';

/**
 * Palette chips (docs/edu/SPEC.md §2 P3, §6.3.1 PaletteItem). A chip adds its block; two lesson-only refinements keep a
 * novice out of dead ends (review r1): a chip with `into: { from, to }` splices its block into that wire while the wire
 * exists (M2 s3: ReLU into Linear 1 → Linear 2), and a Linear chip starts at the width that reaches the model's open
 * end instead of DeepFlow's 768.
 */

/** The root nodes a palette item's `into` names, when the wire between them exists; otherwise null. */
export function spliceTarget(item, graph, templates = {}) {
    if (!item?.into?.from || !item.into.to || !graph?.nodes) return null;
    const options = { templates, scope: withGlobalParameters(graph), graph };
    const from = matchNodes(graph, item.into.from, options)[0];
    const to = matchNodes(graph, item.into.to, options)[0];
    if (!from || !to || !(graph.edges ?? []).some((edge) => edge.source === from.id && edge.target === to.id)) return null;
    return { from, to };
}

/** Where a chip's block goes when it cannot splice: beside the middle of the wire it belongs in, clear of the column. */
export function besideWire(item, graph, templates = {}) {
    if (!item?.into?.from || !item.into.to || !graph?.nodes) return null;
    const options = { templates, scope: withGlobalParameters(graph), graph };
    const from = matchNodes(graph, item.into.from, options)[0];
    const to = matchNodes(graph, item.into.to, options)[0];
    if (!from || !to) return null;
    return { x: Math.max(from.position.x, to.position.x) + 260, y: Math.round((from.position.y + to.position.y) / 2) };
}

/**
 * The feature width at the model's open end: the block furthest from the Input that feeds nothing yet (the Output
 * aside), or null. M2 s7 starts with an Input of width 2; once a Linear 2 → 8 hangs off it, the open end is 8 wide.
 */
export function openEndWidth(graph, templates = {}) {
    if (!graph?.nodes) return null;
    const input = Object.values(graph.nodes).find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Input');
    if (!input) return null;
    const report = cachedValidate(graph, templates, withGlobalParameters(graph), {});
    const next = new Map();
    for (const edge of graph.edges ?? []) {
        if (!next.has(edge.source)) next.set(edge.source, []);
        next.get(edge.source).push(edge.target);
    }
    const depth = new Map([[input.id, 0]]);
    const queue = [input.id];
    while (queue.length) {
        const id = queue.shift();
        for (const target of next.get(id) ?? []) {
            if (depth.has(target)) continue;
            depth.set(target, depth.get(id) + 1);
            queue.push(target);
        }
    }
    const open = [...depth.keys()].filter((id) => !next.has(id) && graph.nodes[id]?.libraryKey !== 'deepflow.io.Output');
    open.sort((left, right) => depth.get(right) - depth.get(left));
    const shape = open.length ? report.shapes?.[open[0]] : null;
    const width = Array.isArray(shape) ? shape.at(-1) : null;
    return Number.isSafeInteger(width) && width > 0 ? width : null;
}

/** The parameters a chip gives its new block: the item's own, and for a Linear the open end's width. */
export function chipParams(item, graph, templates = {}) {
    const params = { ...(item?.params ?? {}) };
    if (item?.block === 'component:linear') {
        const width = openEndWidth(graph, templates);
        if (width && params.in_features === undefined) params.in_features = width;
        if (params.out_features === undefined && (params.in_features ?? width)) params.out_features = params.in_features ?? width;
    }
    return Object.keys(params).length ? params : null;
}
