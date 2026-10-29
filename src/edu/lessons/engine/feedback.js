import { getIncomingEdges } from '../../../domain/graphInputs.js';
import { paramLabel } from './i18n.js';
import { nodeType, orderedNodeIds } from './nodes.js';

/**
 * Friendly validation feedback (docs/edu/SPEC.md §6.4.4). DeepFlow's validator reports errors per root node,
 * including joined "Template … has N internal errors: a; b" strings for components and repeats. Each message is
 * rewritten into a short, specific sentence for the lesson panel; GraphStatusPanel itself is unchanged.
 */
const product = (shape) => shape.slice(1).every((axis) => Number.isSafeInteger(axis) && axis > 0)
    ? shape.slice(1).reduce((total, axis) => total * axis, 1) : null;

function inputShapeOf(graph, report, nodeId) {
    const edge = getIncomingEdges(graph, nodeId)[0];
    return edge ? report?.shapes?.[edge.source] ?? null : null;
}

/** Linear-like width errors become a missing-Flatten hint when the input still has image axes. */
function linearIssue(graph, report, node, expected, actual) {
    const input = inputShapeOf(graph, report, node.id);
    if (Array.isArray(input) && input.length >= 3 && product(input) === expected) {
        return { key: 'fb.missing-flatten', params: { expected, actual } };
    }
    return { key: 'fb.linear-in', params: { expected, actual } };
}

/** Ordered rules: the first that matches a message wins. Patterns follow validator.js and shapes.js. */
const RULES = [
    {
        pattern: /Linear in_features=(\d+) but input has (\d+) features/,
        rewrite: (match, { graph, report, node }) => linearIssue(graph, report, node, Number(match[1]), Number(match[2])),
    },
    {
        // Inside a standard Linear component: input (…, actual) × weightᵀ (expected, out).
        pattern: /MatMul inner dimensions mismatch: (\d+) vs (\d+)/,
        when: ({ type }) => type === 'Linear',
        rewrite: (match, { graph, report, node }) => linearIssue(graph, report, node, Number(match[2]), Number(match[1])),
    },
    {
        pattern: /Conv2d in_channels=(\d+) but input has (\d+) channels/,
        rewrite: (match) => ({ key: 'fb.conv-in', params: { expected: Number(match[1]), actual: Number(match[2]) } }),
    },
    {
        pattern: /Conv2d expects 4D \(B, C, H, W\); got rank (\d+)/,
        rewrite: (match) => ({ key: 'fb.conv-rank', params: { rank: Number(match[1]) } }),
    },
    {
        pattern: /Split dimension (\d+) must be divisible by (\d+) chunks/,
        rewrite: (match) => ({ key: 'fb.heads-divide', params: { dim: Number(match[1]), heads: Number(match[2]) } }),
    },
    {
        pattern: /(BatchNorm2d|BatchNorm1d|LayerNorm) \w+=(\d+) but input has (\d+)/,
        rewrite: (match) => ({ key: 'fb.norm-width', params: { expected: Number(match[2]), actual: Number(match[3]) } }),
    },
    {
        pattern: /Element-wise dim \S+ mismatch: ([\d\s vs]+)/,
        rewrite: (match, { type }) => type === 'Add'
            ? { key: 'fb.add-shapes', params: {} }
            : { key: 'fb.elementwise', params: { widths: match[1].trim().replace(/\s+vs\s+/g, ' vs ') } },
    },
    {
        pattern: /(?:^|[:;]\s*)(\w+) is required/,
        rewrite: (match) => ({ key: 'fb.required', params: { param: paramLabel(match[1]), name: match[1] } }),
    },
];

function rewrite(message, context) {
    for (const rule of RULES) {
        if (rule.when && !rule.when(context)) continue;
        const match = message.match(rule.pattern);
        if (match) return rule.rewrite(match, context);
    }
    return { key: 'fb.generic', params: { message } };
}

/**
 * @returns {{ nodeId: string|null, key: string, params: object, message: string }[]} sorted so the node nearest
 * the Input comes first; graph-wide errors (cycles) come last.
 */
export function friendlyIssues(report, graph, templates = {}) {
    if (!report || !graph?.nodes) return [];
    const order = orderedNodeIds(graph);
    const rank = new Map(order.map((id, index) => [id, index]));
    const issues = [];
    for (const [nodeId, messages] of Object.entries(report.errors ?? {})) {
        const node = graph.nodes[nodeId];
        if (!node) continue;
        const type = nodeType(node, templates);
        for (const message of messages ?? []) {
            const result = rewrite(String(message), { graph, report, node, type, templates });
            issues.push({ nodeId, key: result.key, params: { label: node.label ?? type ?? nodeId, ...result.params }, message: String(message) });
        }
    }
    issues.sort((left, right) => (rank.get(left.nodeId) ?? Infinity) - (rank.get(right.nodeId) ?? Infinity));
    for (const message of report.globalErrors ?? []) {
        issues.push({ nodeId: null, key: 'fb.global', params: { message: String(message) }, message: String(message) });
    }
    return issues;
}
