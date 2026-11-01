import { cachedValidate } from '../../../domain/architecture/levelContext.js';
import { withGlobalParameters } from '../../../domain/expressions.js';
import { getIncomingEdges, getOperationInputIds, getTemplateInputIds } from '../../../domain/graphInputs.js';
import { paramLabel, t } from './i18n.js';
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

// ---- wiring -------------------------------------------------------------------------------------------------------

/** PyTorch blocks that take several tensors; every other torch.nn block, and the Output, takes exactly one. */
const MULTI_INPUT_LAYERS = new Set(['torch.nn.MultiheadAttention', 'torch.nn.LSTM', 'torch.nn.GRU', 'torch.nn.RNN', 'torch.nn.TransformerDecoderLayer', 'torch.nn.Bilinear']);
const SOURCE_OR_SINK = new Set(['deepflow.io.Input', 'deepflow.io.Output', 'deepflow.template.Input', 'deepflow.template.Output', 'deepflow.op.Parameter', 'deepflow.op.Arange']);

/** A block DeepFlow's validator lets take several wires although only the first one is used. */
const takesOneInput = (node) => node?.kind === 'layer'
    && (node.libraryKey === 'deepflow.io.Output' || (node.libraryKey?.startsWith('torch.nn.') && !MULTI_INPUT_LAYERS.has(node.libraryKey)));

const nameOf = (node, templates) => node?.label ?? nodeType(node, templates) ?? node?.id ?? '';
const bold = (text) => `**${text}**`;

/** "**A**", "**A** and **B**", "**A**, **B** and **C**". */
function listOf(names) {
    const items = names.map(bold);
    return items.length < 2 ? items.join('') : `${items.slice(0, -1).join(', ')}${t('value.and')}${items.at(-1)}`;
}

/** Whether a path of wires leads from node `from` to node `to`. */
function reaches(graph, from, to) {
    const seen = new Set([from]);
    const queue = [from];
    while (queue.length) {
        const id = queue.shift();
        for (const edge of graph.edges ?? []) {
            if (edge.source !== id || seen.has(edge.target)) continue;
            if (edge.target === to) return true;
            seen.add(edge.target);
            queue.push(edge.target);
        }
    }
    return false;
}

/**
 * The wire into `nodeId` that is one too many: the one whose source also reaches another source of `nodeId`, so it
 * skips a block (review r1, M2 s3: the old Linear 1 → Linear 2 wire beside the new Linear 1 → ReLU → Linear 2).
 * @returns {{ edge: object, via: object } | null} `via` is the wire it bypasses
 */
export function extraWire(graph, nodeId) {
    const incoming = getIncomingEdges(graph, nodeId);
    for (const edge of incoming) {
        const via = incoming.find((other) => other !== edge && reaches(graph, edge.source, other.source));
        if (via) return { edge, via };
    }
    return null;
}

/** How many inputs a block takes: its template's input ports, an operation's named ports, or one. */
function inputPorts(node, templates) {
    if (node?.kind === 'supernode' || node?.kind === 'repeat') return Math.max(1, getTemplateInputIds(templates?.[node.templateId]?.graph).length);
    return getOperationInputIds(node ?? {})?.length ?? 1;
}

/** A block with more wires coming in than it takes: name the wire that skips a block, or list them all. */
function tooManyInputs({ graph, node, templates }) {
    const incoming = getIncomingEdges(graph, node.id);
    const ports = inputPorts(node, templates);
    const extra = extraWire(graph, node.id);
    if (extra && ports === 1) {
        return {
            key: 'fb.extra-wire', edgeId: extra.edge.id,
            params: { count: incoming.length, from: nameOf(graph.nodes[extra.edge.source], templates), via: nameOf(graph.nodes[extra.via.source], templates) },
        };
    }
    const sources = listOf(incoming.map((edge) => nameOf(graph.nodes[edge.source], templates)));
    return { key: ports === 1 ? 'fb.too-many-inputs' : 'fb.too-many-inputs.ports', params: { count: incoming.length, sources, ports } };
}

/** Two wires on one named input port. */
function portTwice(match, { graph, node, templates }) {
    const port = match[1];
    const label = templates?.[node.templateId]?.graph?.nodes?.[port]?.label ?? port;
    const wires = getIncomingEdges(graph, node.id).filter((edge) => edge.targetHandle === port);
    return { key: 'fb.port-twice', params: { port: label, sources: listOf(wires.map((edge) => nameOf(graph.nodes[edge.source], templates))) }, edgeId: wires.at(-1)?.id };
}

/** Ordered rules: the first that matches a message wins. Patterns follow validator.js and shapes.js. */
const RULES = [
    {
        // Wiring comes first: a second wire into a Linear is the cause of any width error inside it.
        pattern: /Too many unhandled incoming connections/,
        rewrite: (match, context) => tooManyInputs(context),
    },
    {
        pattern: /Input port "([^"]+)" has more than one incoming connection/,
        rewrite: portTwice,
    },
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
            issues.push({
                nodeId, key: result.key, params: { label: node.label ?? type ?? nodeId, ...result.params }, message: String(message),
                ...(result.edgeId ? { edgeId: result.edgeId } : {}),
            });
        }
    }
    // A ReLU or an Output with two wires validates (the first wire wins); for a learner it is a wiring mistake.
    for (const node of Object.values(graph.nodes)) {
        if (!takesOneInput(node) || report.errors?.[node.id]?.length || getIncomingEdges(graph, node.id).length < 2) continue;
        const result = tooManyInputs({ graph, node, templates });
        issues.push({ nodeId: node.id, key: result.key, params: { label: nameOf(node, templates), ...result.params }, message: 'wiring', structural: true, ...(result.edgeId ? { edgeId: result.edgeId } : {}) });
    }
    issues.sort((left, right) => (rank.get(left.nodeId) ?? Infinity) - (rank.get(right.nodeId) ?? Infinity));
    for (const message of report.globalErrors ?? []) {
        issues.push({ nodeId: null, key: 'fb.global', params: { message: String(message) }, message: String(message) });
    }
    return issues;
}

/**
 * Blocks the learner added but did not wire (DeepFlow reports them as warnings, so the graph still validates): the
 * first one, in canvas order, as a hint for the status line. Input and Output are left out: the check names them.
 * @returns {{ nodeId: string, key: string, params: object }[]}
 */
export function wiringHints(report, graph, templates = {}) {
    if (!report || !graph?.nodes) return [];
    const hints = [];
    for (const id of orderedNodeIds(graph)) {
        const node = graph.nodes[id];
        const messages = report.warnings?.[id] ?? [];
        if (!node || SOURCE_OR_SINK.has(node.libraryKey)) continue;
        const noIn = messages.includes('No incoming connections');
        const noOut = messages.includes('No outgoing connections');
        if (!noIn && !noOut) continue;
        hints.push({ nodeId: id, key: noIn && noOut ? 'fb.unwired' : noIn ? 'fb.unwired-in' : 'fb.unwired-out', params: { label: nameOf(node, templates) } });
    }
    return hints;
}

/** Lesson markup (**bold**, `code`) without the marks, for plain-text places (the trainer, code comments). */
export const plainText = (markup) => String(markup).replace(/\*\*([^*]+)\*\*/g, '$1').replace(/`([^`]+)`/g, '$1');

/** The friendly rewrite of a graph's first validation error as plain text, or null when the graph validates. */
export function firstGraphProblem(graph, templates = {}) {
    if (!graph?.nodes) return null;
    const report = cachedValidate(graph, templates, withGlobalParameters(graph), {});
    if (report.ok) return null;
    const issue = friendlyIssues(report, graph, templates)[0];
    return issue ? plainText(t(issue.key, issue.params)) : null;
}
