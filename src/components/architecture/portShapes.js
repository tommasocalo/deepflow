import { activationLabel } from './nodeGeometry.js';

const INPUTS = new Set(['deepflow.io.Input', 'deepflow.template.Input']);
const OUTPUTS = new Set(['deepflow.io.Output', 'deepflow.template.Output']);

/** Shape reports describe outputs; each inlet receives its upstream output. */
export function describePorts(graph, node, shapes) {
    const incoming = graph.edges.filter((edge) => edge.target === node.id);
    const groups = new Map();
    for (const edge of incoming) {
        const label = activationLabel(shapes[edge.source]) ?? '—';
        const group = groups.get(label) ?? { label, count: 0 };
        group.count += 1;
        groups.set(label, group);
    }
    const inputs = [...groups.values()].map(({ label, count }) => count > 1 ? `${count} × [${label}]` : label);
    return {
        input: INPUTS.has(node.libraryKey) ? null : inputs.length ? inputs.join(' · ') : '—',
        // Split has no per-output inference yet: don't present its passthrough
        // validation shape as a computed split-output shape.
        output: OUTPUTS.has(node.libraryKey) || node.libraryKey === 'deepflow.op.Split'
            ? null : activationLabel(shapes[node.id]) ?? '—',
    };
}
