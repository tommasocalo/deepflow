/**
 * Opt-in display metadata plus recognition of the original saved GPT-2 MLP.
 * Recognition is read-only: it never migrates a graph or changes model data.
 */
export function tensorFlowNodeIds(graph) {
    const nodes = Object.values(graph?.nodes ?? {});
    const ids = new Set(nodes.filter((node) => node.visualization === 'tensor-flow').map((node) => node.id));
    const edges = graph?.edges ?? [];
    const outgoing = new Map();
    for (const edge of edges) {
        if (!outgoing.has(edge.source)) outgoing.set(edge.source, new Set());
        outgoing.get(edge.source).add(edge.target);
    }
    const linked = (source, target) => outgoing.get(source.id)?.has(target.id) ?? false;
    const has = (node, key, label) => node.libraryKey === key && node.label === label;
    const expression = (value) => String(value ?? '').replace(/\s+/g, '');
    const isWidth = (value) => expression(value) === '$d_model';
    const isExpandedWidth = (value) => ['4*d_model', '$mlp_width'].includes(expression(value));

    for (const norm of nodes.filter((node) => has(node, 'torch.nn.LayerNorm', 'ln_2') && isWidth(node.parameters?.normalized_shape))) {
        const fc = nodes.find((node) => has(node, 'torch.nn.Linear', 'mlp fc') && linked(norm, node)
            && isWidth(node.parameters?.in_features) && isExpandedWidth(node.parameters?.out_features));
        if (!fc) continue;
        const gelu = nodes.find((node) => has(node, 'torch.nn.GELU', 'gelu') && linked(fc, node));
        if (!gelu) continue;
        const proj = nodes.find((node) => has(node, 'torch.nn.Linear', 'mlp proj') && linked(gelu, node)
            && isExpandedWidth(node.parameters?.in_features) && isWidth(node.parameters?.out_features));
        if (!proj) continue;
        const drop = nodes.find((node) => has(node, 'torch.nn.Dropout', 'dropout') && linked(proj, node));
        if (!drop) continue;
        const residual = nodes.find((node) => has(node, 'deepflow.op.Add', 'residual') && linked(drop, node));
        if (!residual) continue;
        const previousResidual = nodes.find((node) => has(node, 'deepflow.op.Add', 'residual') && linked(node, norm) && linked(node, residual));
        if (!previousResidual) continue;
        const hasAttention = nodes.some((node) => node.libraryKey === 'deepflow.nn.CausalSelfAttention'
            && (linked(node, previousResidual) || nodes.some((dropout) => has(dropout, 'torch.nn.Dropout', 'attention projection dropout')
                && linked(node, dropout) && linked(dropout, previousResidual))));
        const hasOutput = nodes.some((node) => node.libraryKey === 'deepflow.template.Output' && linked(residual, node));
        if (hasAttention && hasOutput) [norm, fc, gelu, proj, drop, residual].forEach((node) => ids.add(node.id));
    }
    return ids;
}
