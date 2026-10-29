import { useMemo } from 'react';
import { useArchitectureStore } from '../../stores/architectureStore.js';
import { useTemplatesStore } from '../../stores/templatesStore.js';
import * as parameterCounting from '../../domain/optimization/parameterCount.js';
import { formatNumber, t } from '../lessons/engine/i18n.js';
import { orderedNodeIds } from '../lessons/engine/nodes.js';

/**
 * The paramCount embed (docs/edu/SPEC.md §6.3.5): the model's total from countArchitectureParameters and, when
 * Stage C's countParametersByNode exists, a per-block breakdown (found by feature detection).
 */
export default function ParamCountReadout() {
    const graph = useArchitectureStore((state) => state.graph);
    const templates = useTemplatesStore((state) => state.templates);
    const { total, byNode } = useMemo(() => {
        const count = parameterCounting.countArchitectureParameters(graph, templates);
        const perNode = typeof parameterCounting.countParametersByNode === 'function'
            ? parameterCounting.countParametersByNode(graph, templates) : null;
        const rows = perNode ? orderedNodeIds(graph).filter((id) => (perNode[id]?.total ?? 0) > 0)
            .map((id) => ({ id, label: graph.nodes[id]?.label ?? id, total: perNode[id].total, complete: perNode[id].complete !== false })) : [];
        return { total: count, byNode: rows };
    }, [graph, templates]);
    return (
        <section className="edu-embed edu-param-count" aria-label={t('paramCount.heading')}>
            <h3 className="edu-section-label">{t('paramCount.heading')}</h3>
            <p className="edu-param-total">
                <strong>{t('paramCount.total', { total: formatNumber(total.total) })}</strong>
                {total.trainable !== total.total && <span className="edu-muted"> · {t('paramCount.trainable', { trainable: formatNumber(total.trainable) })}</span>}
            </p>
            {!total.complete && <p className="edu-muted">{t('paramCount.incomplete')}</p>}
            {byNode.length > 0 && (
                <table className="edu-table">
                    <thead><tr><th scope="col">{t('paramCount.block')}</th><th scope="col">{t('paramCount.count')}</th></tr></thead>
                    <tbody>{byNode.map((row) => <tr key={row.id}><th scope="row">{row.label}</th><td>{formatNumber(row.total)}</td></tr>)}</tbody>
                </table>
            )}
        </section>
    );
}
