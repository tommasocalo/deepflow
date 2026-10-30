import { AlertCircle, AlertTriangle, ChevronDown } from 'lucide-react';
import { shapeSummary } from '../../domain/modelTable/rows.js';
import { ACTIVATION_TITLE, MACS_TITLE, SHAPE_GUESS_NOTE } from '../../domain/modelTable/cellText.js';
import { ANALYSIS_LIMITS, formatBytes, formatCount, formatExact, formatMacs } from '../../domain/metrics/index.js';

/** Where the largest activation sits, in full: `Conv2d in ResNet stages i=2`. */
function largestLocation(largest) {
    const steps = (largest.where ?? []).filter((step) => step.repeat && step.count > 1);
    return [largest.label, ...steps.map((step) => `${step.label} i=${step.index}`)].join(' in ');
}

/** The same, compact: `Conv2d (i=2) · B×256×32×32`. */
function largestSummary(largest) {
    const indices = (largest.where ?? []).filter((step) => step.repeat && step.count > 1).map((step) => step.index);
    const occurrence = indices.length === 1 ? ` (i=${indices[0]})` : indices.length ? ` (${indices.join('.')})` : '';
    return `${largest.label}${occurrence} · ${(shapeSummary([largest.shape]).text ?? '?').replace(/ × /g, '×')}`;
}

/**
 * The model's totals on one line. Their details and the assumptions behind
 * them open below (the table is described by the assumptions either way), so
 * a bottom sheet keeps its height for rows.
 */
export default function ModelTableFooter({ metrics, levelErrorCount, assumptionsId, detailsId, detailsOpen, onToggleDetails, onShowError }) {
    const { totals, issues } = metrics;
    const unresolved = (metric) => totals.unresolved[metric];
    const partial = (metric) => unresolved(metric).size > 0;
    const unresolvedText = (metric) => `${unresolved(metric).size} row${unresolved(metric).size === 1 ? '' : 's'} unresolved`;
    const unresolvedTitle = (metric) => {
        if (!partial(metric)) return '';
        const items = [...new Map([...unresolved(metric).values()].map((item) => [`${item.label}\u001f${item.reason}`, item])).values()];
        return `Not counted:\n${items.slice(0, 8).map((item) => `${item.label}${item.reason ? `: ${item.reason}` : ''}`).join('\n')}${items.length > 8 ? '\n…' : ''}`;
    };
    // A partial total is a lower bound; with nothing counted it is unknown.
    const bounded = (metric, text) => partial(metric) ? (totals[metric] > 0 ? `≥ ${text}` : '?') : text;
    const frozen = totals.params - totals.trainable;
    const largest = totals.largest;
    const inputs = metrics.inputs.map((input) => `${input.label} ${shapeSummary([input.shape]).text ?? '?'}`).join(' · ');
    const errors = issues?.errors ?? [];
    const errorCount = errors.length + (issues?.global?.length ?? 0);
    const firstError = errors[0];
    const errorList = [...(issues?.global ?? []), ...errors.map((error) => `${error.label}: ${error.messages.join('; ')}`)];
    const limits = issues?.limits ?? {};
    // The canvas badge counts the errors of the level it shows, where a container sums up its insides.
    const elsewhere = levelErrorCount !== null && levelErrorCount !== errorCount;
    const counting = `The table counts each failing node once, inside every definition (${errorCount}); the canvas badge counts the errors of the level it shows${levelErrorCount !== null ? ` (${levelErrorCount})` : ''}, where a container stands for what fails inside it.`;
    // Attention and recurrent modules count their products in closed form.
    const macModules = (metrics.macModules ?? []).map((kind) => kind === 'attention' ? ', attention' : ', RNN').join('');
    const stats = [
        {
            label: 'Params', value: bounded('params', formatExact(totals.params)), short: bounded('params', formatCount(totals.params) ?? '0'),
            detail: partial('params') ? unresolvedText('params') : frozen ? `trainable ${formatExact(totals.trainable)}, frozen ${formatExact(frozen)}` : 'all trainable',
            title: unresolvedTitle('params') || 'Learned parameters, every layer counted once per occurrence (no weight tying)',
        },
        {
            label: 'MACs', value: bounded('macs', formatMacs(totals.macs) ?? '0'),
            detail: partial('macs') ? unresolvedText('macs') : `${formatExact(totals.macs)} per sample`,
            title: unresolvedTitle('macs') || MACS_TITLE,
        },
        {
            label: 'Largest act.', value: largest ? bounded('activation', formatBytes(largest.bytes)) : partial('activation') ? '?' : '—',
            detail: largest ? largestSummary(largest) : partial('activation') ? unresolvedText('activation') : null,
            title: `Largest single output tensor at B = 1, fp32${largest ? `: ${largestLocation(largest)}, ${shapeSummary([largest.shape]).text ?? '?'}` : ''}.${partial('activation') ? ` ${unresolvedText('activation')}, so a larger one may exist.` : ''} All activations: ${partial('activation') ? '≥ ' : ''}${formatBytes(totals.activation) ?? '0 B'}. ${ACTIVATION_TITLE}`,
        },
        {
            label: 'Weights', value: bounded('params', formatBytes(totals.params * 4) ?? '0 B'), optional: true,
            detail: `${bounded('params', formatBytes(totals.params * 4) ?? '0 B')}, fp32, 4 bytes each`, title: `${formatExact(totals.params * 4)} bytes, fp32`,
        },
    ];
    return <footer className="df-mt-footer">
        {errorCount > 0 && <p className="df-mt-alert">
            <AlertCircle size={12} strokeWidth={2.6} aria-hidden="true" />
            <span className="df-mt-alert-text" title={`${errorList.slice(0, 12).join('\n')}${errorList.length > 12 ? '\n…' : ''}\n\n${counting}\nTotals use the declared settings. A layer with a shape error of its own, and anything whose size depends on an unknown value, is not counted (?). ${SHAPE_GUESS_NOTE}`}>
                <strong>{errorCount} error{errorCount === 1 ? '' : 's'}</strong>{elsewhere ? ` in the model, ${levelErrorCount || 'none'} on this canvas` : ''}
                <span className="df-mt-alert-detail">{firstError ? ` · ${firstError.label}: ${firstError.messages[0]}` : ` · ${issues.global[0]}`}</span>
            </span>
            {firstError && <button type="button" onClick={() => onShowError(firstError)} title={`Show ${firstError.label} in the table`}>Show</button>}
        </p>}
        {(limits.budget || limits.depth) && <p className="df-mt-alert is-limit">
            <AlertTriangle size={12} strokeWidth={2.6} aria-hidden="true" />
            <span className="df-mt-alert-text" title="Validation expands every repeat and component; past this limit nothing is analysed, so counts (×N) and totals inside are lower bounds.">
                <strong>{limits.budget ? `Analysis stopped at ${ANALYSIS_LIMITS.nodes.toLocaleString('en-US')} expanded nodes` : `Components nested deeper than ${ANALYSIS_LIMITS.depth} levels are not analysed`}</strong> · counts and totals are lower bounds (≥)
            </span>
        </p>}
        <div className="df-mt-totals">
            <dl className="df-mt-stats">
                {stats.map((stat) => <div key={stat.label} className={`df-mt-stat${stat.optional ? ' is-optional' : ''}`} title={stat.title}>
                    <dt>{stat.label}</dt>
                    <dd>{stat.short && stat.short !== stat.value
                        ? <><span className="df-mt-wide">{stat.value}</span><span className="df-mt-narrow">{stat.short}</span></>
                        : stat.value}</dd>
                </div>)}
            </dl>
            <button type="button" className="df-mt-details-toggle" aria-expanded={detailsOpen} aria-controls={detailsId}
                aria-label="B = 1 · fp32: details and assumptions" title="Details of the totals and the assumptions behind them" onClick={onToggleDetails}>
                B = 1 · fp32<ChevronDown size={11} strokeWidth={2.6} aria-hidden="true" />
            </button>
        </div>
        <div id={detailsId} className="df-mt-details" hidden={!detailsOpen}>
            <p className="df-mt-details-text">{stats.filter((stat) => stat.detail).map((stat) => `${stat.label}: ${stat.detail}`).join(' · ')}</p>
            <p id={assumptionsId} className="df-mt-assumptions">
                B = 1{inputs ? ` · ${inputs}` : ''} · fp32 · forward pass · MACs: MatMul, Linear, Conv{macModules} · activations: Σ op outputs, standard modules count one output · no weight tying
            </p>
        </div>
    </footer>;
}
