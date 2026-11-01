import { formatLoss, formatTick, lossAxisMax, previousRuns, runDifference, sampleRows } from '../charts.js';
import { tr } from '../text.js';
import CurveChart from './CurveChart.jsx';
import { useMeasuredWidth } from './hooks.js';

/** The chance-level label: ln C for cross-entropy, ln 2 for BCE. */
function chanceLabel(record) {
    if (!Number.isFinite(record.chance)) return null;
    return record.loss?.type === 'bce'
        ? tr('loss.chanceBce', { value: formatLoss(record.chance) })
        : tr('loss.chance', { classes: record.numClasses, value: formatLoss(record.chance) });
}

export function LossTable({ history }) {
    return (
        <table className="edu-sr-only">
            <caption>{tr('loss.title')}</caption>
            <thead><tr><th scope="col">{tr('table.epoch')}</th><th scope="col">{tr('table.trainLoss')}</th><th scope="col">{tr('table.valLoss')}</th></tr></thead>
            <tbody>
                {sampleRows(history).map((entry) => (
                    <tr key={entry.epoch}><th scope="row">{entry.epoch}</th><td>{formatLoss(entry.trainLoss)}</td><td>{formatLoss(entry.valLoss)}</td></tr>
                ))}
            </tbody>
        </table>
    );
}

/**
 * Training and validation loss per epoch (epoch 0 = before any update), the chance level, the best-validation ▲ and
 * up to `ghostCount` earlier runs on the same dataset, dimmed and labelled with what differed.
 */
export default function LossChart({ record, runs = [], height = 170, compact = false, ghostCount = 5 }) {
    const [ref, width] = useMeasuredWidth(compact ? 220 : 280);
    const history = record?.history ?? [];
    if (!history.length) {
        return <div ref={ref} className="edu-tr-chart-wrap"><p className="edu-tr-empty">{tr('loss.empty')}</p></div>;
    }
    const last = history.at(-1);
    const hasVal = history.some((entry) => Number.isFinite(entry.valLoss));
    const ghosts = previousRuns(runs, record, ghostCount).map((run) => {
        const label = runDifference(run, record);
        return {
            id: run.id,
            label,
            short: label.split(' · ')[0],
            final: run.history.at(-1),
            points: run.history.map((entry) => [entry.epoch, entry.valLoss ?? entry.trainLoss]),
        };
    });
    // While training (and for an epoch-0 check) the axis spans the planned epochs, so the curve grows into it.
    const planned = record.initOnly || record.status === 'running' || record.status === 'stopping' ? record.settings.epochs : last.epoch;
    const xMax = Math.max(1, planned, ...ghosts.map((ghost) => ghost.points.at(-1)?.[0] ?? 0));
    const yMax = lossAxisMax(
        [history.map((entry) => entry.trainLoss), history.map((entry) => entry.valLoss), ...ghosts.map((ghost) => ghost.points.map((point) => point[1]))],
        { chance: record.chance, reference: history[0].trainLoss },
    );
    const series = [{ id: 'train', variant: 'train', label: tr('loss.train'), points: history.map((entry) => [entry.epoch, entry.trainLoss]) }];
    if (hasVal) series.push({ id: 'val', variant: 'val', label: tr('loss.val'), points: history.map((entry) => [entry.epoch, entry.valLoss]) });
    const best = history.length > 1 && hasVal ? record.best : null;
    const desc = [
        tr('loss.summary', { epoch: last.epoch, train: formatLoss(last.trainLoss), val: formatLoss(last.valLoss) }),
        best ? tr('loss.summaryBest', { loss: formatLoss(best.valLoss), epoch: best.epoch }) : '',
        ghosts.length ? tr('loss.previous', { labels: ghosts.map((ghost) => ghost.label).join('; ') }) : '',
    ].filter(Boolean).join(' ');
    return (
        <div ref={ref} className="edu-tr-chart-wrap">
            <CurveChart
                width={width} height={height} xMax={xMax} yMax={yMax} yFormat={formatTick} compact={compact}
                series={series} ghosts={ghosts}
                reference={Number.isFinite(record.chance) ? { y: record.chance, label: chanceLabel(record) } : null}
                marker={best ? { x: best.epoch, y: best.valLoss, title: tr('loss.best', { loss: formatLoss(best.valLoss), epoch: best.epoch }) } : null}
                title={tr('loss.title')} desc={desc}
                table={<LossTable history={history} />}
            />
            {!compact && ghosts.length > 0 && (
                <ul className="edu-tr-ghosts" aria-hidden="true">
                    {ghosts.map((ghost) => (
                        <li key={ghost.id}><span className="edu-tr-ghost-line" />{ghost.label} · {formatLoss(ghost.final?.valLoss ?? ghost.final?.trainLoss)}</li>
                    ))}
                </ul>
            )}
        </div>
    );
}
