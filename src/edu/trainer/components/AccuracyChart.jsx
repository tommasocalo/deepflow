import { formatAccuracy, sampleRows } from '../charts.js';
import { tr } from '../text.js';
import CurveChart from './CurveChart.jsx';
import { useMeasuredWidth } from './hooks.js';

/** Training and validation accuracy per epoch, with the chance level 1/C (classification only). */
export default function AccuracyChart({ record, height = 170 }) {
    const [ref, width] = useMeasuredWidth(280);
    const history = record?.history ?? [];
    if (!history.length) return <div ref={ref} className="edu-tr-chart-wrap"><p className="edu-tr-empty">{tr('acc.empty')}</p></div>;
    if (record.numClasses === null || !history.some((entry) => Number.isFinite(entry.trainAccuracy))) {
        return <div ref={ref} className="edu-tr-chart-wrap"><p className="edu-tr-empty">{tr('acc.regression')}</p></div>;
    }
    const last = history.at(-1);
    const hasVal = history.some((entry) => Number.isFinite(entry.valAccuracy));
    const planned = record.initOnly || record.status === 'running' || record.status === 'stopping' ? record.settings.epochs : last.epoch;
    const series = [{ id: 'train', variant: 'train', label: tr('loss.train'), points: history.map((entry) => [entry.epoch, entry.trainAccuracy]) }];
    if (hasVal) series.push({ id: 'val', variant: 'val', label: tr('loss.val'), points: history.map((entry) => [entry.epoch, entry.valAccuracy]) });
    const chance = record.numClasses >= 2 ? 1 / record.numClasses : null;
    return (
        <div ref={ref} className="edu-tr-chart-wrap">
            <CurveChart
                width={width} height={height} xMax={Math.max(1, planned)} yMin={0} yMax={1} yFormat={formatAccuracy}
                series={series}
                reference={chance ? { y: chance, label: tr('acc.chance', { value: formatAccuracy(chance) }) } : null}
                title={tr('acc.title')}
                desc={tr('acc.summary', { epoch: last.epoch, train: formatAccuracy(last.trainAccuracy), val: formatAccuracy(last.valAccuracy) })}
                table={(
                    <table className="edu-sr-only">
                        <caption>{tr('acc.title')}</caption>
                        <thead><tr><th scope="col">{tr('table.epoch')}</th><th scope="col">{tr('table.trainAcc')}</th><th scope="col">{tr('table.valAcc')}</th></tr></thead>
                        <tbody>
                            {sampleRows(history).map((entry) => (
                                <tr key={entry.epoch}><th scope="row">{entry.epoch}</th><td>{formatAccuracy(entry.trainAccuracy)}</td><td>{formatAccuracy(entry.valAccuracy)}</td></tr>
                            ))}
                        </tbody>
                    </table>
                )}
            />
        </div>
    );
}
