import { useEduUiStore } from '../../shell/uiStore.js';
import { formatAccuracy, formatLoss, previousRuns, runDifference } from '../charts.js';
import { groupIssues, shownRecord, testLine } from '../describe.js';
import { useTrainerStore } from '../store.js';
import { tr } from '../text.js';
import BoundaryView from './BoundaryView.jsx';
import ConfusionMatrix from './ConfusionMatrix.jsx';
import { useStaleReason } from './hooks.js';
import LossChart from './LossChart.jsx';
import { LrField, RunButtons } from './RunSettings.jsx';
import { useRunAnnouncement } from './useAnnouncement.js';
import './trainer.css';

function finalLine(record) {
    const last = record?.history?.at(-1);
    if (!last) return '';
    if (record.initOnly) {
        return tr('card.initial', { loss: formatLoss(last.valLoss ?? last.trainLoss), chance: Number.isFinite(record.chance) ? formatLoss(record.chance) : '–' });
    }
    return Number.isFinite(last.valAccuracy)
        ? tr('card.final', { accuracy: formatAccuracy(last.valAccuracy), loss: formatLoss(last.valLoss) })
        : tr('card.finalRegression', { loss: formatLoss(last.valLoss ?? last.trainLoss) });
}

/**
 * The compact trainer embedded in a lesson step (docs/edu/SPEC.md §3.5, §6.3.5): Train / Stop / Check initial loss,
 * the learning rate, progress, a loss sparkline with the final validation numbers, a boundary thumbnail for 2D data
 * or the confusion matrix, the last runs for comparison, and "Open Training". Props: { locked, lockedReason }.
 */
export default function TrainerCard({ locked = false, lockedReason = '' }) {
    const current = useTrainerStore((state) => state.current);
    const runs = useTrainerStore((state) => state.runs);
    const status = useTrainerStore((state) => state.status);
    const preflightErrors = useTrainerStore((state) => state.preflight.errors);
    const firstError = groupIssues(preflightErrors)[0] ?? null;
    const record = shownRecord(current, runs);
    const running = status === 'running' || status === 'stopping';
    // A run of another model or dataset stays visible for comparison, dimmed, and never reads as this model's result.
    const stale = useStaleReason(running ? null : record);
    const announcement = useRunAnnouncement(current);
    const ghosts = record && !record.initOnly ? previousRuns(runs, record, 3) : [];
    const failure = current?.status === 'failed' ? current.error?.message : null;
    // While a prediction gate is open, a problem line could give the answer away (M3 s6: "… receives 64").
    const problem = locked ? null : failure ?? (!running && firstError ? firstError.message : null);
    return (
        <section className="edu-tr-card" aria-label={tr('card.label')}>
            <RunButtons locked={locked} lockedReason={lockedReason} compact />
            <div className="edu-tr-card-meta">
                <LrField compact disabled={running} />
                {record && !record.initOnly && !stale && (
                    <span className="edu-tr-card-epoch">{tr('card.epoch', { epoch: record.epoch, epochs: record.settings.epochs })}</span>
                )}
            </div>
            {problem && <p className="edu-tr-card-problem">{problem}</p>}
            {stale && record?.history?.length > 0 && <p className="edu-tr-stale" data-edu-stale={stale}>{tr(`stale.${stale}`)}</p>}
            <div className={`edu-tr-card-body${stale ? ' is-stale' : ''}`}>
                <div className="edu-tr-card-curve" data-edu-target="trainer-curves">
                    <LossChart record={record} runs={runs} compact height={64} ghostCount={3} />
                    {record?.history?.length > 0 && <p className="edu-tr-card-final">{finalLine(record)}</p>}
                    {record?.test && <p className="edu-tr-card-test">{testLine(record)}</p>}
                    {record?.history?.length > 0 && !record.initOnly && Number.isFinite(record.chance) && (
                        <p className="edu-tr-card-chance">{tr('card.chance', { chance: formatLoss(record.chance) })}</p>
                    )}
                </div>
                {record?.boundary && <BoundaryView record={record} size={120} thumbnail />}
            </div>
            {record?.history?.length > 0 && !record.boundary && record.confusion && !running && (
                <div className={stale ? 'is-stale' : undefined}><ConfusionMatrix record={record} compact /></div>
            )}
            {ghosts.length > 0 && (
                <div className="edu-tr-card-runs">
                    <p className="edu-tr-card-runs-title">{tr('card.recent')}</p>
                    <ul>
                        {[record, ...ghosts].map((run) => {
                            const final = run.history.at(-1);
                            const label = run === record ? tr(stale ? 'label.last' : 'label.current') : runDifference(run, record);
                            return (
                                <li key={run.id}>
                                    {Number.isFinite(final?.valAccuracy)
                                        ? tr('card.recentRowAcc', { label, loss: formatLoss(final.valLoss), accuracy: formatAccuracy(final.valAccuracy) })
                                        : tr('card.recentRow', { label, loss: formatLoss(final?.valLoss ?? final?.trainLoss) })}
                                </li>
                            );
                        })}
                    </ul>
                </div>
            )}
            <button type="button" className="edu-tr-link edu-tr-open" onClick={() => useEduUiStore.getState().setTab('training')}>
                {tr('card.openTraining')} <span aria-hidden="true">↗</span>
            </button>
            <p className="edu-sr-only" aria-live="polite">{announcement}</p>
        </section>
    );
}
