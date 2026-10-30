import { useState } from 'react';
import { formatAccuracy, formatLoss } from '../charts.js';
import { shownRecord } from '../describe.js';
import { getDataset } from '../../data/registry.js';
import { useEduDataStore } from '../../data/store.js';
import { useTrainerStore } from '../store.js';
import { tr } from '../text.js';
import AccuracyChart from './AccuracyChart.jsx';
import BoundaryView from './BoundaryView.jsx';
import ConfusionMatrix from './ConfusionMatrix.jsx';
import { useBridgeSource } from '../../useBridgeSource.js';
import { useMeasuredWidth } from './hooks.js';
import LossChart from './LossChart.jsx';
import RunSettings from './RunSettings.jsx';
import { useRunAnnouncement } from './useAnnouncement.js';
import './trainer.css';

function ResultColumn({ record }) {
    const [view, setView] = useState('boundary');
    const [ref, width] = useMeasuredWidth(240);
    // Before the first run, a 2D dataset announces the boundary view (so lesson spotlights find it).
    const boundShape = useEduDataStore((state) => getDataset(state.datasetId)?.input?.shape ?? null);
    const hasBoundary = !!record?.boundary;
    const hasConfusion = !!record?.confusion;
    const mode = hasBoundary && hasConfusion ? view : hasBoundary || (!record?.history?.length && boundShape === '(2,)') ? 'boundary' : 'confusion';
    const size = Math.max(140, Math.min(212, width - 4));
    return (
        <div className="edu-tr-col" ref={ref} data-edu-target={mode === 'boundary' ? 'trainer-boundary' : 'trainer-confusion'}>
            <div className="edu-tr-col-head">
                <h3>{tr(mode === 'boundary' ? 'boundary.title' : 'confusion.title')}</h3>
                {hasBoundary && hasConfusion && (
                    <div className="edu-tr-segmented" role="group" aria-label={tr('view.label')}>
                        {['boundary', 'confusion'].map((value) => (
                            <button key={value} type="button" aria-pressed={mode === value} className={mode === value ? 'is-on' : ''} onClick={() => setView(value)}>
                                {tr(`view.${value}`)}
                            </button>
                        ))}
                    </div>
                )}
            </div>
            {!record?.history?.length && <p className="edu-tr-empty">{tr('boundary.empty')}</p>}
            {record?.history?.length > 0 && mode === 'boundary' && hasBoundary && <BoundaryView record={record} size={size} target={false} />}
            {record?.history?.length > 0 && mode === 'confusion' && <ConfusionMatrix record={record} target={false} />}
        </div>
    );
}

/**
 * The Training tab's trainer (docs/edu/SPEC.md §3.5): run settings, the loss and accuracy curves, and the decision
 * boundary (2D) or confusion matrix. Train stays locked while a lesson's prediction gate is open.
 */
export default function TrainerPanel() {
    const current = useTrainerStore((state) => state.current);
    const runs = useTrainerStore((state) => state.runs);
    const lesson = useBridgeSource('lesson');
    const locked = lesson?.mode === 'lesson' && !!lesson?.trainLocked;
    const record = shownRecord(current, runs);
    const last = record?.history?.at(-1);
    const announcement = useRunAnnouncement(current);
    return (
        <section className="edu-tr-host" aria-label={tr('panel.label')}>
            <div className="edu-tr-panel">
                <div className="edu-tr-col edu-tr-run">
                    <div className="edu-tr-col-head"><h3>{tr('run.title')}</h3></div>
                    <RunSettings locked={locked} />
                </div>
                <div className="edu-tr-col" data-edu-target="trainer-curves">
                    <div className="edu-tr-col-head">
                        <h3>{tr('loss.title')}</h3>
                        {last && Number.isFinite(last.valLoss) && (
                            <span className="edu-tr-head-value">
                                {tr('loss.val')} {formatLoss(last.valLoss)}
                                {record.best && record.history.length > 1 ? <> · <span aria-hidden="true">▲</span> {tr('loss.best', { epoch: record.best.epoch })}</> : null}
                            </span>
                        )}
                    </div>
                    <LossChart record={record} runs={runs} />
                </div>
                <div className="edu-tr-col">
                    <div className="edu-tr-col-head">
                        <h3>{tr('acc.title')}</h3>
                        {last && Number.isFinite(last.valAccuracy) && <span className="edu-tr-head-value">{tr('loss.val')} {formatAccuracy(last.valAccuracy)}</span>}
                    </div>
                    <AccuracyChart record={record} />
                </div>
                <ResultColumn record={record} />
            </div>
            <p className="edu-tr-sr" aria-live="polite">{announcement}</p>
        </section>
    );
}
