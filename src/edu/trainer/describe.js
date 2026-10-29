/**
 * Plain-text descriptions of a run for the trainer UI: the status line and the live-region announcement.
 */
import { formatAccuracy, formatLoss } from './charts.js';
import { tr } from './text.js';

/** One line about the latest run: progress, result or failure. */
export function runStatusText(record, status) {
    if (!record) return '';
    const epochs = record.settings?.epochs ?? 0;
    if (record.status === 'failed') return tr('run.failed', { message: record.error?.message ?? '' });
    if (status === 'running' || status === 'stopping') {
        return record.history.length ? tr('run.progress', { epoch: record.epoch, epochs: record.initOnly ? 0 : epochs }) : tr('run.starting');
    }
    if (record.initOnly) return tr('run.initDone', { loss: formatLoss(record.history[0]?.valLoss ?? record.history[0]?.trainLoss) });
    if (record.status === 'stopped') return tr('run.stopped', { epoch: record.epoch });
    if (record.status === 'completed') {
        return tr('run.done', { epochs: record.epoch, time: formatDuration(record.finishedAt - record.startedAt) });
    }
    return '';
}

/** What the live region announces when a run ends (not every epoch). */
export function runAnnouncement(record) {
    if (!record || !['completed', 'stopped', 'failed'].includes(record.status)) return '';
    if (record.status === 'failed') return tr('run.failed', { message: record.error?.message ?? '' });
    const last = record.history.at(-1);
    if (!last) return '';
    if (record.initOnly) return tr('run.initAnnounce', { loss: formatLoss(last.valLoss ?? last.trainLoss) });
    return Number.isFinite(last.valAccuracy)
        ? tr('run.finishedAnnounce', { accuracy: formatAccuracy(last.valAccuracy), loss: formatLoss(last.valLoss) })
        : tr('run.finishedAnnounceRegression', { loss: formatLoss(last.valLoss ?? last.trainLoss) });
}

/** The run whose results are shown: the current one, or the last good run when the latest attempt failed. */
export function shownRecord(current, runs) {
    if (current && current.status !== 'failed') return current;
    return runs.at(-1) ?? null;
}

/** 40 ms, 0.7 s, 12 s. */
export function formatDuration(ms) {
    if (!Number.isFinite(ms) || ms < 0) return '?';
    if (ms < 1000) return `${Math.max(1, Math.round(ms))} ms`;
    return ms < 10000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms / 1000)} s`;
}

/**
 * Preflight errors for display: when a model has many blocks the browser cannot run (GPT-2, CNNs), one line names
 * their types instead of one line per block.
 */
export function groupIssues(errors = []) {
    const unsupported = errors.filter((issue) => issue.code === 'unsupported-op');
    if (unsupported.length <= 2) return errors;
    const types = [...new Set(unsupported.map((issue) => issue.type).filter(Boolean))];
    const listed = types.slice(0, 4).join(', ') + (types.length > 4 ? ', …' : '');
    const merged = { severity: 'error', code: 'unsupported-op', nodeId: unsupported[0].nodeId ?? null,
        message: tr('preflight.unsupportedMany', { count: unsupported.length, types: listed }) };
    return [merged, ...errors.filter((issue) => issue.code !== 'unsupported-op')];
}
