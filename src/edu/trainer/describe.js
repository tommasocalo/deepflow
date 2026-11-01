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

/** "Test accuracy 93.3% (best epoch 41, used once)": the held-out test split, evaluated once at the end of a run. */
export function testLine(record) {
    const test = record?.test;
    if (!test || !Number.isFinite(test.epoch)) return '';
    return Number.isFinite(test.accuracy)
        ? tr('test.accuracy', { accuracy: formatAccuracy(test.accuracy), epoch: test.epoch })
        : tr('test.loss', { loss: formatLoss(test.loss), epoch: test.epoch });
}

/** The run whose results are shown: the current one, or the last good run when the latest attempt failed. */
export function shownRecord(current, runs) {
    if (current && current.status !== 'failed') return current;
    return runs.at(-1) ?? null;
}

/** JSON with sorted keys and without node positions: moving a block does not change the model. */
function canonicalJson(value) {
    return JSON.stringify(value ?? null, function replacer(key, item) {
        if (key === 'position') return undefined;
        if (!item || typeof item !== 'object' || Array.isArray(item)) return item;
        return Object.fromEntries(Object.keys(item).sort().map((name) => [name, item[name]]));
    });
}

/** The templates a graph uses, at any depth (a Linear component, the blocks inside a repeat). */
function usedTemplates(graph, templates = {}) {
    const used = {};
    const visit = (level) => {
        for (const node of Object.values(level?.nodes ?? {})) {
            const id = node.templateId;
            if (!id || used[id] || !templates?.[id]) continue;
            used[id] = templates[id];
            visit(templates[id].graph);
        }
    };
    visit(graph);
    return used;
}

const modelKeys = new WeakMap();
/** A comparable key for a model: its graph and the templates it uses. Cached per graph and templates object. */
function modelKey(graph, templates) {
    if (!graph) return null;
    const cached = modelKeys.get(graph);
    if (cached && cached.templates === templates) return cached.key;
    const key = canonicalJson({ graph, templates: usedTemplates(graph, templates) });
    modelKeys.set(graph, { templates, key });
    return key;
}

/**
 * Why a finished run no longer describes the live workspace (review r1: the card showed a 2-class blobs run as the result
 * of the Iris model): 'model' when the graph or its templates changed (node positions aside), 'dataset' when the
 * dataset, its options, the split or the data seed changed, 'both', or null. `live` is { graph, templates, binding }
 * with the Data tab's binding { spec, options, split, dataSeed }. The optimizer and training settings do not count:
 * comparing learning rates is what the earlier runs are for.
 */
export function staleReason(record, live) {
    if (!record || !live || ['running', 'stopping'].includes(record.status)) return null;
    const model = !!record.graph && !!live.graph && record.graph !== live.graph
        && modelKey(record.graph, record.templates) !== modelKey(live.graph, live.templates);
    const binding = live.binding ?? {};
    const dataset = (record.datasetId ?? null) !== (binding.spec?.id ?? binding.datasetId ?? null)
        || canonicalJson(record.dataOptions ?? {}) !== canonicalJson(binding.options ?? {})
        || canonicalJson(record.split ?? {}) !== canonicalJson(binding.split ?? {})
        || (record.dataSeed ?? null) !== (binding.dataSeed ?? null);
    if (model && dataset) return 'both';
    return model ? 'model' : dataset ? 'dataset' : null;
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
