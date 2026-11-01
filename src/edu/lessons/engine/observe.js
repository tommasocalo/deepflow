import { formatAccuracy, formatLoss } from '../../trainer/charts.js';
import { finalMetrics, isInitRun, trainerRuns } from './checks.js';
import { formatNumber, t } from './i18n.js';
import { matchNodes } from './nodes.js';

/**
 * Predict, observe, explain (docs/edu/SPEC.md §2 P6, review r1). A prediction about something the learner must see
 * happen — a training result, a boundary, a loss — is an outcome prediction: Submit only locks it, and the reveal
 * waits until the step's run has passed, then compares the answer with what that run shows. Predictions about values
 * the tool computes without training (parameter counts, shapes) reveal at once.
 *
 * The rule is the step's check: a step whose check needs a trainer run (`run` or `runs`) holds an outcome prediction.
 * What the run shows is read by an observer: `predict.observe` names one, otherwise it is inferred from the step.
 */
export const OBSERVERS = Object.freeze(['initLoss', 'bestClass', 'boundaryShape', 'bends', 'hiddenWidth', 'bestLr', 'risingCurve', 'tool']);

function findKey(spec, keys) {
    if (!spec || typeof spec !== 'object') return null;
    for (const key of keys) if (Object.hasOwn(spec, key)) return { key, value: spec[key] };
    for (const value of Object.values(spec)) {
        const found = Array.isArray(value) ? value.map((item) => findKey(item, keys)).find(Boolean) : findKey(value, keys);
        if (found) return found;
    }
    return null;
}

/** The step's run query ({ run } or { runs }) wherever it sits in its check. */
export function runQueryOf(step) {
    return findKey(step?.check, ['run', 'runs']);
}

/** An outcome prediction: the step's check needs a training run, so the answer is checked against what it shows. */
export function isOutcomePrediction(step) {
    return !!step?.predict && !!runQueryOf(step);
}

const optionsAre = (predict, ids) => Array.isArray(predict?.options) && predict.options.length > 0 && predict.options.every((option) => ids.includes(option));

/** Which observer reads the run for this prediction (explicit `predict.observe`, else inferred from the step). */
export function observerOf(step) {
    const predict = step?.predict;
    if (!isOutcomePrediction(step)) return null;
    if (OBSERVERS.includes(predict.observe)) return predict.observe;
    const query = runQueryOf(step);
    if (predict.kind === 'number') return query.key === 'run' && query.value?.initOnly ? 'initLoss' : 'tool';
    if (predict.kind !== 'choice') return 'tool';
    if (query.key === 'runs' && Array.isArray(query.value?.lr)) return 'bestLr';
    if (query.value?.overfit && optionsAre(predict, ['val', 'train', 'neither'])) return 'risingCurve';
    if (optionsAre(predict, ['line', 'circle', 'curve'])) return 'boundaryShape';
    if (optionsAre(predict, ['yes', 'no'])) return 'bends';
    return 'bestClass';
}

// ---- reading a run ------------------------------------------------------------------------------------------------

/**
 * The decision boundary's shape on its grid: 'line' when the cells where the predicted class changes lie on one
 * straight line, 'closed' when a class forms an island that touches no edge of the plot, 'curve' otherwise; 'none'
 * when one class covers the whole plot. `boundary` is a run's { size, classes } (trainer/boundary.js).
 */
export function boundaryShape(boundary) {
    const size = boundary?.size;
    const classes = boundary?.classes;
    if (!size || !classes?.length) return null;
    const points = [];
    for (let row = 0; row < size; row++) {
        for (let col = 0; col < size; col++) {
            const value = classes[row * size + col];
            if (col + 1 < size && classes[row * size + col + 1] !== value) points.push([col + 1, row + 0.5]);
            if (row + 1 < size && classes[(row + 1) * size + col] !== value) points.push([col + 0.5, row + 1]);
        }
    }
    if (!points.length) return 'none';
    // A class missing from the border lies inside the others.
    const onEdge = new Set();
    for (let index = 0; index < size; index++) {
        for (const cell of [index, (size - 1) * size + index, index * size, index * size + size - 1]) onEdge.add(classes[cell]);
    }
    if (new Set(classes).size > onEdge.size) return 'closed';
    // Straight: the change points sit within about a cell of their principal line (a staircase on the grid).
    const n = points.length;
    const mx = points.reduce((sum, [x]) => sum + x, 0) / n;
    const my = points.reduce((sum, [, y]) => sum + y, 0) / n;
    let xx = 0, yy = 0, xy = 0;
    for (const [x, y] of points) { xx += (x - mx) ** 2; yy += (y - my) ** 2; xy += (x - mx) * (y - my); }
    xx /= n; yy /= n; xy /= n;
    const spread = (xx + yy) / 2;
    const minor = spread - Math.sqrt(((xx - yy) / 2) ** 2 + xy ** 2);
    return Math.sqrt(Math.max(0, minor)) <= 1.5 ? 'line' : 'curve';
}

/** Validation recall per class from a run's confusion matrix (rows are true classes). */
function recalls(run) {
    return (run?.confusion ?? []).map((row, index) => {
        const total = row.reduce((sum, value) => sum + value, 0);
        const correct = row[index] ?? 0;
        return { index, total, correct, recall: total ? correct / total : null };
    });
}

function optionText(prefix, option) {
    return t(`${prefix}.predict.opt.${option}`);
}

/** The finished runs since the step began, oldest first; the one that passed the check (`runId`) when given. */
function stepRuns({ trainer, enteredAtSeq = 0 }) {
    return trainerRuns(trainer).filter((run) => (run.seq ?? 0) > enteredAtSeq && run.status !== 'failed');
}

function pickRun(context, test = () => true) {
    const runs = stepRuns(context);
    return (context.runId && runs.find((run) => run.id === context.runId && test(run))) ?? [...runs].reverse().find(test) ?? null;
}

const trainingRun = (run) => !isInitRun(run) && (run.history?.length ?? 0) > 1;

/** The number of hidden units of a run's model: the first Linear's Output. */
function hiddenWidth(run) {
    const first = matchNodes(run?.graph ?? { nodes: {}, edges: [] }, { type: 'Linear', nth: 0 }, { templates: run?.templates ?? {}, graph: run?.graph })[0];
    const width = Number(first?.parameterValues?.out_features ?? first?.parameters?.out_features);
    return Number.isFinite(width) ? width : null;
}

/**
 * What the step's run shows about the prediction: `{ option?, value?, text }` (text in lesson markup), or null when
 * no run has shown it yet. `context`: { trainer, enteredAtSeq, runId }; `prefix` is the step's text prefix (m2.s1).
 */
export function observePrediction(step, context = {}, prefix = '') {
    const observer = observerOf(step);
    const predict = step?.predict;
    if (!observer) return null;
    if (observer === 'tool') {
        const value = predict.kind === 'number' ? predict.expected : predict.kind === 'choice' ? predict.correct : null;
        if (value === null || value === undefined) return null;
        return { value, option: predict.kind === 'choice' ? value : undefined, text: predict.kind === 'number' ? formatNumber(value) : optionText(prefix, value), tool: true };
    }
    if (observer === 'initLoss') {
        const run = pickRun(context, (item) => isInitRun(item)) ?? pickRun(context);
        const first = run?.history?.[0];
        const loss = first?.valLoss ?? first?.trainLoss;
        if (!Number.isFinite(loss)) return null;
        return { value: Number(loss.toFixed(2)), text: t('observe.initLoss', { loss: formatLoss(loss) }) };
    }
    if (observer === 'bestLr') {
        const lrs = runQueryOf(step).value.lr;
        const runs = stepRuns(context).filter((run) => trainingRun(run) && run.status === 'completed');
        const best = runs.map((run) => ({ run, lr: Number(run.optimizer?.hyperparameters?.learning_rate), loss: finalMetrics(run)?.valLoss }))
            .filter((item) => Number.isFinite(item.loss) && lrs.some((lr) => Math.abs(lr - item.lr) <= Math.abs(lr) * 0.01))
            .sort((left, right) => left.loss - right.loss)[0];
        if (!best) return null;
        const option = predict.options.find((id) => Math.abs(Number(optionText(prefix, id)) - best.lr) <= Math.abs(best.lr) * 0.01);
        return { option, text: t('observe.bestLr', { lr: option ? optionText(prefix, option) : best.lr, loss: formatLoss(best.loss) }) };
    }
    const run = pickRun(context, trainingRun);
    if (!run) return null;
    const final = finalMetrics(run) ?? {};
    const accuracy = Number.isFinite(final.valAccuracy) ? formatAccuracy(final.valAccuracy) : '–';
    if (observer === 'risingCurve') {
        const losses = (key) => (run.history ?? []).map((entry) => entry[key]).filter(Number.isFinite);
        const rise = (values) => (values.length ? values.at(-1) / Math.max(1e-9, Math.min(...values)) : 1);
        const val = rise(losses('valLoss'));
        const train = rise(losses('trainLoss'));
        const option = val >= 1.1 && train < 1.1 ? 'val' : train >= 1.1 && val < 1.1 ? 'train' : 'neither';
        return { option, text: t(`observe.rising.${option}`, { rise: val.toFixed(1) }) };
    }
    if (observer === 'bestClass') {
        const rows = recalls(run).filter((row) => row.recall !== null);
        if (!rows.length) return null;
        const top = Math.max(...rows.map((row) => row.recall));
        const names = (run.classNames ?? []).map((name) => String(name).toLowerCase());
        const options = rows.filter((row) => row.recall === top).map((row) => predict.options.find((id) => id.toLowerCase() === names[row.index])).filter(Boolean);
        // On a tie the authored answer is one of the best; say so rather than pick another.
        const option = options.includes(predict.correct) ? predict.correct : options[0];
        if (!option) return null;
        const row = rows.find((item) => names[item.index] === option.toLowerCase());
        return { option, text: t('observe.bestClass', { name: optionText(prefix, option), correct: row.correct, total: row.total }) };
    }
    const shape = boundaryShape(run.boundary);
    if (observer === 'boundaryShape') {
        if (!shape) return null;
        const option = shape === 'line' ? 'line' : shape === 'closed' ? 'circle' : shape === 'curve' ? 'curve' : null;
        return { option, text: t(`observe.shape.${shape}`, { accuracy }) };
    }
    if (observer === 'bends') {
        if (!shape) return { text: t('observe.accuracy', { accuracy }) };
        const bends = shape === 'closed' || shape === 'curve';
        return { option: bends ? 'yes' : 'no', text: t(bends ? 'observe.bends.yes' : 'observe.bends.no', { accuracy }) };
    }
    if (observer === 'hiddenWidth') {
        const width = hiddenWidth(run);
        if (width === null) return null;
        const option = predict.options.find((id) => Number(optionText(prefix, id)) === width);
        return { option, value: width, text: t('observe.width', { units: width, accuracy }) };
    }
    return null;
}
