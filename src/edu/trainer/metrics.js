/**
 * Evaluation metrics (docs/edu/SPEC.md §6.10.7): mean loss and accuracy over a split in evaluation mode, the
 * confusion matrix and the best-validation epoch.
 */
import { computeLoss } from './losses.js';
import { forward } from './model.js';
import { Tensor } from './tensor.js';

export const EVAL_CHUNK = 256;

/** How predictions become classes: CE argmax, BCE logit > 0, MSE/L1 on 0/1 labels > 0.5, none for regression. */
export function predictionRule(lossType, dataset) {
    if (dataset.numClasses === null) return null;
    if (lossType === 'cross_entropy') return 'argmax';
    if (lossType === 'bce') return 'logit';
    return 'threshold';
}

export function predictRow(data, offset, width, rule) {
    if (rule === 'argmax') {
        let best = 0;
        for (let c = 1; c < width; c++) if (data[offset + c] > data[offset + best]) best = c;
        return best;
    }
    if (rule === 'logit') return data[offset] > 0 ? 1 : 0;
    return data[offset] > 0.5 ? 1 : 0;
}

/** Rows `start … end − 1` of a split as a batch tensor of shape (B, …featureShape). */
export function batchOf(split, featureShape, start, end, indices = null) {
    const width = featureShape.reduce((product, dim) => product * dim, 1);
    const rows = indices ? indices.length : end - start;
    const data = new Float64Array(rows * width);
    const y = split.y instanceof Float64Array ? new Float64Array(rows) : new Int32Array(rows);
    for (let r = 0; r < rows; r++) {
        const row = indices ? indices[r] : start + r;
        data.set(split.x.subarray(row * width, (row + 1) * width), r * width);
        y[r] = split.y[row];
    }
    return { x: new Tensor([rows, ...featureShape], data), y };
}

/**
 * Generator: evaluate a split in chunks (yielding between them) and return
 * { loss, accuracy, correct, count, predictions }. `loss` is the mean per element, whatever the training reduction.
 */
export function* evaluateSplit(model, dataset, split, objective, { keepPredictions = false } = {}) {
    if (!split.n) return { loss: null, accuracy: null, count: 0, predictions: null };
    const rule = predictionRule(objective.loss.type, dataset);
    const hyper = { ...objective.loss.hyperparameters, reduction: 'sum' };
    let total = 0, elements = 0, correct = 0;
    const predictions = keepPredictions && rule ? new Int32Array(split.n) : null;
    for (let start = 0; start < split.n; start += EVAL_CHUNK) {
        const end = Math.min(split.n, start + EVAL_CHUNK);
        const { x, y } = batchOf(split, dataset.featureShape, start, end);
        const out = forward(model, x, { training: false });
        total += computeLoss(objective.loss.type, out, y, hyper).loss;
        const width = out.shape[out.shape.length - 1] ?? 1;
        elements += objective.loss.type === 'cross_entropy' ? end - start : out.size;
        if (rule) {
            for (let r = 0; r < end - start; r++) {
                const predicted = predictRow(out.data, r * width, width, rule);
                if (predicted === y[r]) correct += 1;
                if (predictions) predictions[start + r] = predicted;
            }
        }
        yield 'eval';
    }
    return { loss: total / Math.max(1, elements), accuracy: rule ? correct / split.n : null, count: split.n, predictions };
}

/** Rows = true class, columns = predicted class. */
export function confusionMatrix(predictions, labels, numClasses) {
    const matrix = Array.from({ length: numClasses }, () => Array(numClasses).fill(0));
    for (let i = 0; i < labels.length; i++) {
        const truth = labels[i], predicted = predictions[i];
        if (truth >= 0 && truth < numClasses && predicted >= 0 && predicted < numClasses) matrix[truth][predicted] += 1;
    }
    return matrix;
}

/** The epoch with the lowest validation loss (training loss when there is no validation split); earliest on ties. */
export function bestEpoch(history) {
    let best = null;
    for (const entry of history) {
        const loss = entry.valLoss ?? entry.trainLoss;
        if (!Number.isFinite(loss)) continue;
        if (!best || loss < best.valLoss) best = { epoch: entry.epoch, valLoss: loss };
    }
    return best;
}

/** −ln(1/C) for C balanced classes (ln 2 for BCE): the loss of an untrained model that guesses uniformly. */
export function chanceLoss(lossType, numClasses) {
    if (lossType === 'bce') return Math.log(2);
    if (lossType === 'cross_entropy' && numClasses >= 2) return Math.log(numClasses);
    return null;
}
