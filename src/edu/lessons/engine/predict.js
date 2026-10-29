import { formatNumber } from './i18n.js';
import { formatShape, parseShapeInput, sameShape } from './nodes.js';

/**
 * Prediction gates (docs/edu/SPEC.md §3.3, §6.3.1). Predictions are never "failed": the verdict is 'exact',
 * 'close' or 'not-quite', and a matching diagnose rule names the misconception behind a wrong answer.
 */

/** Accept "100,480", "100 480", "1.10" and "1,10" (a lone comma is a decimal comma). */
export function parseNumberInput(text) {
    if (typeof text === 'number') return Number.isFinite(text) ? text : null;
    if (typeof text !== 'string') return null;
    let cleaned = text.trim().replace(/[\s_\u00a0\u202f]/g, '');
    if (!cleaned) return null;
    if (/^-?\d{1,3}(,\d{3})+(\.\d+)?$/.test(cleaned)) cleaned = cleaned.replaceAll(',', '');
    else if (/^-?\d+,\d+$/.test(cleaned)) cleaned = cleaned.replace(',', '.');
    const value = Number(cleaned);
    return Number.isFinite(value) ? value : null;
}

export function parsePredictionInput(predict, raw) {
    if (!predict) return null;
    if (predict.kind === 'number') return parseNumberInput(raw);
    if (predict.kind === 'shape') return Array.isArray(raw) ? raw : parseShapeInput(raw);
    if (predict.kind === 'choice') return predict.options?.includes(raw) ? raw : null;
    return null;
}

const near = (left, right, tolerance) => Math.abs(Number(left) - Number(right)) <= tolerance + 1e-9;

function diagnose(predict, value) {
    for (const rule of predict.diagnose ?? []) {
        const expected = rule.equals;
        const hit = Array.isArray(expected) ? sameShape(value, expected)
            : typeof expected === 'number' ? near(value, expected, predict.tolerance ?? 0)
                : value === expected;
        if (hit) return rule.misconception;
    }
    return null;
}

export function expectedValue(predict) {
    if (!predict) return null;
    if (predict.kind === 'number') return predict.expected;
    if (predict.kind === 'shape') return predict.expectedShape;
    return predict.correct;
}

/** { correct, verdict: 'exact'|'close'|'not-quite', misconception } */
export function gradePrediction(predict, value) {
    if (!predict || value === null || value === undefined) return { correct: false, verdict: 'not-quite', misconception: null };
    let correct = false;
    let close = false;
    if (predict.kind === 'number') {
        const expected = Number(predict.expected);
        correct = near(value, expected, predict.tolerance ?? 0);
        close = !correct && Number.isFinite(expected) && expected !== 0 && Math.abs(Number(value) - expected) / Math.abs(expected) <= 0.1;
    } else if (predict.kind === 'shape') {
        correct = sameShape(value, predict.expectedShape);
    } else if (predict.kind === 'choice') {
        correct = value === predict.correct;
    }
    const misconception = correct ? null : diagnose(predict, value);
    return { correct, verdict: correct ? 'exact' : close && !misconception ? 'close' : 'not-quite', misconception };
}

/** Text for an answer value; choices are displayed by the caller through their option text. */
export function formatPredictionValue(predict, value) {
    if (value === null || value === undefined) return '';
    if (predict?.kind === 'number') return formatNumber(Number(value));
    if (predict?.kind === 'shape') return formatShape(value);
    return String(value);
}
