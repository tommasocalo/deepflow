import { validate } from '../../validation/validator.js';
import { countArchitectureParameters } from './parameterCount.js';

export const MAX_CANDIDATES = 48;
export const MAX_SEARCH_AXES = 3;

/** A small, explicit search space keeps validation responsive in the browser. */
export function parseSearchValues(text, type = 'int') {
    const parts = String(text ?? '').split(',').map((part) => part.trim());
    if (!parts.length || parts.some((part) => !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(part))) {
        return { values: [], error: 'Enter numbers separated by commas.' };
    }
    const values = [...new Set(parts.map(Number))];
    if (values.some((value) => !Number.isFinite(value) || (type === 'int' && !Number.isSafeInteger(value)))) {
        return { values: [], error: type === 'int' ? 'Use whole numbers within the safe integer range.' : 'Use finite numbers.' };
    }
    if (values.length > MAX_CANDIDATES) return { values: [], error: `At most ${MAX_CANDIDATES} values per parameter.` };
    return { values, error: null };
}

/** Inclusive numeric range. Rejects grids that would freeze the canvas before expanding them. */
export function parseSearchRange(start, end, step, type = 'int') {
    const raw = [start, end, step].map((value) => String(value ?? '').trim());
    if (raw.some((value) => !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value))) {
        return { values: [], error: 'Enter a numeric start, end and step.' };
    }
    const [first, last, increment] = raw.map(Number);
    if (![first, last, increment].every(Number.isFinite) || increment <= 0 || last < first) {
        return { values: [], error: 'Use a positive step and an end at or above the start.' };
    }
    if (type === 'int' && (![first, last, increment].every(Number.isSafeInteger))) {
        return { values: [], error: 'Use whole numbers within the safe integer range.' };
    }
    const count = Math.floor((last - first) / increment + 1e-9) + 1;
    if (!Number.isSafeInteger(count) || count > MAX_CANDIDATES) {
        return { values: [], error: `A range can contain at most ${MAX_CANDIDATES} values.` };
    }
    const values = Array.from({ length: count }, (_, index) => type === 'int'
        ? first + increment * index
        : Number((first + increment * index).toPrecision(12)));
    if (type === 'int' && values.some((value) => !Number.isSafeInteger(value))) {
        return { values: [], error: 'Use whole numbers within the safe integer range.' };
    }
    return { values, error: null };
}

export function prepareArchitectureSearch(graph, axes) {
    const parameters = graph?.parameters ?? [];
    if (!Array.isArray(axes) || !axes.length || axes.length > MAX_SEARCH_AXES) return { candidates: [], error: `Choose one to ${MAX_SEARCH_AXES} global parameters.` };
    const names = axes.map((axis) => axis.name);
    if (new Set(names).size !== names.length) return { candidates: [], error: 'Choose a different parameter in each row.' };
    const parsedAxes = [];
    for (const axis of axes) {
        const parameter = parameters.find((item) => item.name === axis.name);
        if (!parameter || !['int', 'float'].includes(parameter.type)) return { candidates: [], error: `Choose a numeric global parameter: ${axis.name}.` };
        const parsed = axis.mode === 'range'
            ? parseSearchRange(axis.start, axis.end, axis.step, parameter.type)
            : parseSearchValues(axis.values, parameter.type);
        if (parsed.error) return { candidates: [], error: `${axis.name}: ${parsed.error}` };
        parsedAxes.push({ name: axis.name, values: parsed.values });
    }
    const count = parsedAxes.reduce((product, axis) => product * axis.values.length, 1);
    if (count > MAX_CANDIDATES) return { candidates: [], error: `These settings create ${count} combinations; the limit is ${MAX_CANDIDATES}.` };
    let candidates = [{}];
    for (const axis of parsedAxes) candidates = candidates.flatMap((candidate) => axis.values.map((value) => ({ ...candidate, [axis.name]: value })));
    return { candidates, axes: parsedAxes, error: null };
}

/** Analyze one candidate without changing the editor or its undo history. */
export function evaluateArchitectureCandidate(graph, templates, values) {
    const changed = { ...graph, parameters: (graph.parameters ?? []).map((parameter) =>
        Object.hasOwn(values, parameter.name) ? { ...parameter, default: values[parameter.name] } : parameter) };
    const report = validate(changed, { templates });
    const count = countArchitectureParameters(changed, templates);
    const issues = [...new Set([...report.globalErrors, ...Object.entries(report.errors).flatMap(([id, messages]) =>
        messages.map((message) => `${changed.nodes[id]?.label ?? id}: ${message}`))])];
    return { values, valid: report.ok, parameters: count.complete ? count.total : null,
        trainable: count.complete ? count.trainable : null, unsupported: count.unsupported, issues: issues.slice(0, 8) };
}

export function bestArchitectureCandidate(rows, { objective = 'smallest', budget = null, referenceValues = {} } = {}) {
    const eligible = rows.filter((row) => row.valid && row.parameters !== null && (budget === null || row.parameters <= budget));
    if (!eligible.length) return null;
    const editDistance = (row) => Object.entries(row.values).reduce((distance, [name, value]) => {
        const initial = Number(referenceValues[name]);
        return distance + (Number.isFinite(initial) ? Math.abs(value - initial) / Math.max(1, Math.abs(initial)) : 0);
    }, 0);
    return eligible.reduce((best, row) => {
        if (row.parameters === best.parameters) return editDistance(row) < editDistance(best) ? row : best;
        return objective === 'largest-within-budget'
            ? row.parameters > best.parameters ? row : best
            : row.parameters < best.parameters ? row : best;
    });
}
