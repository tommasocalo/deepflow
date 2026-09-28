import { validate } from '../../validation/validator.js';
import { countArchitectureParameters } from './parameterCount.js';

export const MAX_CANDIDATES = 24;

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

export function prepareArchitectureSearch(graph, axes) {
    const parameters = graph?.parameters ?? [];
    if (!Array.isArray(axes) || !axes.length || axes.length > 2) return { candidates: [], error: 'Choose one or two global parameters.' };
    const names = axes.map((axis) => axis.name);
    if (new Set(names).size !== names.length) return { candidates: [], error: 'Choose two different parameters.' };
    const parsedAxes = [];
    for (const axis of axes) {
        const parameter = parameters.find((item) => item.name === axis.name);
        if (!parameter || !['int', 'float'].includes(parameter.type)) return { candidates: [], error: `Choose a numeric global parameter: ${axis.name}.` };
        const parsed = parseSearchValues(axis.values, parameter.type);
        if (parsed.error) return { candidates: [], error: `${axis.name}: ${parsed.error}` };
        parsedAxes.push({ name: axis.name, values: parsed.values });
    }
    const count = parsedAxes.reduce((product, axis) => product * axis.values.length, 1);
    if (count > MAX_CANDIDATES) return { candidates: [], error: `The two lists create ${count} combinations; the limit is ${MAX_CANDIDATES}.` };
    let candidates = [{}];
    for (const axis of parsedAxes) candidates = candidates.flatMap((candidate) => axis.values.map((value) => ({ ...candidate, [axis.name]: value })));
    return { candidates, error: null };
}

/** Analyze one candidate without changing the editor or its undo history. */
export function evaluateArchitectureCandidate(graph, templates, values) {
    const changed = { ...graph, parameters: (graph.parameters ?? []).map((parameter) =>
        Object.hasOwn(values, parameter.name) ? { ...parameter, default: values[parameter.name] } : parameter) };
    const report = validate(changed, { templates });
    const count = countArchitectureParameters(changed, templates);
    const issues = [...report.globalErrors, ...Object.values(report.errors).flat()];
    return { values, valid: report.ok, parameters: count.complete ? count.total : null,
        trainable: count.complete ? count.trainable : null, unsupported: count.unsupported, issues: issues.slice(0, 4) };
}

export function bestArchitectureCandidate(rows, { objective = 'smallest', budget = null } = {}) {
    const eligible = rows.filter((row) => row.valid && row.parameters !== null && (budget === null || row.parameters <= budget));
    if (!eligible.length) return null;
    return eligible.reduce((best, row) => objective === 'largest-within-budget'
        ? row.parameters > best.parameters ? row : best
        : row.parameters < best.parameters ? row : best);
}
