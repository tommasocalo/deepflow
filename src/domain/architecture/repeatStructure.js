import { evaluateExpression, repeatBindingValue } from '../expressions.js';
import { getEntry } from '../library/index.js';

// Keep scientific literals intact and match whole identifiers, including $refs.
const TOKENS = /((?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)|\$?([a-zA-Z_][\w$]*)/g;
const hasIndex = (value, variable) => Array.isArray(value)
    ? value.some((item) => hasIndex(item, variable))
    : typeof value === 'string' && [...value.matchAll(TOKENS)].some((match) => match[2] === variable);
const sameValue = (left, right) => left === right || JSON.stringify(left) === JSON.stringify(right);

/** Pin only the repeat index; upstream width/head expressions stay symbolic. */
function firstValue(value, variable, parameter) {
    const first = repeatBindingValue(value, parameter, 0);
    const substitute = (item) => {
        if (Array.isArray(item)) return item.map(substitute);
        if (!hasIndex(item, variable)) return item;
        const expression = item.replace(TOKENS, (token, numeric, name) => name === variable ? '0' : token);
        // A comma denotes a shape tuple, never the arithmetic comma operator.
        return expression.includes(',') ? expression : evaluateExpression(expression) ?? expression;
    };
    return substitute(first);
}

/**
 * A shared definition cannot be rewritten just for Single. Detect free uses of
 * its repeat index beyond the instance bindings, respecting nested repeats and
 * template parameters that introduce their own variable with the same name.
 */
export function singleConversionIssue(node, templates) {
    if (node?.kind !== 'repeat') return 'Select a repeated component first.';
    const template = templates[node.templateId];
    if (!template) return 'The component template is missing. Restore it before switching to Single.';
    const variable = node.iterationVariable || 'i';
    const inspect = (definition, bindings = {}, projectDefaults = false, ancestors = new Set()) => {
        if (!definition || ancestors.has(definition.id)) return false;
        const path = new Set(ancestors).add(definition.id);
        for (const parameter of definition.freeParameters ?? []) {
            const raw = bindings[parameter.name];
            const defaults = raw === undefined || (raw === null && !parameter.nullable) || raw === '';
            if (defaults && !projectDefaults && hasIndex(parameter.default, variable)) return true;
            // Once this parameter is resolved it shadows the outer index for
            // following defaults and every operation in this definition.
            if (parameter.name === variable) return false;
        }
        for (const child of Object.values(definition.graph.nodes)) {
            if (child.kind === 'layer') {
                const specs = getEntry(child.libraryKey)?.parameters ?? [];
                if (Object.entries(child.parameters ?? {}).some(([name, value]) => {
                    const spec = specs.find((parameter) => parameter.name === name);
                    // Names and enum choices are labels, not iteration formulas.
                    // Input.shape is the legacy string representation of a tuple.
                    if (spec?.type === 'enum' || (spec?.type === 'string' && name !== 'shape')) return false;
                    return hasIndex(value, variable);
                })) return true;
                continue;
            }
            if (child.kind === 'repeat' && hasIndex(child.count, variable)) return true;
            if (child.kind === 'repeat' && (child.iterationVariable || 'i') === variable) continue;
            const values = child.kind === 'repeat' ? child.bindings : child.parameterValues;
            if (Object.values(values ?? {}).some((value) => hasIndex(value, variable))) return true;
            if (inspect(templates[child.templateId], values, false, path)) return true;
        }
        return false;
    };
    return inspect(template, node.bindings, true)
        ? `This definition uses the repeat index “${variable}” directly. Expose it as a template parameter before switching to Single.`
        : null;
}

export function repeatAsSingle(node, template) {
    const variable = node.iterationVariable || 'i';
    const bindings = node.bindings ?? {};
    const parameters = new Map((template.freeParameters ?? []).map((parameter) => [parameter.name, parameter]));
    const parameterValues = Object.fromEntries(Object.entries(bindings).map(([name, value]) => [name, firstValue(value, variable, parameters.get(name))]));
    for (const parameter of template.freeParameters ?? []) {
        const raw = parameterValues[parameter.name];
        if ((raw === undefined || (raw === null && !parameter.nullable) || raw === '') && hasIndex(parameter.default, variable)) {
            parameterValues[parameter.name] = firstValue(parameter.default, variable, parameter);
        }
    }
    const single = {
        ...node,
        kind: 'supernode',
        parameterValues,
        repeatSettings: { count: node.count, mode: node.mode ?? 'sequential', iterationVariable: variable, bindings, singleValues: { ...parameterValues } },
    };
    delete single.count;
    delete single.mode;
    delete single.iterationVariable;
    delete single.bindings;
    return single;
}

export function instanceAsRepeat(node, count, mode) {
    const remembered = node.repeatSettings;
    const values = node.parameterValues ?? {};
    const bindings = { ...(remembered?.bindings ?? values) };
    if (remembered) {
        for (const name of new Set([...Object.keys(remembered.singleValues ?? {}), ...Object.keys(values)])) {
            if (!Object.hasOwn(values, name)) delete bindings[name];
            else if (!sameValue(values[name], remembered.singleValues?.[name])) bindings[name] = values[name];
        }
    }
    const repeat = {
        ...node,
        kind: 'repeat',
        count: count !== undefined ? count : remembered?.count ?? 2,
        mode: mode ?? remembered?.mode ?? 'sequential',
        iterationVariable: remembered?.iterationVariable || 'i',
        bindings,
    };
    delete repeat.parameterValues;
    delete repeat.repeatSettings;
    return repeat;
}
