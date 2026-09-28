import { buildParameterScope, withGlobalParameters } from '../expressions.js';
import { effectiveTemplateId } from './instanceStructure.js';
import { resolveLevelContext } from './levelContext.js';

/** A deliberately small predicate language. Formulas in the architecture use
 * their own evaluator; rules cannot call functions or read object properties. */
const TOKEN = /\s*(?:((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|([a-zA-Z_][\w]*)|(===|!==|==|!=|<=|>=|&&|\|\||\*\*|[().!<>+*/%-]))/y;

function tokenize(source) {
    const tokens = [];
    let cursor = 0;
    while (cursor < source.length) {
        TOKEN.lastIndex = cursor;
        const match = TOKEN.exec(source);
        if (!match) {
            if (!source.slice(cursor).trim()) break;
            throw new Error(`Unexpected character at position ${cursor + 1}`);
        }
        tokens.push(match[1] ? { kind: 'number', value: Number(match[1]) }
            : match[2] ? { kind: 'name', value: match[2] }
                : { kind: 'operator', value: match[3] });
        cursor = TOKEN.lastIndex;
    }
    return tokens;
}

/** Compile once, then evaluate the same condition for each instance. */
export function compileRepeatRule(source) {
    if (!source?.trim()) throw new Error('Enter a rule, for example position % 2 == 0');
    const tokens = tokenize(source);
    let cursor = 0;
    const peek = () => tokens[cursor]?.value;
    const take = (value) => peek() === value ? (cursor++, true) : false;
    const expect = (value) => { if (!take(value)) throw new Error(`Expected ${value}`); };
    const chain = (inner, operators) => {
        let left = inner();
        while (operators.includes(peek())) {
            const operator = tokens[cursor++].value;
            const right = inner();
            const previous = left;
            left = (environment) => {
                const a = previous(environment);
                if (operator === '||' && a) return true;
                if (operator === '&&' && !a) return false;
                const b = right(environment);
                switch (operator) {
                    case '||': return Boolean(a || b);
                    case '&&': return Boolean(a && b);
                    case '==': case '===': return a === b;
                    case '!=': case '!==': return a !== b;
                    case '<': return a < b;
                    case '<=': return a <= b;
                    case '>': return a > b;
                    case '>=': return a >= b;
                    case '+': return a + b;
                    case '-': return a - b;
                    case '*': return a * b;
                    case '/': return a / b;
                    case '%': return a % b;
                    default: return null;
                }
            };
        }
        return left;
    };
    const primary = () => {
        const token = tokens[cursor++];
        if (!token) throw new Error('The rule ends before the condition is complete');
        if (token.kind === 'number') return () => token.value;
        if (token.value === '(') {
            const expression = or();
            expect(')');
            return expression;
        }
        if (token.kind === 'name') {
            if (token.value === 'true') return () => true;
            if (token.value === 'false') return () => false;
            if (token.value === 'position') return ({ position }) => position;
            if (!['local', 'global'].includes(token.value)) throw new Error(`Unknown variable ${token.value}`);
            expect('.');
            const field = tokens[cursor++];
            if (field?.kind !== 'name') throw new Error(`Expected a parameter after ${token.value}.`);
            return (environment) => {
                const value = environment[token.value]?.[field.value];
                if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error(`${token.value}.${field.value} has no numeric value here`);
                return value;
            };
        }
        throw new Error(`Unexpected ${token.value}`);
    };
    const power = () => {
        const left = primary();
        if (take('**')) {
            const right = unary();
            return (environment) => left(environment) ** right(environment);
        }
        return left;
    };
    const unary = () => {
        if (take('!')) { const right = unary(); return (environment) => !right(environment); }
        if (take('-')) { const right = unary(); return (environment) => -right(environment); }
        if (take('+')) { const right = unary(); return (environment) => +right(environment); }
        return power();
    };
    const multiply = () => chain(unary, ['*', '/', '%']);
    const add = () => chain(multiply, ['+', '-']);
    const compare = () => chain(add, ['<', '<=', '>', '>=']);
    const equal = () => chain(compare, ['==', '===', '!=', '!==']);
    const and = () => chain(equal, ['&&']);
    const or = () => chain(and, ['||']);
    const expression = or();
    if (cursor < tokens.length) throw new Error(`Unexpected ${peek()}`);
    return (environment) => {
        const value = expression(environment);
        if (typeof value !== 'boolean') throw new Error('A rule must return true or false; add a comparison such as == 0');
        return value;
    };
}

/** `position` is one based. `local` contains the visible repeat's free
 * parameters for that occurrence; `global` contains root parameters. */
export function selectRepeatInstances({ graph, path, templates, inspectedIterations = {}, count, rule }) {
    const predicate = compileRepeatRule(rule);
    const parent = resolveLevelContext(graph, path.slice(0, -1), templates, inspectedIterations);
    const repeat = parent.graph?.nodes?.[path.at(-1)];
    if (repeat?.kind !== 'repeat' || !Number.isInteger(count) || count < 1) throw new Error('Open a repeat with a valid instance count');
    const globals = withGlobalParameters(graph);
    const global = Object.fromEntries(Object.entries(globals).filter(([, value]) => typeof value === 'number' && Number.isFinite(value)));
    const indices = [];
    const names = new Set([`local.${repeat.iterationVariable || 'i'}`]);
    for (let index = 0; index < count; index++) {
        const template = templates[effectiveTemplateId(repeat, index)];
        if (!template) throw new Error(`Template missing for instance ${index + 1}`);
        for (const parameter of template.freeParameters ?? []) {
            if (['int', 'float'].includes(parameter.type)) names.add(`local.${parameter.name}`);
        }
        const indexName = repeat.iterationVariable || 'i';
        const scope = buildParameterScope(template.freeParameters ?? [], repeat.bindings ?? {},
            { ...parent.scope, [indexName]: index }, index);
        const local = Object.fromEntries(Object.entries(scope)
            .filter(([name, value]) => (name === indexName || template.freeParameters?.some((parameter) => parameter.name === name))
                && typeof value === 'number' && Number.isFinite(value)));
        try {
            if (predicate({ position: index + 1, local, global })) indices.push(index);
        } catch (error) {
            throw new Error(`Instance ${index + 1}: ${error.message}`);
        }
    }
    return { indices, suggestions: ['position', ...[...names].sort(), ...Object.keys(global).map((name) => `global.${name}`).sort()] };
}

export function repeatRuleSuggestions({ graph, path, templates, inspectedIterations = {} }) {
    const parent = resolveLevelContext(graph, path.slice(0, -1), templates, inspectedIterations);
    const repeat = parent.graph?.nodes?.[path.at(-1)];
    const template = templates[repeat?.templateId];
    return ['position', `local.${repeat?.iterationVariable || 'i'}`,
        ...(template?.freeParameters ?? []).filter((parameter) => ['int', 'float'].includes(parameter.type))
            .map((parameter) => `local.${parameter.name}`),
        ...Object.entries(withGlobalParameters(graph)).filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
            .map(([name]) => `global.${name}`)];
}
