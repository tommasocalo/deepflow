import { STANDARD_COMPONENTS } from '../../../domain/components/index.js';
import { getEntry } from '../../../domain/library/index.js';
import { ACTIONS } from './actions.js';
import { PREDICATES } from './checks.js';
import { KNOWN_MATCH_KEYS } from './nodes.js';
import { isValidTargetRef } from './targets.js';

/**
 * Lesson schema (docs/edu/SPEC.md §6.3).
 *
 * @typedef {string} ModuleId                 'm1' … 'm7' for the curriculum; tests may use other slugs
 * @typedef {'data'|'model'|'shapes'|'objective'|'train'|'evaluate'|'export'} SubgoalId
 * @typedef {'learn'|'architecture'|'data'|'training'} TabId
 *
 * @typedef {Object} LessonModule
 * @property {ModuleId} id
 * @property {number} version              bump when step ids or meaning change (§6.8.2)
 * @property {number} order                1..7
 * @property {number} minutes
 * @property {string[]} datasets           dataset ids used (must exist in the registry)
 * @property {ModuleId[]} [recommendedAfter]
 * @property {Object<string, Fixture>} fixtures
 * @property {string} start                fixture id loaded when the module starts the first time
 * @property {Object<string, Fact>} facts  numbers used in text (§6.3.6)
 * @property {Object<string, (ctx: object) => ({ pass: boolean }|boolean)>} [checks]   custom checks for { custom: name }
 * @property {QuizItem[]} warmup           0–3 items
 * @property {LessonStep[]} steps          4–8 items
 * @property {QuizItem[]} quiz             3–4 items
 * @property {string[]} objectives         LO ids, e.g. ['m1.lo1', …]; text key `<lo>` holds the sentence
 *
 * @typedef {Object} LessonStep
 * @property {string} id                   's1'…; unique in the module
 * @property {'read'|'task'|'predict'} kind
 * @property {SubgoalId} subgoal
 * @property {TabId} [tab]                 tab shown when the step opens
 * @property {TargetRef[]} [target]        spotlight, in order (the first visible one is ringed)
 * @property {ActionSpec[]} [setup]        run once on first entry, before the entry snapshot
 * @property {CheckSpec} check
 * @property {Diagnostic[]} [diagnose]     evaluated in order while the check fails
 * @property {PredictSpec} [predict]       required when kind === 'predict'
 * @property {ActionSpec[]} solution       Show me; must make check pass from the entry state (tested, §8.4)
 * @property {PaletteItem[]} [palette]
 * @property {{ gate?: 'train' }} [locks]
 * @property {EmbedId[]} [embeds]
 * @property {StepButton[]} [buttons]      extra action buttons, e.g. M7 s7 "Load GPT-2 small"
 * @property {string} [tagRun]             tag the run that satisfied a `run` check when the step passes
 * @property {RevealSpec} [reveal]         live values shown after the prediction or the pass
 *
 * @typedef {Object} PredictSpec
 * @property {string} id                   'm1.params'; stored in progress answers
 * @property {'number'|'choice'|'shape'} kind
 * @property {number} [expected]
 * @property {number} [tolerance]          absolute; default 0
 * @property {string[]} [options]          option ids; text `<m>.<s>.predict.opt.<id>`
 * @property {string} [correct]
 * @property {Array<number|null>} [expectedShape]   null = batch axis; the input accepts "B, 1, 28, 28"
 * @property {{ equals: number|string|Array<number|null>, misconception: string }[]} [diagnose]
 *
 * @typedef {Object} QuizItem
 * @property {string} id                   'm1.q1' or 'm2.w1'
 * @property {string} objective            LO id
 * @property {string[]} options            option ids; text `<id>.opt.<option>` and `<id>.why.<option>`
 * @property {string} correct
 * @property {Object<string, string>} [misconceptions]   option id → misconception id
 *
 * @typedef {{ block: BlockRef, count?: number, distractor?: string, into?: { from: object, to: object }, params?: object }} PaletteItem
 *           distractor = misconception id; `into` splices the block into the wire from → to while it exists;
 *           `params` are the new block's starting parameters (a Linear also starts at the open end's width)
 * @typedef {string} BlockRef        'component:linear' | a library key such as 'torch.nn.ReLU' | 'deepflow.op.Add'
 * @typedef {'trainer'|'paramCount'|'softmaxTemperature'} EmbedId
 * @typedef {{ id: string, actions: ActionSpec[] }} StepButton        label text `<m>.<s>.button.<id>`
 * @typedef {{ run?: 'initLoss'|'finalValAccuracy', compare?: { tag: string, metric: 'gap'|'valLoss'|'valAccuracy' } }} RevealSpec
 * @typedef {{ when: CheckSpec, misconception: string }} Diagnostic
 * @typedef {string | { node: object } | { nodeOpen: object }} TargetRef
 * @typedef {{ do: string, [argument: string]: * }} ActionSpec
 * @typedef {Object<string, *>} CheckSpec   exactly one predicate key (§6.4.3)
 *
 * @typedef {Object} Fixture
 * @property {'snapshot'|'storyboard'} kind
 * @property {(helpers: object) => WorkspaceSnapshot} [build]     kind 'snapshot': pure; returns a snapshot (§6.8.3)
 * @property {string} [storyboard]
 * @property {Object} [configuration]      kind 'storyboard': run a STORYBOARDS entry with options
 * @property {ActionSpec[]} [after]        applied after a storyboard (e.g. setOptimizer, useDataset)
 *
 * @typedef {Object} Fact
 * @property {number|string} value
 * @property {(helpers: object) => number|string} [compute]   recomputed by the content test (§8.4)
 * @property {'simulation'|'literature'} [source]              facts that tests cannot recompute
 *
 * @typedef {{ v: 1, savedAt: number, graph: object, templates: Object, training: object, slices: Object<string, *>,
 *             ui?: { showDimensions?: boolean, showGradientFlow?: boolean } }} WorkspaceSnapshot
 */
export const SUBGOALS = Object.freeze(['data', 'model', 'shapes', 'objective', 'train', 'evaluate', 'export']);
export const STEP_KINDS = Object.freeze(['read', 'task', 'predict']);
export const TABS = Object.freeze(['learn', 'architecture', 'data', 'training']);
export const EMBEDS = Object.freeze(['trainer', 'paramCount', 'softmaxTemperature']);
export const PREDICT_KINDS = Object.freeze(['number', 'choice', 'shape']);
const RUN_QUERY_KEYS = ['dataset', 'minEpochs', 'initOnly', 'final', 'overfit', 'model', 'lr'];

export function isValidBlockRef(block) {
    if (typeof block !== 'string') return false;
    if (block.startsWith('component:')) return STANDARD_COMPONENTS.some((component) => component.key === block.slice('component:'.length));
    return !!getEntry(block);
}

function checkNodeMatch(match, where, errors) {
    if (!match || typeof match !== 'object' || Array.isArray(match)) { errors.push(`${where}: a node match must be an object`); return; }
    for (const key of Object.keys(match)) if (!KNOWN_MATCH_KEYS.includes(key)) errors.push(`${where}: unknown node match key "${key}"`);
}

/** Structural validation of a CheckSpec tree. Returns the misconception-free list of problems. */
export function validateCheckSpec(spec, where = 'check', errors = []) {
    if (!spec || typeof spec !== 'object' || Array.isArray(spec)) { errors.push(`${where}: a check must be an object`); return errors; }
    const keys = Object.keys(spec).filter((key) => PREDICATES.includes(key));
    if (keys.length !== 1) { errors.push(`${where}: a check needs exactly one predicate (found ${Object.keys(spec).join(', ') || 'none'})`); return errors; }
    const [name] = keys;
    const value = spec[name];
    const extra = Object.keys(spec).filter((key) => key !== name && !(name === 'has' && ['count', 'deep'].includes(key)) && !(name === 'chain' && key === 'allow'));
    if (extra.length) errors.push(`${where}: unexpected keys ${extra.join(', ')} next to ${name}`);
    if (['all', 'any'].includes(name)) {
        if (!Array.isArray(value) || !value.length) errors.push(`${where}.${name}: needs a non-empty list`);
        else value.forEach((item, index) => validateCheckSpec(item, `${where}.${name}[${index}]`, errors));
    } else if (name === 'not') {
        validateCheckSpec(value, `${where}.not`, errors);
    } else if (['visited', 'tab'].includes(name)) {
        if (!TABS.includes(value)) errors.push(`${where}.${name}: unknown tab ${value}`);
    } else if (['has'].includes(name)) {
        checkNodeMatch(value, `${where}.has`, errors);
    } else if (['param', 'shapeAt'].includes(name)) {
        checkNodeMatch(value?.node, `${where}.${name}.node`, errors);
        if (name === 'param' && typeof value?.name !== 'string') errors.push(`${where}.param: needs a parameter name`);
    } else if (name === 'chain') {
        if (!Array.isArray(value) || value.length < 2) errors.push(`${where}.chain: needs at least two node matches`);
        else value.forEach((item, index) => checkNodeMatch(item, `${where}.chain[${index}]`, errors));
    } else if (name === 'edge') {
        checkNodeMatch(value?.from, `${where}.edge.from`, errors);
        checkNodeMatch(value?.to, `${where}.edge.to`, errors);
    } else if (['inputShape', 'outputShape'].includes(name)) {
        if (!Array.isArray(value)) errors.push(`${where}.${name}: needs a shape array`);
    } else if (name === 'run') {
        for (const key of Object.keys(value ?? {})) if (!RUN_QUERY_KEYS.includes(key)) errors.push(`${where}.run: unknown query key "${key}"`);
        if (value?.model) validateCheckSpec(value.model, `${where}.run.model`, errors);
    } else if (name === 'runs') {
        if (!Array.isArray(value?.lr) || !value.lr.length) errors.push(`${where}.runs: needs a list of learning rates`);
    } else if (name === 'level') {
        checkNodeMatch(value?.open, `${where}.level.open`, errors);
    } else if (name === 'repeat') {
        checkNodeMatch(value?.node, `${where}.repeat.node`, errors);
    } else if (name === 'component') {
        if (!Array.isArray(value?.contains)) errors.push(`${where}.component: needs contains: [...]`);
    } else if (name === 'custom' || name === 'answered' || name === 'answeredCorrectly' || name === 'dataset') {
        if (typeof value !== 'string' || !value) errors.push(`${where}.${name}: needs an id`);
    }
    return errors;
}

export function validateActions(actions, where, errors = [], { module } = {}) {
    if (!Array.isArray(actions)) { errors.push(`${where}: actions must be a list`); return errors; }
    actions.forEach((action, index) => {
        const at = `${where}[${index}]`;
        if (!ACTIONS.includes(action?.do)) { errors.push(`${at}: unknown action ${action?.do}`); return; }
        if (['insertBetween', 'insertAfter'].includes(action.do) && !isValidBlockRef(action.block)) errors.push(`${at}: unknown block ${action.block}`);
        if (action.do === 'buildChain') {
            for (const item of action.blocks ?? []) if (!isValidBlockRef(item.block)) errors.push(`${at}: unknown block ${item.block}`);
        }
        if (action.do === 'loadFixture' && module && !module.fixtures?.[action.id]) errors.push(`${at}: unknown fixture ${action.id}`);
        if (action.do === 'setUi' && action.tab && !TABS.includes(action.tab)) errors.push(`${at}: unknown tab ${action.tab}`);
    });
    return errors;
}

const misconceptionsOf = (module) => {
    const ids = new Set();
    for (const step of module.steps ?? []) {
        for (const rule of step.diagnose ?? []) ids.add(rule.misconception);
        for (const rule of step.predict?.diagnose ?? []) ids.add(rule.misconception);
        for (const item of step.palette ?? []) if (item.distractor) ids.add(item.distractor);
    }
    for (const item of [...(module.warmup ?? []), ...(module.quiz ?? [])]) for (const id of Object.values(item.misconceptions ?? {})) ids.add(id);
    return [...ids].filter(Boolean);
};

/** Required and optional text keys of a module (§6.3.3); the content test also checks that no key is unused. */
export function moduleTextKeys(module) {
    const m = module.id;
    const required = [`${m}.title`, `${m}.goal`, `${m}.recap.1`, `${m}.recap.2`, `${m}.recap.3`, ...(module.objectives ?? [])];
    const optional = [];
    for (const step of module.steps ?? []) {
        const s = `${m}.${step.id}`;
        required.push(`${s}.title`, `${s}.body`, `${s}.hint1`, `${s}.hint2`);
        optional.push(`${s}.why`, `${s}.done`);
        if (step.predict) {
            required.push(`${s}.predict.prompt`, `${s}.predict.reveal`);
            for (const option of step.predict.options ?? []) required.push(`${s}.predict.opt.${option}`);
        }
        for (const button of step.buttons ?? []) required.push(`${s}.button.${button.id}`);
        if (step.reveal?.compare) required.push(`${s}.reveal.better`, `${s}.reveal.notBetter`);
    }
    for (const item of [...(module.warmup ?? []), ...(module.quiz ?? [])]) {
        required.push(`${item.id}.prompt`);
        for (const option of item.options ?? []) required.push(`${item.id}.opt.${option}`, `${item.id}.why.${option}`);
    }
    for (const id of misconceptionsOf(module)) required.push(`mis.${id}`);
    return { required: [...new Set(required)], optional: [...new Set(optional)] };
}

function hasVisited(spec, tab) {
    if (!spec || typeof spec !== 'object') return false;
    if (spec.visited === tab) return true;
    return Object.values(spec).some((value) => (Array.isArray(value) ? value.some((item) => hasVisited(item, tab)) : typeof value === 'object' && hasVisited(value, tab)));
}

/**
 * Validate a module definition and its text bundle (`text` is the flat key → string map of one locale).
 * Options: `datasetIds` (registry ids, checked when given), `limits: false` to skip the 4–8 / 3–4 size rules.
 * @returns {string[]} problems; empty means valid
 */
export function validateModule(module, text = {}, { datasetIds, limits = true } = {}) {
    const errors = [];
    if (!module || typeof module !== 'object') return ['module must be an object'];
    const m = module.id;
    if (typeof m !== 'string' || !/^[a-z][a-z0-9-]*$/.test(m)) errors.push(`id must be a slug (got ${m})`);
    if (!Number.isInteger(module.version) || module.version < 1) errors.push(`${m}: version must be a positive integer`);
    if (!Number.isInteger(module.order) || module.order < 1) errors.push(`${m}: order must be a positive integer`);
    if (!(module.minutes > 0)) errors.push(`${m}: minutes must be positive`);
    if (!Array.isArray(module.datasets)) errors.push(`${m}: datasets must be a list`);
    else if (datasetIds) for (const id of module.datasets) if (!datasetIds.includes(id)) errors.push(`${m}: unknown dataset ${id}`);
    if (!module.fixtures || typeof module.fixtures !== 'object') errors.push(`${m}: fixtures must be an object`);
    else {
        if (!module.fixtures[module.start]) errors.push(`${m}: start fixture "${module.start}" is missing`);
        for (const [id, fixture] of Object.entries(module.fixtures)) {
            const kind = fixture?.kind ?? 'snapshot';
            if (kind === 'snapshot' && typeof fixture?.build !== 'function') errors.push(`${m}: fixture ${id} needs build()`);
            if (kind === 'storyboard' && typeof fixture?.storyboard !== 'string') errors.push(`${m}: fixture ${id} needs a storyboard id`);
            if (fixture?.after) validateActions(fixture.after, `${m}.fixtures.${id}.after`, errors, { module });
        }
    }
    for (const [name, fact] of Object.entries(module.facts ?? {})) {
        if (fact?.value === undefined) errors.push(`${m}: fact ${name} needs a value`);
    }
    const steps = module.steps ?? [];
    const quiz = module.quiz ?? [];
    const warmup = module.warmup ?? [];
    if (limits) {
        if (steps.length < 4 || steps.length > 8) errors.push(`${m}: needs 4–8 steps (has ${steps.length})`);
        if (quiz.length < 3 || quiz.length > 4) errors.push(`${m}: needs 3–4 quiz items (has ${quiz.length})`);
        if (warmup.length > 3) errors.push(`${m}: at most 3 warm-up items (has ${warmup.length})`);
    }
    if (!Array.isArray(module.objectives)) errors.push(`${m}: objectives must be a list`);
    const stepIds = new Set();
    for (const step of steps) {
        const s = `${m}.${step.id}`;
        if (stepIds.has(step.id)) errors.push(`${s}: duplicate step id`);
        stepIds.add(step.id);
        if (!STEP_KINDS.includes(step.kind)) errors.push(`${s}: unknown kind ${step.kind}`);
        if (!SUBGOALS.includes(step.subgoal)) errors.push(`${s}: unknown subgoal ${step.subgoal}`);
        if (step.tab !== undefined && !TABS.includes(step.tab)) errors.push(`${s}: unknown tab ${step.tab}`);
        if (step.tab && hasVisited(step.check, step.tab)) errors.push(`${s}: a step that checks { visited: '${step.tab}' } must not open that tab itself`);
        validateCheckSpec(step.check, `${s}.check`, errors);
        for (const [index, rule] of (step.diagnose ?? []).entries()) {
            validateCheckSpec(rule.when, `${s}.diagnose[${index}].when`, errors);
            if (typeof rule.misconception !== 'string') errors.push(`${s}.diagnose[${index}]: needs a misconception id`);
        }
        if (step.setup) validateActions(step.setup, `${s}.setup`, errors, { module });
        if (!Array.isArray(step.solution)) errors.push(`${s}: needs a solution (Show me)`);
        else validateActions(step.solution, `${s}.solution`, errors, { module });
        for (const button of step.buttons ?? []) validateActions(button.actions, `${s}.button.${button.id}`, errors, { module });
        if (step.kind === 'predict' && !step.predict) errors.push(`${s}: predict steps need a predict spec`);
        if (step.predict) {
            const p = step.predict;
            if (typeof p.id !== 'string') errors.push(`${s}.predict: needs an id`);
            if (!PREDICT_KINDS.includes(p.kind)) errors.push(`${s}.predict: unknown kind ${p.kind}`);
            if (p.kind === 'number' && typeof p.expected !== 'number') errors.push(`${s}.predict: number predictions need expected`);
            if (p.kind === 'shape' && !Array.isArray(p.expectedShape)) errors.push(`${s}.predict: shape predictions need expectedShape`);
            if (p.kind === 'choice' && (!Array.isArray(p.options) || !p.options.includes(p.correct))) errors.push(`${s}.predict: correct must be one of the options`);
        }
        for (const item of step.palette ?? []) {
            if (!isValidBlockRef(item.block)) errors.push(`${s}.palette: unknown block ${item.block}`);
            if (item.into !== undefined) {
                checkNodeMatch(item.into?.from, `${s}.palette.into.from`, errors);
                checkNodeMatch(item.into?.to, `${s}.palette.into.to`, errors);
            }
            if (item.params !== undefined && (!item.params || typeof item.params !== 'object' || Array.isArray(item.params))) errors.push(`${s}.palette: params must be an object`);
        }
        for (const embed of step.embeds ?? []) if (!EMBEDS.includes(embed)) errors.push(`${s}: unknown embed ${embed}`);
        for (const ref of step.target ?? []) if (!isValidTargetRef(ref)) errors.push(`${s}: unknown target ${JSON.stringify(ref)}`);
        if (step.locks?.gate && step.locks.gate !== 'train') errors.push(`${s}: unknown gate ${step.locks.gate}`);
        if (step.locks?.gate && !step.predict) errors.push(`${s}: a train gate needs a prediction`);
    }
    for (const item of [...warmup, ...quiz]) {
        if (typeof item.id !== 'string') errors.push(`${m}: quiz items need ids`);
        if (!Array.isArray(item.options) || !item.options.includes(item.correct)) errors.push(`${item.id}: correct must be one of the options`);
        if (item.objective && !(module.objectives ?? []).includes(item.objective) && item.id.startsWith(`${m}.q`)) errors.push(`${item.id}: objective ${item.objective} is not a module objective`);
    }
    for (const key of moduleTextKeys(module).required) {
        if (typeof text[key] !== 'string' || !text[key].trim()) errors.push(`${m}: missing text ${key}`);
    }
    return errors;
}
