import assert from 'node:assert/strict';
import { test } from 'node:test';

// The curriculum content (docs/edu/SPEC.md §4, §8.4): schema, text, facts, misconceptions and, the key test, that
// every step's check fails on entry and passes after its Show me, module by module, headless, with Stage B's real
// trainer and data stores.

// Install in-memory storage before the stores hydrate (as startup-regressions.test.mjs does).
const memory = new Map();
const storage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: storage };
globalThis.localStorage = storage;

const { MODULES, TEXT } = await import('../src/edu/lessons/modules/index.js');
const { createFactHelpers } = await import('../src/edu/lessons/modules/facts.js');
const { validateModule, moduleTextKeys } = await import('../src/edu/lessons/engine/schema.js');
const { isValidTargetRef } = await import('../src/edu/lessons/engine/targets.js');
const { gradePrediction } = await import('../src/edu/lessons/engine/predict.js');
const { evaluateStep } = await import('../src/edu/lessons/engine/checks.js');
const { buildCheckContext, factValues } = await import('../src/edu/lessons/engine/context.js');
const { interpolate, t } = await import('../src/edu/lessons/engine/i18n.js');
const { revealValues } = await import('../src/edu/lessons/engine/reveal.js');
const { runActions } = await import('../src/edu/lessons/engine/actions.js');
const { captureWorkspace, restoreWorkspace } = await import('../src/edu/lessons/engine/workspace.js');
const { getModules } = await import('../src/edu/lessons/engine/registry.js');
const { DATASETS } = await import('../src/edu/data/registry.js');
await import('../src/edu/data/store.js');
const { useTrainerStore } = await import('../src/edu/trainer/store.js');
const { useLessonSession } = await import('../src/edu/lessons/engine/session.js');
const { useEduUiStore } = await import('../src/edu/shell/uiStore.js');
const { STANDARD_COMPONENTS } = await import('../src/domain/components/index.js');
const { PYTORCH_BLOCKS } = await import('../src/domain/library/pytorchBlocks.js');
const { getEntry } = await import('../src/domain/library/index.js');
const { generateArchitectureModule, generateTrainingScript } = await import('../src/codegen/index.js');
const { readSource } = await import('../src/edu/bridge.js');
const { getDataset } = await import('../src/edu/data/registry.js');
const { useArchitectureStore } = await import('../src/stores/architectureStore.js');
const { useTemplatesStore } = await import('../src/stores/templatesStore.js');
const { useTrainingStore } = await import('../src/stores/trainingStore.js');

const EN = TEXT.en;
const DATASET_IDS = DATASETS.map((spec) => spec.id);
const MODULE_IDS = ['m1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7'];
/** Misconceptions the engine raises on its own (global diagnostics, §6.4.3). */
const GLOBAL_MISCONCEPTIONS = ['softmax-before-ce', 'sigmoid-before-bce'];
/** Live values a step's reveal adds to its text (engine/reveal.js). */
const LIVE_PARAMS = (step) => [
    ...(step.reveal?.run ? ['measured'] : []),
    ...(step.reveal?.compare ? ['before', 'after'] : []),
];
const session = () => useLessonSession.getState();

// ---- helpers ------------------------------------------------------------------------------------------------------

const plain = (text) => String(text).replace(/\*\*|`/g, '');
const words = (text) => plain(text).split(/\s+/).filter((token) => /[\p{L}\p{N}]/u.test(token));
const sentences = (text) => plain(text).split(/\n+|(?<=[.!?])\s+/).map((part) => part.trim()).filter(Boolean);
const placeholders = (text) => [...String(text).matchAll(/\{([A-Za-z_][\w.-]*)\}/g)].map((match) => match[1]);

function walkChecks(spec, visit) {
    if (!spec || typeof spec !== 'object') return;
    visit(spec);
    for (const value of Object.values(spec)) {
        if (Array.isArray(value)) value.forEach((item) => walkChecks(item, visit));
        else if (value && typeof value === 'object') walkChecks(value, visit);
    }
}

function allActions(module) {
    const lists = [];
    for (const fixture of Object.values(module.fixtures)) if (fixture.after) lists.push(fixture.after);
    for (const step of module.steps) {
        lists.push(step.setup ?? [], step.solution ?? []);
        for (const button of step.buttons ?? []) lists.push(button.actions);
        for (const rule of step.diagnose ?? []) lists.push(rule.example ?? []);
    }
    return lists.flat();
}

/** Text keys a module's custom checks return as details (`<m>.check.<name>`), read from their source. */
function customCheckKeys(module) {
    return Object.values(module.checks ?? {}).flatMap((check) => [...String(check).matchAll(/'([a-z0-9]+\.check\.[\w-]+)'/g)].map((match) => match[1]));
}

function moduleMisconceptions(module) {
    const ids = new Set();
    for (const step of module.steps) {
        for (const rule of step.diagnose ?? []) ids.add(rule.misconception);
        for (const rule of step.predict?.diagnose ?? []) ids.add(rule.misconception);
        for (const item of step.palette ?? []) if (item.distractor) ids.add(item.distractor);
    }
    for (const item of [...module.warmup, ...module.quiz]) for (const id of Object.values(item.misconceptions ?? {})) ids.add(id);
    for (const check of Object.values(module.checks ?? {})) for (const match of String(check).matchAll(/misconception: '([\w-]+)'/g)) ids.add(match[1]);
    return ids;
}

/** Every parameter name a block of the library or a standard component exposes (for field:<name> targets). */
const KNOWN_FIELDS = new Set([
    ...PYTORCH_BLOCKS.flatMap((entry) => (entry.parameters ?? []).map((parameter) => parameter.name)),
    'in_features', 'out_features', 'bias', 'num_embeddings', 'embedding_dim', 'count', 'p', 'kernel_size', 'padding',
]);

/** model.py and train.py as the code panel generates them from the live stores (src/components/CodePanel.jsx). */
function liveCode() {
    const graph = useArchitectureStore.getState().graph;
    const templates = useTemplatesStore.getState().templates;
    const binding = readSource('data');
    const spec = binding?.datasetId ? getDataset(binding.datasetId) : null;
    const settings = readSource('trainer')?.settings;
    const data = spec ? {
        spec, options: binding.options ?? {}, split: binding.split ?? {}, dataSeed: binding.dataSeed ?? 1,
        ...(settings ? { settings: { epochs: settings.epochs, batchSize: settings.batchSize, seed: settings.seed } } : {}),
    } : undefined;
    return {
        model: generateArchitectureModule(graph, { templates, className: 'Model' }),
        train: generateTrainingScript(graph, useTrainingStore.getState().graph, { templates, data }),
    };
}

// ---- 1. schema --------------------------------------------------------------------------------------------------

test('the registry holds the seven modules in order, and the engine loads it', () => {
    assert.deepEqual(MODULES.map((module) => module.id), MODULE_IDS);
    assert.deepEqual(MODULES.map((module) => module.order), [1, 2, 3, 4, 5, 6, 7]);
    assert.deepEqual(getModules().map((module) => module.id), MODULE_IDS);
});

test('every module passes validateModule with the registry datasets and the spec sizes', () => {
    for (const module of MODULES) {
        assert.deepEqual(validateModule(module, EN, { datasetIds: DATASET_IDS }), [], module.id);
        assert.ok(module.steps.length >= 4 && module.steps.length <= 8, `${module.id} steps`);
        assert.ok(module.quiz.length >= 3 && module.quiz.length <= 4, `${module.id} quiz`);
        assert.equal(module.warmup.length, 2, `${module.id} warm-up`);
        assert.ok(module.objectives.every((lo) => lo.startsWith(`${module.id}.lo`)), `${module.id} objectives`);
    }
    assert.equal(MODULES.reduce((sum, module) => sum + module.steps.length, 0), 50, 'the spec counts 50 steps');
});

test('an explanation never repeats the verdict the panel prints before it ("Correct. Right.")', () => {
    for (const module of MODULES) {
        for (const item of [...module.warmup, ...module.quiz]) {
            assert.doesNotMatch(EN[`${item.id}.why.${item.correct}`], /^(Right|Correct|Exactly)\b/, `${item.id}.why.${item.correct}`);
        }
        for (const step of module.steps.filter((entry) => entry.predict)) {
            assert.doesNotMatch(EN[`${module.id}.${step.id}.predict.reveal`], /^(Right|Correct|Exactly)\b/, `${module.id}.${step.id}.predict.reveal`);
        }
    }
});

test('quizzes and warm-ups are well formed: one correct answer, every option explained, every distractor diagnosed', () => {
    for (const module of MODULES) {
        const earlier = MODULES.filter((other) => other.order < module.order).flatMap((other) => other.objectives);
        for (const item of [...module.warmup, ...module.quiz]) {
            const where = `${item.id}`;
            assert.equal(new Set(item.options).size, item.options.length, `${where}: duplicate options`);
            assert.equal(item.options.filter((option) => option === item.correct).length, 1, `${where}: exactly one correct option`);
            assert.ok(item.options.length >= 2 && item.options.length <= 4, `${where}: 2-4 options`);
            for (const option of item.options) {
                assert.ok(EN[`${item.id}.opt.${option}`], `${where}: option text ${option}`);
                assert.ok(EN[`${item.id}.why.${option}`], `${where}: explanation for ${option}`);
            }
            for (const [option, id] of Object.entries(item.misconceptions ?? {})) {
                assert.ok(item.options.includes(option) && option !== item.correct, `${where}: misconception on a distractor (${option})`);
                assert.ok(EN[`mis.${id}`], `${where}: misconception ${id} is in the catalogue`);
            }
            const isQuiz = item.id.startsWith(`${module.id}.q`);
            if (isQuiz) {
                assert.ok(module.objectives.includes(item.objective), `${where}: objective`);
                for (const option of item.options) if (option !== item.correct) assert.ok(item.misconceptions?.[option], `${where}: distractor ${option} names a misconception`);
            } else if (module.id !== 'm1') {
                // Retrieval warm-ups ask about earlier modules (§4.0); Module 1 opens with two myths instead.
                assert.ok(earlier.includes(item.objective), `${where}: warm-up objective ${item.objective} belongs to an earlier module`);
            }
        }
        const covered = new Set(module.quiz.map((item) => item.objective));
        assert.deepEqual([...covered].sort(), [...module.objectives].sort(), `${module.id}: the quiz covers every objective`);
    }
});

test('datasets, spotlight targets, blocks and embeds exist', () => {
    for (const module of MODULES) {
        const referenced = new Set(module.datasets);
        for (const step of module.steps) {
            for (const ref of step.target ?? []) {
                assert.ok(isValidTargetRef(ref), `${module.id}.${step.id}: target ${JSON.stringify(ref)}`);
                if (typeof ref === 'string' && ref.startsWith('data.dataset:')) referenced.add(ref.slice('data.dataset:'.length));
                if (typeof ref === 'string' && ref.startsWith('field:')) assert.ok(KNOWN_FIELDS.has(ref.slice('field:'.length)), `${module.id}.${step.id}: field ${ref}`);
            }
            walkChecks(step.check, (spec) => {
                if (spec.dataset && typeof spec.dataset === 'string') referenced.add(spec.dataset);
                if (spec.run?.dataset) referenced.add(spec.run.dataset);
                if (spec.runs?.dataset) referenced.add(spec.runs.dataset);
            });
            for (const item of step.palette ?? []) {
                if (item.block.startsWith('component:')) assert.ok(STANDARD_COMPONENTS.some((component) => `component:${component.key}` === item.block), item.block);
                else assert.ok(getEntry(item.block), item.block);
                if (item.distractor) assert.ok(EN[`mis.${item.distractor}`], `${module.id}.${step.id}: distractor ${item.distractor}`);
            }
        }
        for (const action of allActions(module)) if (action.do === 'useDataset') referenced.add(action.id);
        for (const id of referenced) {
            assert.ok(DATASET_IDS.includes(id), `${module.id}: dataset ${id} exists`);
            assert.ok(module.datasets.includes(id), `${module.id}: dataset ${id} is listed in module.datasets`);
        }
    }
});

// ---- 2. text ----------------------------------------------------------------------------------------------------

test('every text key is used, and every used misconception is in the catalogue', () => {
    const used = new Set(GLOBAL_MISCONCEPTIONS.map((id) => `mis.${id}`));
    for (const module of MODULES) {
        const { required, optional } = moduleTextKeys(module);
        for (const key of [...required, ...optional, ...customCheckKeys(module)]) used.add(key);
        for (const id of moduleMisconceptions(module)) {
            assert.ok(EN[`mis.${id}`], `${module.id}: misconception ${id}`);
            used.add(`mis.${id}`);
        }
        for (const key of customCheckKeys(module)) assert.ok(EN[key], `${module.id}: custom check text ${key}`);
    }
    const unused = Object.keys(EN).filter((key) => !used.has(key));
    assert.deepEqual(unused, [], 'unused text keys');
});

test('copy stays within the text budget: bodies ≤ 60 words, sentences ≤ 20 words, no unresolved placeholders', () => {
    const problems = [];
    const expect = (ok, message) => { if (!ok) problems.push(message); };
    for (const module of MODULES) {
        const facts = factValues(module);
        const check = (key, allowed = [], { body = false } = {}) => {
            const text = EN[key];
            if (text === undefined) return;
            for (const name of placeholders(text)) expect(Object.hasOwn(facts, name) || allowed.includes(name), `${key}: unknown placeholder {${name}}`);
            const shown = interpolate(text, { ...facts, ...Object.fromEntries(allowed.map((name) => [name, '0.0'])) });
            if (body) expect(words(shown).length <= 60, `${key}: ${words(shown).length} words`);
            for (const sentence of sentences(shown)) expect(words(sentence).length <= 20, `${key}: sentence of ${words(sentence).length} words: "${sentence}"`);
        };
        for (const key of [`${module.id}.title`, `${module.id}.goal`, `${module.id}.recap.1`, `${module.id}.recap.2`, `${module.id}.recap.3`, ...module.objectives]) check(key);
        expect(words(EN[`${module.id}.goal`] ?? '').length <= 8, `${module.id}.goal has at most 8 words`);
        for (const step of module.steps) {
            const prefix = `${module.id}.${step.id}`;
            const live = LIVE_PARAMS(step);
            check(`${prefix}.body`, [], { body: true });
            for (const part of ['title', 'why', 'hint1', 'hint2', 'predict.prompt', 'predict.reveal']) check(`${prefix}.${part}`);
            check(`${prefix}.done`, live);
            check(`${prefix}.reveal.better`, live);
            check(`${prefix}.reveal.notBetter`, live);
            for (const option of step.predict?.options ?? []) check(`${prefix}.predict.opt.${option}`);
            for (const button of step.buttons ?? []) check(`${prefix}.button.${button.id}`);
        }
        for (const item of [...module.warmup, ...module.quiz]) {
            check(`${item.id}.prompt`);
            for (const option of item.options) { check(`${item.id}.opt.${option}`); check(`${item.id}.why.${option}`); }
        }
    }
    for (const key of Object.keys(EN).filter((name) => name.startsWith('mis.'))) {
        for (const sentence of sentences(EN[key])) expect(words(sentence).length <= 25, `${key}: "${sentence}"`);
    }
    assert.deepEqual(problems, [], 'copy over the text budget');
});

test('live reveal values only appear where the step has a reveal', () => {
    for (const module of MODULES) {
        for (const step of module.steps) {
            const prefix = `${module.id}.${step.id}`;
            for (const part of ['body', 'why', 'hint1', 'hint2', 'predict.prompt', 'predict.reveal']) {
                for (const name of placeholders(EN[`${prefix}.${part}`] ?? '')) assert.ok(!['measured', 'before', 'after'].includes(name), `${prefix}.${part} uses {${name}} before it exists`);
            }
            if (step.reveal?.run) assert.ok(placeholders(EN[`${prefix}.done`] ?? '').includes('measured'), `${prefix}: reveal.run is shown in done`);
        }
    }
});

// ---- 3. misconceptions in predictions -------------------------------------------------------------------------------

test('every prediction diagnosis fires on its own wrong answer and never on the right one', () => {
    for (const module of MODULES) {
        for (const step of module.steps) {
            const predict = step.predict;
            if (!predict) continue;
            const correctValue = predict.kind === 'number' ? predict.expected : predict.kind === 'shape' ? predict.expectedShape : predict.correct;
            assert.equal(gradePrediction(predict, correctValue).correct, true, `${predict.id}: the expected answer grades as correct`);
            if (predict.kind === 'choice') assert.equal(new Set(predict.options).size, predict.options.length, `${predict.id}: options`);
            for (const rule of predict.diagnose ?? []) {
                const grade = gradePrediction(predict, rule.equals);
                assert.equal(grade.correct, false, `${predict.id}: ${JSON.stringify(rule.equals)} is a wrong answer`);
                assert.equal(grade.misconception, rule.misconception, `${predict.id}: ${JSON.stringify(rule.equals)} → ${rule.misconception}`);
                if (predict.kind === 'choice') assert.ok(predict.options.includes(rule.equals), `${predict.id}: diagnosed option ${rule.equals} exists`);
            }
        }
    }
});

// ---- 4. solvability and facts -------------------------------------------------------------------------------------

/**
 * Before the learner answers, nothing the panel shows may contain the expected number or shape: not the status line
 * (the first failing check or the friendly validation error), and not the trainer card's problem line, which a train
 * gate hides (TrainerCard.jsx).
 */
function assertNoSpoiler(step, result, where) {
    const { predict } = step;
    const values = predict.kind === 'number' ? [predict.expected]
        : predict.kind === 'shape' ? [predict.expectedShape.filter((axis) => axis !== null).join(', ')] : [];
    const shown = [result.issue ? t(result.issue.key, result.issue.params) : result.detail ? t(result.detail.key, result.detail.params) : ''];
    if (step.embeds?.includes('trainer') && step.locks?.gate !== 'train') shown.push(...useTrainerStore.getState().refreshPreflight().errors.map((issue) => issue.message));
    for (const value of values) {
        const number = new RegExp(`(^|[^\\d.])${String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?![\\d])`);
        for (const text of shown) assert.doesNotMatch(plain(text), number, `${where}: "${text}" gives away the answer ${value} before it is given`);
    }
}

const computedFacts = {};

async function computeFacts(module, at, record) {
    const helpers = createFactHelpers(buildCheckContext({ module }), { facts: computedFacts });
    for (const [name, fact] of Object.entries(module.facts)) {
        if (fact.compute && (fact.at ?? 'start') === at) {
            const value = await fact.compute(helpers);
            record[name] = value;
            assert.deepEqual(value, fact.value, `${module.id} fact ${name} (at ${at}) is ${JSON.stringify(value)}, the ledger says ${JSON.stringify(fact.value)}`);
        }
    }
}

test('every step fails on entry and passes after its Show me, module after module (headless, real trainer)', async () => {
    const started = Date.now();
    memory.clear();
    useLessonSession.getState().hydrate();
    session().setHeadless(true);
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false });
    for (const module of MODULES) {
        const record = {};
        computedFacts[module.id] = record;
        assert.equal(await session().startModule(module.id), true, `${module.id} starts`);
        assert.equal(session().mode, 'lesson');
        await computeFacts(module, 'start', record);
        assert.equal(session().phase, 'warmup', `${module.id} opens with its warm-up`);
        for (const item of module.warmup) {
            assert.equal(session().answerQuiz(item.id, item.correct)?.correct, true, item.id);
            await session().nextWarmup();
        }
        for (const [index, step] of module.steps.entries()) {
            const where = `${module.id}.${step.id}`;
            assert.equal(session().phase, 'steps', where);
            assert.equal(session().stepIndex, index, where);
            assert.equal(session().runtime.error, null, `${where}: setup ran without error`);
            if (step.tab) assert.equal(useEduUiStore.getState().activeTab, step.tab, `${where}: opens tab ${step.tab}`);
            const atEntry = session().evaluate();
            assert.equal(atEntry.pass, false, `${where}: the check must fail on entry`);
            if (step.predict) assertNoSpoiler(step, atEntry, where);
            if (where === 'm1.s4') assert.deepEqual(readSource('lesson').datasetUse, { setOutput: false }, 'the Data tab leaves the output size to s5');
            if (where === 'm1.s5') assert.equal(readSource('lesson').datasetUse, null);
            // Each diagnose rule fires on its hand-made wrong state, built from the entry state.
            if (step.diagnose?.length) {
                const entry = captureWorkspace();
                for (const rule of step.diagnose) {
                    assert.ok(rule.example?.length, `${where}: diagnose ${rule.misconception} has an example`);
                    const made = await runActions(rule.example, { module, headless: true, label: 'test' });
                    assert.equal(made.ok, true, `${where}: example for ${rule.misconception}: ${made.error}`);
                    const result = session().evaluate();
                    assert.equal(result.pass, false, `${where}: the example state for ${rule.misconception} must not pass`);
                    assert.equal(result.misconception, rule.misconception, `${where}: diagnose ${rule.misconception}`);
                    restoreWorkspace(entry);
                }
                assert.equal(session().evaluate().pass, false, `${where}: back at the entry state`);
            }
            const shown = await session().showMe();
            assert.equal(shown.ok, true, `${where}: Show me failed: ${shown.error}`);
            const result = session().evaluate();
            assert.equal(result.pass, true, `${where}: the check must pass after Show me (${JSON.stringify(result.detail)})`);
            if (step.reveal) {
                const live = revealValues(step.reveal, {
                    trainer: useTrainerStore.getState(), enteredAtSeq: session().runtime.enteredAtSeq,
                    runTags: Object.fromEntries(Object.entries(session().progress.modules[module.id]?.runTags ?? {})),
                });
                for (const name of LIVE_PARAMS(step)) assert.ok(live.params[name] !== undefined, `${where}: live value {${name}} is available after Show me`);
                if (step.reveal.compare) assert.ok(live.variant, `${where}: a comparison variant is chosen`);
            }
            if (step.codeQuotes) {
                // Lessons that point at a line of the export quote it verbatim; the generated code must contain it.
                const code = liveCode()[step.codeQuotes.tab];
                for (const quote of step.codeQuotes.text) assert.ok(code.includes(quote), `${where}: ${step.codeQuotes.tab === 'train' ? 'train.py' : 'model.py'} contains ${quote}`);
            }
            await computeFacts(module, step.id, record);
            assert.equal(await session().next(), true, `${where}: Next`);
        }
        assert.equal(session().phase, 'quiz', `${module.id}: the quiz follows the last step`);
        for (const item of module.quiz) {
            assert.equal(session().answerQuiz(item.id, item.correct)?.correct, true, item.id);
            session().nextQuiz();
        }
        assert.equal(session().phase, 'recap', `${module.id}: recap`);
        assert.equal(session().progress.modules[module.id].status, 'completed', `${module.id} completed`);
    }
    for (const module of MODULES) {
        for (const [name, fact] of Object.entries(module.facts)) {
            if (fact.compute) assert.ok(Object.hasOwn(computedFacts[module.id], name), `${module.id} fact ${name} was recomputed (at ${fact.at ?? 'start'})`);
            else assert.ok(['simulation', 'literature'].includes(fact.source), `${module.id} fact ${name} has a compute or a source`);
        }
    }
    const seconds = (Date.now() - started) / 1000;
    assert.ok(seconds < 60, `the walk took ${seconds.toFixed(1)} s (budget 60 s)`);
});

test('the M6 residual check guides each wire of the skip connection', async () => {
    const module = MODULES.find((item) => item.id === 'm6');
    const { buildFixtureSnapshot } = await import('../src/edu/lessons/engine/workspace.js');
    const residual = () => module.checks.residualAdd(buildCheckContext({ module }));
    const act = async (actions) => assert.equal((await runActions(actions, { module, headless: true })).ok, true);
    const reset = () => restoreWorkspace(buildFixtureSnapshot(module, module.start));
    reset();
    assert.equal(residual().detail.key, 'm6.check.noAdd');
    await act([{ do: 'insertAfter', block: 'deepflow.op.Add', after: { id: 'r2' } }]);
    assert.equal(residual().detail.key, 'm6.check.addPlace', 'an Add after relu2 is in the wrong place');
    reset();
    await act([{ do: 'insertBetween', block: 'deepflow.op.Add', from: { id: 'b2' }, to: { id: 'r2' } }]);
    assert.equal(residual().detail.key, 'm6.check.addSkip', 'bn2 → Add → relu2 still needs the skip wire');
    await act([{ do: 'connect', from: { id: 'relu0' }, to: { type: 'Add' } }]);
    assert.equal(residual().pass, true);
    await act([{ do: 'connect', from: { id: 'bn0' }, to: { type: 'Add' } }]);
    assert.equal(residual().detail.key, 'm6.check.addTwo', 'a third input is one too many');
    reset();
    await act([{ do: 'insertBetween', block: 'deepflow.op.Add', from: { id: 'b2' }, to: { id: 'r2' } }, { do: 'connect', from: { id: 'relu0' }, to: { type: 'Add' } }, { do: 'disconnect', from: { id: 'c1' }, to: { id: 'b1' } }]);
    assert.equal(residual().detail.key, 'm6.check.blockChain', 'the block chain must stay wired');
    for (const key of ['noAdd', 'addPlace', 'addSkip', 'addTwo', 'blockChain']) assert.ok(EN[`m6.check.${key}`], key);
});

test('every step that reads the code quotes the export it points at', () => {
    for (const module of MODULES) {
        for (const step of module.steps.filter((item) => item.subgoal === 'export')) {
            assert.ok(step.codeQuotes?.text?.length, `${module.id}.${step.id}: codeQuotes`);
            let tab = null;
            walkChecks(step.check, (spec) => { if (spec.ui?.codeTab) tab = spec.ui.codeTab; });
            if (tab) assert.equal(step.codeQuotes.tab, tab, `${module.id}.${step.id}: quotes the tab the check opens`);
        }
    }
});

test('every fact is used in the module text', () => {
    for (const module of MODULES) {
        const text = Object.entries(EN).filter(([key]) => key.startsWith(`${module.id}.`)).map(([, value]) => value).join('\n');
        for (const name of Object.keys(module.facts)) assert.ok(text.includes(`{${name}}`), `${module.id} fact ${name} is used`);
    }
});
