import assert from 'node:assert/strict';
import { afterEach, test } from 'node:test';

const memory = new Map();
const goodStorage = { getItem: (key) => memory.get(key) ?? null, setItem: (key, value) => memory.set(key, String(value)), removeItem: (key) => memory.delete(key), key: (index) => [...memory.keys()][index] ?? null, get length() { return memory.size; } };
globalThis.window = { localStorage: goodStorage };
globalThis.localStorage = goodStorage;

const P = await import('../src/edu/lessons/engine/progress.js');
const storageApi = await import('../src/edu/lessons/engine/storage.js');
const logApi = await import('../src/edu/lessons/engine/log.js');
const { PRODUCT_NAME, CURRICULUM_VERSION } = await import('../src/edu/config.js');
const { TEST_MODULE, TEST_TEXT } = await import('./fixtures/edu-test-module.js');

afterEach(() => {
    globalThis.localStorage = goodStorage;
    globalThis.window.localStorage = goodStorage;
    memory.clear();
    storageApi.resetStorageForTests();
    logApi.resetLogForTests();
});

test('step reducers: enter, pass, skip and monotone hint levels', () => {
    let progress = P.startModule(P.emptyProgress(), 'demo', 1);
    assert.equal(progress.lastModuleId, 'demo');
    assert.equal(progress.modules.demo.status, 'in-progress');
    progress = P.enterStep(progress, 'demo', 's1', 100);
    assert.deepEqual(progress.modules.demo.steps.s1, { status: 'visited', hintLevel: 0, enteredAt: 100 });
    assert.equal(progress.modules.demo.currentStepId, 's1');
    const frozen = structuredClone(progress);
    const passed = P.passStep(progress, 'demo', 's1', 200);
    assert.deepEqual(progress, frozen, 'reducers never mutate their input');
    assert.equal(passed.modules.demo.steps.s1.status, 'passed');
    assert.equal(P.skipStep(passed, 'demo', 's1').modules.demo.steps.s1.status, 'passed', 'skip never downgrades a pass');
    assert.equal(P.skipStep(progress, 'demo', 's2').modules.demo.steps.s2.status, 'skipped');
    let hints = P.raiseHint(progress, 'demo', 's1', 2);
    hints = P.raiseHint(hints, 'demo', 's1', 1);
    assert.equal(hints.modules.demo.steps.s1.hintLevel, 2);
    assert.equal(P.raiseHint(hints, 'demo', 's1', 9).modules.demo.steps.s1.hintLevel, 3);
    assert.equal(P.enterStep(passed, 'demo', 's1', 300).modules.demo.steps.s1.status, 'passed', 'entering again keeps the pass');
});

test('answers, quiz and warm-up records keep the first answer', () => {
    let progress = P.recordAnswer(P.emptyProgress(), 'demo', 'demo.params', 16, false, { now: 1 });
    assert.deepEqual(progress.modules.demo.answers['demo.params'], { value: 16, correct: false, at: 1 });
    assert.equal(P.recordAnswer(progress, 'demo', 'demo.shape', [null, 8], true, { source: 'show-me' }).modules.demo.answers['demo.shape'].source, 'show-me');
    progress = P.clearAnswer(progress, 'demo', 'demo.params');
    assert.equal(progress.modules.demo.answers['demo.params'], undefined);
    progress = P.recordQuiz(progress, 'demo', 'demo.q1', 'sixteen', false);
    progress = P.recordQuiz(progress, 'demo', 'demo.q1', 'twenty-four', true);
    assert.deepEqual(progress.modules.demo.quiz['demo.q1'], { first: 'sixteen', correct: false, attempts: 2, last: 'twenty-four', solved: true });
    progress = P.recordWarmup(progress, 'demo', 'demo.w1', 'false', true);
    assert.equal(progress.modules.demo.warmup['demo.w1'].correct, true);
});

test('module status and completion', () => {
    let progress = P.startModule(P.emptyProgress(), 'demo');
    assert.equal(P.moduleStatus(P.emptyProgress(), TEST_MODULE).status, 'not-started');
    for (const step of TEST_MODULE.steps.slice(0, 3)) progress = P.passStep(P.enterStep(progress, 'demo', step.id), 'demo', step.id);
    progress = P.enterStep(progress, 'demo', 's4');
    const summary = P.moduleStatus(progress, TEST_MODULE);
    assert.deepEqual([summary.status, summary.done, summary.total, summary.currentStepId, summary.firstOpenIndex], ['in-progress', 3, 7, 's4', 3]);
    assert.equal(P.isModuleComplete(progress, TEST_MODULE), false);
    for (const step of TEST_MODULE.steps.slice(3)) progress = P.skipStep(progress, 'demo', step.id);
    assert.equal(P.isModuleComplete(progress, TEST_MODULE), false, 'quiz items still unanswered');
    for (const item of TEST_MODULE.quiz) progress = P.recordQuiz(progress, 'demo', item.id, item.correct, true);
    assert.equal(P.isModuleComplete(progress, TEST_MODULE), true);
    progress = P.completeModule(progress, 'demo', 5);
    assert.equal(P.moduleStatus(progress, TEST_MODULE).status, 'completed');
    assert.equal(progress.modules.demo.completedAt, 5);
});

test('migrate starts an updated module\'s steps again and keeps the answers, quiz and warm-up results whose ids remain', () => {
    let progress = P.startModule(P.emptyProgress(), 'demo', 1);
    for (const id of ['s1', 's2', 's3']) progress = P.passStep(P.enterStep(progress, 'demo', id), 'demo', id);
    progress = P.enterStep(progress, 'demo', 's3');
    progress = P.recordAnswer(progress, 'demo', 'demo.params', 24, true);
    progress = P.recordAnswer(progress, 'demo', 'demo.gone', 1, true);
    progress = P.recordQuiz(progress, 'demo', 'demo.q1', 'twenty-four', true);
    progress = P.recordQuiz(progress, 'demo', 'demo.q2', 'bend', true);
    progress = P.recordWarmup(progress, 'demo', 'demo.w1', 'false', true);
    progress = P.tagRun(progress, 'demo', 'demo.old', { runId: 'r1' });
    progress = P.completeModule(progress, 'demo', 5);
    // Version 2 puts a new step at s2 (review r2: M5's "The final score" at s5) and drops quiz item q2.
    const steps = [...TEST_MODULE.steps];
    steps.splice(1, 0, { ...TEST_MODULE.steps[0], id: 'new' });
    const bumped = {
        ...TEST_MODULE, version: 2,
        steps: steps.map((step, index) => ({ ...step, id: `s${index + 1}` })),
        quiz: TEST_MODULE.quiz.filter((item) => item.id !== 'demo.q2'),
    };
    assert.deepEqual(P.updatedModules(progress, [bumped]).map((module) => module.id), ['demo']);
    const migrated = P.migrate(progress, [bumped]).modules.demo;
    assert.deepEqual([migrated.moduleVersion, migrated.migratedFrom], [2, 1]);
    assert.deepEqual(migrated.steps, {}, 'a step id may now name another step');
    assert.equal(migrated.currentStepId, null);
    assert.equal(migrated.status, 'in-progress', 'a completed module is completed no more');
    assert.equal(migrated.completedAt, undefined);
    assert.deepEqual(Object.keys(migrated.answers), ['demo.params'], 'answers to predictions that still exist');
    assert.deepEqual(Object.keys(migrated.quiz), ['demo.q1']);
    assert.deepEqual(Object.keys(migrated.warmup), ['demo.w1']);
    assert.deepEqual(migrated.runTags, {}, 'no step tags demo.old any more');
    assert.equal(P.moduleStatus({ modules: { demo: migrated } }, bumped).firstOpenIndex, 0);

    assert.equal(P.migrate(progress, [TEST_MODULE]).modules.demo, progress.modules.demo, 'same version: untouched');
    assert.deepEqual(P.updatedModules(progress, [TEST_MODULE]), []);
    const untouched = P.migrate(P.startModule(P.emptyProgress(), 'demo', 1), [bumped]).modules.demo;
    assert.equal(untouched.status, 'in-progress', 'a started module stays started');
    assert.deepEqual(P.migrate({ version: 99 }, [TEST_MODULE]), P.emptyProgress(), 'unknown formats start over');
    assert.deepEqual(P.migrate(null, []), P.emptyProgress());
});

test('storage falls back to memory when setItem throws and sets storageDegraded', () => {
    assert.equal(storageApi.isStorageDegraded(), false);
    assert.equal(storageApi.writeJSON('deepflow-edu:probe', { a: 1 }), true);
    assert.deepEqual(JSON.parse(memory.get('deepflow-edu:probe')), { a: 1 });
    const full = { ...goodStorage, setItem: () => { const error = new Error('quota'); error.name = 'QuotaExceededError'; throw error; } };
    globalThis.localStorage = full;
    const notified = [];
    const stop = storageApi.subscribeStorageDegraded((value) => notified.push(value));
    assert.equal(storageApi.writeJSON('deepflow-edu:progress:v1', { x: 2 }), false);
    stop();
    assert.equal(storageApi.isStorageDegraded(), true);
    assert.deepEqual(notified, [true]);
    assert.deepEqual(storageApi.readJSON('deepflow-edu:progress:v1'), { x: 2 }, 'the value is kept in memory for this tab');
});

test('storage that throws on access, or is missing, never throws to callers', () => {
    globalThis.localStorage = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); }, removeItem: () => { throw new Error('denied'); } };
    assert.equal(storageApi.readJSON('deepflow-edu:session:v1', 'fallback'), 'fallback');
    assert.doesNotThrow(() => storageApi.writeJSON('deepflow-edu:session:v1', { mode: 'lesson' }));
    assert.deepEqual(storageApi.readJSON('deepflow-edu:session:v1'), { mode: 'lesson' });
    assert.doesNotThrow(() => storageApi.removeKey('deepflow-edu:session:v1'));
    storageApi.resetStorageForTests();
    globalThis.localStorage = undefined;
    globalThis.window.localStorage = undefined;
    assert.equal(storageApi.readJSON('deepflow-edu:x', 7), 7);
    assert.equal(storageApi.writeJSON('deepflow-edu:x', 8), false);
    assert.equal(storageApi.readJSON('deepflow-edu:x', 7), 8);
    assert.equal(storageApi.isStorageDegraded(), true);
    memory.set('garbage', '{not json');
    globalThis.localStorage = goodStorage;
    assert.equal(storageApi.readJSON('garbage', 'safe'), 'safe', 'malformed JSON reads as the fallback');
});

test('the event log is a 2,000-event ring buffer and exports with the product name', () => {
    for (let index = 0; index < logApi.LOG_LIMIT + 25; index++) logApi.logEvent('hint', { module: 'demo', step: 's1', data: { level: index } }, index);
    const events = logApi.readLog();
    assert.equal(events.length, logApi.LOG_LIMIT);
    assert.equal(events[0].data.level, 25, 'the oldest events are dropped first');
    assert.deepEqual(events.at(-1), { t: logApi.LOG_LIMIT + 24, type: 'hint', module: 'demo', step: 's1', data: { level: logApi.LOG_LIMIT + 24 } });
    assert.equal(JSON.parse(memory.get(logApi.LOG_KEY)).length, logApi.LOG_LIMIT);
    logApi.resetLogForTests();
    assert.equal(logApi.readLog().length, logApi.LOG_LIMIT, 'the log is read back from storage');
    const payload = logApi.learningLogExport({ version: 1 }, Date.UTC(2026, 9, 29));
    assert.equal(payload.product, PRODUCT_NAME);
    assert.equal(payload.curriculumVersion, CURRICULUM_VERSION);
    assert.equal(payload.exportedAt, '2026-10-29T00:00:00.000Z');
    assert.equal(payload.events.length, logApi.LOG_LIMIT);
    logApi.clearLog();
    assert.deepEqual(logApi.readLog(), []);
});

test('session and progress persist and come back after a reload', async () => {
    const { setModuleRegistry } = await import('../src/edu/lessons/engine/registry.js');
    const { useLessonSession, SESSION_KEY, PROGRESS_KEY } = await import('../src/edu/lessons/engine/session.js');
    setModuleRegistry({ modules: [TEST_MODULE], text: TEST_TEXT });
    const session = useLessonSession;
    session.getState().setHeadless(true);
    session.getState().hydrate();
    await session.getState().startModule('demo');
    await session.getState().skipWarmup();
    await session.getState().showMe();
    await session.getState().next();
    session.getState().submitPrediction('demo.params', 16);
    const saved = JSON.parse(memory.get(SESSION_KEY));
    assert.deepEqual([saved.mode, saved.moduleId, saved.phase, saved.stepIndex], ['lesson', 'demo', 'steps', 1]);
    const progress = JSON.parse(memory.get(PROGRESS_KEY));
    assert.equal(progress.modules.demo.steps.s1.status, 'passed');
    assert.deepEqual(progress.modules.demo.answers['demo.params'].value, 16);
    assert.equal(progress.modules.demo.answers['demo.params'].correct, false);

    // A reload reads both keys back.
    session.setState({ mode: 'sandbox', moduleId: null, stepIndex: 0, progress: P.emptyProgress() });
    memory.set(SESSION_KEY, JSON.stringify(saved));
    memory.set(PROGRESS_KEY, JSON.stringify(progress));
    session.getState().hydrate();
    assert.deepEqual([session.getState().mode, session.getState().moduleId, session.getState().stepIndex], ['lesson', 'demo', 1]);
    assert.equal(session.getState().progress.modules.demo.steps.s1.status, 'passed');
    const logTypes = logApi.readLog().map((event) => event.type);
    for (const type of ['module-start', 'step-enter', 'show-me', 'step-pass', 'predict']) assert.ok(logTypes.includes(type), `${type} is logged`);
});
