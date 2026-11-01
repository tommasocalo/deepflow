import assert from 'node:assert/strict';
import { after, afterEach, beforeEach, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// The Learn home and the lesson panel (docs/edu/SPEC.md §3.2, §3.3, §8.1) with the demo module of
// scripts/fixtures/edu-test-module.js, real engine and stores, rendered in jsdom.
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const jsdomErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => jsdomErrors.push(error));
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/', pretendToBeVisual: true, virtualConsole });
dom.window.DOMMatrixReadOnly = class { constructor() { this.m22 = 1; } };
let narrowScreen = false;
const installedGlobals = new Map();
for (const [key, value] of Object.entries({
    window: dom.window,
    document: dom.window.document,
    navigator: dom.window.navigator,
    HTMLElement: dom.window.HTMLElement,
    Element: dom.window.Element,
    SVGElement: dom.window.SVGElement,
    Node: dom.window.Node,
    CustomEvent: dom.window.CustomEvent,
    localStorage: dom.window.localStorage,
    getComputedStyle: dom.window.getComputedStyle.bind(dom.window),
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} unobserve() {} disconnect() {} },
    // The lesson panel turns into a bottom drawer below 900 px.
    matchMedia: (query) => ({ matches: query === '(max-width: 899px)' && narrowScreen, addEventListener() {}, removeEventListener() {} }),
    // jsdom has no MessageChannel; Node's would let a store's open port keep this test process alive.
    MessageChannel: undefined,
    IS_REACT_ACT_ENVIRONMENT: true,
})) {
    installedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
}

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');

// Stage B's mount points (SPEC §6.2) are stubs with the contract's props: the gate test reads the Train button the
// stub renders from `locked`. Stage B's own tests cover the real TrainerCard.
const MOUNT_POINT_STUBS = {
    'DataWorkspace.jsx': 'export default function DataWorkspace() { return <div data-stub="DataWorkspace">Datasets</div>; }',
    'TrainerPanel.jsx': 'export default function TrainerPanel() { return <div data-stub="TrainerPanel">Trainer</div>; }',
    'TrainerCard.jsx': `export default function TrainerCard({ locked = false, lockedReason = '' }) {
        return <div data-stub="TrainerCard"><button type="button" data-edu-target="trainer-run" disabled={locked}>Train</button>{lockedReason ? <p className="stub-locked">{lockedReason}</p> : null}</div>;
    }`,
};
const stubMountPoints = {
    name: 'edu-mount-point-stubs',
    setup(builder) {
        builder.onResolve({ filter: /(DataWorkspace|TrainerPanel|TrainerCard)\.jsx$/ }, (args) => {
            const name = args.path.split('/').at(-1);
            return /[\\/]edu[\\/](data|trainer)/.test(join(args.resolveDir, args.path)) ? { path: name, namespace: 'edu-stub' } : undefined;
        });
        builder.onLoad({ filter: /.*/, namespace: 'edu-stub' }, (args) => ({ contents: MOUNT_POINT_STUBS[args.path], loader: 'jsx', resolveDir: projectRoot }));
    },
};
const temporaryDirectory = await mkdtemp(join(projectRoot, '.edu-ui-learn-'));
const bundlePath = join(temporaryDirectory, 'learn.mjs');
let bundle;
try {
    await build({
        stdin: {
            contents: `
                export { default as LearnHome } from './src/edu/components/LearnHome.jsx';
                export { default as LessonPanel } from './src/edu/components/LessonPanel.jsx';
                export { useArchitectureStore } from './src/stores/architectureStore.js';
                export { useArchitectureHistory } from './src/stores/architectureHistory.js';
                export { readSource, registerSource } from './src/edu/bridge.js';
                export { useEduUiStore } from './src/edu/shell/uiStore.js';
                export { applyDeepLink, parseLearnHash, syncHash } from './src/edu/shell/deepLink.js';
                export { useLessonSession } from './src/edu/lessons/engine/session.js';
                export { setModuleRegistry } from './src/edu/lessons/engine/registry.js';
                export { restoreWorkspace, buildFixtureSnapshot } from './src/edu/lessons/engine/workspace.js';
                export { readLog, resetLogForTests } from './src/edu/lessons/engine/log.js';
                export { runActions } from './src/edu/lessons/engine/actions.js';
                export { resetStorageForTests } from './src/edu/lessons/engine/storage.js';
                export * as P from './src/edu/lessons/engine/progress.js';
                export { TEST_MODULE, TEST_TEXT } from './scripts/fixtures/edu-test-module.js';
            `,
            resolveDir: projectRoot,
        },
        outfile: bundlePath,
        bundle: true,
        packages: 'external',
        format: 'esm',
        platform: 'node',
        jsx: 'automatic',
        loader: { '.css': 'empty' },
        plugins: [stubMountPoints],
        logLevel: 'silent',
    });
    bundle = await import(pathToFileURL(bundlePath).href);
} finally {
    // Remove the bundle at once (as ui-regressions.test.mjs does), so an interrupted run leaves nothing behind.
    await rm(temporaryDirectory, { recursive: true, force: true });
}
const {
    LearnHome, LessonPanel, useArchitectureStore, useArchitectureHistory, readSource, registerSource, useEduUiStore, applyDeepLink, parseLearnHash, syncHash,
    useLessonSession, setModuleRegistry, restoreWorkspace, buildFixtureSnapshot, readLog, resetLogForTests, resetStorageForTests, P,
    TEST_MODULE, TEST_TEXT, runActions,
} = bundle;

// A second module so Learn shows a not-started card that is recommended after the first.
const NEXT_MODULE = { ...TEST_MODULE, id: 'demo2', order: 2, minutes: 10, recommendedAfter: ['demo'], datasets: ['iris'] };
const TEXT = { en: { ...TEST_TEXT.en, 'demo2.title': 'Second tour', 'demo2.goal': 'Do it again.' } };
const session = () => useLessonSession.getState();
const arch = () => useArchitectureStore.getState();

let container;
let root;
beforeEach(() => {
    jsdomErrors.length = 0;
    narrowScreen = false;
    localStorage.clear();
    resetStorageForTests();
    resetLogForTests();
    setModuleRegistry({ modules: [TEST_MODULE, NEXT_MODULE], text: TEXT });
    useEduUiStore.setState({ activeTab: 'learn', codeOpen: false, codeTab: 'model', drawerOpen: true });
    useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0, collapsed: false });
    session().hydrate();
    // The sandbox the learner leaves when a lesson starts.
    restoreWorkspace(buildFixtureSnapshot(TEST_MODULE, 'relu'));
    container = document.createElement('div');
    document.body.append(container);
    root = createRoot(container);
});
afterEach(async () => {
    if (root) await act(() => root.unmount());
    container?.remove();
    localStorage.clear();
    window.history.replaceState(null, '', '/');
    assert.deepEqual(jsdomErrors, [], 'DOM event handlers must not throw uncaught errors');
});
after(async () => {
    dom.window.close();
    for (const [key, descriptor] of installedGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
    }
});

const render = (element) => act(() => root.render(element));
const settle = (ms = 30) => act(() => new Promise((resolve) => setTimeout(resolve, ms)));
const run = (callback) => act(async () => { await callback(); });
/** Click and keep the act scope open while async handlers (startModule, Show me) finish. */
const click = async (element, ms = 30) => {
    assert.ok(element, 'Expected an element to click');
    await act(async () => {
        element.click();
        await new Promise((resolve) => setTimeout(resolve, ms));
    });
};
const text = (element) => element?.textContent.replace(/\s+/g, ' ').trim() ?? '';
/** Text without decorative glyphs (aria-hidden), as a screen reader reads it. */
const spoken = (element) => {
    if (!element) return '';
    const clone = element.cloneNode(true);
    for (const hidden of clone.querySelectorAll('[aria-hidden="true"]')) hidden.remove();
    return text(clone);
};
const panel = () => container.querySelector('aside[aria-label="Lesson"]');
const button = (name, scope = container) => [...scope.querySelectorAll('button')].find((element) => text(element) === name
    || element.getAttribute('aria-label') === name) ?? null;
const status = () => spoken(panel().querySelector('.edu-status[role="status"]'));
const heading = () => panel().querySelector('h2');
const setInput = (input, value) => act(() => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLInputElement.prototype, 'value').set.call(input, value);
    input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
});
const choose = async (value) => {
    const radio = panel().querySelector(`input[type="radio"][value="${value}"]`);
    assert.ok(radio, `Expected the option ${value}`);
    await click(radio);
};
/** A quiz or warm-up answer: choosing only selects, Check commits it. */
const answer = async (value) => {
    await choose(value);
    await click(button('Check'));
};
const renderPanel = () => render(h(LessonPanel, { checkDelay: 0 }));
async function openStep(index) {
    await run(() => session().startModule('demo'));
    if (session().phase === 'warmup') await run(() => session().skipWarmup());
    if (session().stepIndex !== index) await run(() => session().goToStep(index));
    await renderPanel();
    await settle();
}

test('Learn home: statuses come from progress, the cards are never locked, and Continue resumes the current step', async () => {
    let progress = P.startModule(P.emptyProgress(), 'demo');
    for (const id of ['s1', 's2', 's3']) progress = P.enterStep(progress, 'demo', id);
    progress = P.passStep(P.passStep(progress, 'demo', 's1'), 'demo', 's2');
    useLessonSession.setState({ progress });
    await render(h(LearnHome));
    assert.equal(text(container.querySelector('h1')), 'Learn to build neural networks');
    const [first, second] = container.querySelectorAll('section.edu-module-card');
    assert.equal(text(first.querySelector('.edu-module-number')), '1');
    assert.match(text(first.querySelector('h2')), /Tour of the lesson panel$/);
    assert.match(text(first), /In progress, 2 of 7 steps/, 'the status has a text label for screen readers');
    assert.ok(first.querySelector('svg.edu-status-glyph[aria-hidden="true"] path'), 'the half-filled circle marks in progress');
    assert.equal(text(button('Continue module 1: Tour of the lesson panel')), 'Continue');
    assert.match(text(second), /Not started/);
    assert.match(text(second), /Recommended after 1/);
    assert.ok(!button('Start module 2: Second tour').disabled, 'modules are never locked');
    assert.equal(text(container.querySelector('.edu-continue')), 'Continue · 1 Tour of the lesson panel · step 3 of 7');

    await click(container.querySelector('.edu-continue'));
    assert.equal(session().mode, 'lesson');
    assert.equal(session().moduleId, 'demo');
    assert.equal(session().currentStep().id, 's3', 'Continue resumes at the current step');
    assert.equal(useEduUiStore.getState().activeTab, 'architecture', 'the learner leaves Learn for the step’s tab');

    await act(() => useLessonSession.setState({ progress: P.completeModule(session().progress, 'demo') }));
    assert.match(text(container.querySelector('[data-module="demo"]')), /Done/);
    assert.equal(text(button('Review module 1: Tour of the lesson panel')), 'Review');
    assert.equal(text(container.querySelector('.edu-continue')), 'Start · 2 Second tour', 'Continue moves on to the next module');
});

test('Learn home: a browser that cannot save says so and keeps working', async () => {
    await render(h(LearnHome));
    assert.ok(!container.querySelector('.edu-callout-warning'));
    // Lesson slots are the large writes (§10): let them exceed the quota while DeepFlow's own small keys still fit.
    const setItem = dom.window.Storage.prototype.setItem;
    dom.window.Storage.prototype.setItem = function limited(key, value) {
        if (String(key).startsWith('deepflow-edu:')) throw new dom.window.DOMException('full', 'QuotaExceededError');
        return setItem.call(this, key, value);
    };
    try {
        await click(container.querySelector('.edu-continue'));
    } finally {
        dom.window.Storage.prototype.setItem = setItem;
    }
    assert.equal(session().mode, 'lesson', 'the lesson opened from memory');
    assert.equal(session().phase, 'warmup', 'the module started at its warm-up');
    assert.ok(Object.keys(arch().graph.nodes).includes('lin2'), 'the module workspace loaded');
    await run(() => session().exitLesson());
    await render(h(LearnHome));
    assert.match(text(container.querySelector('.edu-callout-warning')), /not saving progress right now/);
});

test('Lesson panel: the warm-up explains every option; the step heading takes focus; a check passes live with “Step complete”', async () => {
    await run(() => session().startModule('demo'));
    assert.equal(session().phase, 'warmup', 'a first start opens the warm-up');
    await renderPanel();
    await settle();
    assert.ok(document.activeElement === heading(), 'focus moves to the question');
    await choose('true');
    assert.equal(text(panel().querySelector('.edu-quiz-feedback')), '', 'choosing alone commits nothing');
    assert.equal(session().progress.modules.demo?.warmup?.['demo.w1'], undefined);
    await click(button('Check'));
    assert.match(text(panel().querySelector('.edu-quiz-feedback')), /Not quite\. Many people start with arithmetic/);
    await answer('false');
    assert.match(text(panel().querySelector('.edu-quiz-feedback')), /Correct\. Right: arithmetic/);
    await click(button('Next question'));
    await answer('small');
    await click(button('Start the steps'));
    assert.equal(session().phase, 'steps');
    assert.equal(text(heading()), 'Find the loss and the optimizer');
    assert.ok(document.activeElement === heading(), 'focus moves to the step heading');
    assert.equal(text(panel().querySelector('.edu-subgoal')), 'Choose loss & optimizer');
    assert.equal(status(), 'Not yet: Open the Training tab in the top bar.');
    assert.ok(button('Next').disabled, 'Next waits for the check');

    await run(() => useEduUiStore.getState().setTab('training'));
    await settle();
    assert.equal(status(), 'Done — Step complete', 'the live region announces the pass');
    assert.match(text(panel()), /This model trains with Adam and BCE\./, 'the done sentence follows');
    assert.ok(!button('Next').disabled);
    assert.equal(session().stepIndex, 0, 'there is no automatic advance');
    await click(button('Next'));
    assert.equal(text(heading()), 'Count the parameters');
    assert.ok(document.activeElement === heading());
    assert.equal(session().progress.modules.demo.steps.s1.status, 'passed');
});

test('a number prediction is committed, revealed with its misconception and never marked failed', async () => {
    await openStep(1);
    assert.match(text(panel()), /Answer the question to see the count\./, 'the parameter count waits for the prediction');
    const input = panel().querySelector('input.edu-predict-input');
    await setInput(input, 'sixteen');
    await click(button('Submit answer'));
    assert.match(text(panel().querySelector('[role="alert"]')), /Enter a number/);
    await setInput(input, '16');
    await click(button('Submit answer'));
    const reveal = panel().querySelector('.edu-reveal-heading');
    assert.equal(text(reveal), 'Not quite — here is why');
    assert.ok(document.activeElement === reveal, 'focus moves to the reveal');
    assert.match(text(panel()), /You said 16 · The tool shows 24/);
    assert.match(text(panel()), /A common mix-up: Each output unit also has one bias/);
    assert.ok(!panel().querySelector('input.edu-predict-input'), 'the answer is locked');
    assert.match(text(panel().querySelector('.edu-param-count')), /Total: 33/);
    assert.equal(status(), 'Done — Step complete');
    assert.deepEqual(session().progress.modules.demo.answers['demo.params'].value, 16);
    assert.equal(session().progress.modules.demo.answers['demo.params'].correct, false);
});

test('the hint ladder goes Hint → Another hint → Show me, one undoable edit with an Undo toast', async () => {
    await openStep(2);
    assert.equal(status(), 'Not yet: After hidden, connect an activation (such as ReLU).');
    assert.ok(panel().querySelector('[data-edu-target="lesson-palette"]'), 'the palette lists the blocks');
    await click(button('Hint'));
    assert.match(text(panel().querySelector('.edu-hints')), /Hint 1\s*Use the ReLU chip below/);
    await click(button('Another hint'));
    assert.match(text(panel().querySelector('.edu-hints')), /Hint 2\s*Delete the wire from hidden to score/);
    assert.equal(session().progress.modules.demo.steps.s3.hintLevel, 2);
    const undoBefore = useArchitectureHistory.getState().undoCount;
    await click(button('Show me'));
    await settle();
    assert.equal(status(), 'Done — Step complete');
    assert.equal(useArchitectureHistory.getState().undoCount, undoBefore + 1, 'Show me is a single history entry');
    assert.equal(useArchitectureHistory.getState().undoLabel, 'Lesson: show me');
    assert.equal(session().progress.modules.demo.steps.s3.hintLevel, 3);
    assert.match(text(panel().querySelector('.edu-toast')), /Show me made this change for you\./);
    await click(button('Undo', panel().querySelector('.edu-toast')));
    await settle();
    assert.ok(!Object.values(arch().graph.nodes).some((node) => node.libraryKey === 'torch.nn.ReLU'), 'Undo reverts the whole Show me');
    assert.match(status(), /Not yet/);
    assert.deepEqual(readLog().filter((event) => ['hint', 'show-me'].includes(event.type)).map((event) => event.data?.level ?? event.type), [1, 2, 'show-me']);
});

test('a palette chip adds its block even without a mounted canvas, and the insert is logged', async () => {
    await openStep(2);
    const before = Object.keys(arch().graph.nodes).length;
    await click(button('Insert ReLU'));
    await settle(1200);
    const added = Object.values(arch().graph.nodes).filter((node) => node.libraryKey === 'torch.nn.ReLU');
    assert.equal(Object.keys(arch().graph.nodes).length, before + 1);
    assert.equal(added.length, 1);
    assert.deepEqual(arch().selectedNodeIds, [added[0].id]);
    assert.deepEqual(readLog().filter((event) => event.type === 'palette-insert').map((event) => event.data.block), ['torch.nn.ReLU']);
});

test('a prediction gate locks Train until the answer is committed, then shows the reveal', async () => {
    await openStep(5);
    const train = () => panel().querySelector('[data-stub="TrainerCard"] [data-edu-target="trainer-run"]');
    assert.ok(train().disabled, 'Train stays disabled while the prediction is open');
    assert.equal(text(panel().querySelector('.stub-locked')), 'Answer the question first, then train.');
    assert.equal(readSource('lesson').trainLocked, true, 'other stages can read the gate through the bridge');
    await click(button('Submit answer'));
    assert.match(text(panel().querySelector('[role="alert"]')), /Choose an answer/);
    await choose('bce');
    await click(button('Submit answer'));
    assert.equal(text(panel().querySelector('.edu-reveal-heading')), 'Not quite — here is why');
    assert.match(text(panel()), /You said BCE · The tool shows Cross-entropy/);
    assert.match(text(panel()), /With K classes the model needs K outputs/);
    assert.equal(train().disabled, false, 'committing the answer unlocks Train');
    assert.equal(readSource('lesson').trainLocked, false);
    assert.match(status(), /Not yet: The loss is BCE; this step needs Cross-Entropy\./, 'the step still needs its task');
});

test('Reset step restores the entry snapshot and reopens the prediction; Back revisits without rerunning setup', async () => {
    await openStep(3);
    assert.equal(arch().showDimensions, false, 'the setup turned Dimensions off');
    await setInput(panel().querySelector('input.edu-predict-input'), 'B, 8');
    await click(button('Submit answer'));
    assert.equal(text(panel().querySelector('.edu-reveal-heading')), 'Exactly');
    await run(() => arch().setShowDimensions(true));
    await settle();
    assert.equal(status(), 'Done — Step complete');
    await click(button('Reset step'));
    assert.equal(arch().showDimensions, false, 'Reset step restored the entry state');
    assert.ok(panel().querySelector('input.edu-predict-input'), 'the prediction can be tried again');
    assert.match(status(), /Not yet/);

    await run(() => arch().setShowDimensions(true));
    await run(() => session().goToStep(4));
    await settle();
    assert.equal(text(heading()), 'One output per class');
    await click(button('Back'));
    assert.equal(text(heading()), 'Read a shape');
    assert.equal(arch().showDimensions, true, 'Back does not run the setup again');
});

test('the quiz explains each option at once, keeps the first answer and ends in the recap', async () => {
    await run(() => session().startModule('demo'));
    await run(() => session().skipWarmup());
    await act(() => useLessonSession.setState({ phase: 'quiz', quizIndex: 0 }));
    await renderPanel();
    await settle();
    assert.equal(text(heading()), 'How many parameters does a Linear block from 2 to 8 have?');
    // Arrowing past options selects them without logging an attempt; only Check counts.
    await choose('seven');
    await choose('sixteen');
    assert.equal(session().progress.modules.demo.quiz?.['demo.q1'], undefined, 'no attempt before Check');
    await click(button('Check'));
    assert.match(text(panel().querySelector('.edu-quiz-feedback')), /Not quite\. Right number of weights, but add the 8 biases\. Each output unit also has one bias/);
    await answer('seven');
    assert.match(text(panel().querySelector('.edu-quiz-feedback')), /That counts nothing per weight/);
    await answer('twenty-four');
    assert.match(text(panel().querySelector('.edu-quiz-feedback')), /Correct\. Right: 2 × 8 weights \+ 8 biases\./);
    const record = session().progress.modules.demo.quiz['demo.q1'];
    assert.deepEqual([record.first, record.correct, record.attempts, record.solved], ['sixteen', false, 3, true]);
    await click(button('Next question'));
    assert.ok(document.activeElement === heading(), 'each question takes focus');
    assert.equal(button('Next question').disabled, true, 'a quiz question needs an answer');
    await answer('bend');
    await click(button('Next question'));
    await answer('three');
    await click(button('See the recap'));
    assert.equal(session().phase, 'recap');
    assert.equal(text(heading()), 'You finished Tour of the lesson panel.');
    assert.equal(panel().querySelectorAll('.edu-takeaways li').length, 3);
    assert.match(text(panel()), /Some steps were skipped/, 'the steps were never passed here');
    await click(button('Next module: Second tour'));
    assert.equal(session().moduleId, 'demo2');
    assert.equal(session().phase, 'warmup');
});

test('deep links open a step (earlier steps visited, not passed) and the hash follows the lesson', async () => {
    assert.deepEqual(parseLearnHash('#/learn'), { moduleId: null, stepId: null });
    assert.deepEqual(parseLearnHash('#/learn/m3'), { moduleId: 'm3', stepId: null });
    assert.deepEqual(parseLearnHash('#/learn/m3/s4'), { moduleId: 'm3', stepId: 's4' });
    assert.equal(parseLearnHash('#/architecture'), null);
    await run(() => applyDeepLink('#/learn/demo/s3'));
    assert.equal(session().mode, 'lesson');
    assert.equal(session().phase, 'steps', 'a deep link skips the warm-up');
    assert.equal(session().currentStep().id, 's3');
    const steps = session().progress.modules.demo.steps;
    assert.deepEqual([steps.s1.status, steps.s2.status], ['visited', 'visited']);
    syncHash();
    assert.equal(window.location.hash, '#/learn/demo/s3');
    await run(() => session().goToStep(3));
    syncHash();
    assert.equal(window.location.hash, '#/learn/demo/s4');
    await run(() => session().exitLesson());
    syncHash();
    assert.equal(window.location.hash, '', 'outside a lesson the step hash is removed');
    assert.equal(await applyDeepLink('#/learn/nothing'), false);
    assert.equal(useEduUiStore.getState().activeTab, 'learn');
});

test('end to end: the demo module runs from warm-up to recap with Hint, Hint, Show me and Next', async () => {
    await run(() => session().startModule('demo'));
    await renderPanel();
    await settle();
    await answer('false');
    await click(button('Next question'));
    await answer('small');
    await click(button('Start the steps'));
    for (const step of TEST_MODULE.steps) {
        assert.equal(session().currentStep().id, step.id);
        await click(button('Hint'));
        await click(button('Another hint'));
        await click(button('Show me'));
        await settle(60);
        assert.equal(status(), 'Done — Step complete', `${step.id} passes after Show me`);
        await click(button(step === TEST_MODULE.steps.at(-1) ? 'Go to the quiz' : 'Next'));
    }
    for (const item of TEST_MODULE.quiz) {
        await answer(item.correct);
        await click(button(item === TEST_MODULE.quiz.at(-1) ? 'See the recap' : 'Next question'));
    }
    assert.equal(session().phase, 'recap');
    assert.equal(session().progress.modules.demo.status, 'completed');
    assert.ok(!/Some steps were skipped/.test(text(panel())));
    const types = readLog().map((event) => event.type);
    for (const type of ['module-start', 'warmup', 'step-enter', 'hint', 'show-me', 'predict', 'step-pass', 'quiz', 'module-complete']) {
        assert.ok(types.includes(type), `the log records ${type}`);
    }
    await click(button('Back to Learn'));
    assert.equal(session().mode, 'sandbox');
    assert.equal(useEduUiStore.getState().activeTab, 'learn');
    assert.ok(Object.values(arch().graph.nodes).some((node) => node.id === 'relu'), 'the sandbox workspace is back');
});

test('a run tagged in one step feeds the before/after reveal of a later step', async () => {
    // A two-step module in the shape of M5 s3 → s5: the passing run of s1 is tagged, and s2 compares its own run with it.
    const TAGS = {
        id: 'tags', version: 1, order: 3, minutes: 5, datasets: ['blobs'], objectives: [], warmup: [], quiz: [],
        fixtures: TEST_MODULE.fixtures, start: 'relu', facts: {},
        steps: [
            { id: 's1', kind: 'task', subgoal: 'train', tagRun: 'base', check: { all: [{ run: { dataset: 'blobs', minEpochs: 1 } }, { valid: true }] }, solution: [] },
            { id: 's2', kind: 'task', subgoal: 'evaluate', reveal: { compare: { tag: 'base', metric: 'gap' } }, check: { run: { dataset: 'blobs', minEpochs: 1 } }, solution: [] },
        ],
    };
    const words = { 'tags.title': 'Tags', 'tags.goal': 'Compare runs.' };
    for (const id of ['s1', 's2']) Object.assign(words, { [`tags.${id}.title`]: `Step ${id}`, [`tags.${id}.body`]: 'Train.', [`tags.${id}.hint1`]: 'Train.', [`tags.${id}.hint2`]: 'Train.' });
    words['tags.s2.reveal.better'] = 'The gap went from {before} to {after}.';
    words['tags.s2.reveal.notBetter'] = 'The gap did not shrink: {before} to {after}.';
    setModuleRegistry({ modules: [TEST_MODULE, TAGS], text: { en: { ...TEXT.en, ...words } } });
    const runs = [];
    const listeners = new Set();
    const addRun = (valLoss, trainLoss) => {
        const seq = runs.length + 1;
        const history = [{ epoch: 0, trainLoss: 1.2, valLoss: 1.2, trainAccuracy: 0.5, valAccuracy: 0.5 }, { epoch: 1, trainLoss, valLoss, trainAccuracy: 0.9, valAccuracy: 0.9 }];
        runs.push({ id: `run${seq}`, seq, status: 'completed', datasetId: 'blobs', settings: { epochs: 1, batchSize: 32, seed: 1 }, optimizer: { type: 'Adam', hyperparameters: { learning_rate: 0.03 } }, history });
        for (const listener of listeners) listener();
    };
    const unregister = registerSource('trainer', {
        read: () => ({ status: 'idle', seq: runs.length, settings: { epochs: 1, batchSize: 32, seed: 1 }, current: null, runs: [...runs], preflight: { ok: true, errors: [], warnings: [] } }),
        subscribe: (listener) => { listeners.add(listener); return () => listeners.delete(listener); },
        actions: { tagRun: () => {} },
    });
    try {
        await run(() => session().startModule('tags'));
        await renderPanel();
        await settle();
        assert.match(status(), /Press Train/);
        await run(() => addRun(0.9, 0.3));
        await settle();
        assert.equal(status(), 'Done — Step complete');
        await click(button('Next'));
        assert.deepEqual(session().progress.modules.tags.runTags.base.runId, 'run1', 'the run that passed s1 was tagged');
        assert.match(status(), /Press Train/, 'runs from before the step do not count');
        await run(() => addRun(0.5, 0.3));
        await settle();
        assert.equal(status(), 'Done — Step complete');
        assert.equal(text(panel().querySelector('.edu-callout-info')), 'The gap went from 0.600 to 0.200.');
    } finally {
        unregister();
    }
});

test('collapse keeps a rail on wide screens; below 900 px the drawer handle toggles and resizes the lesson', async () => {
    await openStep(0);
    await click(button('Collapse lesson panel'));
    assert.ok(panel().classList.contains('is-collapsed'));
    assert.match(text(panel()), /1\/7/);
    await click(button('Expand lesson panel'));
    assert.equal(text(heading()), 'Find the loss and the optimizer');

    narrowScreen = true;
    await act(() => root.unmount());
    root = createRoot(container);
    await renderPanel();
    await settle();
    const handle = panel().querySelector('.edu-drawer-handle');
    assert.ok(panel().classList.contains('is-drawer'));
    assert.equal(handle.getAttribute('aria-expanded'), 'true');
    assert.ok(button('Exit lesson', panel()), 'exit stays beside the handle');
    await click(handle);
    assert.equal(panel().querySelector('.edu-drawer-handle').getAttribute('aria-expanded'), 'false');
    assert.ok(!heading(), 'the closed drawer shows only its bar');
    assert.match(text(panel().querySelector('.edu-drawer-title')), /Lesson 1\/7 · Tour of the lesson panel/);
    await click(panel().querySelector('.edu-drawer-handle'));
    assert.ok(heading());
    await act(() => panel().querySelector('.edu-drawer-handle').dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true })));
    assert.match(panel().getAttribute('style') ?? '', /height: \d+px/, 'the arrow keys resize the drawer');
});

test('the progress segments are one tab stop, the arrows move between them, and Undo and Redo sit in the panel header', async () => {
    await openStep(2);
    const segments = [...panel().querySelectorAll('.edu-segment')];
    assert.deepEqual(segments.map((segment) => segment.tabIndex), segments.map((_, index) => (index === 2 ? 0 : -1)), 'only the current step is a tab stop');
    segments[2].focus();
    await act(() => segments[2].dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
    assert.ok(document.activeElement === segments[3], 'ArrowRight moves to the next segment');
    await act(() => segments[3].dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Home', bubbles: true })));
    assert.ok(document.activeElement === segments[0], 'Home moves to the first');
    const undo = button('Undo the last edit', panel());
    assert.ok(undo && button('Redo the edit', panel()), 'Undo and Redo are in the lesson panel');
    assert.equal(undo.disabled, true, 'nothing to undo yet');
    await run(() => runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }], { label: 'test' }));
    await settle();
    assert.equal(button('Undo the last edit', panel()).disabled, false);
    await click(button('Undo the last edit', panel()));
    assert.ok(!Object.values(arch().graph.nodes).some((node) => node.libraryKey === 'torch.nn.ReLU'), 'the panel Undo reverts the edit');
    assert.ok(panel().querySelector('.edu-step-dock .edu-status[role="status"]') && panel().querySelector('.edu-step-dock footer.edu-nav'), 'the status and the navigation share the dock');
});

test('a misconception callout waits until the learner signals they are done, and the log records it', async () => {
    await openStep(2);
    // The Sigmoid distractor after the last Linear, the ReLU not placed yet: diagnosed (activation at the output), but
    // the model is not finished, so the callout waits.
    await run(() => runActions([{ do: 'insertBetween', block: 'torch.nn.Sigmoid', from: { id: 'lin2' }, to: { id: 'out' } }], { label: 'test' }));
    await settle(60);
    assert.equal(panel().querySelector('.edu-callout-misconception'), null, 'not while the learner may still be editing');
    assert.doesNotMatch(text(panel().querySelector('.edu-status')), /A common mix-up/);
    await click(button('Hint'));
    await settle(60);
    assert.match(text(panel().querySelector('.edu-callout-misconception')), /A common mix-up:/, 'a hint after the edit shows it at once');
    const events = readLog();
    assert.ok(events.some((event) => event.type === 'misconception-shown' && event.data.id === 'activation-at-output'), 'the shown misconception is logged');
    assert.ok(events.some((event) => event.type === 'status-shown' && event.step === 's3'), 'status messages are logged');
});

test('a finished wrong model is named in the status line at once (review r2: the Sigmoid trap)', async () => {
    await openStep(2);
    // The palette's ReLU placed after the last Linear: valid, nothing left to place, and the rule names it.
    await run(() => runActions([{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin2' }, to: { id: 'out' } }], { label: 'test' }));
    await settle(60);
    const status = panel().querySelector('.edu-status');
    assert.match(text(status), /A common mix-up:/, 'named at once, in the status line');
    assert.ok(status.classList.contains('is-misconception'));
    assert.equal(panel().querySelector('.edu-callout-misconception'), null, 'not twice');
    assert.ok(readLog().some((event) => event.type === 'misconception-shown' && event.data.id === 'activation-at-output'), 'logged');
});
