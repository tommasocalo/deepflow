import assert from 'node:assert/strict';
import { after, test } from 'node:test';
import { mkdtemp, rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { build } from 'esbuild';
import { JSDOM, VirtualConsole } from 'jsdom';

// Predict, observe, explain (review r1): outcome predictions (M1 s3/s7, M2 s1/s2/s5, M3 s6, M5 s1–s3) only lock on
// Submit and reveal "You said X · Your run shows Y" once the step's run has passed, with Y read from that run.
const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const jsdomErrors = [];
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', (error) => jsdomErrors.push(error));
const dom = new JSDOM('<!doctype html><html><body></body></html>', { url: 'http://localhost/', pretendToBeVisual: true, virtualConsole });
const installedGlobals = new Map();
for (const [key, value] of Object.entries({
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement,
    Element: dom.window.Element, Node: dom.window.Node, CustomEvent: dom.window.CustomEvent, localStorage: dom.window.localStorage,
    getComputedStyle: dom.window.getComputedStyle.bind(dom.window), IS_REACT_ACT_ENVIRONMENT: true, MessageChannel: undefined,
})) {
    installedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
}
after(() => {
    dom.window.close();
    for (const [key, descriptor] of installedGlobals) {
        if (descriptor) Object.defineProperty(globalThis, key, descriptor);
        else delete globalThis[key];
    }
});

const { act, createElement: h } = await import('react');
const { createRoot } = await import('react-dom/client');
const temporaryDirectory = await mkdtemp(join(projectRoot, '.edu-observe-'));
let app;
try {
    const bundlePath = join(temporaryDirectory, 'observe.mjs');
    await build({
        stdin: {
            contents: `
                export { default as PredictGate } from './src/edu/components/PredictGate.jsx';
                export { MODULES } from './src/edu/lessons/modules/index.js';
                export { isOutcomePrediction, observerOf, observePrediction, boundaryShape } from './src/edu/lessons/engine/observe.js';
                export { useLessonSession } from './src/edu/lessons/engine/session.js';
                export { useEduUiStore } from './src/edu/shell/uiStore.js';
                export { readSource } from './src/edu/bridge.js';
                export { t } from './src/edu/lessons/engine/i18n.js';
                import './src/edu/data/store.js';
                import './src/edu/trainer/store.js';
            `,
            resolveDir: projectRoot,
        },
        outfile: bundlePath, bundle: true, packages: 'external', format: 'esm', platform: 'node', jsx: 'automatic',
        loader: { '.css': 'empty' }, logLevel: 'silent',
    });
    app = await import(pathToFileURL(bundlePath).href);
} finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
}
const { PredictGate, MODULES, isOutcomePrediction, observerOf, observePrediction, boundaryShape, useLessonSession, useEduUiStore, readSource, t } = app;
const session = () => useLessonSession.getState();
const stepOf = (moduleId, stepId) => MODULES.find((module) => module.id === moduleId).steps.find((step) => step.id === stepId);

test('outcome predictions are the ones whose step needs a training run; each has an observer', () => {
    const outcome = MODULES.flatMap((module) => module.steps.filter(isOutcomePrediction).map((step) => `${module.id}.${step.id}:${observerOf(step)}`));
    assert.deepEqual(outcome, [
        'm1.s3:boundaryShape', 'm1.s7:bestClass', 'm2.s1:bends', 'm2.s2:bends', 'm2.s5:hiddenWidth', 'm3.s6:tool', 'm5.s1:initLoss',
        'm5.s2:bestLr', 'm5.s3:risingCurve',
    ]);
    for (const [moduleId, stepId] of [['m1', 's2'], ['m3', 's1'], ['m4', 's2'], ['m6', 's4'], ['m7', 's6']]) {
        assert.equal(isOutcomePrediction(stepOf(moduleId, stepId)), false, `${moduleId}.${stepId} reveals at once (a value the tool computes)`);
    }
});

test('boundaryShape reads a straight line, an island and a curve on the grid', () => {
    const grid = (size, classOf) => ({ size, classes: Uint8Array.from({ length: size * size }, (_, i) => classOf(i % size, Math.floor(i / size))) });
    assert.equal(boundaryShape(grid(64, (x, y) => (x + 0.5 * y > 40 ? 1 : 0))), 'line');
    assert.equal(boundaryShape(grid(64, (x, y) => ((x - 32) ** 2 + (y - 32) ** 2 < 200 ? 1 : 0))), 'closed');
    assert.equal(boundaryShape(grid(64, (x, y) => (y > 20 + ((x - 32) ** 2) / 40 ? 1 : 0))), 'curve');
    assert.equal(boundaryShape(grid(64, () => 0)), 'none');
});

test('each outcome step, solved by Show me, shows what its own run shows', async () => {
    const expected = {
        'm1.s3': { option: 'line', text: /^a straight line \(validation accuracy \d/ },
        'm1.s7': { option: 'setosa', text: /^Setosa recognised best, \d+ of \d+ right$/ },
        'm2.s1': { option: 'no', text: /^a straight boundary \(validation accuracy \d/ },
        'm2.s2': { option: 'no', text: /^a straight boundary/ },
        'm2.s5': { option: 'three', text: /^3 hidden units reaching \d/ },
        'm3.s6': { option: undefined, text: /^64$/ },
        'm5.s1': { option: undefined, text: /^a first loss of 1\.\d\d$/ },
        'm5.s2': { option: 'medium', text: /^the lowest validation loss with lr 0\.01 \(/ },
        'm5.s3': { option: 'val', text: /^validation loss rising to \d+\.\d× its lowest value while training loss falls$/ },
    };
    for (const [where, want] of Object.entries(expected)) {
        const [moduleId, stepId] = where.split('.');
        localStorage.clear();
        useEduUiStore.setState({ activeTab: 'learn', codeOpen: false });
        useLessonSession.setState({ mode: 'sandbox', moduleId: null, phase: 'steps', stepIndex: 0 });
        session().hydrate();
        session().setHeadless(true);
        await session().startModule(moduleId, { stepId });
        const step = stepOf(moduleId, stepId);
        const context = () => ({ trainer: readSource('trainer'), enteredAtSeq: session().runtime.enteredAtSeq, runId: session().evaluate().runId });
        if (observerOf(step) !== 'tool') assert.equal(observePrediction(step, context(), where), null, `${where}: nothing to observe before a run`);
        assert.equal((await session().showMe()).ok, true, where);
        const seen = observePrediction(step, context(), where);
        assert.ok(seen, `${where}: an observation`);
        if (want.option !== undefined) assert.equal(seen.option, want.option, `${where}: the run shows ${want.option}`);
        assert.match(t('predict.yourRun', { value: seen.text }).replace('Your run shows ', ''), want.text, `${where}: ${seen.text}`);
    }
});

test('PredictGate: an outcome answer is locked first and revealed with the observation', async () => {
    const container = document.createElement('div');
    document.body.append(container);
    const root = createRoot(container);
    const step = stepOf('m2', 's1');
    const answer = { value: 'yes', correct: false, at: 1 };
    const render = (props) => act(() => root.render(h(PredictGate, { predict: step.predict, textPrefix: 'm2.s1', answer, params: {}, outcome: true, ...props })));
    const text = () => container.textContent.replace(/\s+/g, ' ').trim();
    await render({ observed: false });
    assert.match(text(), /Locked: you said Yes/);
    assert.doesNotMatch(text(), /Not quite|The tool shows|One neuron draws one straight line/, 'no verdict, answer or explanation before the run');
    await render({ observed: true, observation: { option: 'no', text: 'a straight boundary (validation accuracy 58.3%)' } });
    assert.match(text(), /Not quite — here is why/);
    assert.match(text(), /You said Yes · Your run shows a straight boundary \(validation accuracy 58\.3%\)/);
    assert.match(text(), /No\. One neuron draws one straight line/);
    assert.ok(container.querySelector('[aria-live="polite"] .edu-reveal-heading'), 'the reveal is announced when it arrives');

    // Static predictions keep the immediate reveal; option text is rendered as markup, not raw backticks.
    const forward = stepOf('m2', 's8');
    await act(() => root.render(h(PredictGate, { predict: forward.predict, textPrefix: 'm2.s8', answer: { value: 'relu', correct: true, at: 1 }, params: {} })));
    assert.match(text(), /Exactly/);
    assert.doesNotMatch(text(), /`/, 'no raw backticks');
    assert.equal(container.querySelector('.edu-reveal-compare code')?.textContent, 'self.ReLU(…)');
    await act(() => root.unmount());
    container.remove();
    assert.deepEqual(jsdomErrors, []);
});
