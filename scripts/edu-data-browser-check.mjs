// Stage B browser check (docs/edu/SPEC.md §7.3 B, §8.2): the Data tab, the TrainerPanel and the lesson TrainerCard in
// Chromium against the dev server. Run by hand (not part of npm test):
//
//   npx vite --host 127.0.0.1 --port 5174 --strictPort &
//   node scripts/edu-data-browser-check.mjs [--url http://127.0.0.1:5174/] [--shots <dir>]
//
// It checks every Stage B spotlight target of §6.6, no console errors and no horizontal page scroll at 1440 × 900
// and 390 × 844, and measures the trainer budget of §6.10.7 (circle 2-8-1 < 0.5 s, spiral3 overfit < 4 s, digits
// 64-128-10 < 4 s, on a 2020 laptop): it reports each run against the budget and fails beyond 1.5 × the budget.
// Screenshots go to --shots (default: the scratch folder named in docs/edu/SPEC.md §7.1).
import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';

const argument = (name, fallback) => {
    const index = process.argv.indexOf(name);
    return index >= 0 ? process.argv[index + 1] : fallback;
};
const URL = argument('--url', 'http://127.0.0.1:5174/');
const SHOTS = argument('--shots', '/tmp/claude-0/-home-user-deepflow/7a8350aa-ff30-52a8-8452-b7a05647a64e/scratchpad/edu/shots');
const EXECUTABLE = argument('--chromium', '/opt/pw-browsers/chromium');
mkdirSync(SHOTS, { recursive: true });

const STAGE_B_TARGETS = ['use-dataset', 'dataset-preview', 'dataset-options', 'trainer-run', 'trainer-lr', 'trainer-curves', 'trainer-boundary', 'trainer-confusion', 'trainer-init-loss'];

/** In-page helpers, installed once per page. */
async function installHelpers(page) {
    await page.evaluate(() => {
        const edu = window.__deepflow.edu;
        window.__stageB = {
            /** Chain Input → blocks → Output with Stage A's fixture helpers; blocks: ['linear', in, out] | ['relu'] | ['flatten'] | ['dropout', p]. */
            load({ input, blocks, loss, lr = 0.01, dataset, options = {}, split = { val: 0.2, test: 0 }, trainer = {} }) {
                const h = edu.engine.createFixtureHelpers();
                const nodes = [h.input('in', input)];
                blocks.forEach(([kind, a, b], index) => {
                    if (kind === 'linear') nodes.push(h.linear(`lin${index}`, a, b));
                    else if (kind === 'relu') nodes.push(h.layer(`relu${index}`, 'torch.nn.ReLU'));
                    else if (kind === 'flatten') nodes.push(h.layer(`flat${index}`, 'torch.nn.Flatten'));
                    else if (kind === 'dropout') nodes.push(h.layer(`drop${index}`, 'torch.nn.Dropout', { p: a }));
                });
                const output = h.output('out');
                nodes.push(output);
                edu.engine.restoreWorkspace(h.snapshot({
                    nodes, edges: h.chain(...nodes), training: [h.optimizer('Adam', { learning_rate: lr }), h.loss(loss, output)],
                    data: h.data(dataset, options, split), trainer: h.trainer(trainer),
                }));
            },
            idle: () => edu.trainer.getState().status === 'idle',
            runs: () => edu.trainer.getState().runs.length,
            async timed(sync) {
                const started = performance.now();
                const record = await edu.trainer.getState().start({ sync });
                return { ms: performance.now() - started, status: record.status, error: record.error?.message ?? null, final: record.history.at(-1) };
            },
        };
    });
}

async function visibleTargets(page) {
    return page.evaluate(() => [...document.querySelectorAll('[data-edu-target]')]
        .filter((element) => { const box = element.getBoundingClientRect(); return box.width > 0 && box.height > 0; })
        .map((element) => element.dataset.eduTarget));
}

async function checkViewport(browser, width, height) {
    const tag = `${width}`;
    const page = await (await browser.newContext({ viewport: { width, height } })).newPage();
    const errors = [];
    page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
    page.on('pageerror', (error) => errors.push(String(error)));
    await page.goto(URL);
    await page.waitForFunction(() => window.__deepflow?.edu?.trainer && window.__deepflow?.edu?.engine && window.__deepflow?.edu?.data);
    await installHelpers(page);
    const seen = new Set();
    const shot = async (name) => {
        await page.waitForTimeout(200);
        for (const name of await visibleTargets(page)) seen.add(name);
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
        assert.ok(overflow <= 0, `${tag} ${name}: horizontal page scroll of ${overflow} px`);
        await page.screenshot({ path: `${SHOTS}/edu-data-${tag}-${name}.png` });
    };
    const waitRuns = (count) => page.waitForFunction((n) => window.__stageB.idle() && window.__stageB.runs() >= n, count, { timeout: 30000 });

    // Data tab: every dataset row, the preview, options and Use this dataset.
    await page.click('[data-tab="data"]');
    await page.waitForSelector('[data-edu-target="dataset-blobs"]');
    const datasetIds = await page.evaluate(() => window.__deepflow.edu.registry.DATASETS.map((spec) => spec.id));
    for (const id of datasetIds) assert.ok(await page.$(`[data-edu-target="dataset-${id}"]`), `row for ${id}`);
    await shot('01-data-blobs');
    await page.evaluate(() => window.__stageB.load({ input: [2], blocks: [['linear', 2, 1]], loss: 'bce', lr: 0.03, dataset: 'blobs', options: { n: 200, noise: 0.1, seed: 1 } }));
    await page.click('[data-edu-target="dataset-iris"]');
    await page.click('[data-edu-target="use-dataset"]');
    await page.waitForFunction(() => window.__deepflow.architectureStore.getState().graph.nodes.in.parameters.shape === '(4,)');
    assert.match(await page.textContent('.edu-data-use-status'), /Input → \(4,\); last Linear \(Linear\) Output → 3/);
    await shot('02-data-iris-used');
    await page.click('[data-edu-target="dataset-digits"]');
    await page.waitForSelector('.edu-data-digit-grid');
    await shot('03-data-digits');

    // Training tab: blobs, a learning-rate comparison, the boundary.
    await page.evaluate(() => window.__stageB.load({ input: [2], blocks: [['linear', 2, 1]], loss: 'bce', lr: 0.03, dataset: 'blobs', options: { n: 200, noise: 0.1, seed: 1 }, trainer: { epochs: 30 } }));
    await page.click('[data-tab="training"]');
    await page.waitForSelector('.edu-tr-panel [data-edu-target="trainer-run"]');
    await page.click('.edu-tr-panel [data-edu-target="trainer-run"]');
    await waitRuns(1);
    await page.fill('.edu-tr-panel [data-edu-target="trainer-lr"]', '0.0001');
    await page.click('.edu-tr-panel [data-edu-target="trainer-run"]');
    await waitRuns(2);
    assert.match(await page.textContent('.edu-tr-ghosts'), /lr 0\.03/);
    await page.locator('.edu-tr-panel').scrollIntoViewIfNeeded();
    await shot('04-training-blobs');

    // Iris: the confusion matrix and Check initial loss.
    await page.evaluate(() => window.__stageB.load({ input: [4], blocks: [['linear', 4, 3]], loss: 'cross_entropy', lr: 0.03, dataset: 'iris', options: { seed: 1 }, trainer: { epochs: 50, batchSize: 16 } }));
    await page.click('.edu-tr-panel [data-edu-target="trainer-init-loss"]');
    await waitRuns(1);
    await page.click('.edu-tr-panel [data-edu-target="trainer-run"]');
    await waitRuns(2);
    assert.equal(await page.locator('[data-edu-target="trainer-confusion"] tbody tr').count(), 3);
    await shot('05-training-iris');

    // The lesson card behind a prediction gate (Stage A's demo module, step s6).
    await page.evaluate(async () => {
        const { TEST_MODULE, TEST_TEXT } = await import('/scripts/fixtures/edu-test-module.js');
        const { engine, session } = window.__deepflow.edu;
        engine.setModuleRegistry({ modules: [TEST_MODULE], text: TEST_TEXT });
        await session.getState().startModule('demo', { stepId: 's6' });
        if (session.getState().phase === 'warmup') await session.getState().skipWarmup();
        await session.getState().goToStep(TEST_MODULE.steps.findIndex((step) => step.id === 's6'));
    });
    await page.waitForSelector('.edu-tr-card');
    assert.equal(await page.isDisabled('.edu-tr-card [data-edu-target="trainer-run"]'), true, 'the gate locks Train');
    await shot('06-card-locked');
    await page.evaluate(() => window.__deepflow.edu.session.getState().submitPrediction('demo.loss', 'cross_entropy'));
    await page.waitForFunction(() => !document.querySelector('.edu-tr-card [data-edu-target="trainer-run"]').disabled);
    await page.click('.edu-tr-card [data-edu-target="trainer-run"]');
    await waitRuns(1);
    await page.locator('.edu-tr-card').scrollIntoViewIfNeeded();
    await shot('07-card-trained');
    await page.evaluate(() => window.__deepflow.edu.session.getState().exitLesson());

    // The trainer budget of §6.10.7, synchronous and sliced (what the learner waits for).
    const budget = {};
    const cases = {
        circle: [{ input: [2], blocks: [['linear', 2, 8], ['relu'], ['linear', 8, 1]], loss: 'bce', lr: 0.03, dataset: 'circle', options: { n: 300, noise: 0, seed: 2 }, trainer: { epochs: 50, batchSize: 32, seed: 2 } }, 500],
        spiralOverfit: [{ input: [2], blocks: [['linear', 2, 128], ['relu'], ['linear', 128, 128], ['relu'], ['linear', 128, 3]], loss: 'cross_entropy', lr: 0.01, dataset: 'spiral3', options: { perClass: 50, noise: 0.3, seed: 5 }, split: { val: 0.5, test: 0 }, trainer: { epochs: 300, batchSize: 16, seed: 5 } }, 4000],
        digits: [{ input: [1, 8, 8], blocks: [['flatten'], ['linear', 64, 128], ['relu'], ['linear', 128, 10]], loss: 'cross_entropy', lr: 0.001, dataset: 'digits', options: { seed: 1 }, trainer: { epochs: 15, batchSize: 32, seed: 3 } }, 4000],
    };
    for (const [name, [config, limit]] of Object.entries(cases)) {
        await page.evaluate((value) => window.__stageB.load(value), config);
        await page.evaluate(() => window.__stageB.timed(true)); // warm-up (digits loads its chunk once)
        const sync = await page.evaluate(() => window.__stageB.timed(true));
        const sliced = await page.evaluate(() => window.__stageB.timed(false));
        assert.equal(sync.status, 'completed', `${name}: ${sync.error}`);
        budget[name] = { syncMs: Math.round(sync.ms), slicedMs: Math.round(sliced.ms), budgetMs: limit, withinBudget: sync.ms < limit, valAccuracy: sync.final?.valAccuracy };
        // The budget is for a 2020 laptop; a slower or busy machine only fails far beyond it.
        assert.ok(sync.ms < 1.5 * limit, `${name} took ${sync.ms.toFixed(0)} ms (budget ${limit} ms on a 2020 laptop)`);
    }

    assert.deepEqual(errors, [], `${tag}: console errors`);
    const missing = STAGE_B_TARGETS.filter((name) => !seen.has(name));
    assert.deepEqual(missing, [], `${tag}: Stage B targets never visible`);
    await page.context().close();
    return budget;
}

const browser = await chromium.launch({ executablePath: EXECUTABLE });
try {
    for (const [width, height] of [[1440, 900], [390, 844]]) {
        const budget = await checkViewport(browser, width, height);
        console.log(`ok ${width}×${height}: targets, no console errors, no horizontal scroll; trainer timings`, JSON.stringify(budget));
    }
} finally {
    await browser.close();
}
