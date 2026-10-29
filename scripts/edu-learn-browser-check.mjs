// Manual browser check for the Learn home and the lesson panel (docs/edu/SPEC.md §8.1). Not part of npm test.
//
//   npx vite --host 127.0.0.1 --port 5174 --strictPort      (in another terminal)
//   node scripts/edu-learn-browser-check.mjs [--base http://127.0.0.1:5174] [--out <screenshot dir>]
//
// For each viewport (1440×900 docked panel, 390×844 drawer) on a fresh profile it checks that the app opens on Learn,
// loads the demo module of scripts/fixtures/edu-test-module.js through window.__deepflow.edu.engine, opens step 3 by
// deep link, and verifies: the spotlight rings "Browse components" (ribbon.components), the panel never overlaps the
// canvas stage, the page has no horizontal scroll, Show me passes the step, the train gate disables the trainer card's
// Train button until the prediction is committed, and the console stays free of errors. Screenshots go to --out.
import { existsSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { chromium } from 'playwright';

const argument = (name, fallback) => {
    const index = process.argv.indexOf(`--${name}`);
    return index > 0 && process.argv[index + 1] ? process.argv[index + 1] : fallback;
};
const base = argument('base', 'http://127.0.0.1:5174');
const out = argument('out', process.env.EDU_SHOTS ?? join(tmpdir(), 'deepflow-edu-shots'));
const executablePath = argument('chromium', process.env.PLAYWRIGHT_CHROMIUM ?? (existsSync('/opt/pw-browsers/chromium') ? '/opt/pw-browsers/chromium' : undefined));
mkdirSync(out, { recursive: true });

const VIEWPORTS = [
    { name: '1440', width: 1440, height: 900, layout: 'docked' },
    { name: '390', width: 390, height: 844, layout: 'drawer' },
];
const failures = [];
const notes = [];
const check = (condition, message) => { if (!condition) failures.push(message); return !!condition; };

const browser = await chromium.launch(executablePath ? { executablePath } : {});
try {
    for (const viewport of VIEWPORTS) {
        const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height } });
        const page = await context.newPage();
        const consoleErrors = [];
        page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
        page.on('pageerror', (error) => consoleErrors.push(error.message));
        const label = (text) => `[${viewport.name}] ${text}`;
        const shot = (name) => page.screenshot({ path: join(out, `edu-learn-${viewport.name}-${name}.png`) });
        const noHorizontalScroll = async (where) => {
            const { scroll, client } = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
            check(scroll <= client, label(`horizontal page scroll at ${where}: ${scroll} > ${client}`));
        };

        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForFunction(() => !!window.__deepflow?.edu?.engine, null, { timeout: 15000 });
        check(await page.locator('.df-toolbar-tabs [data-tab="learn"][aria-current="page"]').count() === 1, label('a fresh profile opens on Learn'));
        check(await page.locator('.df-toolbar-tabs [data-tab]').evaluateAll((tabs) => tabs.map((tab) => tab.dataset.tab).join(',')) === 'learn,architecture,data,training',
            label('tabs are Learn, Architecture, Data, Training'));
        await shot('01-learn');
        await noHorizontalScroll('Learn');

        await page.evaluate(async () => {
            const { TEST_MODULE, TEST_TEXT } = await import('/scripts/fixtures/edu-test-module.js');
            window.__deepflow.edu.engine.setModuleRegistry({ modules: [TEST_MODULE], text: TEST_TEXT });
        });
        await page.getByRole('button', { name: /^Start · 1 Tour of the lesson panel/ }).waitFor();
        await shot('02-learn-demo');

        // Deep link to step 3, whose spotlight target is ribbon.components.
        await page.evaluate(() => { window.location.hash = '#/learn/demo/s3'; });
        const panel = page.getByRole('complementary', { name: 'Lesson' });
        await panel.getByRole('heading', { name: 'Add a non-linearity' }).waitFor({ timeout: 10000 });
        check(await page.evaluate(() => window.location.hash) === '#/learn/demo/s3', label('the hash follows the step'));
        await page.waitForSelector('[data-edu-spotlight]', { timeout: 5000 }).catch(() => null);
        const ring = await page.evaluate(() => {
            const spotlight = document.querySelector('[data-edu-spotlight]');
            const target = window.__deepflow.edu.engine.resolveTarget('ribbon.components');
            if (!spotlight || !target) return { spotlight: !!spotlight, target: !!target };
            const a = spotlight.getBoundingClientRect();
            const b = target.getBoundingClientRect();
            return {
                spotlight: true, target: true, label: target.getAttribute('aria-label'), hidden: spotlight.getAttribute('aria-hidden'),
                pointer: getComputedStyle(spotlight).pointerEvents, text: spotlight.textContent,
                surrounds: a.left <= b.left && a.top <= b.top && a.right >= b.right && a.bottom >= b.bottom,
            };
        });
        check(ring.target && ring.label === 'Browse components', label('ribbon.components resolves to "Browse components"'));
        check(ring.spotlight && ring.surrounds, label('the spotlight ring surrounds the target'));
        check(ring.hidden === 'true' && ring.pointer === 'none' && ring.text === '', label('the spotlight is aria-hidden, takes no pointer events and has no text'));

        const boxes = await page.evaluate(() => {
            const rect = (element) => { const r = element?.getBoundingClientRect(); return r ? { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height } : null; };
            return { stage: rect(document.querySelector('.edu-stage')), panel: rect(document.querySelector('aside[aria-label="Lesson"]')), innerHeight: window.innerHeight };
        });
        if (viewport.layout === 'docked') {
            check(Math.round(boxes.panel.width) === 360, label(`the panel is 360 px wide at 1440 (got ${boxes.panel.width})`));
            check(boxes.stage.right <= boxes.panel.left + 0.5, label('the docked panel sits beside the stage, never over it'));
        } else {
            check(boxes.stage.bottom <= boxes.panel.top + 0.5, label('the drawer sits below the stage, never over it'));
            check(Math.abs(boxes.panel.height - boxes.innerHeight * 0.45) <= 2, label(`the drawer is 45vh tall (got ${boxes.panel.height})`));
        }
        await noHorizontalScroll('step 3');
        await shot('03-lesson-step3');

        for (const name of ['Hint', 'Another hint', 'Show me']) await panel.getByRole('button', { name, exact: true }).click();
        await panel.getByText('Step complete').waitFor({ state: 'attached', timeout: 5000 });
        check(await page.evaluate(() => window.__deepflow.edu.session.getState().evaluate().pass), label('Show me passes step 3'));
        await shot('04-lesson-step3-done');

        // The train gate of step 6 against the real TrainerCard (Stage B), when it is mounted.
        await page.evaluate(() => window.__deepflow.edu.session.getState().goToStep(5));
        await panel.getByRole('heading', { name: 'Pick the loss' }).waitFor();
        const train = panel.locator('[data-edu-target="trainer-run"]').first();
        if (await train.count()) {
            check(await train.isDisabled(), label('Train is disabled while the prediction is open'));
            await panel.getByRole('radio', { name: 'Cross-entropy' }).check();
            await panel.getByRole('button', { name: 'Submit answer' }).click();
            await panel.getByRole('heading', { name: 'Exactly' }).waitFor();
            await page.waitForFunction(() => !window.__deepflow.edu.session.getState().trainLocked());
            check(await page.evaluate(() => document.activeElement?.textContent) === 'Exactly', label('focus moves to the reveal heading'));
            notes.push(label(`trainer card after the answer: Train ${await train.isDisabled() ? 'disabled (preflight)' : 'enabled'}`));
        } else {
            notes.push(label('the trainer card is still the placeholder: train gate not checked'));
        }
        await noHorizontalScroll('step 6');
        await shot('05-lesson-gate');

        check(consoleErrors.length === 0, label(`console errors: ${consoleErrors.join(' | ')}`));
        await context.close();
    }
} finally {
    await browser.close();
}

console.log(JSON.stringify({ ok: failures.length === 0, failures, notes, screenshots: out }, null, 2));
process.exit(failures.length ? 1 : 0);
