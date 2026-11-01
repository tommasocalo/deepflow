// Stage E walkthrough (docs/edu/SPEC.md §8.5): every module as a learner who uses the whole help ladder. Not part of
// npm test; it takes about ten minutes per viewport.
//
//   npx vite --host 127.0.0.1 --port 5174 --strictPort      (in another terminal)
//   node scripts/edu-walkthrough-browser-check.mjs [--base http://127.0.0.1:5174] [--out <screenshot dir>]
//        [--sizes 1440x900,390x844] [--modules m1,m2,…]
//
// For each viewport on a fresh profile: Learn → each module's Start, Skip warm-up, then per step Hint, Hint, Show me,
// "✓ Done" and Next; the quiz answered; the recap. It asserts no console errors, no horizontal page scroll, and that
// every step with an unused target rings it at entry, or, when it is below the fold (a phone's drawer, the ribbon's
// edge), offers "Where is it?", which brings it into view. After the walk it reloads in the middle of module 2 (the
// same step must come back) and exits the lesson (the sandbox must show the GPT-2 example).
// Screenshots of each module start and each step after Show me go to --out.
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
const sizes = argument('sizes', '1440x900,390x844').split(',').map((size) => size.split('x').map(Number));
const modules = argument('modules', 'm1,m2,m3,m4,m5,m6,m7').split(',');
const executablePath = argument('chromium', process.env.PLAYWRIGHT_CHROMIUM ?? (existsSync('/opt/pw-browsers/chromium') ? '/opt/pw-browsers/chromium' : undefined));
mkdirSync(out, { recursive: true });

const failures = [];
const notes = [];
const check = (condition, message) => { if (!condition) failures.push(message); return !!condition; };

const browser = await chromium.launch(executablePath ? { executablePath } : {});
try {
    for (const [width, height] of sizes) {
        const context = await browser.newContext({ viewport: { width, height } });
        const page = await context.newPage();
        const consoleErrors = [];
        page.on('console', (message) => { if (message.type() === 'error') consoleErrors.push(message.text()); });
        page.on('pageerror', (error) => consoleErrors.push(error.message));
        const label = (text) => `[${width}] ${text}`;
        const panel = page.getByRole('complementary', { name: 'Lesson' });
        const textOf = async (locator) => ((await locator.count()) ? ((await locator.first().textContent()) ?? '').trim() : '');
        const shot = (name) => page.screenshot({ path: join(out, `edu-walk-${width}-${name}.png`) });
        const settle = (ms = 400) => page.waitForTimeout(ms);
        const session = () => page.evaluate(() => {
            const state = window.__deepflow.edu.session.getState();
            return { mode: state.mode, moduleId: state.moduleId, phase: state.phase, step: state.currentStep()?.id ?? null };
        });
        const noHorizontalScroll = async (where) => {
            const { scroll, client } = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
            check(scroll <= client, label(`horizontal page scroll at ${where}: ${scroll} > ${client}`));
        };
        const learnTab = () => page.locator('.df-toolbar-tabs [data-tab="learn"]').click();

        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForFunction(() => !!window.__deepflow?.edu?.engine, null, { timeout: 20000 });
        check(await page.locator('.df-toolbar-tabs [data-tab="learn"][aria-current="page"]').count() === 1, label('a fresh profile opens on Learn'));
        for (const moduleId of modules) {
            const module = await page.evaluate((id) => {
                const found = window.__deepflow.edu.engine.getModule(id);
                return found ? { order: found.order, steps: found.steps.map((step) => ({ id: step.id, targets: (step.target ?? []).length })), quiz: found.quiz.map((item) => item.correct) } : null;
            }, moduleId);
            if (!check(module, label(`module ${moduleId} exists`))) continue;
            await learnTab();
            await page.locator(`[data-module="${moduleId}"] .edu-module-actions button`).click();
            await settle(900);
            await shot(`${moduleId}-start`);
            const skip = panel.getByRole('button', { name: /Skip warm-up/i });
            if (await skip.count()) { await skip.click(); await settle(600); }
            for (const step of module.steps) {
                const where = `${moduleId}.${step.id}`;
                await settle(1500);
                check((await session()).step === step.id, label(`${where}: the lesson is on this step`));
                const ring = await page.locator('[data-edu-spotlight]').count();
                const lost = await panel.getByRole('button', { name: 'Where is it?' }).count();
                const used = await page.evaluate((id) => {
                    const state = window.__deepflow.edu.session.getState();
                    const target = state.currentStep();
                    const { isTargetDone } = window.__deepflow.edu.engine;
                    return (target?.target ?? []).every((ref) => isTargetDone(ref, document, { visitedTabs: state.runtime.visitedTabs }));
                }, moduleId);
                if (step.targets && !used && ring === 0) {
                    // Below the fold (a phone's drawer, the ribbon's edge): "Where is it?" must bring the target on screen.
                    const finder = panel.getByRole('button', { name: 'Where is it?' });
                    await finder.waitFor({ timeout: 3000 }).catch(() => {});
                    if (check(await finder.count() === 1, label(`${where}: neither a ring nor "Where is it?" at entry`))) {
                        await finder.click();
                        await settle(1200);
                        check(await page.locator('[data-edu-spotlight]').count() === 1, label(`${where}: "Where is it?" brings the target into view`));
                        notes.push(label(`${where}: target below the fold, found with "Where is it?"`));
                    }
                } else {
                    if (step.targets && !used) check(ring === 1, label(`${where}: a spotlight ring at entry`));
                    check(lost === 0, label(`${where}: no "Where is it?" while the target is on screen`));
                }
                for (let level = 0; level < 3; level++) {
                    const button = panel.locator('.edu-hint-button');
                    if (!(await button.count())) break;
                    const name = await button.getAttribute('aria-label');
                    await button.click();
                    await settle(name?.startsWith('Show me') ? 600 : 250);
                    if (name?.startsWith('Show me')) break;
                }
                const done = await panel.locator('.edu-status.is-done').first().waitFor({ timeout: 40000 }).then(() => true, () => false);
                check(done, label(`${where}: ✓ Done after Show me (${await textOf(panel.locator('.edu-status'))})`));
                check(!(await textOf(panel.locator('.edu-callout-error'))), label(`${where}: no action error`));
                await shot(`${where}`);
                await noHorizontalScroll(where);
                await panel.locator('footer.edu-nav button.df-button-primary').last().click();
                await settle(600);
            }
            for (const correct of module.quiz) {
                // Options are shown in the learner's seeded order; Check commits the choice.
                await panel.locator(`label.edu-option[data-option="${correct}"]`).click();
                await panel.getByRole('button', { name: 'Check', exact: true }).click();
                await settle(200);
                await panel.locator('footer.edu-nav button').last().click();
                await settle(300);
            }
            check((await session()).phase === 'recap', label(`${moduleId}: the quiz ends in the recap`));
            const status = await page.evaluate((id) => window.__deepflow.edu.session.getState().progress.modules[id]?.status, moduleId);
            check(status === 'completed', label(`${moduleId}: completed`));
            notes.push(label(`${moduleId}: ${module.steps.length} steps and ${module.quiz.length} quiz items done`));
        }
        // Reload in the middle of module 2, then leave for the sandbox.
        await learnTab();
        await page.locator('[data-module="m2"] .edu-module-actions button').click();
        await settle(900);
        await page.evaluate(() => { window.location.hash = '#/learn/m2/s3'; });
        await settle(1500);
        await page.reload({ waitUntil: 'networkidle' });
        await page.waitForFunction(() => !!window.__deepflow?.edu?.engine, null, { timeout: 20000 });
        await settle(1200);
        const resumed = await session();
        check(resumed.mode === 'lesson' && resumed.moduleId === 'm2' && resumed.step === 's3', label(`a reload resumes M2 s3 (got ${JSON.stringify(resumed)})`));
        await panel.getByRole('button', { name: 'Exit lesson' }).first().click();
        await settle(800);
        await page.getByRole('button', { name: /Open the sandbox/ }).click();
        await settle(1200);
        const sandbox = await page.evaluate(() => Object.values(window.__deepflow.architectureStore.getState().graph.nodes).map((node) => node.label));
        check(sandbox.includes('token embedding'), label('Exit lesson restores the GPT-2 sandbox'));
        check(consoleErrors.length === 0, label(`console errors: ${consoleErrors.slice(0, 3).join(' | ')}`));
        await context.close();
    }
} finally {
    await browser.close();
}
console.log(JSON.stringify({ ok: failures.length === 0, failures, notes, screenshots: out }, null, 2));
process.exit(failures.length ? 1 : 0);
