/**
 * Drives each of the three headline storyboards (ResNet, DANN, GAN) and
 * captures architecture-tab + training-tab screenshots for each, plus a
 * generated-code screenshot.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5174/';
const STORYBOARDS = ['resnet', 'dann', 'gan'];

async function shot(page, name) {
    const out = `screenshots/${name}.png`;
    await page.screenshot({ path: out, fullPage: false });
    console.log(`screenshot → ${out}`);
}

async function setViewport(page, key, viewport) {
    await page.evaluate(({ k, v }) => {
        try { localStorage.setItem(k, JSON.stringify(v)); } catch {}
    }, { k: key, v: viewport });
}

async function main() {
    const browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => {
        if (msg.type() === 'error') console.error('[browser:error]', msg.text());
    });

    for (const sbId of STORYBOARDS) {
        await page.goto(URL, { waitUntil: 'domcontentloaded' });
        await page.evaluate(() => { try { localStorage.clear(); } catch {} });
        // Pre-set zoom so the whole architecture is visible
        await setViewport(page, 'deepflow:architecture-viewport:root', { x: 60, y: 0, zoom: 0.6 });
        await setViewport(page, 'deepflow:training-viewport',          { x: 0, y: 0, zoom: 0.7 });
        await page.goto(URL, { waitUntil: 'networkidle' });
        await page.waitForTimeout(500);

        // Run the storyboard
        await page.evaluate((id) => {
            const wf = window.__deepflow;
            // Storyboards are pulled in lazily via the navbar menu. Trigger
            // by importing the module directly through window.
            return import('/src/storyboards/index.js').then(({ STORYBOARDS }) => {
                const sb = STORYBOARDS.find((s) => s.id === id);
                sb.run({
                    architectureStore: wf.architectureStore,
                    templatesStore:    wf.templatesStore,
                    trainingStore:     wf.trainingStore,
                });
            });
        }, sbId);
        await page.waitForTimeout(500);
        await shot(page, `100-${sbId}-arch`);

        // Switch to training tab
        await page.locator('button', { hasText: /^Training/ }).first().click({ force: true });
        await page.waitForTimeout(500);
        await shot(page, `101-${sbId}-train`);

        // Show generated code
        await page.locator('button', { hasText: /^Code$/ }).first().click({ force: true });
        await page.waitForTimeout(400);
        await page.locator('button', { hasText: /train\.py/ }).first().click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, `102-${sbId}-code`);
    }

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
