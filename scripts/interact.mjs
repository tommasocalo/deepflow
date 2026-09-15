/**
 * Drives the deepflow-v2 canvas through a small scripted scenario:
 *   1. Load with a clean localStorage
 *   2. Add a Conv2d via the palette (click)
 *   3. Take a screenshot
 *   4. Hover the left sidebar to expand it
 *   5. Take another screenshot
 */
import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';

const URL = process.env.URL ?? 'http://localhost:5174/';

async function shot(page, name) {
    const out = path.resolve(`screenshots/${name}.png`);
    await fs.mkdir(path.dirname(out), { recursive: true });
    await page.screenshot({ path: out, fullPage: false });
    console.log(`screenshot → ${out}`);
}

async function main() {
    const browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();

    /** @type {string[]} */
    const errors = [];
    page.on('pageerror', (err) => errors.push(err.stack ?? String(err)));
    page.on('console', (msg) => {
        if (msg.type() === 'error') errors.push(`console: ${msg.text()}`);
    });

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);

    // Step 1: initial state
    await shot(page, '10-initial');

    // Step 2: click the Conv2d block in the palette to add it to the canvas
    const conv2d = page.locator('button', { hasText: 'Conv2d' }).first();
    if (await conv2d.count()) {
        await conv2d.click();
        await page.waitForTimeout(400);
        await shot(page, '11-after-add-conv2d');
    } else {
        console.warn('Conv2d button not found');
    }

    // Step 3: hover the left sidebar — it expands on hover
    const sidebar = page.locator('div').filter({ hasText: /^Overview$|Settings/i }).first();
    if (await sidebar.count()) {
        await sidebar.hover();
        await page.waitForTimeout(300);
        await shot(page, '12-sidebar-hover');
    }

    // Step 4: click the Input node to open the inspector. Force the click
    // because xyflow background svg can intercept hit testing in headless.
    const inputNode = page.locator('.react-flow__node', { hasText: 'Input' }).first();
    if (await inputNode.count()) {
        await inputNode.click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, '13-input-selected');
    }

    // Step 5: click the Conv2d node so we can see error rendering on inspector
    const conv2dNode = page.locator('.react-flow__node', { hasText: 'Conv2d' }).first();
    if (await conv2dNode.count()) {
        await conv2dNode.click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, '14-conv2d-selected');
    }

    // Step 6: open the validation status dropdown
    const pill = page.locator('button', { hasText: /errors?,/ }).first();
    if (await pill.count()) {
        await pill.click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, '15-validation-dropdown');
    }

    if (errors.length) {
        console.error('Errors:');
        for (const e of errors) console.error('  ' + e);
    }

    await browser.close();
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
