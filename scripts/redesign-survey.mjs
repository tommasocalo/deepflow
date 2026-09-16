/**
 * Visual survey across the major UI states. Saves under screenshots/r-*.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5173/';

async function shot(page, name) {
    const out = `screenshots/${name}.png`;
    await page.screenshot({ path: out, fullPage: false });
    console.log(`screenshot → ${out}`);
}

async function main() {
    const browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();
    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    await shot(page, 'r01-arch-empty');

    await page.evaluate(() => {
        const { architectureStore } = window.__deepflow;
        const arch = architectureStore.getState();
        const inputId  = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.patchLayerParameters(inputId, { shape: '(3, 224, 224)' });
        const convId = arch.addLayerByLibraryKey('torch.nn.Conv2d', { x: 220, y: 220 });
        arch.patchLayerParameters(convId, { in_channels: 3, out_channels: 64 });
        arch.connect({ source: inputId, target: convId });
        arch.connect({ source: convId, target: outputId });
    });
    await page.waitForTimeout(300);
    await shot(page, 'r02-arch-with-conv');

    // Click Conv2d body to expand
    const convNode = page.locator('.react-flow__node', { hasText: 'Conv2d' }).first();
    if (await convNode.count()) {
        await convNode.click({ force: true });
        await page.waitForTimeout(400);
        await shot(page, 'r03-arch-conv-expanded');
    }

    // Collapse via store, then select an edge to surface its menu
    await page.evaluate(() => {
        window.__deepflow.architectureStore.getState().clearExpanded();
    });
    await page.waitForTimeout(200);

    const edgePath = page.locator('.react-flow__edge').first();
    if (await edgePath.count()) {
        await edgePath.click({ force: true, position: { x: 10, y: 10 } });
        await page.waitForTimeout(400);
        await shot(page, 'r04-edge-selected');
    }

    // Expand the Convolution category in the palette
    const convCategory = page.locator('button', { hasText: /^CONVOLUTION$/i }).first();
    if (await convCategory.count()) {
        await convCategory.click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, 'r07-palette-category-expanded');
    }

    // Switch to Training tab
    await page.locator('button', { hasText: /^TRAINING$/i }).first().click({ force: true });
    await page.waitForTimeout(400);
    await shot(page, 'r05-training-empty');

    await page.evaluate(() => {
        const { trainingStore } = window.__deepflow;
        trainingStore.getState().addOptimizer('Adam', { x: 200, y: 200 });
    });
    await page.waitForTimeout(300);
    const optNode = page.locator('.react-flow__node', { hasText: /optimizer/i }).first();
    if (await optNode.count()) {
        await optNode.click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, 'r06-training-optimizer-expanded');
    }

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
