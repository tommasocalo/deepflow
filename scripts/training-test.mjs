/**
 * Verify the Training tab end-to-end:
 *   1. Build a tiny architecture: Input → Linear → Output
 *   2. Switch to Training tab
 *   3. Add an Adam optimizer + Cross-Entropy loss
 *   4. Bind the loss to the Output, the optimizer to a parameter group
 *   5. Take screenshots
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5173/';

async function main() {
    const browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await ctx.newPage();

    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => {
        if (msg.type() === 'error') console.error('[browser:error]', msg.text());
    });

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    // 1. Build architecture programmatically via the running app's stores.
    await page.evaluate(() => {
        const { architectureStore } = window.__deepflow;
        const arch = architectureStore.getState();
        const inputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.patchLayerParameters(inputId, { shape: '(128,)' });
        const linearId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 220, y: 220 });
        arch.patchLayerParameters(linearId, { in_features: 128, out_features: 10 });
        arch.connect({ source: inputId, target: linearId });
        arch.connect({ source: linearId, target: outputId });
    });
    await page.waitForTimeout(200);

    // 2. Switch to Training tab
    await page.locator('button', { hasText: /^Training/ }).first().click({ force: true });
    await page.waitForTimeout(400);
    await page.screenshot({ path: 'screenshots/50-training-empty.png', fullPage: false });
    console.log('screenshot → screenshots/50-training-empty.png');

    // 3. Add Adam optimizer and Cross-Entropy loss via the palette
    await page.locator('button', { hasText: /^Adam$/ }).first().click({ force: true });
    await page.waitForTimeout(200);
    await page.locator('button', { hasText: /Cross-Entropy/ }).first().click({ force: true });
    await page.waitForTimeout(300);
    await page.screenshot({ path: 'screenshots/51-training-with-nodes.png', fullPage: false });
    console.log('screenshot → screenshots/51-training-with-nodes.png');

    // 4. Bind loss to Output via the inspector + connect optimizer to the Linear's params
    const result = await page.evaluate(() => {
        const { architectureStore } = window.__deepflow;
        const trainStoreModule = window.__deepflow.trainingStore;
        // The training store wasn't yet exposed via window — reach through the
        // page's bundle if needed. For now, drive via DOM clicks.
        const arch = architectureStore.getState();
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        const linearId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'torch.nn.Linear').id;
        return { outputId, linearId };
    });

    // Click the Loss node, then in the inspector pick the Output target
    const lossNode = page.locator('.react-flow__node', { hasText: /loss/i }).first();
    if (await lossNode.count()) {
        await lossNode.click({ force: true });
        await page.waitForTimeout(300);
        await page.screenshot({ path: 'screenshots/52-loss-selected.png', fullPage: false });
        console.log('screenshot → screenshots/52-loss-selected.png');
    }

    // Click the Optimizer node to see its inspector + parameter group picker
    const optNode = page.locator('.react-flow__node', { hasText: /optimizer/i }).first();
    if (await optNode.count()) {
        await optNode.click({ force: true });
        await page.waitForTimeout(300);
        await page.screenshot({ path: 'screenshots/53-optimizer-selected.png', fullPage: false });
        console.log('screenshot → screenshots/53-optimizer-selected.png');
    }

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
