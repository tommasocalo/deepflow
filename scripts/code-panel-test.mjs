/**
 * Build a small parametric architecture, open the Code panel, and screenshot
 * each tab to verify the generated code renders correctly in the UI.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5174/';

async function main() {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => {
        if (msg.type() === 'error') console.error('[browser:error]', msg.text());
    });

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    await page.evaluate(() => {
        const { architectureStore, templatesStore, trainingStore } = window.__deepflow;
        const arch = architectureStore.getState();
        const tplStore = templatesStore.getState();
        const train = trainingStore.getState();
        const inputId  = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.patchLayerParameters(inputId, { shape: '(128,)' });

        const linearId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 220, y: 220 });
        arch.patchLayerParameters(linearId, { in_features: 128, out_features: 64 });
        arch.connect({ source: inputId, target: linearId });
        arch.connect({ source: linearId, target: outputId });

        const tplId = arch.groupSelection([linearId], 'DenseBlock');
        tplStore.addFreeParameter(tplId, { name: 'in', type: 'int', default: 128 });
        tplStore.addFreeParameter(tplId, { name: 'out', type: 'int', default: 64 });
        tplStore.patchTemplateLayerParameters(tplId, linearId, { in_features: '$in', out_features: '$out' });

        const root = architectureStore.getState().graph;
        const inst = Object.values(root.nodes).find((n) => n.kind === 'supernode');
        const repeatId = arch.convertInstanceToRepeat(inst.id, 3);
        arch.patchRepeatBinding(repeatId, 'in',  '128 / 2 ** i');
        arch.patchRepeatBinding(repeatId, 'out', '64 / 2 ** i');

        const optId = train.addOptimizer('Adam', { x: 60, y: 100 });
        train.patchHyperparameters(optId, { learning_rate: 1e-3 });
        train.addLoss('cross_entropy', { x: 360, y: 100 }, outputId);
    });
    await page.waitForTimeout(400);

    // Click "Code" button
    await page.locator('button', { hasText: /^Code$/ }).first().click({ force: true });
    await page.waitForTimeout(400);
    await page.screenshot({ path: 'screenshots/90-code-panel-model.png', fullPage: false });
    console.log('screenshot → screenshots/90-code-panel-model.png');

    // Switch to train.py tab
    await page.locator('button', { hasText: /train\.py/ }).first().click({ force: true });
    await page.waitForTimeout(300);
    await page.screenshot({ path: 'screenshots/91-code-panel-train.png', fullPage: false });
    console.log('screenshot → screenshots/91-code-panel-train.png');

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
