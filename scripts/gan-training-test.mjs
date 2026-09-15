/**
 * Build a GAN-style training graph end-to-end:
 *   - One generator + one discriminator on the architecture canvas
 *   - Two optimizers (G_opt, D_opt) bound to their respective parameter groups
 *   - Two losses (G_loss, D_loss) bound to the architecture outputs
 *   - Two phases (G_step, D_step) alternating each batch
 * Capture screenshots after each major step.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5174/';

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
    page.on('console', (msg) => {
        if (msg.type() === 'error') console.error('[browser:error]', msg.text());
    });

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    // Build architecture: a tiny G + D split.
    const archIds = await page.evaluate(() => {
        const { architectureStore } = window.__deepflow;
        const arch = architectureStore.getState();
        const inputId  = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.patchLayerParameters(inputId, { shape: '(64,)' });
        arch.patchNode(outputId, { label: 'D-logits' });

        // Generator: Linear(64 → 128)
        const gId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 200, y: 180 });
        arch.patchNode(gId, { label: 'G' });
        arch.patchLayerParameters(gId, { in_features: 64, out_features: 128 });

        // Discriminator: Linear(128 → 1)
        const dId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 200, y: 280 });
        arch.patchNode(dId, { label: 'D' });
        arch.patchLayerParameters(dId, { in_features: 128, out_features: 1 });

        arch.connect({ source: inputId, target: gId });
        arch.connect({ source: gId, target: dId });
        arch.connect({ source: dId, target: outputId });

        return { inputId, outputId, gId, dId };
    });
    await page.waitForTimeout(200);

    // Switch to Training tab
    await page.locator('button', { hasText: /^Training/ }).first().click({ force: true });
    await page.waitForTimeout(300);

    // Build the training graph programmatically.
    await page.evaluate((archIds) => {
        const { trainingStore } = window.__deepflow;
        const t = trainingStore.getState();

        const gOptId = t.addOptimizer('Adam', { x: 60, y: 100 });
        t.patchNode(gOptId, { label: 'G optimizer', parameterGroup: [archIds.gId] });
        t.patchHyperparameters(gOptId, { learning_rate: 2e-4 });

        const dOptId = t.addOptimizer('Adam', { x: 60, y: 260 });
        t.patchNode(dOptId, { label: 'D optimizer', parameterGroup: [archIds.dId] });
        t.patchHyperparameters(dOptId, { learning_rate: 2e-4 });

        const gLossId = t.addLoss('bce', { x: 360, y: 100 }, archIds.outputId);
        t.patchNode(gLossId, { label: 'G loss', weight: 1 });

        const dLossId = t.addLoss('bce', { x: 360, y: 260 }, archIds.outputId);
        t.patchNode(dLossId, { label: 'D loss', weight: 1 });

        const gPhaseId = t.addPhase({ x: 660, y: 100 }, {
            label: 'G step',
            optimizerIds: [gOptId],
            lossIds: [gLossId],
            everyNSteps: 1,
            stepsPerCycle: 1,
        });
        const dPhaseId = t.addPhase({ x: 660, y: 260 }, {
            label: 'D step',
            optimizerIds: [dOptId],
            lossIds: [dLossId],
            everyNSteps: 1,
            stepsPerCycle: 1,
        });

        // Wire loss → optimizer (gradient flow) and optimizer → phase
        t.connect({ source: gLossId, target: gOptId });
        t.connect({ source: dLossId, target: dOptId });
        t.connect({ source: gOptId,  target: gPhaseId });
        t.connect({ source: dOptId,  target: dPhaseId });
    }, archIds);
    await page.waitForTimeout(400);
    await shot(page, '60-gan-built');

    // Click the G phase to see its inspector
    const gPhaseLocator = page.locator('.react-flow__node', { hasText: /G step/ }).first();
    if (await gPhaseLocator.count()) {
        await gPhaseLocator.click({ force: true });
        await page.waitForTimeout(300);
        await shot(page, '61-phase-selected');
    }

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
