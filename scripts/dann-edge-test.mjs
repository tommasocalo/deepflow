/**
 * DANN-style scenario: a feature extractor branches into a class predictor
 * and a domain predictor; the domain branch carries a gradient-reversal
 * edge. Confirm the GradientEdge renders the badge for `reverse(λ)`.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5174/';

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
    // pre-set viewport so the page mounts already zoomed appropriately
    await page.evaluate(() => {
        localStorage.setItem('deepflow:architecture-viewport:root',
            JSON.stringify({ x: 60, y: 0, zoom: 0.65 }));
    });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    const dump = await page.evaluate(() => {
        const { architectureStore } = window.__deepflow;
        const arch = architectureStore.getState();

        const inputId  = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.moveNode(inputId,  { x: 320, y: 40 });
        arch.moveNode(outputId, { x: 80,  y: 740 });
        arch.patchLayerParameters(inputId, { shape: '(64,)' });
        arch.patchNode(outputId, { label: 'class logits' });

        const featId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 320, y: 220 });
        arch.patchNode(featId, { label: 'feature extractor' });
        arch.patchLayerParameters(featId, { in_features: 64, out_features: 128 });

        const classId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 80, y: 540 });
        arch.patchNode(classId, { label: 'class head' });
        arch.patchLayerParameters(classId, { in_features: 128, out_features: 10 });

        const domId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 600, y: 540 });
        arch.patchNode(domId, { label: 'domain head' });
        arch.patchLayerParameters(domId, { in_features: 128, out_features: 2 });

        const domOutId = arch.addLayerByLibraryKey('deepflow.io.Output', { x: 600, y: 740 });
        arch.patchNode(domOutId, { label: 'domain logits' });

        arch.connect({ source: inputId, target: featId });
        arch.connect({ source: featId, target: classId });
        arch.connect({ source: classId, target: outputId });

        const grlEdgeId = arch.connect({ source: featId, target: domId });
        arch.connect({ source: domId, target: domOutId });

        // Mark feat → domain head as a gradient-reversal edge with λ = 0.5
        if (grlEdgeId) arch.setEdgeGradientOp(grlEdgeId, 'reverse', 0.5);

        const updated = architectureStore.getState().graph.edges.find((e) => e.id === grlEdgeId);
        return { grlOp: updated?.gradientOp, grlLambda: updated?.gradientOpLambda };
    });
    console.log('GRL edge state:', dump);

    // Persist a zoomed-out viewport so the whole DANN structure fits.
    await page.evaluate(() => {
        localStorage.setItem('deepflow:architecture-viewport:root',
            JSON.stringify({ x: 100, y: 0, zoom: 0.65 }));
    });
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(700);
    await page.waitForTimeout(400);
    await page.screenshot({ path: 'screenshots/70-dann-with-grl.png', fullPage: false });
    console.log('screenshot → screenshots/70-dann-with-grl.png');

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
