/**
 * Adds a Conv2d, then dumps the architecture graph and validation state
 * via the page's globals so we can see what the validator computed.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5174/';

async function main() {
    const browser = await chromium.launch();
    const page = await browser.newPage();

    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => console.log(`[${msg.type()}]`, msg.text()));

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(700);

    // Click Conv2d in the palette
    await page.locator('button', { hasText: 'Conv2d' }).first().click();
    await page.waitForTimeout(300);

    // Dump the store state plus validation result. Zustand stores are
    // imported by modules; we expose them through window for probing.
    const dump = await page.evaluate(async () => {
        const { architectureStore, validation: valMod } = window.__deepflow;
        const state = architectureStore.getState();
        const report = valMod.validate(state.graph);
        return {
            nodes: Object.values(state.graph.nodes).map((n) => ({
                id: n.id,
                kind: n.kind,
                libraryKey: n.libraryKey,
                label: n.label,
                parameters: n.parameters,
            })),
            edges: state.graph.edges,
            errors: report.errors,
            warnings: report.warnings,
            globalErrors: report.globalErrors,
            ok: report.ok,
        };
    });
    console.log('DUMP:', JSON.stringify(dump, null, 2));

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
