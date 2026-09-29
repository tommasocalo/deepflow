import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { levelTrigger } from './level-navigation-browser-helper.mjs';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/structure-scope';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    const measurements = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const main = page.locator('[data-graph-pane="main"]');
    const scope = ribbon.getByRole('switch', { name: 'Shared structure', exact: true });
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const fit = async () => { await ribbon.getByRole('button', { name: 'Fit active graph', exact: true }).click(); await page.waitForTimeout(300); };
    const graph = () => page.evaluate(() => window.__deepflow.architectureStore.getState().getViewGraph());
    const currentScope = () => page.evaluate(() => window.__deepflow.architectureStore.getState().getStructureScope());
    const openInstances = async (pane = main) => {
        await levelTrigger(pane).focus();
        const trigger = ribbon.locator('.df-ribbon-instance-button');
        if (await trigger.getAttribute('aria-expanded') !== 'true') await trigger.click();
    };
    const checked = async (value, pane = main) => {
        await openInstances(pane);
        assert.equal(await scope.getAttribute('aria-checked'), String(value));
    };
    const switchTo = async (shared, pane = main) => {
        await openInstances(pane);
        if (await scope.getAttribute('aria-checked') !== String(shared)) await scope.click();
    };
    const screenshot = async (name, pane = main) => {
        await settle();
        await openInstances(pane);
        const size = await ribbon.evaluate((element) => ({ viewport: innerWidth, width: element.clientWidth, content: element.scrollWidth }));
        measurements.push({ name, ...size });
        assert.ok(size.content <= size.width, `${name}: ribbon fits without horizontal overflow ${JSON.stringify(size)}`);
        const bounds = await scope.boundingBox();
        assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= size.viewport, `${name}: the scope switch remains visible`);
        await page.screenshot({ path: `${directory}/${name}.png` });
    };

    await ribbon.waitFor();
    assert.equal(await scope.count(), 0, 'The root graph has no instance switch');
    assert.equal(await ribbon.locator('.df-ribbon-scope').count(), 0);
    const groupOrder = await ribbon.locator('.df-ribbon-group > h2').allTextContents();
    assert.equal(groupOrder.indexOf('Edit scope'), -1);
    assert.equal(groupOrder.indexOf('History'), groupOrder.indexOf('View') + 1);
    await fit();
    await main.getByRole('button', { name: 'Inspect Transformer blocks', exact: true }).click();
    assert.equal(await scope.count(), 0, 'Selecting a repeat does not add a switch outside its body');
    await main.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    await fit();
    await checked(true);
    assert.equal(await ribbon.locator('.df-iteration-title').count(), 1, 'Switch is beside the repeat controls');
    assert.equal(await main.locator('.df-iteration-bar').count(), 0, 'Repeat controls do not reserve canvas height');
    await screenshot('01-open-transformer-shared-1440');
    await switchTo(false);
    await checked(false);
    await fit();
    assert.equal((await currentScope()).local, true);
    assert.equal(await page.getByText('Editing shared structure', { exact: true }).count(), 0);
    const original = await graph();
    const originalCount = Object.keys(original.nodes).length;
    await ribbon.getByRole('button', { name: 'Insert operations', exact: true }).click();
    await page.getByRole('textbox', { name: 'Search operations', exact: true }).fill('Scale');
    await page.getByRole('textbox', { name: 'Search operations', exact: true }).press('Enter');
    const edited = await graph();
    const added = Object.values(edited.nodes).find((node) => !original.nodes[node.id]);
    assert.equal(added.libraryKey, 'deepflow.op.Scale');
    assert.equal(Object.keys(edited.nodes).length, originalCount + 1);
    await switchTo(true);
    assert.equal(Object.keys((await graph()).nodes).length, originalCount, 'Shared restores the unmodified common structure');
    assert.equal((await currentScope()).hasSavedLocal, true, 'The local structure remains saved');
    await switchTo(false);
    assert.deepEqual(await graph(), edited, 'Reactivating This instance restores the exact edited graph');
    await openInstances();
    await ribbon.getByRole('button', { name: 'Next instance', exact: true }).click();
    await checked(true);
    assert.equal(await ribbon.getByRole('spinbutton', { name: 'Inspected instance number', exact: true }).inputValue(), '2');
    assert.equal((await graph()).nodes[added.id], undefined, 'A sibling repeat instance keeps the shared graph');
    await ribbon.getByRole('button', { name: 'Previous instance', exact: true }).click();
    await checked(false);
    assert.ok((await graph()).nodes[added.id], 'The first instance retains its private addition');
    await switchTo(true);
    await ribbon.getByRole('button', { name: 'Undo', exact: true }).click();
    await checked(false);
    assert.ok((await graph()).nodes[added.id], 'Undo restores the active local structure');
    await ribbon.getByRole('button', { name: 'Redo', exact: true }).click();
    await checked(true);
    assert.equal((await graph()).nodes[added.id], undefined);
    await switchTo(false);
    await fit();
    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 900 });
        await fit();
        await screenshot(`02-local-transformer-${width}`);
    }
    await page.setViewportSize({ width: 1440, height: 960 });
    await ribbon.getByRole('button', { name: 'Open to the side', exact: true }).click();
    const side = page.locator('[data-graph-pane="side"]');
    await side.waitFor();
    await checked(false, side);
    await ribbon.getByRole('button', { name: 'Next instance', exact: true }).click();
    await checked(true, side);
    assert.equal(await ribbon.getAttribute('data-active-pane'), 'side');
    await screenshot('03-side-pane-scope-1440', side);
    await checked(false, main);
    assert.equal(await ribbon.getByRole('spinbutton', { name: 'Inspected instance number', exact: true }).inputValue(), '1');
    await checked(true, side);
    assert.equal(await ribbon.getByRole('spinbutton', { name: 'Inspected instance number', exact: true }).inputValue(), '2');
    await levelTrigger(main).focus();
    await checked(false);
    assert.equal(await ribbon.getAttribute('data-active-pane'), 'main');
    await screenshot('03-independent-pane-scopes-1440');
    await side.getByRole('button', { name: 'Close side view', exact: true }).click();
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        state.openLevel([]);
        const embedding = Object.values(state.getViewGraph().nodes).find((node) => node.label === 'token embedding');
        state.enterSupernode(embedding.id);
    });
    assert.equal(await scope.count(), 0, 'An ordinary component has no instance switch');
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        state.openLevel([]);
        for (const label of ['Transformer blocks', 'causal self-attention', 'Attention heads']) {
            const current = window.__deepflow.architectureStore.getState();
            const node = Object.values(current.getViewGraph().nodes).find((candidate) => candidate.label === label);
            current.enterSupernode(node.id);
        }
    });
    await fit();
    await openInstances();
    assert.equal(await ribbon.locator('.df-iteration-title').count(), 1);
    await checked(true);
    await switchTo(false);
    await checked(false);
    await screenshot('04-parallel-instance-local-1440');
    assert.deepEqual(errors, []);
    await writeFile(`${directory}/measurements.json`, `${JSON.stringify(measurements, null, 2)}\n`);
    console.log(`Structure scope checks passed: repeat-only inline placement, per-instance isolation, non-destructive switching, Undo/Redo, independent pane controls, and 1440/1024/820 layout. Screenshots: ${directory}`);
} finally {
    await browser.close();
}
