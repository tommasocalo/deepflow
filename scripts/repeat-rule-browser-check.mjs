import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/repeat-rule';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.evaluate(() => {
        const store = window.__deepflow.architectureStore.getState();
        const repeat = Object.values(store.graph.nodes).find((node) => node.label === 'Transformer blocks');
        if (!repeat) throw new Error('Default Transformer blocks repeat is missing');
        store.openLevel([repeat.id]);
    });
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    await ribbon.getByRole('button', { name: 'Inspect Block 1 of 12', exact: true }).click();
    assert.equal(await page.locator('.df-iteration-bar').count(), 0, 'Repeat inspection reserves no row above the canvas');
    const ruleButton = page.getByRole('button', { name: /Select by rule/ });
    await ruleButton.waitFor();
    await ruleButton.click();
    const input = page.getByRole('textbox', { name: 'Instance selection rule' });
    await input.fill('position % 2 == 0');
    await page.getByText('6 of 12 in preview').waitFor();
    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 900 });
        const panel = page.getByRole('group', { name: 'Select instances by rule' });
        const bounds = await panel.boundingBox();
        assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= width, `Rule panel fits at ${width}px: ${JSON.stringify(bounds)}`);
        await page.screenshot({ path: `${directory}/01-rule-preview-${width}.png` });
    }
    await page.getByRole('group', { name: 'Select instances by rule' }).getByRole('button', { name: 'Select', exact: true }).click();
    assert.equal(await page.getByRole('combobox', { name: 'Apply parameter changes to' }).inputValue(), 'selected');
    const selected = await page.evaluate(() => {
        const store = window.__deepflow.architectureStore.getState();
        return store.repeatSelections[JSON.stringify(store.currentPath)]?.indices;
    });
    assert.deepEqual(selected, [1, 3, 5, 7, 9, 11]);
    assert.equal(await page.locator('.df-iteration-strip button.is-selected').count(), 6);
    const menu = page.getByRole('dialog', { name: 'Transformer blocks instances', exact: true });
    const bounds = await menu.boundingBox();
    assert.ok(bounds.x >= 0 && bounds.x + bounds.width <= 820, 'Instance controls fit at 820px');
    await page.screenshot({ path: `${directory}/02-selected-820.png` });
    assert.deepEqual(errors, []);
    console.log(`Repeat rule menu, six selected instances and 1440/1024/820 screenshots passed: ${directory}`);
} finally {
    await browser.close();
}
