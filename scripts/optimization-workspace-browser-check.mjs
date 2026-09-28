import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/optimization-workspace';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.setDefaultTimeout(15000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.parameters.some((item) => item.name === 'd_model'));
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 900 });
        if (width !== 1440) {
            await page.reload();
            await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.parameters.some((item) => item.name === 'd_model'));
        }
        await ribbon.getByRole('button', { name: 'Explore architecture settings' }).click();
        const dialog = page.getByRole('dialog', { name: 'Explore configurations' });
        await dialog.waitFor();
        const box = await dialog.boundingBox();
        assert.ok(box, `dialog visible at ${width}px`);
        assert.ok(box.x >= 0 && box.x + box.width <= width, `dialog stays within ${width}px`);
        assert.ok(box.width > Math.min(width - 80, 700), `dialog is a workspace rather than a small popover at ${width}px`);
        assert.match(await dialog.innerText(), /3 combinations/);
        await page.screenshot({ path: `${directory}/${width}-setup.png` });
        if (width === 1440) {
            await dialog.getByRole('button', { name: 'Add parameter' }).click();
            await dialog.getByRole('combobox', { name: 'Search parameter 2' }).selectOption('n_heads');
            await dialog.getByRole('group', { name: 'Values mode for n_heads' }).getByRole('button', { name: 'List' }).click();
            await dialog.getByRole('textbox', { name: 'n_heads values' }).fill('5, 8, 12');
            assert.match(await dialog.innerText(), /9 combinations/);
            await dialog.getByRole('button', { name: 'Compare configurations' }).click();
            await page.waitForFunction(() => document.querySelectorAll('.df-opt-candidate').length === 9);
            await dialog.getByText('Structural comparison').waitFor();
            assert.equal(await dialog.locator('.df-opt-candidate').count(), 9);
            const invalid = dialog.locator('.df-opt-candidate').filter({ hasText: '5' }).first();
            await invalid.click();
            assert.match(await dialog.getByRole('region', { name: 'Selected candidate details' }).innerText(), /Shape issues/);
            await page.screenshot({ path: `${directory}/${width}-results.png` });
            await dialog.locator('.df-opt-candidate.is-best').click();
            await dialog.getByRole('button', { name: 'Apply selected' }).click();
            const changed = await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.parameters.filter((item) => ['d_model', 'n_heads'].includes(item.name)).map((item) => [item.name, item.default]));
            assert.deepEqual(changed, [['d_model', 384], ['n_heads', 12]]);
            await dialog.getByRole('button', { name: 'Close optimization' }).click();
            await ribbon.getByRole('button', { name: 'Undo' }).click();
            const restored = await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.parameters.filter((item) => ['d_model', 'n_heads'].includes(item.name)).map((item) => [item.name, item.default]));
            assert.deepEqual(restored, [['d_model', 768], ['n_heads', 12]]);
        } else {
            await dialog.getByRole('button', { name: 'Compare configurations' }).click();
            await page.waitForFunction(() => document.querySelectorAll('.df-opt-candidate').length === 3);
            await page.screenshot({ path: `${directory}/${width}-results.png` });
            if (width === 1024) {
                await dialog.getByRole('button', { name: /Use budget/ }).click();
                await dialog.getByRole('spinbutton', { name: 'Parameter limit in millions' }).fill('200');
                assert.match(await dialog.locator('.df-opt-best').innerText(), /163\.04M/);
                await dialog.getByRole('spinbutton', { name: 'Parameter limit in millions' }).fill('70');
                assert.match(await dialog.locator('.df-opt-best').innerText(), /60\.28M/);
                await page.screenshot({ path: `${directory}/${width}-budget.png` });
                await dialog.getByRole('spinbutton', { name: 'Parameter limit in millions' }).fill('');
                assert.equal(await dialog.getByRole('button', { name: 'Apply selected' }).isDisabled(), true,
                    'A missing required budget cannot apply a candidate');
            }
            if (width === 820) {
                await dialog.getByRole('button', { name: 'Add parameter' }).click();
                await dialog.getByRole('combobox', { name: 'Search parameter 2' }).selectOption('n_heads');
                await dialog.getByRole('group', { name: 'Values mode for n_heads' }).getByRole('button', { name: 'List' }).click();
                await dialog.getByRole('textbox', { name: 'n_heads values' }).fill('5, 8, 12');
                await dialog.getByRole('button', { name: 'Compare again' }).click();
                await page.waitForFunction(() => document.querySelectorAll('.df-opt-candidate').length === 9);
                assert.ok(await dialog.getByText('Shape issue', { exact: true }).first().isVisible());
                await page.screenshot({ path: `${directory}/${width}-comparison.png` });
                await dialog.getByRole('button', { name: 'Remove search parameter 2' }).click();
                await dialog.getByRole('spinbutton', { name: 'd_model range start' }).fill('384');
                await dialog.getByRole('spinbutton', { name: 'd_model range end' }).fill('1136');
                await dialog.getByRole('spinbutton', { name: 'd_model range step' }).fill('16');
                assert.match(await dialog.innerText(), /48 combinations/);
                await dialog.getByRole('button', { name: 'Compare again' }).click();
                await dialog.getByRole('progressbar', { name: 'Configurations checked' }).waitFor();
                await page.keyboard.press('Escape');
                await dialog.waitFor({ state: 'detached' });
                await ribbon.getByRole('button', { name: 'Explore architecture settings' }).click();
                await dialog.getByRole('button', { name: 'Compare again' }).waitFor();
                assert.equal(await dialog.getByRole('button', { name: 'Stop' }).count(), 0, 'Escape resets the running state');
            }
            await dialog.getByRole('button', { name: 'Close optimization' }).click();
        }
    }
    assert.deepEqual(errors, []);
    console.log(`Optimization workspace passed at 1440, 1024, 820 px; screenshots: ${directory}`);
} finally {
    await browser.close();
}
