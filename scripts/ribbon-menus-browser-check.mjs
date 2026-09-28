import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/ribbon-menus';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.setDefaultTimeout(6000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    await ribbon.waitFor();

    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 900 });
        const triggers = [
            ['Browse components', 'components'],
            ['Insert operations', 'operations'],
            ['Browse templates', 'templates'],
        ];
        for (const [label, menu] of triggers) {
            const trigger = ribbon.getByRole('button', { name: label, exact: true });
            await trigger.click();
            const popover = ribbon.locator(`[data-ribbon-menu="${menu}"]`);
            const [buttonBox, menuBox] = await Promise.all([trigger.boundingBox(), popover.boundingBox()]);
            assert.ok(buttonBox && menuBox, `${label} is visible at ${width}px`);
            assert.ok(Math.abs(menuBox.y - buttonBox.y - buttonBox.height - 3) < 2, `${menu} opens below its own trigger at ${width}px`);
            assert.ok(menuBox.x >= 0 && menuBox.x + menuBox.width <= width, `${menu} stays on screen at ${width}px`);
            await page.screenshot({ path: `${directory}/${width}-${menu}.png` });
            if (menu === 'components') {
                const catalog = popover.getByRole('menu', { name: 'Insert components' });
                for (const name of ['Linear', 'Embedding', 'Norm']) {
                    assert.equal(await catalog.getByRole('menuitem', { name: `Add ${name}`, exact: true }).count(), 1, `${name} is available in the catalog`);
                    assert.equal(await ribbon.locator('.df-ribbon-row').getByRole('button', { name: `Add ${name}`, exact: true }).count(), 0, `${name} has no separate pinned shortcut`);
                }
                assert.equal(await ribbon.getByText('Pinned components').count(), 0);
            }
            await trigger.click();
        }

        const selectedComponent = await page.evaluate(() => {
            const state = window.__deepflow.architectureStore.getState();
            const component = Object.values(state.graph.nodes).find((node) => node.kind === 'repeat' || node.kind === 'supernode');
            if (component) state.selectOnly(component.id);
            return !!component;
        });
        assert.ok(selectedComponent, 'The example graph contains a component');
        const parametersTrigger = ribbon.getByRole('button', { name: 'Edit parameters', exact: true });
        await parametersTrigger.click();
        const dialog = ribbon.getByRole('dialog', { name: /parameters/ });
        const [buttonBox, dialogBox] = await Promise.all([parametersTrigger.boundingBox(), dialog.boundingBox()]);
        assert.ok(Math.abs(dialogBox.y - buttonBox.y - buttonBox.height - 3) < 2, `Parameters opens below its trigger at ${width}px`);
        assert.ok(dialogBox.x >= 0 && dialogBox.x + dialogBox.width <= width, `Parameters stays on screen at ${width}px`);
        const tabs = dialog.getByRole('tablist', { name: 'Parameter scope' });
        assert.equal(await tabs.getByRole('tab', { name: 'Component' }).getAttribute('aria-selected'), 'true');
        assert.ok((await dialog.getByRole('textbox', { name: 'Component name' }).inputValue()).length > 0);
        await page.screenshot({ path: `${directory}/${width}-parameters-component.png` });
        await tabs.getByRole('tab', { name: 'Global' }).click();
        assert.equal(await dialog.getAttribute('aria-label'), 'Global parameters');
        assert.equal(await tabs.getByRole('tab', { name: 'Global' }).getAttribute('aria-selected'), 'true');
        await page.screenshot({ path: `${directory}/${width}-parameters-global.png` });
        await dialog.getByRole('button', { name: 'Close parameters' }).click();
        assert.equal(await dialog.count(), 0);
    }
    assert.deepEqual(errors, []);
    console.log(`Ribbon menus and Parameters passed at 1440, 1024 and 820px. Screenshots: ${directory}`);
} finally {
    await browser.close();
}
