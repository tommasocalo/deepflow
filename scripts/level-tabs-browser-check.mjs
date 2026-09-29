import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { levelEntry, levelTrigger, openLevelMenu, selectLevel } from './level-navigation-browser-helper.mjs';

const directory = 'screenshots/compact-level-navigation';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(6000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');

    const main = page.locator('[data-graph-pane="main"]');
    const open = main.getByRole('button', { name: 'Open Transformer blocks', exact: true });
    const trigger = levelTrigger(main);
    const settle = async () => {
        await page.evaluate(() => document.fonts.ready);
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };

    assert.equal(await main.locator('.df-graph-pane-header, .df-levels').count(), 0, 'The full-width level strip is gone');
    assert.match(await trigger.getAttribute('aria-label'), /Current level: main/);
    await open.click();
    await main.getByRole('button', { name: 'Inspect mlp fc', exact: true }).waitFor();
    assert.match(await trigger.getAttribute('aria-label'), /Transformer blocks/);
    await selectLevel(main, 'main');
    await open.waitFor();
    let menu = await openLevelMenu(main);
    assert.equal(await levelEntry(menu, 'Transformer blocks').count(), 1, 'Returning to main retains the open level');
    assert.equal(await levelEntry(menu, 'main').getAttribute('aria-current'), 'page');
    assert.equal(await levelEntry(menu, 'Transformer blocks').getAttribute('aria-current'), null);
    await settle();
    await page.screenshot({ path: `${directory}/01-main-menu-1440.png` });
    await menu.getByRole('button', { name: 'Close levels menu', exact: true }).click();
    await trigger.focus();
    await trigger.press('ArrowDown');
    menu = main.getByRole('dialog', { name: 'Open levels', exact: true });
    assert.equal(await levelEntry(menu, 'main').evaluate((element) => element === document.activeElement), true,
        'ArrowDown opens the menu and focuses its first level');
    await page.keyboard.press('Tab');
    assert.equal(await menu.isVisible(), true, 'Keyboard Tab can reach actions inside the level menu');
    await page.keyboard.press('Escape');
    assert.equal(await menu.count(), 0);
    assert.equal(await trigger.evaluate((element) => element === document.activeElement), true, 'Escape restores trigger focus');
    menu = await openLevelMenu(main);

    await levelEntry(menu, 'Transformer blocks').click();
    await main.getByRole('button', { name: 'Inspect mlp fc', exact: true }).click();
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('MLP projection');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await selectLevel(main, 'main');
    await page.getByRole('button', { name: 'Training', exact: true }).click();
    await page.getByRole('button', { name: 'Architecture', exact: true }).click();
    menu = await openLevelMenu(main);
    assert.equal(await levelEntry(menu, 'Transformer blocks').count(), 1, 'Changing workspaces preserves the open level');
    await page.setViewportSize({ width: 820, height: 900 });
    await settle();
    await page.screenshot({ path: `${directory}/02-main-menu-820.png` });
    await menu.getByRole('button', { name: 'Close levels menu', exact: true }).click();

    await page.reload();
    await open.waitFor();
    menu = await openLevelMenu(main);
    assert.equal(await levelEntry(menu, 'Transformer blocks').count(), 1, 'Open levels persist across reloads');
    await levelEntry(menu, 'Transformer blocks').click();
    await main.getByRole('button', { name: 'Inspect MLP projection', exact: true }).waitFor();
    await selectLevel(main, 'main');
    menu = await openLevelMenu(main);
    await menu.getByRole('button', { name: 'Close Transformer blocks', exact: true }).click();
    assert.equal(await levelEntry(menu, 'Transformer blocks').count(), 0);
    assert.equal(await levelEntry(menu, 'main').getAttribute('aria-current'), 'page', 'Closing an inactive level leaves the view alone');
    await menu.getByRole('button', { name: 'Close levels menu', exact: true }).click();

    await open.click();
    await selectLevel(main, 'main');
    await open.click();
    menu = await openLevelMenu(main);
    assert.equal(await levelEntry(menu, 'Transformer blocks').count(), 1, 'Reopening does not duplicate an existing level');
    await menu.getByRole('button', { name: 'Open Transformer blocks to the side', exact: true }).click();
    const side = page.locator('[data-graph-pane="side"]');
    await side.waitFor();
    assert.match(await levelTrigger(side).getAttribute('aria-label'), /Transformer blocks/, 'The side pane exposes its own current level');
    assert.match(await trigger.getAttribute('aria-label'), /main/, 'The main pane retains its own current level');
    menu = await openLevelMenu(side);
    const bounds = await Promise.all([side.boundingBox(), menu.boundingBox()]);
    assert.ok(bounds[1].x >= bounds[0].x && bounds[1].x + bounds[1].width <= bounds[0].x + bounds[0].width,
        'The level menu fits inside a narrow side pane');
    await settle();
    await page.screenshot({ path: `${directory}/03-split-menu-820.png` });
    const divider = page.getByRole('separator', { name: 'Resize graph panes', exact: true });
    await divider.focus();
    await divider.press('End');
    menu = await openLevelMenu(side);
    let paneBounds = await side.boundingBox();
    let menuBounds = await menu.boundingBox();
    assert.ok(menuBounds.x >= paneBounds.x && menuBounds.x + menuBounds.width <= paneBounds.x + paneBounds.width,
        'The menu fits when the side pane is 30% wide');
    const statusBounds = await side.locator('.df-graph-status-trigger').boundingBox();
    const triggerBounds = await levelTrigger(side).boundingBox();
    assert.ok(triggerBounds.y >= statusBounds.y + statusBounds.height || triggerBounds.x >= statusBounds.x + statusBounds.width,
        'The compact level control does not cover graph status in the narrow pane');
    await settle();
    await page.screenshot({ path: `${directory}/04-narrow-side-menu-820.png` });
    await divider.focus();
    await divider.press('Home');
    menu = await openLevelMenu(main);
    paneBounds = await main.boundingBox();
    menuBounds = await menu.boundingBox();
    assert.ok(menuBounds.x >= paneBounds.x && menuBounds.x + menuBounds.width <= paneBounds.x + paneBounds.width,
        'The menu fits when the main pane is 30% wide');
    await settle();
    await page.screenshot({ path: `${directory}/05-narrow-main-menu-820.png` });
    await side.getByRole('button', { name: 'Close side view', exact: true }).click();
    await selectLevel(main, 'Transformer blocks');
    await settle();
    await page.screenshot({ path: `${directory}/06-reopened-level-820.png` });

    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('button', { name: /^DANN / }).click();
    await main.getByRole('button', { name: 'Inspect feature extractor', exact: true }).waitFor();
    menu = await openLevelMenu(main);
    assert.equal(await menu.locator('.df-level-entry').count(), 1, 'Replacing the architecture clears old open levels');
    assert.equal(await levelEntry(menu, 'main').getAttribute('aria-current'), 'page');

    assert.deepEqual(errors, []);
    console.log('Compact level menu passed: persistence, editing, closing, split panes, narrow layout and architecture replacement.');
} finally {
    await browser.close();
}
