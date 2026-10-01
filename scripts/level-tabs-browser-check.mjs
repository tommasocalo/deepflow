import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { activeLevelCard, levelCard, levelCards, selectLevel } from './level-navigation-browser-helper.mjs';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/restored-level-cards';
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
    const cards = levelCards(main);
    const settle = async () => {
        await page.evaluate(() => document.fonts.ready);
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    const assertHeaderFits = async (pane) => {
        const bounds = await Promise.all([pane.boundingBox(), pane.locator('.df-graph-pane-header').boundingBox()]);
        assert.ok(bounds[1].x >= bounds[0].x && bounds[1].x + bounds[1].width <= bounds[0].x + bounds[0].width + 1,
            'The persistent card strip fits inside its pane');
        assert.equal(await pane.locator('.df-iteration-panel, .df-iteration-bar').count(), 0,
            'Instance controls do not reserve a row beside the level cards or canvas');
    };

    await open.waitFor();
    assert.equal(await main.locator('.df-graph-pane-header').count(), 1, 'Level cards have their own persistent strip');
    assert.equal(await cards.getByRole('tab').count(), 1);
    assert.equal(await levelCard(main, 'main').getAttribute('aria-selected'), 'true');
    await open.click();
    await main.getByRole('button', { name: 'Inspect mlp fc', exact: true }).waitFor();
    assert.match(await activeLevelCard(main).innerText(), /Transformer blocks/);
    await selectLevel(main, 'main');
    await open.waitFor();
    assert.equal(await levelCard(main, 'Transformer blocks').count(), 1, 'Returning to main retains the open card');
    assert.equal(await levelCard(main, 'main').getAttribute('aria-selected'), 'true');
    assert.equal(await levelCard(main, 'Transformer blocks').getAttribute('aria-selected'), 'false');
    await assertHeaderFits(main);
    await settle();
    await page.screenshot({ path: `${directory}/01-main-cards-1440.png` });

    await activeLevelCard(main).focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await levelCard(main, 'Transformer blocks').getAttribute('aria-selected'), 'true', 'Arrow keys navigate the cards');
    assert.equal(await activeLevelCard(main).evaluate((element) => element === document.activeElement), true);
    await page.keyboard.press('Home');
    assert.equal(await levelCard(main, 'main').getAttribute('aria-selected'), 'true', 'Home selects the root card');
    await page.keyboard.press('End');
    await main.getByRole('button', { name: 'Inspect mlp fc', exact: true }).click();
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('MLP projection');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await selectLevel(main, 'main');
    await page.getByRole('button', { name: 'Training', exact: true }).click();
    await page.getByRole('button', { name: 'Architecture', exact: true }).click();
    assert.equal(await levelCard(main, 'Transformer blocks').count(), 1, 'Changing workspaces preserves the open card');
    await page.setViewportSize({ width: 820, height: 900 });
    await assertHeaderFits(main);
    await settle();
    await page.screenshot({ path: `${directory}/02-main-cards-820.png` });

    await page.reload();
    await open.waitFor();
    assert.equal(await levelCard(main, 'Transformer blocks').count(), 1, 'Open cards persist across reloads');
    await selectLevel(main, 'Transformer blocks');
    await main.getByRole('button', { name: 'Inspect MLP projection', exact: true }).waitFor();
    await selectLevel(main, 'main');
    await cards.getByRole('button', { name: 'Close Transformer blocks', exact: true }).click();
    assert.equal(await levelCard(main, 'Transformer blocks').count(), 0);
    assert.equal(await levelCard(main, 'main').getAttribute('aria-selected'), 'true', 'Closing an inactive card leaves the current view alone');

    await open.click();
    await selectLevel(main, 'main');
    await open.click();
    assert.equal(await levelCard(main, 'Transformer blocks').count(), 1, 'Reopening does not duplicate a card');
    await selectLevel(main, 'main');
    await levelCard(main, 'Transformer blocks').click({ button: 'right' });
    await page.getByRole('menuitem', { name: /Open to the side/ }).click();
    const side = page.locator('[data-graph-pane="side"]');
    await side.getByRole('button', { name: 'Inspect MLP projection', exact: true }).waitFor();
    assert.match(await activeLevelCard(side).innerText(), /Transformer blocks/, 'The side pane has its own active card');
    assert.match(await activeLevelCard(main).innerText(), /^main$/, 'The main pane retains its active root card');
    await assertHeaderFits(main);
    await assertHeaderFits(side);
    await settle();
    await page.screenshot({ path: `${directory}/03-split-cards-820.png` });
    const divider = page.getByRole('separator', { name: 'Resize graph panes', exact: true });
    await divider.focus();
    await divider.press('End');
    await assertHeaderFits(side);
    await selectLevel(side, 'main');
    assert.match(await activeLevelCard(main).innerText(), /^main$/, 'Side navigation keeps the main card selected');
    assert.equal(await levelCard(side, 'Transformer blocks').count(), 1, 'The side pane also retains inactive cards');
    await selectLevel(side, 'Transformer blocks');
    await settle();
    await page.screenshot({ path: `${directory}/04-narrow-side-cards-820.png` });
    await divider.focus();
    await divider.press('Home');
    await assertHeaderFits(main);
    await side.getByRole('button', { name: 'Close side view', exact: true }).click();
    await selectLevel(main, 'Transformer blocks');
    await settle();
    await page.screenshot({ path: `${directory}/05-reopened-level-820.png` });

    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('button', { name: /^DANN / }).click();
    await main.getByRole('button', { name: 'Inspect feature extractor', exact: true }).waitFor();
    assert.equal(await cards.getByRole('tab').count(), 1, 'Replacing the architecture clears old open cards');
    assert.equal(await levelCard(main, 'main').getAttribute('aria-selected'), 'true');

    assert.deepEqual(errors, []);
    console.log('Level cards passed: persistence, keyboard navigation, editing, closing, independent panes, narrow layout and architecture replacement.');
} finally {
    await browser.close();
}
