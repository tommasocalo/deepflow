import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');

    const main = page.getByRole('tab', { name: 'main', exact: true });
    const transformer = page.getByRole('tab', { name: /^Transformer blocks/ });
    const open = page.getByRole('button', { name: 'Open Transformer blocks', exact: true });
    const close = page.getByRole('button', { name: 'Close Transformer blocks', exact: true });
    const settle = async () => {
        await page.evaluate(() => document.fonts.ready);
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };

    await open.click();
    await page.getByRole('button', { name: 'Inspect mlp fc', exact: true }).waitFor();
    await main.click();
    await open.waitFor();
    assert.equal(await transformer.count(), 1, 'Returning to main keeps the supernode card');
    assert.equal(await main.getAttribute('aria-selected'), 'true');
    assert.equal(await transformer.getAttribute('aria-selected'), 'false');
    assert.equal(await page.locator('.df-levels-hint').count(), 0);
    await page.mouse.move(340, 80);
    await settle();
    await page.screenshot({ path: 'screenshots/persistent-level-tabs/01-main-with-open-supernode.png' });
    await page.locator('.df-levels').screenshot({ path: 'screenshots/persistent-level-tabs/02-open-cards.png' });

    await transformer.click();
    await page.getByRole('button', { name: 'Inspect mlp fc', exact: true }).click();
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('MLP projection');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await main.click();
    await page.getByRole('button', { name: 'Training', exact: true }).click();
    await page.getByRole('button', { name: 'Architecture', exact: true }).click();
    assert.equal(await transformer.count(), 1, 'Changing workspaces keeps open cards');

    await page.reload();
    await open.waitFor();
    assert.equal(await transformer.count(), 1, 'Open cards survive reload while main is active');
    await transformer.click();
    await page.getByRole('button', { name: 'Inspect MLP projection', exact: true }).waitFor();
    await main.click();
    await close.click();
    assert.equal(await transformer.count(), 0);
    assert.equal(await main.getAttribute('aria-selected'), 'true', 'Closing an inactive card does not navigate');
    await open.waitFor();

    await open.click();
    await main.click();
    await open.click();
    assert.equal(await transformer.count(), 1, 'Reopening focuses the existing card without duplicates');
    await close.click();
    assert.equal(await page.getByRole('tab').count(), 1);
    assert.equal(await main.getAttribute('aria-selected'), 'true');
    await open.click();
    await page.getByRole('button', { name: 'Inspect MLP projection', exact: true }).waitFor();
    await page.mouse.move(340, 80);
    await settle();
    await page.screenshot({ path: 'screenshots/persistent-level-tabs/03-reopened-supernode.png' });

    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('button', { name: /^DANN / }).click();
    await page.getByRole('button', { name: 'Inspect feature extractor', exact: true }).waitFor();
    assert.equal(await page.getByRole('tab').count(), 1, 'Replacing the architecture clears its old cards');
    assert.equal(await main.getAttribute('aria-selected'), 'true');

    assert.deepEqual(errors, []);
    console.log('Persistent tabs passed: main round trip, workspace switch, reload, closing active/inactive cards, no duplicates, edits preserved, architecture replacement.');
} finally {
    await browser.close();
}
