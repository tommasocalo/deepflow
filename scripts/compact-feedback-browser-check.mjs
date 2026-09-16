import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = 'screenshots/compact-feedback';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const settle = async () => {
        await page.mouse.move(350, 80);
        await page.evaluate(() => document.fonts.ready);
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    const selection = () => page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        return state.graph.nodes[state.selectedNodeIds[0]];
    });
    const inspector = page.locator('.df-inspector');
    const graphChecks = page.getByRole('button', { name: /^Graph checks:/ });

    await page.getByRole('button', { name: 'Add Pooling', exact: true }).click();
    const pool = await selection();
    assert.equal(await inspector.count(), 0, 'Click adds just the node');
    await page.getByRole('button', { name: 'Inspect MaxPool2d', exact: true }).click();
    const kernel = page.getByLabel('kernel_size', { exact: true });
    assert.equal(await kernel.getAttribute('aria-invalid'), 'true');
    assert.equal(await inspector.locator('.df-param-errors').textContent(), 'Required');
    assert.equal(await inspector.locator('.df-inspector-issue.is-error').count(), 0);
    assert.equal(await inspector.locator('.df-inspector-issue.is-warning').count(), 1);
    await settle();
    await page.screenshot({ path: `${directory}/02-inline-settings.png` });
    await inspector.screenshot({ path: `${directory}/03-pool-detail.png` });
    const poolSize = await inspector.boundingBox();
    assert.equal(poolSize.width, 168);
    assert.ok(poolSize.height < 210, 'Required field and warnings fit a short inspector');

    await page.getByRole('combobox', { name: 'Operation type', exact: true }).selectOption('torch.nn.AdaptiveAvgPool2d');
    await settle();
    await inspector.screenshot({ path: `${directory}/04-type-detail.png` });
    await page.getByRole('combobox', { name: 'Operation type', exact: true }).selectOption('torch.nn.MaxPool2d');
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('Downsample');

    // A second addition also closes the previously open inspector.
    await page.getByRole('button', { name: 'Add Convolution', exact: true }).click();
    assert.equal(await inspector.count(), 0);
    const conv = await selection();
    await page.getByRole('button', { name: 'Inspect Conv2d', exact: true }).click();
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('Feature extractor');
    assert.equal(await inspector.locator('.df-param-errors').count(), 2);
    assert.equal(await inspector.locator('.df-inspector-issue.is-error').count(), 0);
    await settle();
    await inspector.screenshot({ path: `${directory}/05-convolution-detail.png` });
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();

    await graphChecks.click();
    const issues = page.getByRole('region', { name: 'Graph issues', exact: true });
    const poolGroup = issues.locator(`[data-issue-node-id="${pool.id}"]`);
    const convGroup = issues.locator(`[data-issue-node-id="${conv.id}"]`);
    assert.equal(await issues.locator('.df-graph-issue-group').count(), 2);
    assert.equal(await poolGroup.locator('.df-graph-issue-heading').innerText(), 'Downsample');
    assert.equal(await convGroup.locator('.df-graph-issue-heading').innerText(), 'Feature extractor');
    assert.equal(await poolGroup.locator('.df-graph-issue-message').count(), 3);
    assert.equal(await convGroup.locator('.df-graph-issue-message').count(), 4);
    assert.ok((await issues.boundingBox()).width <= 224);
    await settle();
    await page.screenshot({ path: `${directory}/06-grouped-overview.png` });
    await issues.screenshot({ path: `${directory}/07-grouped-detail.png` });
    await poolGroup.hover();
    assert.ok(await page.locator(`.react-flow__node[data-id="${pool.id}"]`).evaluate((el) => el.classList.contains('deepflow-hover-issue')));
    await poolGroup.click();
    assert.equal(await page.getByRole('textbox', { name: 'Block name', exact: true }).inputValue(), 'Downsample');
    assert.equal(await issues.count(), 0);
    assert.equal(await page.locator('.deepflow-hover-issue').count(), 0);
    await kernel.fill('2');
    await kernel.press('Tab');
    assert.equal(await kernel.getAttribute('aria-invalid'), 'false');
    assert.equal(await inspector.locator('.df-param-errors').count(), 0);
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await graphChecks.click();
    await issues.locator(`[data-issue-node-id="${pool.id}"]`).focus();
    await page.keyboard.press('Escape');
    assert.equal(await issues.count(), 0);
    assert.ok(await graphChecks.evaluate((el) => el === document.activeElement));

    await page.setViewportSize({ width: 1050, height: 760 });
    await graphChecks.click();
    await settle();
    assert.ok(await issues.evaluate((el) => el.scrollWidth <= el.clientWidth));
    await page.screenshot({ path: `${directory}/08-laptop.png` });
    assert.deepEqual(errors, []);
    console.log('Compact feedback passed: closed-on-add, inline field errors, compact warnings, named issue groups, hover/click/keyboard navigation and laptop layout.');
} finally {
    await browser.close();
}
