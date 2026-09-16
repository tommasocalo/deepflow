import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch();
const errors = [];
const settle = async (page) => {
    await page.evaluate(() => document.fonts.ready);
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
};
const overlaps = (a,b) => a.x < b.x+b.width && a.x+a.width > b.x && a.y < b.y+b.height && a.y+a.height > b.y;
async function verifyPanel(page, nodeName, pair) {
    await settle(page);
    const node = page.getByRole('button', { name: `Inspect ${nodeName}`, exact: true });
    const panel = page.getByRole('complementary', { name: `${nodeName} inspector`, exact: true });
    const panelBox = await panel.boundingBox();
    assert.ok(panelBox.height <= 180, 'Closed-options layer inspector fits in 180px');
    const shapeBoxes = await node.locator('.df-port-shape').evaluateAll(labels => labels.filter(label => getComputedStyle(label).visibility !== 'hidden').map(label => {const r=label.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height};}));
    for (const box of [await node.boundingBox(), ...shapeBoxes]) assert.ok(!overlaps(box, panelBox), 'Inspector avoids the node and its port dimensions');
    if (pair) {
        const a = await panel.getByRole('textbox', { name: pair[0], exact: true }).boundingBox();
        const b = await panel.getByRole('textbox', { name: pair[1], exact: true }).boundingBox();
        assert.ok(Math.abs(a.y-b.y) < 1 && a.x+a.width < b.x, 'Primary parameter fields share one row');
    }
    const icon = await node.locator('.df-operation-icon').getAttribute('src');
    assert.equal(await panel.locator('.df-operation-icon').getAttribute('src'), icon, 'Inspector uses the identical operation mark');
    return { node: await node.evaluate(el => ({ width: el.offsetWidth, height: el.offsetHeight })), panel: panelBox };
}
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Inspect positional embedding', exact: true }).click();
    const positional = await verifyPanel(page, 'positional embedding', ['max_len', 'embedding_dim']);
    await page.mouse.move(340,100);
    await page.screenshot({ path: 'screenshots/compact-review/03-positional-embedding.png' });

    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('button', { name: /^DANN / }).click();
    await page.getByRole('button', { name: /fit view/i }).click();
    const domain = page.getByRole('button', { name: 'Inspect domain head', exact: true });
    await domain.click();
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('Domain Head');
    const domainMetrics = await verifyPanel(page, 'Domain Head', ['in_features', 'out_features']);
    assert.equal(await page.getByRole('button', { name: 'Inspect Domain Head', exact: true }).locator('.df-block-name').textContent(), 'Domain Head');
    await page.mouse.move(340,100);
    await page.screenshot({ path: 'screenshots/compact-review/04-domain-head.png' });

    // Long names and expressions keep focus, fit their panel and preserve editing.
    const name = 'Domain Head with a deliberately long descriptive name';
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill(name);
    const longNode = page.getByRole('button', { name: `Inspect ${name}`, exact: true });
    await settle(page);
    assert.ok(await longNode.evaluate(el => el.offsetWidth <= 240));
    assert.ok(await longNode.locator('.df-block-name').evaluate(el => el.scrollWidth > el.clientWidth));
    assert.equal(await page.getByRole('textbox', { name: 'Block name', exact: true }).evaluate(el => el === document.activeElement), true);
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await longNode.focus();
    await page.keyboard.press('Enter');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).waitFor();
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('Domain Head');
    await page.locator('.df-inspector summary').click();
    await page.getByRole('checkbox', { name: 'bias', exact: true }).uncheck();
    assert.equal(await page.getByRole('checkbox', { name: 'bias', exact: true }).isChecked(), false);
    await page.locator('.df-inspector summary').click();
    const input = page.getByRole('textbox', { name: 'in_features', exact: true });
    await input.fill('128');
    await input.blur();
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect Domain Head', exact: true }).click();
    assert.equal(await input.inputValue(), '128');
    await page.setViewportSize({ width: 1050, height: 760 });
    await page.getByRole('button', { name: /fit view/i }).click();
    await settle(page);
    const narrowPanel = await page.locator('.df-inspector').boundingBox();
    assert.ok(narrowPanel.x >= 0 && narrowPanel.x+narrowPanel.width <= 1050);
    assert.ok(narrowPanel.y >= 134 && narrowPanel.y+narrowPanel.height <= 760);
    const palette = await page.locator('.df-library').boundingBox();
    assert.ok(!overlaps(narrowPanel, palette), 'Panel avoids the palette in the smaller viewport');
    await page.mouse.move(340,100);
    await page.screenshot({ path: 'screenshots/compact-review/05-smaller-viewport.png' });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({ positional, domainMetrics, narrowPanel }, null, 2));
    console.log('Compact layout review passed: paired fields, shared icons, port-label clearance, naming, keyboard, options, smaller viewport.');
} finally {
    await browser.close();
}
