import assert from 'node:assert/strict';
import { chromium } from 'playwright';

// Isolated local browser session: never modifies a user's browser workspace.
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).waitFor();
    await page.mouse.move(350, 100);
    await page.screenshot({ path: 'screenshots/features-after-overview.png' });
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect causal self-attention', exact: true }).click();
    await page.mouse.move(350, 100);
    await page.screenshot({ path: 'screenshots/features-after-inspector.png' });
    const fields = await page.locator('.df-inspector .df-param-input').evaluateAll((inputs) => inputs.filter(i => i.getBoundingClientRect().height).map(i => ({width:i.getBoundingClientRect().width, height:i.getBoundingClientRect().height})));
    assert.ok(fields.length >= 2);
    for (const field of fields) { assert.ok(field.width <= 264); assert.equal(field.height, 28); }
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await page.mouse.move(350, 100);
    await page.screenshot({ path: 'screenshots/features-after-transformer.png' });
    await page.getByRole('button', { name: 'Zoom In', exact: true }).click();
    await page.getByRole('button', { name: 'Zoom In', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect mlp fc', exact: true }).click();
    await page.mouse.move(350, 100);
    await page.screenshot({ path: 'screenshots/features-after-mlp.png' });
    const mlp = page.locator('.react-flow__node').filter({ has: page.getByRole('button', { name: 'Inspect mlp fc', exact: true }) });
    assert.equal(await mlp.locator('.df-tensor-grid rect').count(), 192);
    const projection = page.locator('.react-flow__node').filter({ has: page.getByRole('button', { name: 'Inspect mlp proj', exact: true }) });
    assert.equal(await projection.locator('.df-tensor-grid rect').count(), 48);
    assert.equal(await page.locator('.df-tensor-edge').count(), 0);
    assert.ok(await page.locator('.df-feature-stream path').count() > 0);
    const before = await mlp.boundingBox();
    const grip = await mlp.locator('.df-tensor-grid').boundingBox();
    await page.mouse.move(grip.x + grip.width / 2, grip.y + grip.height / 2);
    await page.mouse.down();
    await page.mouse.move(grip.x + grip.width / 2 - 60, grip.y + grip.height / 2, { steps: 20 });
    await page.mouse.up();
    const after = await mlp.boundingBox();
    assert.ok(after.x < before.x - 40, 'Feature grid itself remains draggable');
    assert.ok(Math.abs(after.width - before.width) < .1, 'Dragging preserves the matrix width');
    assert.deepEqual(errors, []);
    console.log('Visual checks passed: grouped node previews, compact inspector, feature streams, drag.');
    console.log(JSON.stringify({ fields, screenshots: ['features-after-overview.png', 'features-after-transformer.png', 'features-after-inspector.png', 'features-after-mlp.png'] }));
} finally {
    await browser.close();
}
