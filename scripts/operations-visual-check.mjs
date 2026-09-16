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
    await page.screenshot({ path: 'screenshots/compact-review/final-overview.png' });
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect causal self-attention', exact: true }).waitFor();
    await page.mouse.move(350, 100);
    await page.screenshot({ path: 'screenshots/compact-review/final-transformer.png' });
    assert.equal(await page.locator('.df-tensor-grid, .df-feature-stream, .df-activation').count(), 0);
    assert.equal(await page.getByText('Template parameters', { exact: true }).count(), 0);
    assert.ok(await page.locator('.df-block img').evaluateAll((icons) => icons.length && icons.every((image) => image.complete && image.naturalWidth > 0)), 'Generated operation assets load');
    await page.getByRole('button', { name: 'Inspect causal self-attention', exact: true }).click();
    await page.getByRole('button', { name: 'Close inspector', exact: true }).waitFor();
    await page.screenshot({ path: 'screenshots/compact-review/final-attention-inspector.png' });
    const fields = await page.locator('.df-inspector .df-param-input').evaluateAll((inputs) => inputs.filter(i => i.getBoundingClientRect().height).map(i => ({width:i.getBoundingClientRect().width, height:i.getBoundingClientRect().height})));
    assert.ok(fields.length >= 2);
    for (const field of fields) assert.ok(field.width <= 250 && field.height <= 30);
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await page.getByRole('button', { name: 'Zoom In', exact: true }).click();
    await page.getByRole('button', { name: 'Zoom In', exact: true }).click();
    const mlp = page.locator('.react-flow__node').filter({ has: page.getByRole('button', { name: 'Inspect mlp fc', exact: true }) });
    await mlp.locator('.df-block-name').click();
    await page.getByRole('button', { name: 'Close inspector', exact: true }).waitFor();
    // Wait for panel measurement and automatic placement to settle.
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const blockBox = await mlp.boundingBox();
    const panel = await page.locator('.df-inspector').boundingBox();
    const canvas = await page.locator('[data-architecture-canvas]').boundingBox();
    const overlaps = (a,b) => a.x < b.x+b.width && a.x+a.width > b.x && a.y < b.y+b.height && a.y+a.height > b.y;
    assert.ok(!overlaps(panel,blockBox), 'Floating inspector leaves the selected block visible');
    assert.ok(panel.x >= canvas.x && panel.y >= canvas.y && panel.x+panel.width <= canvas.x+canvas.width && panel.y+panel.height <= canvas.y+canvas.height, 'Inspector stays in the canvas');
    assert.equal(await mlp.locator('.df-block').evaluate(el => getComputedStyle(el).outlineStyle), 'none', 'No external selection outline');
    assert.ok(await mlp.locator('.df-block').evaluate(el => el.offsetWidth < 110), 'Short names use a content-sized card');
    await page.mouse.move(350, 100);
    await page.screenshot({ path: 'screenshots/compact-review/final-mlp-inspector.png' });
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    const grip = await mlp.locator('.df-block-name').boundingBox();
    await page.mouse.move(grip.x + grip.width / 2, grip.y + grip.height / 2);
    await page.mouse.down();
    await page.mouse.move(grip.x + grip.width / 2 - 70, grip.y + grip.height / 2, { steps: 20 });
    await page.mouse.up();
    const after = await mlp.boundingBox();
    assert.ok(after.x < blockBox.x - 50, 'Operation card remains draggable');
    assert.ok(Math.abs(after.width - blockBox.width) < .1, 'Dragging preserves node width');
    // A separate fresh page checks variant controls without changing screenshot fixtures.
    const variantPage = await browser.newPage({ viewport: { width: 1200, height: 900 } });
    variantPage.on('pageerror', (error) => errors.push(error.message));
    await variantPage.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await variantPage.getByRole('textbox', { name: 'Search blocks', exact: true }).fill('embedding');
    assert.equal(await variantPage.locator('.df-library-tile').count(), 1, 'Variant search resolves to its family');
    await variantPage.getByRole('button', { name: 'Add Linear', exact: true }).click();
    await variantPage.getByRole('button', { name: 'Inspect Linear', exact: true }).click();
    const variant = variantPage.getByRole('combobox', { name: 'Operation type', exact: true });
    await variant.selectOption('torch.nn.Embedding');
    assert.equal(await variant.inputValue(), 'torch.nn.Embedding');
    assert.ok(await variantPage.getByLabel('num_embeddings').count(), 'Embedding exposes its own parameters');
    await variant.selectOption('deepflow.nn.PositionalEmbedding');
    assert.equal(await variant.inputValue(), 'deepflow.nn.PositionalEmbedding');
    assert.ok(await variantPage.getByLabel('max_len').count(), 'Positional embedding exposes its own parameters');
    await variantPage.getByRole('textbox', { name: 'Search blocks', exact: true }).fill('causal');
    await variantPage.getByRole('button', { name: 'Add Attention', exact: true }).click();
    const addedAttention = await variantPage.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        return state.graph.nodes[state.selectedNodeIds[0]];
    });
    await variantPage.getByRole('button', { name: `Inspect ${addedAttention.label}`, exact: true }).click();
    await variant.selectOption('torch.nn.MultiheadAttention');
    assert.equal(await variant.inputValue(), 'torch.nn.MultiheadAttention');
    assert.deepEqual(errors, []);
    console.log('Visual checks passed: generated icons, dimensions, floating inspector, no outlines, node drag.');
    console.log(JSON.stringify({ fields, panel, blockBox }));
} finally {
    await browser.close();
}
