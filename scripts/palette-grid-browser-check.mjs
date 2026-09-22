import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? '/tmp/deepflow-palette-grid';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const library = page.getByRole('dialog', { name: 'Block library', exact: true });
    const add = (name) => page.getByRole('button', { name: `Add ${name}`, exact: true });
    const settle = async () => {
        await page.evaluate(() => document.fonts.ready);
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    const nodeCount = () => page.evaluate(() => Object.keys(window.__deepflow.architectureStore.getState().graph.nodes).length);
    const selected = () => page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        return state.graph.nodes[state.selectedNodeIds[0]];
    });
    const openSelected = async () => {
        assert.equal(await page.locator('.df-inspector').count(), 0, 'Adding only places the node');
        const node = await selected();
        await page.locator(`.react-flow__node[data-id="${node.id}"]`).getByRole('button', { name: `Inspect ${node.label}`, exact: true }).click();
    };
    await add('Normalization').waitFor();
    await settle();
    for (const family of ['Linear', 'Attention', 'Normalization', 'Activation', 'Convolution', 'Pooling', 'Recurrent']) {
        assert.ok(await add(family).isVisible(), `${family} is available directly from the ribbon`);
    }
    assert.equal(await page.locator('.df-library').count(), 0, 'The graph starts without a floating palette');
    await page.screenshot({ path: `${directory}/02-ribbon-overview.png` });
    await ribbon.getByRole('button', { name: 'Search all blocks' }).click();
    assert.equal(await page.locator('.df-library-category').count(), 0);
    assert.ok(await library.locator('.df-library-grid').evaluateAll((grids) => grids.every((element) => getComputedStyle(element).gridTemplateColumns.split(' ').length === 5)));
    assert.ok(await library.locator('.df-library-icon').evaluateAll((icons) => icons.filter((element) => element.tagName === 'IMG').every((element) => element.complete && element.naturalWidth > 0)));
    assert.ok(await library.locator('.df-library-name').evaluateAll((labels) => labels.every((element) => element.scrollWidth <= element.clientWidth)));
    await library.screenshot({ path: `${directory}/03-library-detail.png` });

    // The transient catalog stays within laptop bounds and can scroll to every entry.
    await page.setViewportSize({ width: 1050, height: 760 });
    await settle();
    const libraryBounds = await library.boundingBox();
    assert.ok(libraryBounds.y + libraryBounds.height <= 760);
    await library.locator('.df-library-tile').last().scrollIntoViewIfNeeded();
    assert.ok(await library.locator('.df-library-tile').last().isVisible());
    await page.screenshot({ path: `${directory}/06-laptop.png` });
    await page.setViewportSize({ width: 1440, height: 960 });

    // Search by a variant, then choose that variant on the actual node.
    const search = page.getByRole('textbox', { name: 'Search blocks', exact: true });
    await search.fill('group norm');
    assert.equal(await library.locator('.df-library-tile').count(), 1);
    await search.press('ArrowDown');
    const result = library.getByRole('button', { name: 'Add Normalization', exact: true });
    assert.ok(await result.evaluate((element) => element === document.activeElement));
    const before = await nodeCount();
    await page.keyboard.press('Enter');
    assert.equal(await nodeCount(), before + 1);
    assert.equal(await library.count(), 0, 'Insertion dismisses the library');
    await openSelected();
    const variant = page.getByRole('combobox', { name: 'Operation type', exact: true });
    const name = page.getByRole('textbox', { name: 'Block name', exact: true });
    await name.fill('Feature norm');
    await variant.selectOption('torch.nn.GroupNorm');
    assert.equal((await selected()).label, 'Feature norm');
    assert.equal((await selected()).libraryKey, 'torch.nn.GroupNorm');
    await page.getByLabel('num_groups', { exact: true }).fill('8');
    await page.getByLabel('num_groups', { exact: true }).press('Tab');
    await page.getByLabel('num_channels', { exact: true }).fill('64');
    await page.getByLabel('num_channels', { exact: true }).press('Tab');
    await page.mouse.move(350, 80);
    await settle();
    await page.screenshot({ path: `${directory}/04-normalization-settings.png` });

    for (const [family, key] of [
        ['Activation', 'torch.nn.GELU'],
        ['Convolution', 'torch.nn.ConvTranspose2d'],
        ['Pooling', 'torch.nn.AdaptiveAvgPool2d'],
        ['Recurrent', 'torch.nn.GRU'],
    ]) {
        await add(family).click();
        await openSelected();
        await variant.selectOption(key);
        assert.equal((await selected()).libraryKey, key);
        assert.ok(await variant.evaluate((element) => {
            const style = getComputedStyle(element);
            const context = document.createElement('canvas').getContext('2d');
            context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            return context.measureText(element.selectedOptions[0].text).width + 24 <= element.clientWidth;
        }), `${family} variant name fits the compact inspector`);
        if (family === 'Pooling') {
            await page.mouse.move(350, 80);
            await settle();
            await page.screenshot({ path: `${directory}/05-pooling-settings.png` });
        }
    }

    // The full encoder block remains distinct from the attention family.
    await ribbon.getByRole('button', { name: 'Search all blocks' }).click();
    await search.fill('Transformer');
    await library.getByRole('button', { name: 'Add Transformer', exact: true }).click();
    await openSelected();
    assert.equal((await selected()).libraryKey, 'torch.nn.TransformerEncoderLayer');
    assert.equal(await variant.count(), 0);
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();

    // Exercise the real HTML drag payload and canvas drop handler.
    const beforeDrag = await nodeCount();
    await add('Convolution').dragTo(page.locator('.react-flow__pane'), { targetPosition: { x: 250, y: 200 } });
    assert.equal(await nodeCount(), beforeDrag + 1);
    assert.equal(await page.locator('.df-inspector').count(), 0, 'Dropping only places the node');
    assert.ok(await page.evaluate(() => Object.values(window.__deepflow.architectureStore.getState().graph.nodes).some((node) => node.libraryKey === 'torch.nn.Conv2d')));

    // Only usable templates are offered at each level.
    await ribbon.getByRole('button', { name: 'Browse templates' }).click();
    const template = page.getByRole('button', { name: 'Add Transformer block template', exact: true });
    await template.click();
    assert.equal((await selected()).kind, 'supernode');
    assert.equal(await page.locator('.df-inspector').count(), 0);
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    await ribbon.getByRole('button', { name: 'Browse templates' }).click();
    assert.equal(await template.count(), 0, 'A template cannot contain itself, so its library entry is hidden');
    await library.getByRole('button', { name: 'Close block library' }).click();
    await page.getByRole('tab', { name: 'main', exact: true }).click();
    await ribbon.getByRole('button', { name: 'Browse templates' }).click();
    assert.ok(await template.isEnabled());
    assert.deepEqual(errors, []);
    console.log('Library checks passed: direct family tiles, five-column popup catalog, laptop bounds/scrolling, search/keyboard, variants, readable settings, drag/drop and recursion-safe templates.');
} finally {
    await browser.close();
}
