import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = 'screenshots/palette-grid';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const add = (name) => page.getByRole('button', { name: `Add ${name}`, exact: true });
    const settle = async () => {
        await page.evaluate(() => document.fonts.ready);
        await page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    };
    const nodeCount = () => page.evaluate(() => Object.keys(window.__deepflow.architectureStore.getState().graph.nodes).length);
    const selected = () => page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        return state.graph.nodes[state.expandedNodeId];
    });
    await add('Normalization').waitFor();
    await settle();
    for (const family of ['Linear', 'Attention', 'Normalization', 'Activation', 'Convolution', 'Pooling', 'Recurrent']) {
        assert.ok(await add(family).isVisible(), `${family} is available without expanding a category`);
    }
    assert.equal(await page.locator('.df-library-category').count(), 0);
    assert.ok(await page.locator('.df-library-grid').evaluateAll((grids) => grids.every((el) => getComputedStyle(el).gridTemplateColumns.split(' ').length === 4)));
    assert.ok(await page.locator('.df-library-icon').evaluateAll((icons) => icons.filter((el) => el.tagName === 'IMG').every((el) => el.complete && el.naturalWidth > 0)));
    assert.ok(await page.locator('.df-library-name').evaluateAll((labels) => labels.every((el) => el.scrollWidth <= el.clientWidth)));
    await page.screenshot({ path: `${directory}/02-grid-overview.png` });
    await page.locator('.df-library').screenshot({ path: `${directory}/03-grid-detail.png` });

    // The complete default catalog and reusable template also fit a laptop viewport.
    await page.setViewportSize({ width: 1050, height: 760 });
    await settle();
    assert.ok(await page.locator('.df-library-scroll').evaluate((el) => el.scrollHeight <= el.clientHeight + 1));
    await page.screenshot({ path: `${directory}/06-laptop.png` });
    await page.setViewportSize({ width: 1440, height: 960 });

    // Search by a variant, then choose that variant on the actual node.
    const search = page.getByRole('textbox', { name: 'Search blocks', exact: true });
    await search.fill('group norm');
    assert.equal(await page.locator('.df-library-tile').count(), 1);
    await search.press('ArrowDown');
    assert.ok(await add('Normalization').evaluate((el) => el === document.activeElement));
    const before = await nodeCount();
    await page.keyboard.press('Enter');
    assert.equal(await nodeCount(), before + 1);
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
    await search.fill('');
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
        await variant.selectOption(key);
        assert.equal((await selected()).libraryKey, key);
        // Selected variant names must fit alongside the native select arrow.
        assert.ok(await variant.evaluate((el) => {
            const style = getComputedStyle(el);
            const context = document.createElement('canvas').getContext('2d');
            context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            return context.measureText(el.selectedOptions[0].text).width + 24 <= el.clientWidth;
        }), `${family} variant name fits the compact inspector`);
        if (family === 'Pooling') {
            await page.mouse.move(350, 80);
            await settle();
            await page.screenshot({ path: `${directory}/05-pooling-settings.png` });
        }
    }

    // The full encoder block remains distinct from the attention family.
    await add('Transformer').click();
    assert.equal((await selected()).libraryKey, 'torch.nn.TransformerEncoderLayer');
    assert.equal(await variant.count(), 0);
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();

    // Exercise the real HTML drag payload and canvas drop handler.
    const beforeDrag = await nodeCount();
    await add('Convolution').dragTo(page.locator('.react-flow__pane'), { targetPosition: { x: 250, y: 200 } });
    assert.equal(await nodeCount(), beforeDrag + 1);
    assert.ok(await page.evaluate(() => Object.values(window.__deepflow.architectureStore.getState().graph.nodes).some((node) => node.libraryKey === 'torch.nn.Conv2d')));

    // Templates are immediately available, including the recursive-template guard.
    const template = page.getByRole('button', { name: 'Add Transformer block template', exact: true });
    await template.click();
    assert.equal((await selected()).kind, 'supernode');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    assert.ok(await template.isDisabled(), 'A template cannot contain itself');
    await page.getByRole('tab', { name: 'main', exact: true }).click();
    assert.ok(await template.isEnabled());
    assert.deepEqual(errors, []);
    console.log('Palette checks passed: complete four-column catalog, laptop fit, search/keyboard, all new variants, readable settings, drag/drop and reusable templates.');
} finally {
    await browser.close();
}
