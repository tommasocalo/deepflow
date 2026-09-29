import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { selectLevel } from './level-navigation-browser-helper.mjs';

const directory = process.env.SCREENSHOT_DIR ?? '/tmp/deepflow-insert-menus';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const components = page.getByRole('menu', { name: 'Insert components', exact: true });
    const selected = () => page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        return state.graph.nodes[state.selectedNodeIds[0]];
    });
    const nodeCount = () => page.evaluate(() => Object.keys(window.__deepflow.architectureStore.getState().graph.nodes).length);
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const openSelected = async () => {
        assert.equal(await page.locator('.df-inspector').count(), 0);
        const node = await selected();
        await page.locator(`.react-flow__node[data-id="${node.id}"]`).getByRole('button', { name: `Inspect ${node.label}`, exact: true }).click();
    };
    const addComponent = async (name) => {
        await ribbon.getByRole('button', { name: 'Browse components', exact: true }).click();
        await components.getByRole('menuitem', { name: `Add ${name}`, exact: true }).click();
    };
    await ribbon.waitFor();
    assert.equal(await page.locator('.df-library').count(), 0, 'There is no detached block palette');
    await ribbon.getByRole('button', { name: 'Browse components', exact: true }).click();
    for (const name of ['Linear', 'Embedding', 'Attention', 'Norm', 'Learned positions', 'Sinusoidal positions', 'Activation', 'Convolution', 'Pooling', 'Recurrent', 'Dropout']) {
        assert.equal(await components.getByRole('menuitem', { name: `Add ${name}`, exact: true }).count(), 1, `${name} remains available`);
    }
    assert.equal(await components.getByRole('menuitem', { name: 'Add MatMul', exact: true }).count(), 0, 'Math operations have their own menu');
    await page.screenshot({ path: `${directory}/01-components.png` });
    await page.setViewportSize({ width: 1024, height: 760 });
    const bounds = await page.locator('[data-ribbon-menu="components"]').boundingBox();
    assert.ok(bounds.y + bounds.height <= 760);
    await components.getByRole('menuitem').last().scrollIntoViewIfNeeded();
    assert.ok(await components.getByRole('menuitem').last().isVisible());
    await page.screenshot({ path: `${directory}/02-laptop-components.png` });
    await page.setViewportSize({ width: 1440, height: 960 });

    const search = page.getByRole('textbox', { name: 'Search components', exact: true });
    await search.fill('layer norm');
    assert.equal(await components.getByRole('menuitem').count(), 1);
    await search.press('ArrowDown');
    assert.equal(await components.getByRole('menuitem', { name: 'Add Norm', exact: true }).evaluate((element) => element === document.activeElement), true);
    const before = await nodeCount();
    await page.keyboard.press('Enter');
    assert.equal(await nodeCount(), before + 1);
    const norm = await selected();
    assert.equal(norm.kind, 'supernode');
    assert.equal(await page.evaluate((id) => window.__deepflow.templatesStore.getState().templates[id].standardKey, norm.templateId), 'layernorm');
    await page.getByRole('button', { name: 'Open Norm', exact: true }).click();
    assert.ok(await page.getByRole('button', { name: 'Inspect mean', exact: true }).isVisible(), 'Norm exposes its primitive graph');
    await selectLevel(page.locator('[data-graph-pane="main"]'), 'main');

    // Other normalization types remain editable layer modules.
    await addComponent('GroupNorm');
    await openSelected();
    const variant = page.getByRole('combobox', { name: 'Operation type', exact: true });
    await page.getByRole('textbox', { name: 'Block name', exact: true }).fill('Feature norm');
    await page.getByLabel('num_groups', { exact: true }).fill('8');
    await page.getByLabel('num_groups', { exact: true }).press('Tab');
    await page.getByLabel('num_channels', { exact: true }).fill('64');
    await page.getByLabel('num_channels', { exact: true }).press('Tab');
    assert.equal((await selected()).label, 'Feature norm');
    assert.equal((await selected()).parameters.num_groups, 8);
    for (const [family, key] of [
        ['Activation', 'torch.nn.GELU'], ['Convolution', 'torch.nn.ConvTranspose2d'],
        ['Pooling', 'torch.nn.AdaptiveAvgPool2d'], ['Recurrent', 'torch.nn.GRU'],
    ]) {
        await addComponent(family);
        await openSelected();
        await variant.selectOption(key);
        assert.equal((await selected()).libraryKey, key);
        assert.ok(await variant.evaluate((element) => {
            const style = getComputedStyle(element);
            const context = document.createElement('canvas').getContext('2d');
            context.font = `${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
            return context.measureText(element.selectedOptions[0].text).width + 24 <= element.clientWidth;
        }), `${family} variant label fits`);
    }
    await addComponent('Transformer');
    await openSelected();
    assert.equal((await selected()).libraryKey, 'torch.nn.TransformerEncoderLayer');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    const beforeDrag = await nodeCount();
    await ribbon.getByRole('button', { name: 'Browse components' }).click();
    await components.getByRole('menuitem', { name: 'Add Convolution', exact: true }).dragTo(page.locator('.react-flow__pane'), { targetPosition: { x: 800, y: 240 } });
    await settle();
    assert.equal(await nodeCount(), beforeDrag + 1);
    assert.equal(await components.count(), 0);

    await ribbon.getByRole('button', { name: 'Browse templates' }).click();
    const templates = page.getByRole('menu', { name: 'Insert templates', exact: true });
    assert.equal(await templates.getByRole('menuitem', { name: 'Add LayerNorm template', exact: true }).count(), 0, 'Standard catalog entries do not duplicate the project template list');
    await templates.getByRole('menuitem', { name: 'Add Transformer block template', exact: true }).click();
    assert.equal((await selected()).kind, 'supernode');
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    await ribbon.getByRole('button', { name: 'Browse templates' }).click();
    assert.equal(await templates.getByRole('menuitem', { name: 'Add Transformer block template', exact: true }).count(), 0, 'Recursive templates are unavailable');
    await page.getByRole('button', { name: 'Close templates menu', exact: true }).click();
    await selectLevel(page.locator('[data-graph-pane="main"]'), 'main');
    await ribbon.getByRole('button', { name: 'Browse templates' }).click();
    assert.ok(await templates.getByRole('menuitem', { name: 'Add Transformer block template', exact: true }).isEnabled());
    assert.deepEqual(errors, []);
    console.log('Insert menus passed: all editable standard components, separate operations/templates, laptop bounds, keyboard search, primitive definitions, remaining layer variants, drag/drop and template recursion protection.');
} finally {
    await browser.close();
}
