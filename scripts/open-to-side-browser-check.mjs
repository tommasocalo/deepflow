import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = 'screenshots/open-to-side';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const main = page.locator('[data-graph-pane="main"]');
    const side = page.locator('[data-graph-pane="side"]');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const root = () => page.evaluate(() => window.__deepflow.architectureStore.getState().graph);
    const template = () => page.evaluate(() => {
        const repeat = Object.values(window.__deepflow.architectureStore.getState().graph.nodes).find((node) => node.kind === 'repeat');
        return window.__deepflow.templatesStore.getState().templates[repeat.templateId].graph;
    });
    const focusedPane = () => page.evaluate(() => document.activeElement.closest('[data-graph-pane]')?.dataset.graphPane);
    const center = async (locator) => {
        const box = await locator.boundingBox();
        assert.ok(box);
        return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    };

    await main.locator('.df-repeat-face').click({ button: 'right' });
    await page.screenshot({ path: `${directory}/01-node-menu.png` });
    await page.getByRole('menuitem', { name: /Open to the side/ }).click();
    await side.getByRole('button', { name: 'Inspect mlp fc', exact: true }).waitFor();
    await settle();
    assert.equal(await main.locator('.react-flow__node').count(), 9);
    assert.equal(await side.locator('.react-flow__node').count(), 11);
    assert.equal(await focusedPane(), 'side', 'The opened pane receives keyboard focus');
    assert.deepEqual(await page.evaluate(() => window.__deepflow.architectureStore.getState().currentPath), []);
    await page.screenshot({ path: `${directory}/02-two-panes.png` });

    // Selection/inspector state is private, while edits use shared templates.
    await main.locator('.df-repeat-face').click();
    const repeat = Object.values((await root()).nodes).find((node) => node.kind === 'repeat');
    await side.getByRole('button', { name: 'Inspect mlp fc', exact: true }).click();
    assert.equal(await main.locator('.df-inspector').count(), 1);
    await side.getByRole('textbox', { name: 'Block name', exact: true }).fill('mlp expansion');
    assert.ok(Object.values((await template()).nodes).some((node) => node.label === 'mlp expansion'));
    assert.deepEqual(await page.evaluate(() => window.__deepflow.architectureStore.getState().selectedNodeIds), [repeat.id]);
    await page.screenshot({ path: `${directory}/03-independent-inspectors.png` });
    const fc = side.getByRole('button', { name: 'Inspect mlp expansion', exact: true });
    const start = await center(fc);
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    await page.mouse.move(start.x - 80, start.y + 15, { steps: 12 });
    await page.mouse.up();
    await settle();
    assert.equal(await side.locator('.df-inspector').count(), 0);
    assert.equal(await main.locator('.df-inspector').count(), 1);
    assert.deepEqual((await root()).nodes[repeat.id].position, repeat.position);

    // Shared ribbon shortcuts and deletion affect only the pane being used.
    await page.keyboard.press('Control+k');
    assert.equal(await page.getByRole('textbox', { name: 'Search blocks' }).evaluate((element) => element === document.activeElement), true);
    assert.equal(await ribbon.getAttribute('data-active-pane'), 'side');
    assert.equal(await ribbon.count(), 1);
    const beforeAdd = await template();
    await page.getByRole('dialog', { name: 'Block library', exact: true }).getByRole('button', { name: 'Add Linear', exact: true }).click();
    const added = Object.values((await template()).nodes).find((node) => !beforeAdd.nodes[node.id]);
    assert.ok(added);
    await side.locator(`.react-flow__node[data-id="${added.id}"]`).click();
    await page.keyboard.press('Delete');
    await settle();
    assert.equal((await template()).nodes[added.id], undefined);
    assert.ok((await root()).nodes[repeat.id], 'Inactive main selection is not deleted');

    // Cards can open the same level in both views; data remains shared.
    await main.getByRole('tab', { name: 'main', exact: true }).click({ button: 'right' });
    await page.getByRole('menuitem', { name: /Open to the side/ }).click();
    await side.getByRole('button', { name: 'Inspect tokens', exact: true }).click();
    await side.getByRole('textbox', { name: 'Block name', exact: true }).fill('shared tokens');
    assert.equal(await main.getByRole('button', { name: 'Inspect shared tokens', exact: true }).count(), 1);
    const mainViewport = await main.locator('.react-flow__viewport').getAttribute('style');
    await side.locator('.react-flow__controls-zoomin').click();
    await settle();
    assert.equal(await main.locator('.react-flow__viewport').getAttribute('style'), mainViewport, 'Zoom stays independent');
    await side.getByRole('tab', { name: /Transformer blocks/ }).click();
    await settle();

    // A cross-pane drop activates its destination and rejects recursive templates.
    await main.getByRole('tab', { name: 'main', exact: true }).click();
    await ribbon.getByRole('button', { name: 'Browse templates', exact: true }).click();
    const countBeforeDrop = Object.keys((await template()).nodes).length;
    const canvas = side.locator('.react-flow__pane');
    await page.getByRole('button', { name: 'Add Transformer block template', exact: true }).dragTo(canvas, { targetPosition: { x: 140, y: 160 } });
    await settle();
    assert.equal(Object.keys((await template()).nodes).length, countBeforeDrop, 'A template cannot be dropped inside itself from the other pane');
    await ribbon.getByRole('button', { name: 'Add Linear', exact: true }).dragTo(canvas, { targetPosition: { x: 140, y: 160 } });
    await settle();
    assert.equal(Object.keys((await template()).nodes).length, countBeforeDrop + 1);
    assert.equal(await focusedPane(), 'side');
    assert.equal(await side.evaluate((element) => element.classList.contains('is-active')), true);
    await page.keyboard.press('Delete');
    await settle();
    assert.equal(Object.keys((await template()).nodes).length, countBeforeDrop);
    assert.ok((await root()).nodes[repeat.id]);
    await main.getByRole('button', { name: 'Close inspector' }).click();

    const divider = page.getByRole('separator', { name: 'Resize graph panes' });
    const initialWidth = (await main.boundingBox()).width;
    const handle = await center(divider);
    await page.mouse.move(handle.x, handle.y);
    await page.mouse.down();
    await page.mouse.move(handle.x + 120, handle.y, { steps: 10 });
    await page.mouse.up();
    assert.ok((await main.boundingBox()).width > initialWidth + 90);
    await page.screenshot({ path: `${directory}/04-resized.png` });
    await divider.dblclick();
    await side.getByRole('button', { name: 'Close side view' }).click();
    assert.equal(await side.count(), 0);
    assert.ok((await main.boundingBox()).width > 1400);

    // Keyboard node entry point, with focus handed to the reopened view.
    const repeatWrapper = main.locator(`.react-flow__node[data-id="${repeat.id}"]`);
    await repeatWrapper.focus();
    await page.keyboard.press('Alt+Enter');
    await side.getByRole('button', { name: 'Inspect mlp expansion', exact: true }).waitFor();
    await settle();
    assert.equal(await focusedPane(), 'side');
    await page.screenshot({ path: `${directory}/05-reopened-by-shortcut.png` });
    assert.deepEqual(errors, []);
    console.log('Open to the side passed: node/card/keyboard entry points, independent navigation/zoom/selection, shared edits, active-pane deletion/search/drop, recursion guard, resize, close/reopen.');
} finally {
    await browser.close();
}
