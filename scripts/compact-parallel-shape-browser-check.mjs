import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/parameter-flow';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('group', { name: 'Example architectures' }).getByRole('button', { name: /^Multi-head attention/ }).click();
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const fit = async () => { await page.getByRole('button', { name: 'Fit active graph', exact: true }).click(); await page.waitForTimeout(300); };
    await page.locator('.df-parallel').waitFor();
    await fit();
    const footprint = await page.locator('.df-parallel').evaluate((element) => ({ width: element.offsetWidth, height: element.offsetHeight }));
    assert.equal(await page.locator('.df-parallel-instance').count(), 3);
    assert.ok(footprint.width < 400, 'Three named input ports no longer force a 660px card');
    const ports = await page.locator('.df-parallel .df-port-shape--input').evaluateAll((elements) => elements.map((element) => {
        const rect = element.getBoundingClientRect();
        return { left: rect.left, right: rect.right };
    }));
    assert.ok(ports.every((port, index) => index === 0 || port.left > ports[index - 1].right), 'Always-visible input dimension frames do not overlap');
    await page.screenshot({ path: `${directory}/01-compact-parallel.png` });

    // Configure a valid larger fixture through the same stores used by the UI.
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        const path = state.currentPath;
        state.goToLevel(0);
        const input = Object.values(state.graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
        state.patchLayerParameters(input.id, { shape: '(128, 800)' });
        state.patchNode(path[0], { parameterValues: { H: 200 } });
        state.openLevel(path);
    });
    await settle();
    assert.match(await page.locator('.df-parallel .df-repeat-count').innerText(), /200/);
    assert.equal(await page.locator('.df-parallel-instance').count(), 3);
    assert.deepEqual(await page.locator('.df-parallel').evaluate((element) => ({ width: element.offsetWidth, height: element.offsetHeight })), footprint);
    assert.match(await page.locator('.df-parallel .df-port-shape--output').innerText(), /200 tensors \[B × 128 × 4\]/);
    await page.screenshot({ path: `${directory}/02-two-hundred-heads.png` });
    await page.getByRole('button', { name: 'Open Attention heads instance 200', exact: true }).click();
    const instance = page.getByRole('spinbutton', { name: 'Inspected instance number', exact: true });
    assert.equal(await instance.inputValue(), '200');
    await instance.fill('137');
    assert.equal(await page.getByRole('tab', { name: 'Attention heads × 200', exact: true }).count(), 1, 'Open tabs show the resolved automatic count');
    assert.equal(await page.evaluate(() => Object.values(window.__deepflow.architectureStore.getState().inspectedIterations).at(-1)), 136);
    await fit();
    await page.screenshot({ path: `${directory}/03-inspect-instance-137.png` });

    await page.getByRole('tab', { name: 'Multi-head attention', exact: true }).click();
    assert.equal(await page.getByRole('tab', { name: 'Attention heads × 200', exact: true }).count(), 1, 'The resolved count remains visible on inactive tabs');
    await page.getByRole('button', { name: 'Open Q projection', exact: true }).click();
    await fit();
    await page.getByRole('button', { name: 'Inspect weight', exact: true }).click();
    const first = page.getByRole('textbox', { name: 'shape dimension 1', exact: true });
    const second = page.getByRole('textbox', { name: 'shape dimension 2', exact: true });
    assert.equal(await first.inputValue(), '$out_features');
    assert.equal(await second.inputValue(), '$in_features');
    const assertContained = async () => {
        const layout = await page.locator('.df-inspector').evaluate((element) => {
            const box = element.getBoundingClientRect();
            return { width: box.width, body: element.querySelector('.df-inspector-body').scrollWidth,
                cells: [...element.querySelectorAll('.df-shape-cell')].map((cell) => ({ left: cell.getBoundingClientRect().left - box.left, right: cell.getBoundingClientRect().right - box.left })) };
        });
        assert.ok(layout.cells.every((cell) => cell.left >= 0 && cell.right <= layout.width), 'Dimension cells fit inside the compact inspector');
        assert.ok(layout.body <= layout.width, 'The inspector has no horizontal overflow');
    };
    await assertContained();
    await page.screenshot({ path: `${directory}/04-weight-dimensions.png` });
    await page.getByRole('button', { name: 'Add shape dimension', exact: true }).click();
    await page.getByRole('textbox', { name: 'shape dimension 3', exact: true }).fill('2 * 4');
    await page.getByRole('button', { name: 'Remove shape dimension 2', exact: true }).click();
    assert.equal(await second.inputValue(), '2 * 4');
    const saved = await page.evaluate(() => Object.values(window.__deepflow.architectureStore.getState().getViewGraph().nodes).find((node) => node.label === 'weight').parameters.shape);
    assert.deepEqual(saved, ['$out_features', '2 * 4']);
    await assertContained();
    await page.screenshot({ path: `${directory}/05-edited-weight-dimensions.png` });
    assert.deepEqual(errors, []);
    console.log('Compact UI browser checks passed: 12/200 heads same footprint, non-overlapping port frames, arbitrary instance navigation, editable weight dimensions without overflow.');
} finally { await browser.close(); }
