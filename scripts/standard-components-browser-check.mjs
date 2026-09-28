import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/standard-components';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1050 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const graph = () => page.evaluate(() => window.__deepflow.architectureStore.getState().getViewGraph());
    await page.getByRole('button', { name: 'Open positional embedding', exact: true }).waitFor();
    await settle();
    await page.screenshot({ path: `${directory}/01-standard-gpt2.png` });
    await page.getByRole('button', { name: 'Open positional embedding', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect position indices', exact: true }).waitFor();
    await settle();
    assert.equal(await page.getByRole('button', { name: 'Open position table', exact: true }).count(), 1);
    await page.screenshot({ path: `${directory}/02-learned-positions.png` });
    await page.getByRole('button', { name: 'Open position table', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect embedding table', exact: true }).waitFor();
    await ribbon.getByRole('button', { name: 'Show gradient flow', exact: true }).click();
    await settle();
    assert.equal(await page.locator('.df-gradient-edge--flow').count(), 2);
    assert.equal(await page.locator('.df-gradient-edge--constant').count(), 1, 'Lookup indices have no derivative');
    await page.screenshot({ path: `${directory}/03-embedding-gradients.png` });
    const before = await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        const rootPosition = Object.values(state.graph.nodes).find((node) => node.label === 'positional embedding');
        return { id: rootPosition.templateId, templates: JSON.parse(JSON.stringify(window.__deepflow.templatesStore.getState().templates)) };
    });
    await ribbon.getByRole('button', { name: 'Make copy', exact: true }).click();
    await settle();
    const copied = await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        const node = Object.values(state.graph.nodes).find((node) => node.label === 'positional embedding');
        return node.templateId;
    });
    assert.notEqual(copied, before.id, 'Copying a nested component forks its containing shared definition');
    await page.getByRole('button', { name: 'Inspect embedding table', exact: true }).click();
    const trainable = page.getByRole('checkbox', { name: 'requires_grad', exact: true });
    assert.equal(await trainable.isChecked(), true);
    await trainable.uncheck();
    await settle();
    assert.equal(await page.locator('.df-gradient-edge--flow').count(), 0);
    assert.equal(await page.locator('.df-gradient-edge--constant').count(), 3);
    const originals = await page.evaluate((ids) => Object.fromEntries(ids.map((id) => [id, window.__deepflow.templatesStore.getState().templates[id]])), Object.keys(before.templates));
    assert.deepEqual(JSON.parse(JSON.stringify(originals)), before.templates, 'Editing the copied weight preserves every original template');
    await page.screenshot({ path: `${directory}/04-frozen-copy.png` });
    await trainable.check();
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    const current = await graph();
    const weight = Object.values(current.nodes).find((node) => node.libraryKey === 'deepflow.op.Parameter');
    const connection = current.edges.find((edge) => edge.source === weight.id);
    const path = page.locator(`.react-flow__edge[data-id="${connection.id}"] .react-flow__edge-path`);
    const point = await path.evaluate((element) => {
        const position = element.getPointAtLength(element.getTotalLength() * .55);
        const screen = new DOMPoint(position.x, position.y).matrixTransform(element.getScreenCTM());
        return { x: screen.x, y: screen.y };
    });
    await page.mouse.click(point.x, point.y);
    await page.getByRole('radio', { name: 'Stop', exact: true }).check();
    await settle();
    assert.equal(await page.locator('.df-gradient-edge--stopped').count(), 1);
    assert.equal((await graph()).edges.find((edge) => edge.id === connection.id).gradientOp, 'stop');
    await page.screenshot({ path: `${directory}/05-stop-gradient.png` });
    await ribbon.getByRole('button', { name: 'Insert operations', exact: true }).click();
    await page.getByRole('menu', { name: 'Insert operations', exact: true }).waitFor();
    await page.screenshot({ path: `${directory}/06-operations-menu.png` });
    await page.keyboard.press('Escape');

    // A separate, valid fixture makes the non-learned alternative inspectable.
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        state.goToLevel(0); state.resetCurrentGraph();
        const root = window.__deepflow.architectureStore.getState();
        const input = Object.values(root.graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Input');
        const output = Object.values(root.graph.nodes).find((node) => node.libraryKey === 'deepflow.io.Output');
        root.patchLayerParameters(input.id, { shape: '(128,)', dtype: 'int64' });
        const component = root.addStandardComponent('sinusoidal-position', { x: 400, y: 200 });
        root.patchNode(component, { parameterValues: { D: 768 } });
        root.connect({ source: input.id, target: component }); root.connect({ source: component, target: output.id });
        root.enterSupernode(component); root.setShowGradientFlow(true);
    });
    await page.getByRole('button', { name: 'Inspect sine', exact: true }).waitFor();
    await settle();
    assert.equal(await page.locator('.df-gradient-edge--flow').count(), 0, 'Sinusoidal positions contain no trainable values');
    await page.screenshot({ path: `${directory}/07-sinusoidal-positions.png` });
    assert.deepEqual(errors, []);
    console.log('Standard component browser checks passed: nested position definition, correct gradient paths, isolated copy, freezing, connection stop, operations menu, sinusoidal math.');
} finally { await browser.close(); }
