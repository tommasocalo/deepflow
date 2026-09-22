import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/clipboard-history';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    const page = await context.newPage();
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).waitFor();
    const modifier = await page.evaluate(() => /Mac/.test(navigator.platform) ? 'Meta' : 'Control');
    const main = page.locator('[data-graph-pane="main"]');
    const side = page.locator('[data-graph-pane="side"]');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const graph = () => page.evaluate(() => window.__deepflow.architectureStore.getState().graph);
    const history = () => page.evaluate(() => window.__deepflow.architectureHistory.getState().undoCount);
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await page.evaluate(() => {
        const state = window.__deepflow;
        const scale = (id, x, y) => ({ id, kind: 'layer', label: id, libraryKey: 'deepflow.op.Scale', parameters: { factor: 2 }, position: { x, y } });
        const component = { id: 'box', kind: 'supernode', templateId: 'body', label: 'Box', parameterValues: {}, position: { x: 600, y: 220 } };
        state.templatesStore.setState({ templates: { body: { id: 'body', name: 'Box', freeParameters: [], graph: { nodes: { inner: scale('inner', 100, 100) }, edges: [] } } } });
        state.architectureStore.setState({ graph: { nodes: { a: scale('a', 200, 100), b: scale('b', 200, 260), box: component }, edges: [{ id: 'ab', source: 'a', target: 'b', gradientOp: 'none' }] }, currentPath: [], openPaths: [], selectedNodeIds: ['a', 'b'], expandedNodeId: 'a' });
        state.architectureHistory.getState().clear();
    });
    await settle();
    await main.focus();
    const original = await graph();
    await page.keyboard.press(`${modifier}+c`);
    await page.keyboard.press(`${modifier}+v`);
    await settle();
    const first = await graph();
    assert.equal(Object.keys(first.nodes).length, 5, 'Real keyboard paste duplicates both selected nodes');
    assert.equal(first.edges.length, 2, 'Internal cable is duplicated');
    assert.equal(await history(), 1, 'One paste is one undo step');
    assert.equal(await page.getByRole('button', { name: 'Close inspector', exact: true }).count(), 0, 'Paste leaves settings closed');
    const firstIds = await page.evaluate(() => window.__deepflow.architectureStore.getState().selectedNodeIds);
    assert.equal(first.nodes[firstIds[0]].position.x, original.nodes.a.position.x + 36);
    await page.keyboard.press(`${modifier}+v`);
    const second = await graph();
    const secondIds = await page.evaluate(() => window.__deepflow.architectureStore.getState().selectedNodeIds);
    assert.equal(second.nodes[secondIds[0]].position.x, original.nodes.a.position.x + 72);
    assert.equal(Object.keys(second.nodes).length, 7);
    assert.equal(await history(), 2);
    await ribbon.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.deepEqual(await graph(), first);
    await ribbon.getByRole('button', { name: 'Redo', exact: true }).click();
    assert.deepEqual(await graph(), second);

    // Native field copy/paste must never copy or paste selected graph nodes.
    await page.evaluate(() => { const state = window.__deepflow.architectureStore.getState(); state.selectOnly('a'); state.setExpanded('a'); });
    const name = page.getByRole('textbox', { name: 'Block name', exact: true });
    await name.fill('Native clipboard');
    await name.selectText();
    await page.keyboard.press(`${modifier}+c`);
    await name.fill('');
    await page.keyboard.press(`${modifier}+v`);
    assert.equal(await name.inputValue(), 'Native clipboard');
    assert.equal(Object.keys((await graph()).nodes).length, 7);
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await main.focus();
    await page.keyboard.press(`${modifier}+v`);
    assert.equal(Object.keys((await graph()).nodes).length, 7, 'Unrelated plain text does not paste stale graph nodes');

    // Copy a root instance and paste into the active side pane only.
    await page.evaluate(() => window.__deepflow.architectureStore.getState().selectOnly('a'));
    await main.focus();
    await page.keyboard.press(`${modifier}+c`);
    await page.evaluate(() => window.__deepflow.architectureStore.getState().selectOnly('box'));
    await ribbon.getByRole('button', { name: 'Open to the side', exact: true }).click();
    await side.waitFor();
    await side.focus();
    const rootBeforeSide = await graph();
    await page.keyboard.press(`${modifier}+v`);
    await settle();
    assert.deepEqual(await graph(), rootBeforeSide);
    assert.equal(await page.evaluate(() => Object.keys(window.__deepflow.templatesStore.getState().templates.body.graph.nodes).length), 2);
    assert.equal(await ribbon.getAttribute('data-active-pane'), 'side');
    assert.equal(await ribbon.getByRole('button', { name: 'Make copy', exact: true }).count(), 0);
    assert.deepEqual(await ribbon.locator('.df-ribbon-group').evaluateAll((elements) => elements.map((element) => element.getAttribute('aria-label'))), ['Insert', 'Structure', 'Component', 'View', 'History']);

    for (const width of [1440, 1024]) {
        await page.setViewportSize({ width, height: 1000 });
        await settle();
        const bounds = await ribbon.evaluate((element) => ({ client: element.clientWidth, scroll: element.scrollWidth, historyRight: element.querySelector('.df-ribbon-history').getBoundingClientRect().right }));
        assert.ok(bounds.scroll <= bounds.client + 1, `Ribbon fits ${width}px: ${JSON.stringify(bounds)}`);
        assert.ok(bounds.historyRight <= width, `History remains visible at ${width}px`);
        await page.screenshot({ path: `${directory}/history-${width}.png` });
    }
    assert.deepEqual(errors, []);
    console.log('Clipboard keyboard, native fields, split panes, history and responsive ribbon checks passed.');
} finally { await browser.close(); }
