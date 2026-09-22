import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/close-inspector-on-drag';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 }, deviceScaleFactor: 2 });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const panel = page.locator('.df-inspector');
    const assertClosed = async (message) => {
        assert.equal(await panel.count(), 0, message);
        assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().expandedNodeId), null);
    };
    const paths = () => page.locator('.react-flow__edge').evaluateAll((edges) => Object.fromEntries(edges.map((edge) => [edge.dataset.id, edge.querySelector('.react-flow__edge-path').getAttribute('d')])));
    const graph = () => page.evaluate(() => window.__deepflow.architectureStore.getState().getViewGraph());
    const block = (id) => page.locator(`.react-flow__node[data-id="${id}"] .df-block`);
    const center = async (locator) => {
        const box = await locator.boundingBox();
        assert.ok(box);
        const point = { x: box.x + box.width / 2, y: box.y + box.height / 2 };
        const viewport = locator.page().viewportSize();
        assert.ok(point.x > 0 && point.x < viewport.width && point.y > 0 && point.y < viewport.height,
            'A drag must start on a node inside the visible viewport');
        return point;
    };

    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    // A first visit must frame the destination level, without a manual Fit.
    // Checking before clicking a node also prevents Playwright's scrolling
    // from hiding an incorrect viewport restored from the previous level.
    await page.waitForFunction(() => {
        const canvas = document.querySelector('[data-architecture-canvas]').getBoundingClientRect();
        const nodes = [...document.querySelectorAll('.react-flow__node')];
        return nodes.length > 0 && nodes.every((node) => {
            const rect = node.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0 && rect.top >= canvas.top && rect.bottom <= canvas.bottom
                && rect.left >= canvas.left && rect.right <= canvas.right;
        });
    });
    await page.getByRole('button', { name: 'Inspect mlp fc', exact: true }).click();
    await settle();
    const initial = await graph();
    const fc = Object.values(initial.nodes).find((node) => node.label === 'mlp fc');
    const unrelated = initial.edges.filter((edge) => edge.source !== fc.id && edge.target !== fc.id);
    const originalPaths = await paths();
    assert.equal(await panel.isVisible(), true);
    const start = await center(block(fc.id));
    await page.screenshot({ path: `${directory}/01-before-drag.png` });
    await page.mouse.move(start.x, start.y);
    await page.mouse.down();
    assert.equal(await panel.isVisible(), true, 'Pressing without dragging keeps the inspector open');
    for (const [dx, dy] of [[-280, -80], [-150, 50], [160, 60], [100, 25]]) {
        await page.mouse.move(start.x + dx, start.y + dy, { steps: 10 });
        await settle();
        await assertClosed('Inspector closes as soon as a node moves and stays closed throughout the drag');
        const movedPaths = await paths();
        for (const edge of unrelated) assert.equal(movedPaths[edge.id], originalPaths[edge.id], 'Unrelated wire stays identical');
    }
    await page.screenshot({ path: `${directory}/02-during-drag.png` });
    await page.mouse.up();
    await settle();
    await assertClosed('Release does not reopen the inspector');
    const committed = await graph();
    assert.notDeepEqual(committed.nodes[fc.id].position, initial.nodes[fc.id].position);
    const releasedPaths = await paths();
    for (const edge of unrelated) assert.equal(releasedPaths[edge.id], originalPaths[edge.id], 'Unrelated wire also stays identical after commit');
    await page.screenshot({ path: `${directory}/03-after-release.png` });
    await block(fc.id).click();
    await settle();
    assert.equal(await panel.isVisible(), true, 'An explicit click reopens the inspector');
    await page.screenshot({ path: `${directory}/04-reopened-on-click.png` });

    // Dragging a different node also dismisses the open inspector.
    const gelu = Object.values(committed.nodes).find((node) => node.label === 'gelu');
    const otherStart = await center(block(gelu.id));
    await page.mouse.move(otherStart.x, otherStart.y);
    await page.mouse.down();
    await page.mouse.move(otherStart.x - 200, otherStart.y + 20, { steps: 15 });
    await settle();
    await assertClosed('Dragging a different node closes the open inspector');
    await page.mouse.up();
    await settle();
    await assertClosed('Dragging a different node does not reopen the inspector on release');

    // An isolated fixture reproduces the old geometric false-positive: a
    // disconnected node moves through the corridor of a stationary wire.
    const fixture = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    fixture.setDefaultTimeout(5000);
    fixture.on('pageerror', (error) => errors.push(error.message));
    await fixture.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ids = await fixture.evaluate(() => {
        const store = window.__deepflow.architectureStore;
        store.setState({ graph: { nodes: {}, edges: [] }, currentPath: [], selectedNodeIds: [], expandedNodeId: null });
        const state = store.getState();
        const a = state.addLayerByLibraryKey('torch.nn.Linear', { x: 250, y: 80 });
        const b = state.addLayerByLibraryKey('torch.nn.Linear', { x: 250, y: 400 });
        const stray = state.addLayerByLibraryKey('torch.nn.Linear', { x: 700, y: 220 });
        const edge = state.connect({ source: a, target: b });
        state.patchNode(a, { label: 'Source' });
        state.patchNode(b, { label: 'Target' });
        state.patchNode(stray, { label: 'Unconnected' });
        localStorage.setItem('deepflow:architecture-viewport:root', JSON.stringify({ x: 60, y: 60, zoom: 1 }));
        return { a, b, stray, edge };
    });
    await fixture.reload();
    await fixture.getByRole('button', { name: 'Inspect Source', exact: true }).click();
    const wire = fixture.locator(`.react-flow__edge[data-id="${ids.edge}"] .react-flow__edge-path`);
    const original = await wire.getAttribute('d');
    const stray = fixture.getByRole('button', { name: 'Inspect Unconnected', exact: true });
    const strayStart = await center(stray);
    const fixturePanel = fixture.locator('.df-inspector');
    assert.equal(await fixturePanel.isVisible(), true);
    await fixture.mouse.move(strayStart.x, strayStart.y);
    await fixture.mouse.down();
    await fixture.mouse.move(strayStart.x - 450, strayStart.y, { steps: 25 });
    assert.equal(await fixturePanel.count(), 0);
    assert.equal(await wire.getAttribute('d'), original);
    await fixture.mouse.up();
    await fixture.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    assert.equal(await wire.getAttribute('d'), original, 'Unconnected node cannot reclassify a stationary wire on drop');

    assert.equal(await fixturePanel.count(), 0);

    // Cancellation resets positions without reopening the inspector.
    await fixture.getByRole('button', { name: 'Inspect Source', exact: true }).click();
    assert.equal(await fixturePanel.isVisible(), true);
    const cancelStart = await center(stray);
    await fixture.mouse.move(cancelStart.x, cancelStart.y);
    await fixture.mouse.down();
    await fixture.mouse.move(cancelStart.x + 180, cancelStart.y + 90, { steps: 10 });
    await fixture.evaluate(() => window.dispatchEvent(new Event('blur')));
    await fixture.mouse.up();
    await fixture.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const restored = await center(stray);
    assert.ok(Math.abs(restored.x - cancelStart.x) < 1 && Math.abs(restored.y - cancelStart.y) < 1);
    assert.equal(await fixturePanel.count(), 0);
    await stray.click();
    assert.equal(await fixturePanel.isVisible(), true, 'A click can inspect the node again after cancellation');
    assert.deepEqual(errors, []);
    console.log('Stable dragging passed: inspector closes on movement, stays closed after release/cancel, reopens on click; unrelated wires remain stable.');
} finally {
    await browser.close();
}
