import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.waitForFunction(() => !!window.__deepflow?.architectureStore);
    const ids = await page.evaluate(() => {
        const store = window.__deepflow.architectureStore;
        store.setState({ graph: { nodes: {}, edges: [] }, currentPath: [], selectedNodeIds: [], expandedNodeId: null });
        const arch = store.getState();
        const a = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 300, y: 100 });
        const b = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 300, y: 420 });
        const c = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 700, y: 420 });
        const d = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 700, y: 100 });
        const edge = arch.connect({ source: a, target: b });
        arch.setEdgeGradientOp(edge, 'reverse', .5);
        localStorage.setItem('deepflow:architecture-viewport:root', JSON.stringify({ x: 80, y: 80, zoom: 1 }));
        return { a, b, c, d, edge };
    });
    await page.reload();
    const edgeLocator = () => page.locator(`.react-flow__edge[data-id="${ids.edge}"]`);
    await edgeLocator().waitFor();
    await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const edgeState = () => page.evaluate((id) => window.__deepflow.architectureStore.getState().graph.edges.find(edge => edge.id === id), ids.edge);
    const center = async (locator) => {
        const box = await locator.boundingBox();
        assert.ok(box, 'Drag target is visible');
        return { x: box.x + box.width / 2, y: box.y + box.height / 2 };
    };
    const port = (id, kind) => page.locator(`.react-flow__node[data-id="${id}"] .react-flow__handle.${kind}`);
    const endpoint = (kind) => edgeLocator().locator(`.react-flow__edgeupdater-${kind}`);
    const drag = async (from, to, during) => {
        const start = await center(from);
        const end = typeof to.x === 'number' ? to : await center(to);
        await page.mouse.move(start.x, start.y);
        await page.mouse.down();
        await page.mouse.move(end.x, end.y, { steps: 16 });
        await during?.();
        await page.mouse.up();
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
    };
    await drag(endpoint('target'), port(ids.c, 'target'));
    assert.equal((await edgeState()).target, ids.c, 'Target reconnects');
    await drag(endpoint('source'), port(ids.d, 'source'));
    assert.equal((await edgeState()).source, ids.d, 'Source reconnects');
    assert.equal((await edgeState()).gradientOp, 'reverse');
    assert.equal((await edgeState()).gradientOpLambda, .5);
    assert.equal(await page.locator('.df-feature-stream, .df-activation').count(), 0);

    const stable = await edgeState();
    await drag(endpoint('source'), port(ids.c, 'source'));
    assert.deepEqual(await edgeState(), stable, 'Self-connection cancels');
    await page.evaluate(({ d, b }) => window.__deepflow.architectureStore.getState().connect({ source: d, target: b }), ids);
    await drag(endpoint('target'), port(ids.b, 'target'));
    assert.deepEqual(await edgeState(), stable, 'Duplicate cancels');
    await drag(endpoint('target'), page.locator(`.react-flow__node[data-id="${ids.b}"] .df-block`));
    assert.deepEqual(await edgeState(), stable, 'Dropping on a node body cancels');
    await endpoint('target').click();
    assert.deepEqual(await edgeState(), stable, 'A click alone does not disconnect');
    await drag(endpoint('target'), { x: 1100, y: 750 }, () => page.keyboard.press('Escape'));
    assert.deepEqual(await edgeState(), stable, 'Escape cancels a detach');
    await drag(endpoint('target'), { x: 1100, y: 750 });
    assert.equal(await edgeState(), undefined, 'Empty canvas drop disconnects');
    assert.equal(await page.locator('.react-flow__edge').count(), 1, 'Other connections remain');

    await drag(port(ids.a, 'source'), port(ids.b, 'target'));
    const hasNew = await page.evaluate(({ a, b }) => window.__deepflow.architectureStore.getState().graph.edges.some(edge => edge.source === a && edge.target === b), ids);
    assert.equal(hasNew, true, 'New connections still work');
    assert.deepEqual(errors, []);
    console.log('Browser connectors passed: both endpoints, metadata, invalid/self/duplicate drops, click, Escape, detach, new connection.');
} finally {
    await browser.close();
}
