import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.waitForFunction(() => !!window.__deepflow?.architectureStore);
    const ids = await page.evaluate(() => {
        const store = window.__deepflow.architectureStore;
        store.setState({ graph: { nodes: {}, edges: [] }, currentPath: [], selectedNodeIds: [], expandedNodeId: null, showGradientFlow: true });
        const state = store.getState();
        const a = state.addLayerByLibraryKey('torch.nn.Linear', { x: 250, y: 80 });
        const b = state.addLayerByLibraryKey('torch.nn.Linear', { x: 250, y: 400 });
        const c = state.addLayerByLibraryKey('torch.nn.Linear', { x: 700, y: 80 });
        const d = state.addLayerByLibraryKey('torch.nn.Linear', { x: 700, y: 400 });
        const first = state.connect({ source: a, target: b });
        const second = state.connect({ source: c, target: d });
        localStorage.setItem('deepflow:architecture-viewport:root', JSON.stringify({ x: 60, y: 60, zoom: 1 }));
        return { a, b, c, d, first, second };
    });
    await page.reload();
    await page.locator(`.react-flow__edge[data-id="${ids.second}"]`).waitFor();
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await settle();
    const selected = () => page.locator('.react-flow__edge.selected').evaluateAll((elements) => elements.map((element) => element.dataset.id).sort());
    const blank = { x: 1300, y: 800 };
    const clickBlank = () => page.mouse.click(blank.x, blank.y);
    const clickEdge = async (id) => {
        const point = await page.locator(`.react-flow__edge[data-id="${id}"] .react-flow__edge-path`).evaluate((path) => {
            const local = path.getPointAtLength(path.getTotalLength() * 0.4);
            const screen = new DOMPoint(local.x, local.y).matrixTransform(path.getScreenCTM());
            return { x: screen.x, y: screen.y };
        });
        await page.mouse.click(point.x, point.y);
        await settle();
    };

    // On macOS Control-click is the native context-menu gesture. Exercise
    // Command there; Control is also exercised on the other platforms.
    const additiveModifiers = process.platform === 'darwin' ? ['Meta'] : ['Meta', 'Control'];
    for (const modifier of additiveModifiers) {
        await clickEdge(ids.first);
        await page.keyboard.down(modifier);
        await clickEdge(ids.second);
        assert.deepEqual(await selected(), [ids.first, ids.second].sort(), `${modifier} adds another connection`);
        await clickBlank();
        assert.deepEqual(await selected(), [ids.first, ids.second].sort(), `${modifier} + missed connection preserves selection`);
        await clickEdge(ids.first);
        assert.deepEqual(await selected(), [ids.second], `${modifier} still toggles a selected connection off`);
        await page.keyboard.up(modifier);
        await clickBlank();
        assert.deepEqual(await selected(), [], 'A normal blank click clears selection');
    }

    await clickEdge(ids.first);
    await page.keyboard.down('Shift');
    await clickBlank();
    assert.deepEqual(await selected(), [ids.first], 'Shift + blank click preserves the existing selection');
    await page.keyboard.up('Shift');

    // The alternate pan interaction uses ReactFlow's click handler rather
    // than its selection pointer-up handler. Both must preserve selection.
    await page.getByRole('button', { name: 'Pan canvas', exact: true }).click();
    await page.keyboard.down('Meta');
    await clickBlank();
    assert.deepEqual(await selected(), [ids.first], 'Modifier + blank click in pan mode preserves selection');
    await page.keyboard.up('Meta');
    await clickBlank();
    assert.deepEqual(await selected(), [], 'Normal blank click in pan mode clears selection');

    // A real marquee remains a selection gesture, including its cleanup.
    await page.getByRole('button', { name: 'Select nodes', exact: true }).click();
    const boxes = await Promise.all([ids.a, ids.b].map((id) => page.locator(`.react-flow__node[data-id="${id}"]`).boundingBox()));
    const left = Math.min(...boxes.map((box) => box.x)) - 30;
    const top = Math.min(...boxes.map((box) => box.y)) - 25;
    const right = Math.max(...boxes.map((box) => box.x + box.width)) + 30;
    const bottom = Math.max(...boxes.map((box) => box.y + box.height)) + 25;
    await page.keyboard.down('Shift');
    await page.mouse.move(left, top);
    await page.mouse.down();
    await page.mouse.move(right, bottom, { steps: 15 });
    await page.mouse.up();
    await page.keyboard.up('Shift');
    await settle();
    assert.deepEqual(await page.evaluate(() => [...window.__deepflow.architectureStore.getState().selectedNodeIds].sort()), [ids.a, ids.b].sort(), 'Shift marquee still selects the enclosed nodes');
    assert.deepEqual(await selected(), [ids.first], 'Shift marquee still selects their connection');
    await clickBlank();
    assert.deepEqual(await selected(), [], 'The click after a marquee clears selection normally');
    assert.equal(await page.locator('.react-flow__selection').count(), 0, 'No stale marquee remains');

    await page.getByRole('button', { name: 'Pan canvas', exact: true }).click();
    await page.mouse.move(blank.x, blank.y);
    await page.mouse.down();
    await page.mouse.move(blank.x - 150, blank.y - 80, { steps: 10 });
    await page.mouse.up();
    const beforeFit = await page.locator('.react-flow__viewport').getAttribute('style');
    await page.mouse.dblclick(blank.x, blank.y);
    await page.waitForTimeout(300);
    const afterFit = await page.locator('.react-flow__viewport').getAttribute('style');
    assert.notEqual(afterFit, beforeFit, 'Double-clicking empty canvas still fits the graph');
    assert.deepEqual(errors, []);
    console.log(`Gradient selection passed: ${additiveModifiers.join('/')} add and toggle, modifier misses preserve selection, normal deselection, pan mode, Shift marquee, double-click fit.`);
} finally {
    await browser.close();
}
