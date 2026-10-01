import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { levelCard, selectLevel } from './level-navigation-browser-helper.mjs';

const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.setDefaultTimeout(5000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).waitFor();
    await page.evaluate(() => document.fonts.ready);
    const ids = await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        const first = Object.values(state.graph.nodes).find((node) => node.kind === 'repeat');
        const second = state.addSuperNodeInstance(first.templateId, { x: 800, y: 300 }, 'Second instance');
        const template = window.__deepflow.templatesStore.getState().templates[first.templateId];
        const fc = Object.values(template.graph.nodes).find((node) => node.label === 'mlp fc');
        if (fc?.kind !== 'supernode') throw new Error('Viewport fixture requires an editable mlp fc component');
        return { first: first.id, second };
    });
    const main = page.locator('[data-graph-pane="main"]');
    const side = page.locator('[data-graph-pane="side"]');
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const view = (pane) => pane.locator('.react-flow__viewport').getAttribute('style');
    const framed = () => page.waitForFunction(() => {
        const pane = document.querySelector('[data-graph-pane="main"]');
        const canvas = pane.querySelector('[data-architecture-canvas]').getBoundingClientRect();
        return [...pane.querySelectorAll('.react-flow__node')].every((node) => {
            const rect = node.getBoundingClientRect();
            return rect.height > 0 && rect.top >= canvas.top && rect.bottom <= canvas.bottom
                && rect.left >= canvas.left && rect.right <= canvas.right;
        });
    });
    const waitView = (pane, expected) => page.waitForFunction(({ pane, expected }) =>
        document.querySelector(`[data-graph-pane="${pane}"] .react-flow__viewport`)?.getAttribute('style') === expected,
    { pane, expected });
    const openLevel = (id) => page.evaluate((id) => window.__deepflow.architectureStore.getState().openLevel([id]), id);

    await openLevel(ids.first);
    await framed();
    await main.locator('.react-flow__controls-zoomin').click();
    await settle();
    const firstView = await view(main);

    // Both instance paths resolve to the exact same graph and node IDs.
    // Readiness must react to the level key even when no measurement changes.
    await openLevel(ids.second);
    await framed();
    assert.notEqual(await view(main), firstView, 'A new instance frames its own level instead of inheriting the previous zoom');
    await main.locator('.react-flow__controls-zoomout').click();
    await settle();
    const secondView = await view(main);
    await selectLevel(main, 'Transformer blocks');
    await waitView('main', firstView);
    await selectLevel(main, 'Second instance');
    await waitView('main', secondView);

    await levelCard(main, 'Transformer blocks').click({ button: 'right' });
    await page.getByRole('menuitem', { name: /Open to the side/ }).click();
    await side.getByRole('button', { name: 'Inspect mlp fc', exact: true }).waitFor();
    await side.locator('.react-flow__controls-zoomin').click();
    await settle();
    const sideView = await view(side);
    const untouchedMain = await view(main);
    await selectLevel(side, 'main');
    await side.getByRole('button', { name: 'Open Transformer blocks', exact: true }).waitFor();
    await selectLevel(side, 'Transformer blocks');
    await waitView('side', sideView);
    assert.equal(await view(main), untouchedMain, 'Restoring a side-pane viewport preserves the main pane viewport');
    await side.getByRole('button', { name: 'Close side view', exact: true }).click();
    const firstKey = `deepflow:architecture-viewport:${ids.first}`;
    const savedFirst = await page.evaluate((key) => JSON.parse(localStorage.getItem(key)), firstKey);
    assert.ok(savedFirst.width > 0 && savedFirst.height > 0, 'Viewport records include the canvas size');
    await page.setViewportSize({ width: 820, height: 900 });
    await selectLevel(main, 'Transformer blocks');
    const resized = await main.locator('[data-architecture-canvas]').boundingBox();
    const expectedX = savedFirst.x + (resized.width - savedFirst.width) / 2;
    await page.waitForFunction((x) => {
        const viewport = document.querySelector('[data-graph-pane="main"] .react-flow__viewport');
        return viewport && Math.abs(new DOMMatrix(getComputedStyle(viewport).transform).m41 - x) < 2;
    }, expectedX);
    const visible = await main.locator('.react-flow__node').evaluateAll((nodes) => {
        const canvas = document.querySelector('[data-graph-pane="main"] [data-architecture-canvas]').getBoundingClientRect();
        return nodes.filter((node) => {
            const box = node.getBoundingClientRect();
            return box.right > canvas.left && box.left < canvas.right && box.bottom > canvas.top && box.top < canvas.bottom;
        }).length;
    });
    assert.ok(visible > 2, 'A level saved at desktop width remains visible after reopening at 820px');

    await selectLevel(main, 'Second instance');
    await page.evaluate((key) => localStorage.setItem(key, JSON.stringify({ x: 99999, y: 99999, zoom: 1 })), firstKey);
    await selectLevel(main, 'Transformer blocks');
    await page.waitForFunction(() => {
        const pane = document.querySelector('[data-graph-pane="main"]');
        const canvas = pane.querySelector('[data-architecture-canvas]').getBoundingClientRect();
        return [...pane.querySelectorAll('.react-flow__node')].some((node) => {
            const box = node.getBoundingClientRect();
            return box.right > canvas.left && box.left < canvas.right && box.bottom > canvas.top && box.top < canvas.bottom;
        });
    });
    assert.deepEqual(errors, []);
    console.log('Level viewport checks passed: first-visit framing, independent panes, resized restores and legacy viewport fallback.');
} finally {
    await browser.close();
}
