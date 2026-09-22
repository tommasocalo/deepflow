import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/binding-layout';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('group', { name: 'Example architectures' }).getByRole('button', { name: /^DANN/ }).click();
    const fit = async () => {
        await page.getByRole('button', { name: 'Fit active graph', exact: true }).click();
        await page.waitForTimeout(300);
    };
    await fit();
    await page.getByRole('button', { name: 'Open domain head', exact: true }).click();
    await fit();
    await page.getByRole('button', { name: 'Inspect bias', exact: true }).click();
    const scale = page.locator('.df-inspector [data-parameter="scale"]');
    const result = scale.locator('.df-parameter-result > strong');
    assert.equal(await scale.locator('.df-param-label').innerText(), 'Bound (±)');
    assert.equal(await scale.getByLabel('scale', { exact: true }).inputValue(), '1 / (in_features ** 0.5)');
    assert.equal(await result.innerText(), '≈ 0.0883883');
    assert.equal(await result.getAttribute('title'), `Exact value: ${1 / Math.sqrt(128)}`);
    assert.match(await scale.locator('.df-parameter-result').innerText(), /Result:\s+≈ 0.0883883/);

    const checkLayout = async () => {
        const geometry = await scale.evaluate((field) => {
            const source = field.querySelector('.df-parameter-result');
            const value = source.querySelector('strong');
            const state = source.querySelector('span');
            const input = field.querySelector('input');
            const panel = field.closest('.df-inspector');
            const rect = (element) => { const box = element.getBoundingClientRect(); return { x: box.x, y: box.y, right: box.right, bottom: box.bottom, width: box.width }; };
            return { field: rect(field), value: rect(value), state: rect(state), panel: rect(panel),
                valueWidth: value.clientWidth, valueScroll: value.scrollWidth, inputWidth: input.clientWidth, inputScroll: input.scrollWidth,
                viewport: innerWidth, bodyWidth: panel.querySelector('.df-inspector-body').clientWidth, bodyScroll: panel.querySelector('.df-inspector-body').scrollWidth };
        });
        assert.ok(geometry.state.right + 3 <= geometry.value.x, 'State and value have visible separation');
        assert.ok(geometry.valueScroll <= geometry.valueWidth, 'The rounded result is fully visible');
        assert.ok(geometry.inputScroll <= geometry.inputWidth, 'The entire formula is visible');
        assert.ok(geometry.bodyScroll <= geometry.bodyWidth, 'No inspector horizontal overflow');
        assert.ok(geometry.panel.x >= 0 && geometry.panel.right <= geometry.viewport, 'Panel fits the viewport');
    };
    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 1000 });
        await fit();
        await checkLayout();
        await page.screenshot({ path: `${directory}/02-bias-after-${width}.png` });
    }
    await page.locator('.df-inspector-options > summary').filter({ hasText: /^Options$/ }).click();
    await checkLayout();
    await page.screenshot({ path: `${directory}/03-bias-options-820.png` });
    assert.equal(await page.locator('.df-parameter-source').count(), 0, 'No Linked or provenance rows');
    assert.equal(await page.getByText('Expose a field…', { exact: true }).count(), 0, 'No detached exposure menu');
    await scale.getByRole('button', { name: 'Bound (±) in domain head settings', exact: true }).click();
    const saved = await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        const template = window.__deepflow.templatesStore.getState().templates[state.getLevel().templateId];
        const node = Object.values(template.graph.nodes).find((candidate) => candidate.label === 'bias');
        const reference = node.parameters.scale;
        return { reference, defaultValue: template.freeParameters.find((parameter) => `$${parameter.name}` === reference)?.default };
    });
    assert.equal(saved.defaultValue, '1 / (in_features ** 0.5)', 'Exposure keeps the formula, never the rounded display');
    assert.equal(await scale.getByLabel('scale', { exact: true }).inputValue(), saved.reference);
    assert.equal(await scale.locator('.df-parameter-result').count(), 0, 'A direct variable needs no repeated result');
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.equal(await scale.getByLabel('scale', { exact: true }).inputValue(), '1 / (in_features ** 0.5)');
    await scale.getByLabel('scale', { exact: true }).fill('1 / (10 ** 10)');
    assert.equal(await result.innerText(), '1e-10', 'Small values never appear rounded to zero');
    assert.equal(await result.getAttribute('title'), 'Exact value: 1e-10');
    assert.deepEqual(errors, []);
    console.log('Bias binding layout passed at 1440/1024/820: visible formula and result, no overlap or duplicate links, eye exposure preserves exact formula, Undo and small numbers.');
} finally {
    await browser.close();
}
