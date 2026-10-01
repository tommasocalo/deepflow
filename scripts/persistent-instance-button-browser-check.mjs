import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/persistent-instance-button';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [];
    const measurements = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const button = ribbon.locator('.df-ribbon-instance-button');
    const popup = ribbon.locator('.df-iteration-popover');
    const measure = () => ribbon.evaluate((element) => {
        const bounds = element.querySelector('.df-ribbon-instance-button').getBoundingClientRect();
        const parameters = element.querySelector('.df-ribbon-component').getBoundingClientRect();
        const label = element.querySelector('.df-ribbon-instance-label');
        return { x: bounds.x, width: bounds.width, parametersX: parameters.x, labelWidth: label.clientWidth, labelContentWidth: label.scrollWidth };
    });
    const root = () => page.evaluate(() => window.__deepflow.architectureStore.getState().openLevel([]));
    const enter = (label) => page.evaluate((label) => {
        const state = window.__deepflow.architectureStore.getState();
        const node = Object.values(state.getViewGraph().nodes).find((candidate) => candidate.label === label);
        state.enterSupernode(node.id);
    }, label);
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    await button.waitFor();
    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 960 });
        await root();
        await settle();
        assert.equal(await button.isDisabled(), true, 'Instances remains visible but disabled at the root');
        assert.equal(await button.innerText(), 'Instances');
        await button.evaluate((element) => element.click());
        assert.equal(await popup.count(), 0, 'Disabled control cannot open a popup');
        const disabled = await measure();
        if (width === 1440) await page.screenshot({ path: `${directory}/01-root-disabled-1440.png` });
        await enter('Transformer blocks');
        await settle();
        assert.equal(await button.isEnabled(), true);
        assert.equal(await button.innerText(), 'Block 1/12');
        const active = await measure();
        assert.equal(active.width, disabled.width, 'Instances uses the same width in both states');
        assert.equal(active.parametersX, disabled.parametersX, 'Later ribbon groups do not shift when Instances enables');
        assert.ok(active.labelContentWidth <= active.labelWidth, 'The active count fits its label');
        await button.click();
        await popup.waitFor();
        await page.screenshot({ path: `${directory}/02-repeat-active-${width}.png` });
        await root();
        await settle();
        assert.equal(await popup.count(), 0, 'Leaving the repeat closes its popup');
        await enter('Transformer blocks');
        await settle();
        assert.equal(await popup.count(), 0, 'Re-entering does not reopen the old popup');
        measurements.push({ viewport: width, disabled, active });
    }
    await root();
    await enter('token embedding');
    await settle();
    assert.equal(await button.isDisabled(), true, 'An ordinary component has the same disabled control');
    await root();
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        const block = Object.values(state.getViewGraph().nodes).find((node) => node.label === 'Transformer blocks');
        state.patchRepeatNode(block.id, { label: 'Instances', count: 256 });
        state.enterSupernode(block.id);
    });
    await settle();
    const large = await measure();
    assert.ok(large.labelContentWidth <= large.labelWidth, 'A count of 256 remains readable');
    assert.equal(await button.innerText(), 'Instance 1/256');
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        state.setInspectedIteration(state.currentPath, 255);
    });
    await settle();
    assert.equal(await button.innerText(), 'Instance 256/256');
    const last = await measure();
    assert.ok(last.labelContentWidth <= last.labelWidth, 'The last of 256 instances remains readable');
    assert.deepEqual(errors, []);
    await writeFile(`${directory}/measurements.json`, `${JSON.stringify(measurements, null, 2)}\n`);
    console.log(`Persistent instance control passed: disabled root/ordinary component, stable layout at 1440/1024/820, active controls, dismissal, and 256 instances. Screenshots: ${directory}`);
} finally {
    await browser.close();
}
