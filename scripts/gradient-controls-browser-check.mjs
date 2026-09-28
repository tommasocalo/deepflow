import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/gradient-factor';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    const measurements = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    const detail = page.getByRole('complementary', { name: 'Connection gradient settings', exact: true });
    const rules = detail.getByRole('group', { name: 'Connection gradient rule', exact: true });
    const radio = (name) => rules.getByRole('radio', { name, exact: true });
    const factor = detail.getByLabel('Gradient reversal factor', { exact: true });
    const commitFactor = async (value) => { await factor.fill(value); await factor.press('Tab'); await settle(); };
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const graph = () => page.evaluate(() => window.__deepflow.architectureStore.getState().getViewGraph());
    const history = () => page.evaluate(() => window.__deepflow.architectureHistory.getState().undoCount);
    const selected = () => page.locator('.react-flow__edge.selected').evaluateAll((elements) => elements.map((element) => element.dataset.id).sort());
    const fit = async () => {
        await ribbon.getByRole('button', { name: 'Fit active graph', exact: true }).click();
        await page.waitForTimeout(300);
    };
    const selectEdge = async (id, modifier) => {
        const edge = page.locator(`.react-flow__edge[data-id="${id}"] .react-flow__edge-path`);
        const point = await edge.evaluate((element) => {
            for (const fraction of [.55, .4, .7, .3, .8, .2]) {
                const local = element.getPointAtLength(element.getTotalLength() * fraction);
                const screen = new DOMPoint(local.x, local.y).matrixTransform(element.getScreenCTM());
                const hit = document.elementFromPoint(screen.x, screen.y)?.closest('.react-flow__edge');
                if (hit?.dataset.id === element.closest('.react-flow__edge').dataset.id) return { x: screen.x, y: screen.y };
            }
            return null;
        });
        assert.ok(point, `An exposed section of edge ${id} can be clicked`);
        if (modifier) await page.keyboard.down(modifier);
        try { await page.mouse.click(point.x, point.y); }
        finally { if (modifier) await page.keyboard.up(modifier); }
        await detail.waitFor();
        await settle();
    };
    const checkRule = async (name) => {
        await radio(name).check();
        await settle();
        assert.equal(await radio(name).isChecked(), true, `${name} is selected`);
        assert.equal(await rules.locator('input:checked').count(), 1);
    };
    const capture = async (name) => {
        await settle();
        await page.waitForFunction(() => [...document.querySelectorAll('.df-gradient-choice img')].length === 4
            && [...document.querySelectorAll('.df-gradient-choice img')].every((img) => img.complete && img.naturalWidth > 0));
        const measure = await detail.evaluate((element) => {
            const box = element.getBoundingClientRect();
            const choices = [...element.querySelectorAll('.df-gradient-choice')].map((choice) => {
                const rect = choice.getBoundingClientRect();
                return { x: rect.x, y: rect.y, width: rect.width, height: rect.height, client: choice.clientWidth, content: choice.scrollWidth };
            });
            return { viewport: innerWidth, x: box.x, y: box.y, width: box.width, height: box.height, client: element.clientWidth, content: element.scrollWidth,
                pageWidth: document.documentElement.scrollWidth, choices };
        });
        assert.ok(measure.x >= 0 && measure.x + measure.width <= measure.viewport, `${name}: inspector stays in viewport`);
        assert.ok(measure.content <= measure.client, `${name}: inspector has no horizontal overflow`);
        assert.ok(measure.pageWidth <= measure.viewport, `${name}: page has no horizontal overflow`);
        assert.equal(measure.choices.length, 4);
        assert.ok(measure.choices.every((choice) => Math.abs(choice.y - measure.choices[0].y) < 1), `${name}: all four choices occupy one row`);
        assert.ok(measure.choices.every((choice) => choice.content <= choice.client), `${name}: labels fit their tiles`);
        measurements.push({ name, ...measure });
        await page.screenshot({ path: `${directory}/${name}.png` });
    };

    // Load the shipped example through the same menu used in the editor.
    await page.getByRole('button', { name: 'Examples', exact: true }).click();
    await page.getByRole('group', { name: 'Example architectures', exact: true }).getByRole('button', { name: /^DANN/ }).click();
    await ribbon.getByRole('button', { name: 'Show gradient flow', exact: true }).click();
    await fit();
    const initial = await graph();
    const node = (label) => Object.values(initial.nodes).find((candidate) => candidate.label === label);
    const reverse = initial.edges.find((edge) => edge.gradientOp === 'reverse');
    const ordinary = initial.edges.find((edge) => edge.target === node('class head').id);
    const inputEdge = initial.edges.find((edge) => edge.source === node('Input').id);
    assert.ok(reverse && ordinary && inputEdge);
    assert.equal(await page.getByRole('note', { name: 'Gradient path legend', exact: true }).count(), 0);
    assert.equal(await page.getByText('Gradient paths', { exact: true }).count(), 0);

    await selectEdge(reverse.id);
    assert.deepEqual(await rules.getByRole('radio').evaluateAll((elements) => elements.map((element) => element.getAttribute('aria-label'))), ['Flow', 'Stop', 'Detach', 'Reverse']);
    assert.equal(await radio('Reverse').isChecked(), true);
    assert.equal(await factor.inputValue(), '1');
    await capture('02-reverse-1440');
    await commitFactor('0.5');
    await checkRule('Flow');
    assert.equal(await factor.count(), 0);
    await checkRule('Reverse');
    assert.equal(await factor.inputValue(), '0.5', 'Returning to Reverse retains its factor');
    await commitFactor('0');
    await settle();
    assert.equal(await radio('Reverse').isChecked(), true);
    assert.equal(await detail.getByText('Stopped', { exact: true }).count(), 0, 'Redundant status captions are absent');
    assert.equal(await page.locator(`.react-flow__edge[data-id="${reverse.id}"] .df-gradient-edge--stopped`).count(), 1,
        'A zero reversal factor still stops gradient analysis');
    await commitFactor('0.5');

    // Native radio arrows update the value without deleting or moving graph elements.
    await radio('Flow').check();
    await radio('Flow').focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await radio('Stop').isChecked(), true);
    await page.keyboard.press('ArrowRight');
    assert.equal(await radio('Detach').isChecked(), true);
    assert.equal((await graph()).edges.find((edge) => edge.id === reverse.id).gradientOp, 'detach');
    await page.keyboard.press('ArrowRight');
    assert.equal(await radio('Reverse').isChecked(), true);
    assert.equal(await radio('Reverse').evaluate((element) => element === document.activeElement), true);

    await selectEdge(ordinary.id, 'Meta');
    assert.deepEqual(await selected(), [ordinary.id, reverse.id].sort());
    assert.equal(await detail.getByText('Mixed rules', { exact: true }).count(), 1);
    assert.equal(await rules.locator('input:checked').count(), 0, 'Mixed rules do not imply one selected operation');
    await capture('03-mixed-1440');
    const beforeBatch = await history();
    await checkRule('Stop');
    assert.equal(await history(), beforeBatch + 1, 'Editing multiple connections records one undo');
    assert.ok((await graph()).edges.filter((edge) => [ordinary.id, reverse.id].includes(edge.id)).every((edge) => edge.gradientOp === 'stop'));
    await ribbon.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.equal((await graph()).edges.find((edge) => edge.id === reverse.id).gradientOp, 'reverse');
    assert.equal((await graph()).edges.find((edge) => edge.id === ordinary.id).gradientOp, 'none');
    await checkRule('Reverse');
    assert.equal(await factor.inputValue(), '', 'Different reversal factors use an empty Mixed input');
    assert.equal(await factor.getAttribute('placeholder'), 'Mixed');
    await capture('04-mixed-factors-1440');
    const beforeFactor = await history();
    await commitFactor('0.25');
    assert.equal(await history(), beforeFactor + 1, 'Changing a shared factor records one undo');
    assert.ok((await graph()).edges.filter((edge) => [ordinary.id, reverse.id].includes(edge.id)).every((edge) => edge.gradientOpLambda === .25));
    await ribbon.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.equal(await factor.inputValue(), '');

    // A rule describes what to do. An analyzed tensor state is not repeated as a competing caption.
    await detail.getByRole('button', { name: 'Close connection settings', exact: true }).click();
    await page.getByRole('button', { name: 'Inspect Input', exact: true }).click();
    await page.getByRole('combobox', { name: 'dtype', exact: true }).selectOption('int64');
    await page.getByRole('button', { name: 'Close inspector', exact: true }).click();
    await selectEdge(inputEdge.id);
    assert.equal(await radio('Flow').isChecked(), true);
    assert.equal(await detail.getByText('No gradient', { exact: true }).count(), 0, 'Flow does not show a conflicting status caption');
    await capture('05-flow-no-status-1440');
    await detail.getByRole('button', { name: 'Close connection settings', exact: true }).click();
    await ribbon.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.equal((await graph()).nodes[node('Input').id].parameters.dtype, 'float32');

    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 900 });
        await fit();
        await selectEdge(reverse.id);
        assert.equal(await radio('Reverse').isChecked(), true);
        await capture(`06-compact-${width}`);
        await detail.getByRole('button', { name: 'Close connection settings', exact: true }).click();
    }
    await selectEdge(reverse.id);
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        for (const [name, value] of Object.entries({ lambda: .75, alpha: .1, beta: .2, gamma: .3, delta: .4,
            scale: .5, sigma: .6, temperature: .7 })) state.addGlobalParameter({ name, type: 'float', default: value });
    });
    for (const width of [1440, 820]) {
        await detail.getByRole('button', { name: 'Close connection settings', exact: true }).click();
        await page.setViewportSize({ width, height: 900 });
        await fit();
        await selectEdge(reverse.id);
        const before = await detail.boundingBox();
        await factor.fill('$');
        const option = page.getByRole('option', { name: /\$lambda/ });
        await option.waitFor();
        assert.ok(await page.getByRole('listbox', { name: 'Available parameters' }).getByRole('option').count() >= 8);
        await settle();
        const popup = await page.locator('.df-variable-suggestions.is-floating').evaluate((element) => {
            const panel = document.querySelector('.df-connection-inspector');
            const panelRect = panel.getBoundingClientRect();
            const box = element.getBoundingClientRect();
            return { position: getComputedStyle(element).position, insidePanel: panel.contains(element),
                panel: { x: panelRect.x, y: panelRect.y, width: panelRect.width, height: panelRect.height },
                x: box.x, y: box.y, width: box.width, height: box.height,
                viewportWidth: document.documentElement.clientWidth, viewportHeight: document.documentElement.clientHeight };
        });
        assert.equal(popup.position, 'fixed', 'Suggestions float above the canvas');
        assert.equal(popup.insidePanel, false, 'Suggestions escape the clipped inspector frame');
        for (const dimension of ['x', 'y', 'width', 'height']) assert.ok(Math.abs(popup.panel[dimension] - before[dimension]) < 1,
            `Opening suggestions does not move or resize the inspector (${dimension})`);
        assert.ok(popup.x >= 0 && popup.x + popup.width <= popup.viewportWidth, `${width}: suggestions fit horizontally`);
        assert.ok(popup.y >= 0 && popup.y + popup.height <= popup.viewportHeight, `${width}: suggestions fit vertically`);
        assert.ok(popup.height <= 180, `${width}: suggestions scroll inside their own bounded menu`);
        await page.screenshot({ path: `${directory}/07-suggestions-${width}.png` });
        const viewportBefore = await page.locator('.react-flow__viewport').getAttribute('style');
        for (let index = 0; index < 7; index += 1) await factor.press('ArrowDown');
        assert.ok(await page.locator('.df-variable-suggestions.is-floating').evaluate((element) => element.scrollTop) > 0,
            `${width}: keyboard navigation scrolls inside the suggestion menu`);
        assert.equal(await page.locator('.react-flow__viewport').getAttribute('style'), viewportBefore,
            `${width}: keyboard navigation does not pan the graph`);
        await factor.press('Escape');
    }
    await factor.fill('');
    await factor.fill('$la');
    await page.getByRole('option', { name: /\$lambda/ }).click();
    await factor.press('Tab');
    await settle();
    assert.equal((await graph()).edges.find((edge) => edge.id === reverse.id).gradientOpLambda, '$lambda');
    assert.equal(await detail.getByText('= 0.75', { exact: true }).count(), 1);
    await capture('07-variable-factor-820');
    await factor.fill('$missing');
    await factor.press('Tab');
    assert.equal(await detail.getByRole('alert').count(), 1, 'Invalid references show an inline error');
    assert.equal((await graph()).edges.find((edge) => edge.id === reverse.id).gradientOpLambda, '$lambda', 'Invalid references are not stored');
    await factor.fill('$lambda');
    await factor.press('Tab');
    assert.equal(await detail.getByRole('alert').count(), 0);
    assert.deepEqual(errors, []);
    await writeFile(`${directory}/measurements.json`, `${JSON.stringify(measurements, null, 2)}\n`);
    console.log(`Gradient controls checks passed: four direct choices, loaded icons, keyboard navigation, mixed rules/factors, variable completion, atomic undo, no conflicting status caption and compact layout at 1440/1024/820. Screenshots: ${directory}`);
} finally { await browser.close(); }
