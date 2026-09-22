import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/inspector-redesign';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
    page.setDefaultTimeout(8000);
    const errors = [];
    const measurements = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.waitForSelector('.react-flow');
    const settle = () => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const fit = async () => { await page.getByRole('button', { name: 'Fit active graph', exact: true }).click(); await page.waitForTimeout(300); };
    const field = (name) => page.locator(`.df-inspector input[aria-label="${name}"]`);
    const inspect = async (name) => {
        const close = page.getByRole('button', { name: 'Close inspector', exact: true });
        if (await close.count()) await close.click();
        await page.getByRole('button', { name: `Inspect ${name}`, exact: true }).click();
        await settle();
    };
    // Navigation uses the actual document stores; all edits use visible inputs.
    const navigate = async (labels) => {
        await page.evaluate((names) => {
            const state = window.__deepflow.architectureStore.getState();
            state.openLevel([]);
            for (const name of names) {
                const current = window.__deepflow.architectureStore.getState();
                const node = Object.values(current.getViewGraph().nodes).find((candidate) => candidate.label === name);
                if (!node) throw new Error(`No component named ${name}`);
                current.enterSupernode(node.id);
            }
        }, labels);
        await fit();
    };
    const contained = async (name) => {
        const layout = await page.locator('.df-inspector').evaluate((element) => {
            const box = element.getBoundingClientRect();
            const body = element.querySelector('.df-inspector-body');
            const list = element.querySelector('[role="listbox"]')?.getBoundingClientRect();
            return { left: box.left, right: box.right, top: box.top, bottom: box.bottom, width: box.width, viewport: window.innerWidth, height: window.innerHeight, scrollWidth: body.scrollWidth, clientWidth: body.clientWidth, list: list && { left: list.left, right: list.right, top: list.top, bottom: list.bottom } };
        });
        measurements.push({ name, ...layout });
        assert.ok(layout.left >= 0 && layout.right <= layout.viewport, `${name}: inspector is within viewport ${JSON.stringify(layout)}`);
        assert.ok(layout.top >= 0 && layout.bottom <= layout.height + 1, `${name}: inspector stays within viewport height`);
        assert.ok(layout.scrollWidth <= layout.clientWidth + 1, `${name}: inspector has no horizontal overflow ${JSON.stringify(layout)}`);
        if (layout.list) {
            assert.ok(layout.list.left >= layout.left && layout.list.right <= layout.right + 1, `${name}: suggestions stay inside inspector width`);
            assert.ok(layout.list.top >= layout.top && layout.list.bottom <= layout.bottom + 1, `${name}: suggestions are not clipped at the inspector bottom`);
        }
        return layout;
    };
    const screenshot = async (name) => {
        await contained(name);
        await page.screenshot({ path: `${directory}/${name}.png` });
    };

    await fit();
    await inspect('Transformer blocks');
    for (const name of ['count', 'd_model', 'n_heads']) {
        const geometry = await field(name).evaluate((input) => {
            const label = input.closest('.df-param-row').querySelector('label');
            const inputBox = input.getBoundingClientRect();
            const labelBox = label.getBoundingClientRect();
            return { inputTop: inputBox.top, labelBottom: labelBox.bottom, inputLeft: inputBox.left, labelLeft: labelBox.left };
        });
        assert.ok(geometry.labelBottom <= geometry.inputTop, `${name}: label is above its control`);
        assert.ok(Math.abs(geometry.inputLeft - geometry.labelLeft) < 1, `${name}: label and input align`);
    }
    assert.equal(await page.getByRole('button', { name: 'Make copy', exact: true }).count(), 0);
    await screenshot('01-transformer-settings-1440');

    await navigate(['positional embedding', 'position table']);
    await inspect('embedding table');
    const first = field('shape dimension 1');
    const second = field('shape dimension 2');
    assert.equal(await first.inputValue(), '$num_embeddings');
    assert.equal(await second.inputValue(), '$embedding_dim');
    const shapeGeometry = () => page.locator('.df-shape-cell').evaluateAll((elements) => elements.map((element) => {
        const rect = element.getBoundingClientRect();
        return { width: rect.width, top: rect.top, left: rect.left, right: rect.right };
    }));
    const initialShape = await shapeGeometry();
    assert.equal(initialShape.length, 2);
    assert.equal(initialShape[0].top, initialShape[1].top, 'Two dimensions are on one row');
    assert.ok(initialShape.every((cell) => cell.width >= 126), 'Symbolic dimensions have stable readable widths');
    assert.equal(await page.locator('.df-param-shape [aria-invalid="true"]').count(), 0, 'Valid inherited references are not marked as errors');
    await screenshot('02-position-table-dimensions-1440');
    await second.fill('$emb');
    const matching = page.getByRole('listbox', { name: 'Available parameters' });
    assert.match(await matching.innerText(), /\$embedding_dim\s*768/);
    assert.match(await matching.innerText(), /position table/);
    assert.deepEqual(await shapeGeometry(), initialShape, 'Typing a reference does not resize or wrap shape cells');
    await screenshot('03-shape-variable-suggestions-1440');
    await second.press('Enter');
    assert.equal(await second.inputValue(), '$embedding_dim');
    assert.equal(await matching.count(), 0);
    assert.equal(await page.locator('.df-param-shape [aria-invalid="true"]').count(), 0);
    await first.fill('2 * $num + 0');
    await first.evaluate((input) => { input.setSelectionRange(8, 8); input.dispatchEvent(new Event('select', { bubbles: true })); });
    await first.press('ArrowLeft');
    await first.press('ArrowRight');
    await first.press('Enter');
    assert.equal(await first.inputValue(), '2 * $num_embeddings + 0', 'Completion preserves arithmetic around a reference');
    await first.fill('$num_embeddings');
    await first.press('Escape');
    await first.blur();
    assert.equal(await page.locator('.df-param-shape [aria-invalid="true"]').count(), 0);
    await page.getByRole('button', { name: 'Add shape dimension', exact: true }).click();
    const third = field('shape dimension 3');
    await third.fill('2');
    assert.equal((await shapeGeometry())[2].top, initialShape[0].top, 'Additional dimensions scroll on the same row');
    await contained('three-axis-shape');
    await page.getByRole('button', { name: 'Remove shape dimension 3', exact: true }).click();

    await page.setViewportSize({ width: 1024, height: 768 });
    await fit();
    await screenshot('04-position-table-dimensions-1024');
    await second.fill('$');
    assert.equal(await page.locator('.df-param-shape [aria-invalid="true"]').count(), 0, 'A partial reference is not painted red while typing');
    assert.match(await page.locator('.df-parameter-source').innerText(), /Unresolved/);
    assert.equal(await page.getByRole('button', { name: 'Unlink shape', exact: true }).count(), 0, 'An incomplete reference cannot detach into an invalid local value');
    await screenshot('05-shape-variable-suggestions-1024');
    await second.fill('$embedding_dim');
    await second.press('Escape');
    await second.blur();

    await page.setViewportSize({ width: 1440, height: 1000 });
    await navigate(['Transformer blocks', 'causal self-attention']);
    await inspect('Attention heads');
    assert.equal(await field('dropout').inputValue(), '$dropout');
    const source = page.locator('.df-parameter-source');
    assert.match(await source.innerText(), /Linked\s+0.1\s+from causal self-attention/);
    assert.equal(await source.getByRole('button').count(), 1, 'Only the explicit Unlink action remains');
    assert.equal(await page.getByRole('button', { name: /Parameter options/ }).count(), 0);
    assert.equal(await page.getByRole('link').count(), 0);
    await screenshot('06-attention-heads-linked-1440');

    await navigate(['Transformer blocks', 'causal self-attention', 'Attention heads']);
    await inspect('attention dropout');
    assert.equal(await field('p').inputValue(), '$dropout');
    assert.match(await source.innerText(), /Linked\s+0.1/);
    await screenshot('07-dropout-link-1440');
    await page.getByRole('button', { name: 'Unlink p', exact: true }).click();
    assert.equal(await field('p').inputValue(), '0.1');
    assert.equal(await source.count(), 0);
    await field('p').fill('0.25');
    await field('p').blur();
    assert.equal(await field('p').inputValue(), '0.25');
    await field('p').fill('$dr');
    assert.match(await matching.innerText(), /\$dropout\s*0.1/);
    await field('p').press('Enter');
    assert.equal(await field('p').inputValue(), '$dropout');
    assert.match(await source.innerText(), /Linked\s+0.1/);

    await page.getByRole('button', { name: 'Expose a field…', exact: true }).click();
    await page.getByRole('combobox', { name: 'Field to expose', exact: true }).selectOption('p');
    await page.getByRole('textbox', { name: 'Exposed parameter name', exact: true }).fill('attention_probability');
    await screenshot('08-expose-single-field-1440');
    await page.getByRole('button', { name: 'Expose parameter', exact: true }).click();
    assert.equal(await field('p').inputValue(), '$attention_probability');
    assert.match(await source.innerText(), /Linked\s+0.1/);
    await page.getByRole('button', { name: 'Undo', exact: true }).click();
    assert.equal(await field('p').inputValue(), '$dropout');

    await page.setViewportSize({ width: 1024, height: 768 });
    await fit();
    await screenshot('09-dropout-link-1024');
    await navigate([]);
    await inspect('Add');
    assert.equal(await page.locator('.df-inspector-title').innerText(), 'Operation');
    const beforeIcon = await page.locator('.df-inspector-header .df-operation-icon').innerHTML();
    await page.getByRole('combobox', { name: 'Operation type', exact: true }).selectOption('deepflow.op.Multiply');
    assert.equal(await page.locator('.df-inspector-title').innerText(), 'Operation');
    const afterIcon = await page.locator('.df-inspector-header .df-operation-icon').innerHTML();
    assert.notEqual(afterIcon, beforeIcon, 'Changing operation Type changes its symbol');
    await screenshot('10-operation-type-1024');
    assert.deepEqual(errors, []);
    await writeFile(`${directory}/measurements.json`, JSON.stringify(measurements, null, 2));
    console.log('Inspector redesign browser checks passed: aligned labels, stable one-row shapes, scoped autocomplete, inherited validation, explicit unlink/relink, expose+Undo, operation icon/type, 1440/1024 containment.');
} finally { await browser.close(); }
