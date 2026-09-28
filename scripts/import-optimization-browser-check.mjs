import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/import-optimization';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    page.setDefaultTimeout(12000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const commit = '607a30d783dfa663caf39e06633721c8d4cfcd7e';
    await page.route('https://huggingface.co/api/models?*', (route) => route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify([
            { id: 'distilbert/distilgpt2', downloads: 420000, pipeline_tag: 'text-generation' },
            { id: 'google-bert/bert-base-uncased', downloads: 1000, pipeline_tag: 'fill-mask' },
        ]),
    }));
    await page.route('https://huggingface.co/api/models/distilbert/distilgpt2/revision/main', (route) => route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({ id: 'distilbert/distilgpt2', sha: commit }),
    }));
    await page.route(`https://huggingface.co/distilbert/distilgpt2/resolve/${commit}/config.json`, (route) => route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({
            model_type: 'gpt2', architectures: ['GPT2LMHeadModel'], vocab_size: 50257,
            n_positions: 1024, n_embd: 768, n_layer: 6, n_head: 12,
            activation_function: 'gelu_new', embd_pdrop: 0.1, attn_pdrop: 0.1,
            resid_pdrop: 0.1, layer_norm_epsilon: 1e-5,
        }),
    }));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    const ribbon = page.getByRole('region', { name: /Architecture ribbon/ });
    await ribbon.waitFor();
    await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.parameters.some((item) => item.name === 'd_model'));

    for (const width of [1440, 1024, 820]) {
        await page.setViewportSize({ width, height: 900 });
        const trigger = ribbon.getByRole('button', { name: 'Explore architecture settings' });
        await trigger.click();
        const dialog = page.getByRole('dialog', { name: 'Explore configurations' });
        const dialogBox = await dialog.boundingBox();
        assert.ok(dialogBox, `Optimization visible at ${width}px`);
        assert.ok(dialogBox.width > Math.min(width - 80, 700), `Optimization uses a workspace at ${width}px`);
        assert.ok(dialogBox.x >= 0 && dialogBox.x + dialogBox.width <= width, `Optimization stays on screen at ${width}px`);
        await page.screenshot({ path: `${directory}/${width}-optimization.png` });
        if (width === 1440) {
            await dialog.getByRole('button', { name: 'Compare configurations' }).click();
            await page.waitForFunction(() => document.querySelectorAll('.df-opt-candidate').length === 3);
            assert.equal(await dialog.locator('.df-opt-candidate').count(), 3);
            assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.parameters.find((item) => item.name === 'd_model').default), 768);
            await page.screenshot({ path: `${directory}/1440-optimization-results.png` });
            await dialog.getByRole('button', { name: 'Apply selected' }).click();
            assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.parameters.find((item) => item.name === 'd_model').default), 384);
            await dialog.getByRole('button', { name: 'Close optimization' }).click();
            await ribbon.getByRole('button', { name: 'Undo' }).click();
            assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.parameters.find((item) => item.name === 'd_model').default), 768);
        } else await dialog.getByRole('button', { name: 'Close optimization' }).click();
    }

    for (const width of [1440, 820]) {
        await page.setViewportSize({ width, height: 900 });
        await ribbon.getByRole('button', { name: 'Import from Hugging Face' }).click();
        const dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
        await dialog.getByRole('button', { name: /distilbert\/distilgpt2/ }).click();
        await dialog.getByText('Supported', { exact: true }).waitFor();
        assert.match(await dialog.innerText(), /pretrained weights are not loaded/);
        await page.screenshot({ path: `${directory}/${width}-hugging-face.png` });
        if (width === 820) {
            await dialog.getByRole('button', { name: 'Import architecture' }).click();
            await dialog.waitFor({ state: 'detached' });
            const imported = await page.evaluate(() => {
                const graph = window.__deepflow.architectureStore.getState().graph;
                return { layers: graph.parameters.find((item) => item.name === 'n_layers')?.default,
                    source: graph.importSource?.repoId };
            });
            assert.deepEqual(imported, { layers: 6, source: 'distilbert/distilgpt2' });
        } else await dialog.getByRole('button', { name: 'Close Hugging Face import' }).click();
    }
    assert.deepEqual(errors, []);
    console.log(`Optimization and Hugging Face import passed at 1440, 1024 and 820px. Screenshots: ${directory}`);
} finally {
    await browser.close();
}
