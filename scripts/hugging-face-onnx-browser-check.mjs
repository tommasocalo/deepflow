import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { activeLevelCard } from './level-navigation-browser-helper.mjs';

const directory = 'screenshots/hugging-face-onnx';
await mkdir(directory, { recursive: true });
const sha = '1234567890abcdef1234567890abcdef12345678';
const cnn = await readFile(new URL('./fixtures/tiny-residual-cnn.onnx', import.meta.url));
const unmapped = await readFile(new URL('./fixtures/tiny-unsupported.onnx', import.meta.url));
const downloads = [];
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.route('https://huggingface.co/**', (route) => {
        const url = new URL(route.request().url());
        if (url.pathname === '/api/models') return route.fulfill({ json: [
            { id: 'deepflow-fixtures/residual-cnn', pipeline_tag: 'image-classification' },
            { id: 'deepflow-fixtures/model-parts', pipeline_tag: 'image-to-text' },
        ] });
        const multi = url.pathname.includes('/model-parts/');
        if (url.pathname.endsWith('/revision/main')) return route.fulfill({ json: {
            id: multi ? 'deepflow-fixtures/model-parts' : 'deepflow-fixtures/residual-cnn', sha,
            siblings: (multi ? ['encoder/model.onnx', 'decoder/model.onnx', 'broken/model.onnx'] : ['onnx/model.onnx'])
                .map((rfilename) => ({ rfilename })),
        } });
        if (url.pathname.endsWith('/config.json')) return route.fulfill({ status: 404, body: 'No config in ONNX repository' });
        if (url.pathname.endsWith('.onnx')) {
            downloads.push(url.pathname);
            assert.ok(url.pathname.includes(`/resolve/${sha}/`), 'Source graph uses the pinned revision');
            if (url.pathname.includes('/broken/')) return route.fulfill({ status: 503, body: 'Temporary failure' });
            return route.fulfill({ contentType: 'application/octet-stream', body: url.pathname.includes('/decoder/') ? unmapped : cnn });
        }
        return route.fulfill({ status: 404, body: 'Unexpected request' });
    });
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.nodes);
    assert.equal(await page.getByRole('button', { name: 'Import ONNX file', exact: true }).count(), 0);
    await page.getByRole('button', { name: 'Import from Hugging Face', exact: true }).click();
    const dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    await dialog.getByRole('button', { name: /deepflow-fixtures\/residual-cnn/ }).click();
    await dialog.getByText('Supported', { exact: true }).waitFor();
    assert.equal(downloads.length, 1, 'Single published graph is resolved automatically');
    assert.equal(await dialog.getByText('Inspect config.json').count(), 0, 'An ONNX-only repo does not require config.json');
    await dialog.getByText('Inspect source mapping').click();
    assert.match(await dialog.locator('.df-hf-source-map').innerText(), /Conv2d/);
    assert.match(await dialog.locator('.df-hf-source-map').innerText(), /Add/);
    await page.screenshot({ path: `${directory}/01-automatic-bridge-1440.png` });
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.waitFor({ state: 'detached' });
    const imported = await page.evaluate(() => {
        const graph = window.__deepflow.architectureStore.getState().graph;
        return { source: graph.importSource, keys: Object.values(graph.nodes).map((node) => node.libraryKey),
            fanOut: Object.values(graph.nodes).some((node) => graph.edges.filter((edge) => edge.source === node.id).length > 1) };
    });
    assert.equal(imported.source.provider, 'huggingface');
    assert.equal(imported.source.intermediateFormat, 'onnx');
    assert.equal(imported.source.repoId, 'deepflow-fixtures/residual-cnn');
    assert.equal(imported.source.commit, sha);
    assert.equal(imported.source.filePath, 'onnx/model.onnx');
    assert.equal(imported.source.weightsLoaded, false);
    assert.ok(imported.keys.includes('torch.nn.Conv2d'));
    assert.ok(imported.keys.includes('deepflow.op.Add'));
    assert.equal(imported.fanOut, true);
    await page.getByRole('button', { name: 'Fit active graph', exact: true }).click();
    await page.waitForTimeout(350);
    assert.match(await activeLevelCard(page.locator('[data-graph-pane="main"]')).innerText(), /residual-cnn/);
    await page.screenshot({ path: `${directory}/02-imported-cnn-1440.png` });
    await page.reload();
    await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.importSource?.intermediateFormat === 'onnx');
    const before = await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph));
    await page.setViewportSize({ width: 820, height: 900 });
    await page.getByRole('button', { name: 'Import from Hugging Face', exact: true }).click();
    await dialog.getByRole('button', { name: /deepflow-fixtures\/model-parts/ }).click();
    await dialog.getByText('Choose graph', { exact: true }).waitFor();
    assert.equal(downloads.length, 1, 'Multiple graph files require an explicit choice');
    assert.equal(await dialog.getByRole('button', { name: 'Import architecture' }).isDisabled(), true);
    const source = dialog.getByRole('combobox', { name: 'Hugging Face source graph' });
    await source.selectOption('decoder/model.onnx');
    await dialog.getByText('Not mapped yet', { exact: true }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Import architecture' }).isDisabled(), true);
    assert.match(await dialog.locator('.df-hf-unsupported').innerText(), /Abs/);
    assert.equal(await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph)), before);
    await page.screenshot({ path: `${directory}/03-unmapped-part-820.png` });
    await source.selectOption('encoder/model.onnx');
    await dialog.getByText('Supported', { exact: true }).waitFor();
    assert.match(await dialog.innerText(), /Imports only encoder\/model.onnx/);
    await page.screenshot({ path: `${directory}/04-part-selection-820.png` });
    await source.selectOption('broken/model.onnx');
    await dialog.getByRole('alert').filter({ hasText: '503' }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Import architecture' }).isDisabled(), true, 'Failed new choice cannot import old graph');
    assert.equal(await source.inputValue(), 'broken/model.onnx', 'Failed choice stays selected for a clear retry');
    assert.equal(await dialog.getByLabel('Source graph summary').count(), 0, 'Failed choice clears the old graph preview');
    assert.equal(await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph)), before);
    await source.selectOption('encoder/model.onnx');
    await dialog.getByText('Supported', { exact: true }).waitFor();
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.importSource.filePath), 'encoder/model.onnx');
    assert.deepEqual(errors, []);
    console.log('Hugging Face → ONNX → editable residual CNN, multiple graph choices, failure preservation and reload pass.');
} finally { await browser.close(); }
