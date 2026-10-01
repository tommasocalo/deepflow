import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { activeLevelCard, levelCards } from './level-navigation-browser-helper.mjs';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/hugging-face-import';
await mkdir(directory, { recursive: true });
const sha = '607a30d783dfa663caf39e06633721c8d4cfcd7e';
const models = {
    'openai-community/gpt2': 12,
    'distilbert/distilgpt2': 6,
};
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.route('https://huggingface.co/**', (route) => {
        const url = new URL(route.request().url());
        if (url.pathname === '/api/models') {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(Object.keys(models).map((id) => ({ id, pipeline_tag: 'text-generation' }))) });
        }
        const id = Object.keys(models).find((value) => url.pathname.includes(value));
        if (id && url.pathname.endsWith('/revision/main')) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id, sha }) });
        }
        if (id && url.pathname.endsWith(`/resolve/${sha}/config.json`)) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
                model_type: 'gpt2', architectures: ['GPT2LMHeadModel'], vocab_size: 50257,
                n_positions: 1024, n_embd: 768, n_layer: models[id], n_head: 12,
                activation_function: 'gelu_new', embd_pdrop: 0.1, attn_pdrop: 0.1,
                resid_pdrop: 0.1, layer_norm_epsilon: 1e-5,
            }) });
        }
        if (url.pathname.includes('google-bert/bert-base-uncased') && url.pathname.endsWith('/revision/main')) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 'google-bert/bert-base-uncased', sha }) });
        }
        if (url.pathname.includes('google-bert/bert-base-uncased') && url.pathname.endsWith(`/resolve/${sha}/config.json`)) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
                model_type: 'bert', architectures: ['BertForMaskedLM'], vocab_size: 30522,
                max_position_embeddings: 512, hidden_size: 768, num_hidden_layers: 12,
                num_attention_heads: 12, intermediate_size: 3072, type_vocab_size: 2,
                pad_token_id: 0, hidden_act: 'gelu', hidden_dropout_prob: 0.1,
                attention_probs_dropout_prob: 0.1, layer_norm_eps: 1e-12,
            }) });
        }
        if (url.pathname.includes('nlpconnect/vit-gpt2-image-captioning') && url.pathname.endsWith('/revision/main')) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ id: 'nlpconnect/vit-gpt2-image-captioning', sha }) });
        }
        if (url.pathname.includes('nlpconnect/vit-gpt2-image-captioning') && url.pathname.endsWith(`/resolve/${sha}/config.json`)) {
            return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
                model_type: 'vision-encoder-decoder', architectures: ['VisionEncoderDecoderModel'],
                encoder: { model_type: 'vit' }, decoder: { model_type: 'gpt2', add_cross_attention: true },
            }) });
        }
        return route.fulfill({ status: 404, body: 'Unexpected Hugging Face request' });
    });
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.parameters?.some((item) => item.name === 'n_layers'));
    const currentBlocks = () => page.evaluate(() => window.__deepflow.architectureStore.getState().graph.parameters.find((item) => item.name === 'n_layers').default);
    assert.equal(await currentBlocks(), 12);

    await page.getByRole('button', { name: 'Import from Hugging Face' }).click();
    let dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    await dialog.getByRole('button', { name: /openai-community\/gpt2/ }).click();
    await dialog.getByText('Supported', { exact: true }).waitFor();
    assert.match(await dialog.innerText(), /Dimensions match.*GPT-2 components.*identical layout may look unchanged/s);
    assert.equal(await dialog.evaluate((element) => getComputedStyle(element).backgroundColor), 'rgb(255, 255, 255)');
    assert.ok(await dialog.locator('.df-hf-search').evaluate((element) => element.getBoundingClientRect().height >= 44), 'Search control has a comfortable height');
    await page.screenshot({ path: `${directory}/01-same-config-1440.png` });
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await currentBlocks(), 12);
    await page.getByRole('link', { name: 'View imported Hugging Face source openai-community/gpt2' }).waitFor();
    assert.match(await activeLevelCard(page.locator('[data-graph-pane="main"]')).innerText(), /gpt2/, 'Imported model names the root level');
    assert.match(await page.locator('.df-import-source').first().innerText(), /openai-community\/gpt2/);
    assert.equal(await page.locator('.df-import-source').first().getAttribute('href'),
        `https://huggingface.co/openai-community/gpt2/tree/${sha}`);
    const sourcePlacement = await page.locator('.df-import-source').first().evaluate((element) => {
        const canvas = element.closest('.df-architecture-body').getBoundingClientRect();
        const link = element.getBoundingClientRect();
        return { inHeader: !!element.closest('.df-graph-pane-header'), rightGap: canvas.right - link.right,
            bottomGap: canvas.bottom - link.bottom, width: link.width };
    });
    assert.equal(sourcePlacement.inHeader, false, 'Source is outside the navigation tabs');
    assert.ok(sourcePlacement.rightGap >= 0 && sourcePlacement.rightGap <= 16
        && sourcePlacement.bottomGap >= 0 && sourcePlacement.bottomGap <= 16 && sourcePlacement.width >= 170,
        `Source sits quietly at the lower right of the canvas: ${JSON.stringify(sourcePlacement)}`);
    await page.screenshot({ path: `${directory}/02-imported-source-1440.png` });
    await page.getByRole('button', { name: 'Open Transformer blocks', exact: true }).click();
    assert.equal(await page.locator('.df-import-source').count(), 0, 'Provenance is hidden inside component details');
    await levelCards(page.locator('[data-graph-pane="main"]')).getByRole('tab').first().click();
    assert.equal(await page.locator('.df-import-source').count(), 1, 'Provenance returns at the root');

    const beforeFailure = await page.evaluate(() => {
        const store = window.__deepflow.architectureStore;
        const node = Object.values(store.getState().graph.nodes).find((item) => item.label === 'token embedding');
        store.getState().moveNode(node.id, { x: node.position.x + 40, y: node.position.y });
        const originalReset = store.getState().resetCurrentGraph;
        window.__hfOriginalReset = originalReset;
        store.setState({ resetCurrentGraph: (...args) => { originalReset(...args); throw new Error('Injected import failure after graph reset'); } });
        return { graph: JSON.stringify(store.getState().graph), canUndo: window.__deepflow.architectureHistory.getState().canUndo,
            nodeId: node.id, originalX: node.position.x, movedX: node.position.x + 40 };
    });
    assert.equal(beforeFailure.canUndo, true);

    await page.setViewportSize({ width: 820, height: 900 });
    await page.getByRole('button', { name: 'Import from Hugging Face' }).click();
    dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    const search = dialog.getByRole('searchbox', { name: 'Search Hugging Face models or paste a model URL' });
    await search.fill('distilbert/distilgpt2');
    await search.press('Enter');
    await dialog.getByText('Supported', { exact: true }).waitFor();
    assert.match(await dialog.innerText(), /blocks 12 → 6/);
    assert.ok(await dialog.locator('.df-hf-search').evaluate((element) => element.getBoundingClientRect().height >= 44), 'Search control remains tall at 820px');
    await page.screenshot({ path: `${directory}/03-changed-config-820.png` });
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.getByText('Import failed', { exact: true }).waitFor();
    assert.match(await dialog.innerText(), /Injected import failure after graph reset/);
    assert.equal(await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph)), beforeFailure.graph);
    assert.equal(await page.evaluate(() => window.__deepflow.architectureHistory.getState().canUndo), true);
    const restoredHistory = await page.evaluate((nodeId) => {
        const { architectureStore, architectureHistory } = window.__deepflow;
        const undone = architectureHistory.getState().undo();
        const undoX = architectureStore.getState().graph.nodes[nodeId].position.x;
        const redone = architectureHistory.getState().redo();
        const redoX = architectureStore.getState().graph.nodes[nodeId].position.x;
        return { undone, redone, undoX, redoX };
    }, beforeFailure.nodeId);
    assert.deepEqual(restoredHistory, { undone: true, redone: true, undoX: beforeFailure.originalX, redoX: beforeFailure.movedX });
    await page.screenshot({ path: `${directory}/04-import-error-820.png` });
    await page.evaluate(() => {
        window.__deepflow.architectureStore.setState({ resetCurrentGraph: window.__hfOriginalReset });
        delete window.__hfOriginalReset;
    });
    await dialog.getByRole('button', { name: 'Try again' }).click();
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await currentBlocks(), 6);
    assert.match(await page.locator('.df-import-source').first().getAttribute('title'), /distilbert\/distilgpt2/);
    assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.importSource.repoId), 'distilbert/distilgpt2');
    await page.screenshot({ path: `${directory}/05-changed-graph-820.png` });
    await page.getByRole('button', { name: 'Open to the side' }).click();
    await page.getByRole('button', { name: 'Import from Hugging Face' }).click();
    dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    await dialog.getByRole('button', { name: /openai-community\/gpt2/ }).click();
    await dialog.getByText('Supported', { exact: true }).waitFor();
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await currentBlocks(), 12);
    assert.equal(await page.locator('.df-import-source').count(), 1, 'Only the main root shows the imported source');
    assert.equal(await page.locator('[data-graph-pane="side"] .df-import-source').count(), 0);
    assert.match(await page.locator('.df-import-source').first().getAttribute('title'), /openai-community\/gpt2/);
    const paneSourceBounds = await page.locator('[data-graph-pane="main"]').evaluate((pane) => {
        const canvas = pane.querySelector('.df-architecture-body').getBoundingClientRect();
        const source = pane.querySelector('.df-import-source').getBoundingClientRect();
        return { rightGap: canvas.right - source.right, bottomGap: canvas.bottom - source.bottom,
            sourceWidth: source.width, sourceLeft: source.left, canvasLeft: canvas.left };
    });
    assert.equal(paneSourceBounds.sourceLeft >= paneSourceBounds.canvasLeft
        && paneSourceBounds.rightGap >= 0 && paneSourceBounds.rightGap <= 12
        && paneSourceBounds.bottomGap >= 0 && paneSourceBounds.bottomGap <= 12 && paneSourceBounds.sourceWidth >= 60, true,
    `Source link fits at the bottom right of the main canvas: ${JSON.stringify(paneSourceBounds)}`);
    await page.screenshot({ path: `${directory}/06-side-import-820.png` });
    await page.getByRole('button', { name: 'Import from Hugging Face' }).click();
    dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    await dialog.getByRole('button', { name: 'BERT masked LM' }).click();
    await dialog.getByText('Supported', { exact: true }).waitFor();
    assert.match(await dialog.innerText(), /google-bert\/bert-base-uncased/);
    assert.match(await dialog.innerText(), /Dimensions match.*BERT components/);
    await page.screenshot({ path: `${directory}/07-bert-example-820.png` });
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.importSource.repoId), 'google-bert/bert-base-uncased');
    assert.ok(await page.evaluate(() => Object.keys(window.__deepflow.architectureStore.getState().graph.nodes).length >= 15));
    await page.getByRole('button', { name: 'Close side view' }).click();
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.getByRole('button', { name: 'Fit active graph' }).click();
    await page.waitForTimeout(300);
    await page.screenshot({ path: `${directory}/08-bert-import-1440.png` });

    await page.setViewportSize({ width: 820, height: 900 });
    await page.getByRole('button', { name: 'Import from Hugging Face' }).click();
    dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    const visionSearch = dialog.getByRole('searchbox', { name: 'Search Hugging Face models or paste a model URL' });
    await visionSearch.fill('nlpconnect/vit-gpt2-image-captioning');
    await visionSearch.press('Enter');
    await dialog.getByText('Not mapped yet', { exact: true }).waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Import architecture' }).isDisabled(), true);
    await dialog.getByText('Inspect config.json').click();
    assert.match(await dialog.locator('.df-hf-config pre').innerText(), /vision-encoder-decoder/);
    await page.screenshot({ path: `${directory}/09-vision-unmapped-820.png` });
    assert.deepEqual(errors, []);
    console.log(`Hugging Face import: same and changed configurations pass. Screenshots: ${directory}`);
} finally {
    await browser.close();
}
