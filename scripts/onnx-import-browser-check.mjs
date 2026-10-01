import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? 'screenshots/onnx-import';
const fixture = (name) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    page.setDefaultTimeout(10000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Load', exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Import ONNX file' }).count(), 0, 'ONNX is a Load format, not an Insert item');
    const loadFile = async (file) => {
        const chooser = page.waitForEvent('filechooser');
        await page.getByRole('button', { name: 'Load', exact: true }).click();
        await (await chooser).setFiles(file);
    };
    const before = await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph));
    await page.screenshot({ path: `${directory}/01-load-in-toolbar-1440.png` });
    await loadFile(fixture('tiny-unsupported.onnx'));
    let dialog = page.getByRole('dialog', { name: 'Import ONNX' });
    await dialog.getByText('Operator view only').waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Import editable graph' }).isDisabled(), true);
    assert.match(await dialog.innerText(), /Abs.*no editable mapping/s);
    assert.equal(await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph)), before);
    await page.screenshot({ path: `${directory}/02-unmapped-abs-1440.png` });

    await dialog.getByLabel('Choose an ONNX file').setInputFiles(fixture('tiny-linear-relu.onnx'));
    await dialog.getByText('Editable mapping complete').waitFor();
    assert.equal(await dialog.getByRole('button', { name: 'Import editable graph' }).isEnabled(), true);
    assert.match(await dialog.locator('.df-onnx-operator-list').innerText(), /Gemm.*Relu/s);
    assert.match(await dialog.locator('.df-onnx-component-list').innerText(), /Linear.*Relu/s);
    assert.match(await dialog.innerText(), /All 2 operators mapped/);
    await page.screenshot({ path: `${directory}/03-mapped-preview-1440.png` });
    await page.setViewportSize({ width: 820, height: 900 });
    const bounds = await dialog.evaluate((element) => ({ right: element.getBoundingClientRect().right,
        viewport: window.innerWidth, scrollWidth: element.scrollWidth, width: element.clientWidth }));
    assert.ok(bounds.right <= bounds.viewport && bounds.scrollWidth <= bounds.width + 1, `Dialog fits at 820px: ${JSON.stringify(bounds)}`);
    await page.screenshot({ path: `${directory}/04-mapped-preview-820.png` });

    await dialog.getByRole('button', { name: 'Import editable graph' }).click();
    await dialog.waitFor({ state: 'detached' });
    const imported = await page.evaluate(() => {
        const { architectureStore, trainingStore } = window.__deepflow;
        const graph = architectureStore.getState().graph;
        return { source: graph.importSource, keys: Object.values(graph.nodes).map((node) => node.libraryKey),
            edges: graph.edges.length, trainingNodes: Object.keys(trainingStore.getState().graph.nodes).length };
    });
    assert.equal(imported.source.provider, 'onnx');
    assert.equal(imported.source.fileName, 'tiny-linear-relu.onnx');
    assert.equal(imported.source.weightsLoaded, false);
    assert.equal(imported.source.mappedOpCount, 2);
    assert.deepEqual(imported.keys, ['deepflow.io.Input', 'torch.nn.Linear', 'torch.nn.ReLU', 'deepflow.io.Output']);
    assert.equal(imported.edges, 3);
    assert.equal(imported.trainingNodes, 0);
    await page.waitForFunction(() => [...document.querySelectorAll('.react-flow__node')].every((node) => {
        const bounds = node.getBoundingClientRect();
        return bounds.left >= 0 && bounds.right <= window.innerWidth && bounds.top >= 0 && bounds.bottom <= window.innerHeight;
    }));
    await page.getByRole('button', { name: 'Inspect dense' }).click();
    const sourceDetails = page.locator('details.df-inspector-onnx-source');
    await sourceDetails.waitFor();
    assert.match(await sourceDetails.innerText(), /Imported ONNX.*Gemm/s);
    await sourceDetails.locator('summary').click();
    assert.match(await sourceDetails.innerText(), /dense.*x, W, b/s);
    await page.screenshot({ path: `${directory}/05-imported-node-820.png` });

    await page.reload();
    await page.waitForFunction(() => window.__deepflow?.architectureStore.getState().graph.importSource?.provider === 'onnx');
    assert.deepEqual(await page.evaluate(() => window.__deepflow.architectureStore.getState().graph.importSource.operatorGraph.nodes.map((node) => node.opType)),
        ['Gemm', 'Relu'], 'The original operator map survives reload');
    await page.getByRole('button', { name: 'Inspect dense' }).click();
    await sourceDetails.locator('summary').click();
    assert.match(await sourceDetails.innerText(), /dense.*x, W, b/s);
    await page.screenshot({ path: `${directory}/06-source-after-reload-820.png` });

    const beforeFailure = await page.evaluate(() => {
        const { architectureStore, architectureHistory, trainingStore } = window.__deepflow;
        const node = architectureStore.getState().graph.nodes['onnx-mapped:1'];
        architectureStore.getState().moveNode(node.id, { x: node.position.x + 24, y: node.position.y });
        window.__onnxOriginalTrainingReset = trainingStore.getState().resetGraph;
        trainingStore.setState({ resetGraph: () => { throw new Error('Injected ONNX import failure'); } });
        return { graph: JSON.stringify(architectureStore.getState().graph), canUndo: architectureHistory.getState().canUndo };
    });
    assert.equal(beforeFailure.canUndo, true);
    await loadFile(fixture('tiny-linear-relu.onnx'));
    dialog = page.getByRole('dialog', { name: 'Import ONNX' });
    await dialog.getByText('Editable mapping complete').waitFor();
    await dialog.getByRole('button', { name: 'Import editable graph' }).click();
    await dialog.getByText(/Injected ONNX import failure/).waitFor();
    assert.equal(await page.evaluate(() => JSON.stringify(window.__deepflow.architectureStore.getState().graph)), beforeFailure.graph);
    assert.equal(await page.evaluate(() => window.__deepflow.architectureHistory.getState().canUndo), true);
    await page.screenshot({ path: `${directory}/07-import-error-820.png` });
    await page.evaluate(() => {
        window.__deepflow.trainingStore.setState({ resetGraph: window.__onnxOriginalTrainingReset });
        delete window.__onnxOriginalTrainingReset;
    });
    await dialog.getByRole('button', { name: 'Close ONNX import' }).click();

    // Save/Load still round-trips DeepFlow projects through the same button.
    const saved = await page.evaluate(() => ({
        architecture: window.__deepflow.architectureStore.getState().graph,
        templates: window.__deepflow.templatesStore.getState().templates,
        training: window.__deepflow.trainingStore.getState().graph,
    }));
    const downloadPromise = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Save', exact: true }).click();
    const download = await downloadPromise;
    const downloadPath = await download.path();
    await page.evaluate(() => {
        const state = window.__deepflow.architectureStore.getState();
        state.moveNode('onnx-mapped:1', { x: -2000, y: 3000 });
    });
    await loadFile({ name: 'deepflow-graph.json', mimeType: 'application/json', buffer: await readFile(downloadPath) });
    await page.waitForFunction((expected) => JSON.stringify(window.__deepflow.architectureStore.getState().graph) === expected, JSON.stringify(saved.architecture));
    assert.deepEqual(await page.evaluate(() => ({
        architecture: window.__deepflow.architectureStore.getState().graph,
        templates: window.__deepflow.templatesStore.getState().templates,
        training: window.__deepflow.trainingStore.getState().graph,
    })), saved, 'DeepFlow JSON still restores architecture, templates and training');
    assert.equal(await page.evaluate(() => window.__deepflow.architectureHistory.getState().canUndo), false);
    assert.equal(await page.getByRole('dialog', { name: 'Import ONNX' }).count(), 0);
    await page.screenshot({ path: `${directory}/08-json-loaded-820.png` });

    assert.deepEqual(errors, [], `Browser errors: ${errors.join('\n')}`);
    process.stdout.write('ONNX import browser check passed.\n');
} finally {
    await browser.close();
}
