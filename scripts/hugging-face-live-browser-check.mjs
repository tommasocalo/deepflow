// Deliberately separate from npm test: this check downloads a public Hub model.
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = 'screenshots/hugging-face-onnx-live';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch();
try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.getByRole('button', { name: 'Import from Hugging Face' }).click();
    const dialog = page.getByRole('dialog', { name: 'Import from Hugging Face' });
    await dialog.getByRole('button', { name: 'LeNet vision', exact: true }).click();
    await dialog.getByText('Supported', { exact: true }).waitFor({ timeout: 60000 });
    assert.match(await dialog.innerText(), /vtemplier\/LeNet_MNIST/);
    assert.equal(await dialog.getByLabel('Source graph summary').count(), 1);
    await page.screenshot({ path: `${directory}/01-lenet-preview-1440.png` });
    await dialog.getByRole('button', { name: 'Import architecture' }).click();
    await dialog.waitFor({ state: 'detached' });
    const summary = await page.evaluate(() => {
        const graph = window.__deepflow.architectureStore.getState().graph;
        return { repoId: graph.importSource.repoId, commit: graph.importSource.commit,
            filePath: graph.importSource.filePath, format: graph.importSource.intermediateFormat,
            weightsLoaded: graph.importSource.weightsLoaded, operations: graph.importSource.opCount,
            nodes: Object.values(graph.nodes).map((node) => node.libraryKey),
        };
    });
    assert.equal(summary.repoId, 'vtemplier/LeNet_MNIST');
    assert.equal(summary.format, 'onnx');
    assert.ok(summary.nodes.includes('torch.nn.Conv2d'));
    assert.equal(summary.weightsLoaded, false);
    await page.getByText('No issues detected', { exact: true }).waitFor();
    await page.getByRole('button', { name: 'Fit active graph' }).click();
    await page.waitForTimeout(350);
    await page.screenshot({ path: `${directory}/02-lenet-imported-1440.png` });
    await page.setViewportSize({ width: 820, height: 900 });
    await page.getByRole('button', { name: 'Fit active graph' }).click();
    await page.waitForTimeout(350);
    await page.screenshot({ path: `${directory}/03-lenet-imported-820.png` });
    assert.deepEqual(errors, []);
    await writeFile(`${directory}/verified-source.json`, JSON.stringify(summary, null, 2) + '\n');
    console.log(`Live Hugging Face import passed: ${summary.repoId} @ ${summary.commit} (${summary.operations} ONNX operators).`);
} finally { await browser.close(); }
