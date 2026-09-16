/**
 * End-to-end codegen test: build a parametric architecture + a 2-phase
 * GAN-style training graph, then dump the generated Python so we can sanity-
 * check it.
 */
import { chromium } from 'playwright';
import fs from 'node:fs/promises';

const URL = process.env.URL ?? 'http://localhost:5173/';

async function main() {
    const browser = await chromium.launch();
    const page = await browser.newPage();
    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => {
        if (msg.type() === 'error') console.error('[browser:error]', msg.text());
    });

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    const result = await page.evaluate(() => {
        const { architectureStore, templatesStore, trainingStore, codegen } = window.__deepflow;
        const arch = architectureStore.getState();
        const tplStore = templatesStore.getState();
        const train = trainingStore.getState();

        // 1. Build a parametric DenseBlock + a Repeat ×3
        const inputId  = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.patchLayerParameters(inputId, { shape: '(128,)' });

        const linearId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 220, y: 220 });
        arch.patchLayerParameters(linearId, { in_features: 128, out_features: 64 });
        arch.connect({ source: inputId, target: linearId });
        arch.connect({ source: linearId, target: outputId });

        const tplId = arch.groupSelection([linearId], 'DenseBlock');
        tplStore.addFreeParameter(tplId, { name: 'in',  type: 'int', default: 128 });
        tplStore.addFreeParameter(tplId, { name: 'out', type: 'int', default: 64 });
        tplStore.patchTemplateLayerParameters(tplId, linearId, {
            in_features: '$in',
            out_features: '$out',
        });

        const rootGraph = architectureStore.getState().graph;
        const instance = Object.values(rootGraph.nodes).find((n) => n.kind === 'supernode');
        const repeatId = arch.convertInstanceToRepeat(instance.id, 3);
        arch.patchRepeatBinding(repeatId, 'in',  '128 / 2 ** i');
        arch.patchRepeatBinding(repeatId, 'out', '64 / 2 ** i');

        // 2. Build a simple training graph
        const optId = train.addOptimizer('Adam', { x: 60, y: 100 });
        train.patchHyperparameters(optId, { learning_rate: 1e-3 });

        const lossId = train.addLoss('cross_entropy', { x: 360, y: 100 }, outputId);
        train.connect({ source: lossId, target: optId });

        const finalArch = architectureStore.getState().graph;
        const finalTpls = templatesStore.getState().templates;
        const finalTrain = trainingStore.getState().graph;

        return {
            modelSrc: codegen.generateArchitectureModule(finalArch, { templates: finalTpls, className: 'ResNetLikeMLP' }),
            trainSrc: codegen.generateTrainingScript(finalArch, finalTrain, { templates: finalTpls }),
        };
    });

    await fs.writeFile('screenshots/80-model.py', result.modelSrc);
    await fs.writeFile('screenshots/81-train.py', result.trainSrc);
    console.log('wrote screenshots/80-model.py');
    console.log('wrote screenshots/81-train.py');

    console.log('\n========== model.py ==========');
    console.log(result.modelSrc);
    console.log('\n========== train.py ==========');
    console.log(result.trainSrc);

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
