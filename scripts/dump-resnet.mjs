import { chromium } from 'playwright';
import fs from 'node:fs/promises';

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto('http://localhost:5174/', { waitUntil: 'domcontentloaded' });
await page.evaluate(() => { try { localStorage.clear(); } catch {} });
await page.goto('http://localhost:5174/', { waitUntil: 'networkidle' });
await page.waitForTimeout(500);

const result = await page.evaluate(() => {
    const wf = window.__deepflow;
    return import('/src/storyboards/index.js').then(({ STORYBOARDS }) => {
        const sb = STORYBOARDS.find((s) => s.id === 'resnet');
        sb.run({
            architectureStore: wf.architectureStore,
            templatesStore: wf.templatesStore,
            trainingStore: wf.trainingStore,
        });
        const arch = wf.architectureStore.getState().graph;
        const tpls = wf.templatesStore.getState().templates;
        const train = wf.trainingStore.getState().graph;
        const validation = wf.validation.validate(arch, { templates: tpls });
        return {
            model: wf.codegen.generateArchitectureModule(arch, { templates: tpls, className: 'ResNetMLP' }),
            train: wf.codegen.generateTrainingScript(arch, train, { templates: tpls }),
            validationOK: validation.ok,
            errorCount: Object.values(validation.errors).reduce((a, e) => a + e.length, 0),
            warningCount: Object.values(validation.warnings).reduce((a, w) => a + w.length, 0),
        };
    });
});

console.log('validation: ok=' + result.validationOK + ' errors=' + result.errorCount + ' warnings=' + result.warningCount);
await fs.writeFile('screenshots/110-resnet-model.py', result.model);
await fs.writeFile('screenshots/111-resnet-train.py', result.train);
console.log('wrote screenshots/110-resnet-model.py');
console.log('wrote screenshots/111-resnet-train.py');
await browser.close();
