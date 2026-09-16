import { chromium } from 'playwright';
import fs from 'node:fs/promises';

const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto('http://localhost:5173/', { waitUntil: 'domcontentloaded' });
await page.evaluate(() => { try { localStorage.clear(); } catch {} });
await page.goto('http://localhost:5173/', { waitUntil: 'networkidle' });
await page.waitForTimeout(500);

const result = await page.evaluate(() => {
    const wf = window.__deepflow;
    return import('/src/storyboards/index.js').then(({ STORYBOARDS }) => {
        const sb = STORYBOARDS.find((s) => s.id === 'gan');
        sb.run({
            architectureStore: wf.architectureStore,
            templatesStore: wf.templatesStore,
            trainingStore: wf.trainingStore,
        });
        const arch = wf.architectureStore.getState().graph;
        const tpls = wf.templatesStore.getState().templates;
        const train = wf.trainingStore.getState().graph;
        return {
            train: wf.codegen.generateTrainingScript(arch, train, { templates: tpls }),
        };
    });
});

await fs.writeFile('screenshots/120-gan-train.py', result.train);
console.log('wrote screenshots/120-gan-train.py');
await browser.close();
