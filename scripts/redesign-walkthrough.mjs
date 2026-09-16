import { chromium } from 'playwright';
const URL = process.env.URL ?? 'http://localhost:5173/';
const OUT = process.env.OUT ?? 'screenshots';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push('PAGEERR ' + (e.stack ?? e)));
page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE ' + m.text()); });
await page.goto(URL, { waitUntil: 'domcontentloaded' });
await page.evaluate(() => { try { localStorage.clear(); } catch {} });
await page.goto(URL, { waitUntil: 'networkidle' });
await page.waitForTimeout(900);
await page.screenshot({ path: `${OUT}/v3-01-default-gpt2.png` });
// open the repeat's template via its chevron
await page.locator('.df-repeat .df-container-open').first().click({ force: true });
await page.waitForTimeout(700);
await page.screenshot({ path: `${OUT}/v3-02-block-level.png` });
// select the attention block → inspector
await page.locator('.df-block', { hasText: 'causal self-attention' }).first().click({ force: true });
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/v3-03-inspector.png` });
// back to main via card close
await page.locator('.df-level-close').first().click();
await page.waitForTimeout(500);
await page.locator('.df-repeat').first().click({ force: true });
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/v3-04-main-repeat-inspector.png` });
// code
await page.locator('.df-action-segment', { hasText: 'Code' }).first().click();
await page.waitForTimeout(400);
await page.screenshot({ path: `${OUT}/v3-05-code.png` });
const code = await page.evaluate(() => {
    const wf = window.__deepflow;
    return wf.codegen.generateArchitectureModule(wf.architectureStore.getState().graph, { templates: wf.templatesStore.getState().templates, className: 'GPT2' });
});
console.log(code.split('\n').slice(0, 200).join('\n'));
console.log('errors:', errors.length); for (const e of errors) console.log(e);
await browser.close();
