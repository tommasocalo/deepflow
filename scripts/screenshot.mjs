import { chromium } from 'playwright';
import fs from 'node:fs/promises';
import path from 'node:path';

const URL = process.env.URL ?? 'http://localhost:5173/';
const OUT = process.env.OUT ?? 'screenshots/state.png';
const VIEWPORT = { width: 1440, height: 900 };

async function main() {
    const browser = await chromium.launch();
    const ctx = await browser.newContext({ viewport: VIEWPORT });
    const page = await ctx.newPage();

    /** @type {{type:string, text:string}[]} */
    const consoleEntries = [];
    page.on('console', (msg) => consoleEntries.push({ type: msg.type(), text: msg.text() }));
    /** @type {string[]} */
    const pageErrors = [];
    page.on('pageerror', (err) => pageErrors.push(err.stack ?? String(err)));

    if (process.env.CLEAR_STORAGE === '1') {
        await page.goto(URL, { waitUntil: 'domcontentloaded' });
        await page.evaluate(() => {
            try { localStorage.clear(); } catch {}
        });
    }
    await page.goto(URL, { waitUntil: 'networkidle' });
    // Give react/zustand/xyflow time to mount fully
    await page.waitForTimeout(800);

    const outAbs = path.resolve(OUT);
    await fs.mkdir(path.dirname(outAbs), { recursive: true });
    await page.screenshot({ path: outAbs, fullPage: false });

    const logAbs = outAbs.replace(/\.png$/, '.log.txt');
    const lines = [];
    lines.push(`=== URL: ${URL}`);
    lines.push(`=== Page errors (${pageErrors.length}):`);
    for (const e of pageErrors) lines.push(e);
    lines.push(`=== Console entries (${consoleEntries.length}):`);
    for (const c of consoleEntries) lines.push(`[${c.type}] ${c.text}`);
    await fs.writeFile(logAbs, lines.join('\n'));

    console.log(`screenshot → ${outAbs}`);
    console.log(`log        → ${logAbs}`);
    console.log(`page errors: ${pageErrors.length}, console entries: ${consoleEntries.length}`);

    await browser.close();
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
