// Layout checks for the Model table in a real browser: rows visible in two
// panes, the ribbon on one row, number cells that never cut their units, and
// the keyboard way back to the canvas. Run against a dev or preview server:
//   URL=http://127.0.0.1:5173/ node scripts/model-table-browser-check.mjs
// CHROMIUM_PATH selects a Chromium build when Playwright's own is not installed.
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';

const directory = process.env.SCREENSHOT_DIR ?? '/tmp/deepflow-model-table';
await mkdir(directory, { recursive: true });
const browser = await chromium.launch(process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const settle = (page) => page.evaluate(() => new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const table = (page, pane = 'main') => page.locator(`[data-graph-pane="${pane}"] [aria-label="Model table"]`);
const radio = (page, name) => table(page).getByRole('radio', { name, exact: true });

async function open({ width, height, model = 'GPT-2 small' }) {
    const page = await browser.newPage({ viewport: { width, height } });
    page.setDefaultTimeout(8000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(process.env.URL ?? 'http://127.0.0.1:4183/');
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await page.getByRole('button', { name: 'Examples' }).click();
    await page.locator('[aria-label="Example architectures"] button', { hasText: model }).first().click();
    await page.getByRole('button', { name: 'Fit active graph' }).click();
    await settle(page);
    return { page, errors };
}
async function showTable(page, pane = 'main') {
    // The ribbon acts on the pane used last.
    await page.locator(`[data-graph-pane="${pane}"]`).click({ position: { x: 100, y: 300 } });
    await page.getByRole('button', { name: 'Show model table' }).click();
    await table(page, pane).locator('tbody tr').first().waitFor();
    await settle(page);
}
/** Rows drawn whole inside the scroll area, below the sticky header. */
const layout = (page, pane) => table(page, pane).evaluate((aside) => {
    const scroll = aside.querySelector('.df-mt-scroll').getBoundingClientRect();
    const head = aside.querySelector('thead').getBoundingClientRect();
    const rows = [...aside.querySelectorAll('tbody tr[role="row"]')].filter((row) => {
        const box = row.getBoundingClientRect();
        return box.top >= head.bottom - 0.5 && box.bottom <= scroll.bottom + 0.5;
    }).length;
    const height = (selector) => aside.querySelector(selector).getBoundingClientRect().height;
    const grip = aside.querySelector('.df-mt-grip').getBoundingClientRect();
    return {
        rows, sheet: aside.classList.contains('is-sheet'), canvas: aside.parentElement.querySelector('[data-architecture-canvas]').getBoundingClientRect().height,
        toolbar: height('.df-mt-toolbar'), totals: height('.df-mt-totals'), grip: grip.width * grip.height > 0,
    };
});
/** Number cells and headers wider than their column. */
const overflowing = (page) => table(page).evaluate((aside) => {
    const cut = [];
    for (const cell of aside.querySelectorAll('tbody [data-col="1"], tbody [data-col="2"], tbody [data-col="4"], tbody [data-col="5"], tbody [data-col="6"], thead th')) {
        const style = getComputedStyle(cell);
        const room = cell.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
        const inner = cell.firstElementChild;
        if (cell.scrollWidth > cell.clientWidth + 0.5 || (inner && inner.getBoundingClientRect().width > room + 0.5)) cut.push(`${cell.closest('tr').querySelector('.df-mt-label')?.textContent ?? 'header'}: ${cell.textContent.trim()}`);
    }
    return cut;
});

try {
    // Two panes on common laptop screens: each bottom sheet shows rows, the canvas keeps room.
    for (const [width, height, least] of [[1280, 800, 8], [1024, 768, 5]]) {
        const { page, errors } = await open({ width, height });
        await page.getByRole('button', { name: 'Open to the side' }).click();
        await page.locator('[data-graph-pane="side"]').waitFor();
        await showTable(page, 'main');
        await showTable(page, 'side');
        for (const pane of ['main', 'side']) {
            const measured = await layout(page, pane);
            assert.ok(measured.sheet, `${width}×${height} ${pane}: a bottom sheet`);
            assert.ok(measured.rows >= least, `${width}×${height} ${pane}: ${measured.rows} rows visible, expected at least ${least}`);
            assert.ok(measured.canvas >= 200, `${width}×${height} ${pane}: the canvas keeps ${measured.canvas}px`);
            assert.ok(measured.toolbar < 40, `${width}×${height} ${pane}: the toolbar is one line`);
            assert.ok(measured.totals < 26, `${width}×${height} ${pane}: the totals are one line`);
            assert.ok(measured.grip, 'The resize grip is drawn');
        }
        await page.screenshot({ path: `${directory}/two-panes-${width}x${height}.png` });
        assert.deepEqual(errors, []);
        await page.close();
    }

    // The ribbon keeps one row from 1040px up, with one pane and with two, and
    // 10px labels from 1191px. Two panes always show which pane it acts on.
    {
        const { page } = await open({ width: 1440, height: 900 });
        const widths = [940, 1000, 1191, 1301, 1305, 1309, 1310];
        for (let width = 1040; width <= 1600; width += 20) widths.push(width);
        for (const panes of ['one', 'two']) {
            if (panes === 'two') {
                await page.setViewportSize({ width: 1440, height: 900 });
                await page.getByRole('button', { name: 'Open to the side' }).click();
                await page.locator('[data-graph-pane="side"]').waitFor();
            }
            for (const width of widths.sort((a, b) => a - b)) {
                await page.setViewportSize({ width, height: 900 });
                await settle(page);
                const ribbon = await page.evaluate(() => {
                    const row = document.querySelector('.df-ribbon-row');
                    const visible = (element) => !!element && element.getBoundingClientRect().width > 2 && getComputedStyle(element).display !== 'none';
                    return {
                        overflow: row.scrollWidth - row.clientWidth,
                        label: parseFloat(getComputedStyle(document.querySelector('.df-ribbon-button')).fontSize),
                        cue: visible(document.querySelector('.df-ribbon-pane-cue')) || visible(document.querySelector('.df-graph-pane.is-active .df-pane-cue')),
                    };
                });
                if (width >= 1040) assert.ok(ribbon.overflow <= 0, `${panes} pane(s) at ${width}px: the ribbon overflows by ${ribbon.overflow}px`);
                if (width >= 1191) assert.ok(ribbon.label >= 10, `${panes} pane(s) at ${width}px: ribbon labels are ${ribbon.label}px`);
                if (panes === 'two') assert.ok(ribbon.cue, `Two panes at ${width}px: no visible cue names the pane the ribbon acts on`);
            }
        }
        await page.close();
    }

    // Number cells keep their units for both reference models, every depth, lens and cell mode.
    for (const model of ['GPT-2 small', 'ResNet']) {
        const { page } = await open({ width: 1440, height: 900, model });
        await showTable(page);
        for (const depth of ['Root', 'Blocks', 'Ops']) {
            for (const lens of ['Params', 'MACs', 'Activations']) {
                for (const cells of ['Symbols', 'Values']) {
                    for (const name of [depth, lens, cells]) await radio(page, name).click();
                    await settle(page);
                    assert.deepEqual(await overflowing(page), [], `${model} ${depth} · ${lens} · ${cells}: no number is cut`);
                }
            }
        }
        if (model === 'ResNet') await table(page).screenshot({ path: `${directory}/resnet-activations.png` });
        await page.close();
    }

    // Escape in the grid returns to the canvas; resizing is the last Tab stop of the table.
    {
        const { page } = await open({ width: 1440, height: 900 });
        await showTable(page);
        const last = await table(page).evaluate((aside) => [...aside.querySelectorAll('button:not([tabindex="-1"]), [tabindex="0"]')].filter((element) => !element.disabled && !element.closest('[hidden]')).at(-1).getAttribute('role'));
        assert.equal(last, 'separator');
        await table(page).locator('tbody [data-col="0"]').first().focus();
        await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(() => !!document.activeElement.closest('[data-graph-pane="main"] [data-architecture-canvas]')), true, 'Escape focuses the canvas');
        await page.close();
    }
    console.log('Model table browser checks passed');
} finally {
    await browser.close();
}
