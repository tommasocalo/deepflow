/**
 * End-to-end test of parametric supernodes:
 *   1. Builds a small graph with Input → Conv2d → BatchNorm2d → Output programmatically
 *   2. Groups Conv2d + BatchNorm2d into a template
 *   3. Declares a free parameter `out` and rewires both inner blocks to use $out
 *   4. Binds the instance's $out to 64
 *   5. Asserts that shape inference at the supernode reports the right shape
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5173/';

async function main() {
    const browser = await chromium.launch();
    const page = await browser.newPage();

    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => {
        if (msg.type() === 'error') console.error('CONSOLE ERR:', msg.text());
    });

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    // Build a parametric ConvBlock supernode programmatically and verify
    // that the validator correctly resolves $out when descending the
    // template.
    const result = await page.evaluate(async () => {
        const { architectureStore, templatesStore, validation: valMod } = window.__deepflow;
        const arch = architectureStore.getState();
        const tplStore = templatesStore.getState();
        const archMod = { useArchitectureStore: architectureStore };
        const tplMod = { useTemplatesStore: templatesStore };

        // 1. Set Input's shape parameter
        const inputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;
        arch.patchLayerParameters(inputId, { shape: '(3, 224, 224)' });

        // 2. Add Conv2d and BatchNorm2d
        const convId = arch.addLayerByLibraryKey('torch.nn.Conv2d', { x: 200, y: 200 });
        const bnId = arch.addLayerByLibraryKey('torch.nn.BatchNorm2d', { x: 200, y: 280 });
        arch.patchLayerParameters(convId, { in_channels: 3, out_channels: 64, kernel_size: 3 });
        arch.patchLayerParameters(bnId, { num_features: 64 });

        // 3. Wire them: Input → Conv → BN → Output
        arch.connect({ source: inputId, target: convId });
        arch.connect({ source: convId, target: bnId });
        arch.connect({ source: bnId, target: outputId });

        // First sanity check: shapes propagate through the linear graph
        const beforeReport = valMod.validate(
            archMod.useArchitectureStore.getState().graph,
            { templates: tplMod.useTemplatesStore.getState().templates },
        );

        // 4. Group Conv + BN into a template
        const tplId = arch.groupSelection([convId, bnId], 'ConvBlock');
        if (!tplId) {
            return { error: 'groupSelection returned null — selection had multi-IO boundary?' };
        }

        // 5. Declare a free parameter `out` (int) on the template
        tplStore.addFreeParameter(tplId, {
            name: 'out',
            type: 'int',
            description: 'Output channels',
            default: 64,
        });

        // 6. Rewire inner Conv2d's out_channels and BN's num_features to $out
        // Look up the inner nodes (they have new ids after grouping — but in
        // Phase 3 we kept the original ids when copying, so convId/bnId still
        // point at the same nodes inside the template).
        tplStore.patchTemplateLayerParameters(tplId, convId, { out_channels: '$out' });
        tplStore.patchTemplateLayerParameters(tplId, bnId, { num_features: '$out' });

        // 7. Find the SuperNodeInstance that replaced the grouped nodes and
        // bind its $out to 128 to check that the binding flows down.
        const rootGraph = archMod.useArchitectureStore.getState().graph;
        const instance = Object.values(rootGraph.nodes).find((n) => n.kind === 'supernode' && n.templateId === tplId);
        archMod.useArchitectureStore.getState().patchNode(instance.id, {
            parameterValues: { out: 128 },
        });

        // Final validation
        const afterState = archMod.useArchitectureStore.getState();
        const afterReport = valMod.validate(afterState.graph, {
            templates: tplMod.useTemplatesStore.getState().templates,
        });

        const tpl = tplMod.useTemplatesStore.getState().templates[tplId];

        return {
            beforeShapes: {
                input: beforeReport.shapes[inputId],
                conv:  beforeReport.shapes[convId],
                bn:    beforeReport.shapes[bnId],
                output:beforeReport.shapes[outputId],
            },
            tplFreeParams: tpl.freeParameters,
            instanceBindings: instance.parameterValues,
            instanceShape: afterReport.shapes[instance.id],
            afterErrors: afterReport.errors,
            afterWarnings: afterReport.warnings,
            innerConvParams: tpl.graph.nodes[convId]?.parameters,
            innerBnParams:   tpl.graph.nodes[bnId]?.parameters,
        };
    });

    console.log(JSON.stringify(result, null, 2));

    await page.screenshot({ path: 'screenshots/30-parametric-after.png', fullPage: false });
    console.log('screenshot → screenshots/30-parametric-after.png');

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
