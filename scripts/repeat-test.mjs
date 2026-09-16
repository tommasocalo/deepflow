/**
 * End-to-end test for the RepeatNode primitive:
 *   1. Build a parametric DenseBlock template (Linear with $in / $out)
 *   2. Wire Input → DenseBlock instance → Output
 *   3. Convert the instance into a RepeatNode with N=3 and per-iteration formulas
 *   4. Confirm shape inference walks the unrolled chain
 *      Linear(128 → 64) → Linear(64 → 32) → Linear(32 → 16)
 *      and the final shape is (?, 16) with no errors.
 */
import { chromium } from 'playwright';

const URL = process.env.URL ?? 'http://localhost:5173/';

async function main() {
    const browser = await chromium.launch();
    const page = await browser.newPage();

    page.on('pageerror', (err) => console.error('PAGEERR:', err.stack ?? err));
    page.on('console', (msg) => console.error(`[browser:${msg.type()}]`, msg.text()));

    await page.goto(URL, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { try { localStorage.clear(); } catch {} });
    await page.goto(URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(500);

    const result = await page.evaluate(async () => {
        // Use the running app's singletons rather than dynamic-importing
        // (HMR can otherwise hand us a parallel module instance with empty
        // state). See main.jsx for where __deepflow is exposed.
        const { architectureStore, templatesStore, validation: valMod } = window.__deepflow;
        const arch = architectureStore.getState();
        const tplStore = templatesStore.getState();
        const archMod = { useArchitectureStore: architectureStore };
        const tplMod = { useTemplatesStore: templatesStore };

        const inputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
        const outputId = Object.values(arch.graph.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;

        // 1. Input shape — flat 128-vec
        arch.patchLayerParameters(inputId, { shape: '(128,)' });

        // 2. Linear with concrete sizes
        const linearId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 220, y: 220 });
        arch.patchLayerParameters(linearId, { in_features: 128, out_features: 64 });

        // 3. Wire Input → Linear → Output
        arch.connect({ source: inputId, target: linearId });
        arch.connect({ source: linearId, target: outputId });

        // 4. Group the Linear into a template
        const tplId = arch.groupSelection([linearId], 'DenseBlock');
        if (!tplId) return { error: 'groupSelection returned null' };

        // 5. Declare $in and $out free params
        tplStore.addFreeParameter(tplId, { name: 'in',  type: 'int', description: 'Input dim',  default: 128 });
        tplStore.addFreeParameter(tplId, { name: 'out', type: 'int', description: 'Output dim', default: 64 });

        // 6. Rewire inner Linear to use $in / $out
        tplStore.patchTemplateLayerParameters(tplId, linearId, {
            in_features: '$in',
            out_features: '$out',
        });

        // 7. Find the instance and convert it to a RepeatNode (count = 3)
        let rootGraph = archMod.useArchitectureStore.getState().graph;
        const instance = Object.values(rootGraph.nodes).find((n) => n.kind === 'supernode' && n.templateId === tplId);
        const repeatId = arch.convertInstanceToRepeat(instance.id, 3);

        // 8. Set per-iteration formulas
        arch.patchRepeatBinding(repeatId, 'in',  '128 / 2 ** i');
        arch.patchRepeatBinding(repeatId, 'out', '64 / 2 ** i');

        // 9. Validate
        const finalState = archMod.useArchitectureStore.getState();
        const finalReport = valMod.validate(finalState.graph, {
            templates: tplMod.useTemplatesStore.getState().templates,
        });
        const repeatNode = finalState.graph.nodes[repeatId];

        const allTemplates = tplMod.useTemplatesStore.getState().templates;
        return {
            repeatKind: repeatNode.kind,
            repeatId,
            repeatTemplateId: repeatNode.templateId,
            repeatCount: repeatNode.count,
            iterationVar: repeatNode.iterationVariable,
            bindings: repeatNode.bindings,
            templateInStore: !!allTemplates[repeatNode.templateId],
            templateKeys: Object.keys(allTemplates),
            inputId,
            outputId,
            shapes: {
                input:  finalReport.shapes[inputId],
                repeat: finalReport.shapes[repeatId],
                output: finalReport.shapes[outputId],
            },
            errors:   finalReport.errors,
            warnings: finalReport.warnings,
        };
    });

    console.log(JSON.stringify(result, null, 2));

    await page.screenshot({ path: 'screenshots/40-repeat-after.png', fullPage: false });
    console.log('screenshot → screenshots/40-repeat-after.png');

    await browser.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
