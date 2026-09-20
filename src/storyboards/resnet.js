/**
 * ResNet storyboard — parametric ConvBlock repeated 3 times with channel
 * doubling. Demonstrates the RepeatNode primitive.
 *
 * Final shape (assuming Input (3, 32, 32)): (?, 256, 32, 32) at the
 * end of the repeated stack, ignoring inner downsampling.
 */
export function resnetStoryboard({ architectureStore, templatesStore, trainingStore }) {
    const arch = architectureStore.getState();
    const tpl = templatesStore.getState();
    const train = trainingStore.getState();

    arch.resetCurrentGraph();
    train.resetGraph();

    const root = architectureStore.getState().graph;
    const inputId = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
    const outputId = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;

    arch.moveNode(inputId,  { x: 320, y: 40 });
    arch.moveNode(outputId, { x: 320, y: 720 });
    arch.patchLayerParameters(inputId, { shape: '(3, 32, 32)' });

    // Build a parametric ConvBlock = Conv2d → BatchNorm2d → ReLU
    const convId = arch.addLayerByLibraryKey('torch.nn.Conv2d',      { x: 320, y: 200 });
    const bnId   = arch.addLayerByLibraryKey('torch.nn.BatchNorm2d', { x: 320, y: 330 });
    const reluId = arch.addLayerByLibraryKey('torch.nn.ReLU',        { x: 320, y: 460 });
    arch.patchLayerParameters(convId, { in_channels: 3, out_channels: 64, kernel_size: 3, padding: 1, bias: false });
    arch.patchLayerParameters(bnId,   { num_features: 64 });
    arch.connect({ source: inputId, target: convId });
    arch.connect({ source: convId, target: bnId });
    arch.connect({ source: bnId, target: reluId });
    arch.connect({ source: reluId, target: outputId });

    const tplId = arch.groupSelection([convId, bnId, reluId], 'ConvBlock');

    // Grouping inserts generic boundary positions; align this example's
    // boundaries with its layers and maintain the same 130px row spacing.
    const templateNodes = Object.values(tpl.getTemplate(tplId).graph.nodes);
    const templateInput = templateNodes.find((n) => n.libraryKey === 'deepflow.template.Input');
    const templateOutput = templateNodes.find((n) => n.libraryKey === 'deepflow.template.Output');
    tpl.moveTemplateNode(tplId, templateInput.id, { x: 200, y: 30 });
    tpl.moveTemplateNode(tplId, templateOutput.id, { x: 200, y: 550 });

    // Free parameters: in_channels, out_channels
    tpl.addFreeParameter(tplId, { name: 'in',  type: 'int', description: 'Input channels',  default: 3 });
    tpl.addFreeParameter(tplId, { name: 'out', type: 'int', description: 'Output channels', default: 64 });

    // Inner conv2d uses $in / $out, BN's num_features = $out
    tpl.patchTemplateLayerParameters(tplId, convId, { in_channels: '$in', out_channels: '$out' });
    tpl.patchTemplateLayerParameters(tplId, bnId,   { num_features: '$out' });

    // Convert to RepeatNode with 3 iterations and channel doubling
    const rootAfter = architectureStore.getState().graph;
    const instance = Object.values(rootAfter.nodes).find((n) => n.kind === 'supernode' && n.templateId === tplId);
    const repeatId = arch.convertInstanceToRepeat(instance.id, 3);
    arch.patchRepeatNode(repeatId, { label: 'ResNet stages' });
    arch.patchRepeatBinding(repeatId, 'in',  'i === 0 ? 3 : 64 * 2 ** (i - 1)');
    arch.patchRepeatBinding(repeatId, 'out', '64 * 2 ** i');

    // Training graph
    const optId  = train.addOptimizer('Adam', { x: 60, y: 100 });
    train.patchHyperparameters(optId, { learning_rate: 1e-3 });
    train.patchNode(optId, { label: 'Adam optimizer' });
    const lossId = train.addLoss('cross_entropy', { x: 360, y: 100 }, outputId);
    train.connect({ source: lossId, target: optId });
}
