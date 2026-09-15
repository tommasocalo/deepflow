/**
 * GAN storyboard — generator + discriminator with two optimizers, two
 * losses, and two alternating phases.
 *
 * Demonstrates the multi-phase scheduling primitive.
 */
export function ganStoryboard({ architectureStore, templatesStore, trainingStore }) {
    const arch = architectureStore.getState();
    const train = trainingStore.getState();

    arch.resetCurrentGraph();
    train.resetGraph();

    const root = architectureStore.getState().graph;
    const inputId  = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
    const outputId = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;

    arch.moveNode(inputId,  { x: 280, y: 40 });
    arch.moveNode(outputId, { x: 280, y: 760 });
    arch.patchLayerParameters(inputId, { shape: '(64,)' });
    arch.patchNode(inputId,  { label: 'noise z' });
    arch.patchNode(outputId, { label: 'D logits' });

    // Generator: Linear(64 → 128) → ReLU → Linear(128 → 784)
    const g1Id = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 280, y: 180 });
    arch.patchNode(g1Id, { label: 'G linear 1' });
    arch.patchLayerParameters(g1Id, { in_features: 64, out_features: 128 });
    const g1ReluId = arch.addLayerByLibraryKey('torch.nn.ReLU', { x: 280, y: 300 });
    const g2Id = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 280, y: 380 });
    arch.patchNode(g2Id, { label: 'G linear 2' });
    arch.patchLayerParameters(g2Id, { in_features: 128, out_features: 784 });

    // Discriminator: Linear(784 → 128) → LeakyReLU → Linear(128 → 1)
    const d1Id = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 280, y: 500 });
    arch.patchNode(d1Id, { label: 'D linear 1' });
    arch.patchLayerParameters(d1Id, { in_features: 784, out_features: 128 });
    const d1LeakyId = arch.addLayerByLibraryKey('torch.nn.LeakyReLU', { x: 280, y: 600 });
    arch.patchLayerParameters(d1LeakyId, { negative_slope: 0.2 });
    const d2Id = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 280, y: 680 });
    arch.patchNode(d2Id, { label: 'D linear 2' });
    arch.patchLayerParameters(d2Id, { in_features: 128, out_features: 1 });

    arch.connect({ source: inputId, target: g1Id });
    arch.connect({ source: g1Id, target: g1ReluId });
    arch.connect({ source: g1ReluId, target: g2Id });
    arch.connect({ source: g2Id, target: d1Id });
    arch.connect({ source: d1Id, target: d1LeakyId });
    arch.connect({ source: d1LeakyId, target: d2Id });
    arch.connect({ source: d2Id, target: outputId });

    // Training: two optimizers (one per network), two losses, two alternating phases
    const gOptId = train.addOptimizer('Adam', { x: 40, y: 60 });
    train.patchHyperparameters(gOptId, { learning_rate: 2e-4, betas: '(0.5, 0.999)' });
    train.patchNode(gOptId, { label: 'G optimizer', parameterGroup: [g1Id, g2Id] });

    const dOptId = train.addOptimizer('Adam', { x: 40, y: 240 });
    train.patchHyperparameters(dOptId, { learning_rate: 2e-4, betas: '(0.5, 0.999)' });
    train.patchNode(dOptId, { label: 'D optimizer', parameterGroup: [d1Id, d2Id] });

    const gLossId = train.addLoss('bce', { x: 320, y: 60 }, outputId);
    train.patchNode(gLossId, { label: 'G loss' });

    const dLossId = train.addLoss('bce', { x: 320, y: 240 }, outputId);
    train.patchNode(dLossId, { label: 'D loss' });

    const gPhaseId = train.addPhase({ x: 600, y: 60 }, {
        label: 'G step',
        optimizerIds: [gOptId],
        lossIds: [gLossId],
        everyNSteps: 1,
        stepsPerCycle: 1,
    });
    const dPhaseId = train.addPhase({ x: 600, y: 240 }, {
        label: 'D step',
        optimizerIds: [dOptId],
        lossIds: [dLossId],
        everyNSteps: 1,
        stepsPerCycle: 1,
    });

    train.connect({ source: gLossId, target: gOptId });
    train.connect({ source: dLossId, target: dOptId });
    train.connect({ source: gOptId,  target: gPhaseId });
    train.connect({ source: dOptId,  target: dPhaseId });
}
