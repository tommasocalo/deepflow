/**
 * DANN storyboard — Ganin et al.'s domain-adversarial network with a
 * gradient-reversal layer on the domain-classifier branch.
 *
 * Demonstrates the gradient operator on architecture edges.
 */
export function dannStoryboard({ architectureStore, trainingStore }) {
    const arch = architectureStore.getState();
    const train = trainingStore.getState();

    arch.resetCurrentGraph();
    train.resetGraph();

    const root = architectureStore.getState().graph;
    const inputId  = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Input').id;
    const classOutId = Object.values(root.nodes).find((n) => n.libraryKey === 'deepflow.io.Output').id;

    arch.moveNode(inputId, { x: 320, y: 40 });
    arch.moveNode(classOutId, { x: 80, y: 740 });
    arch.patchLayerParameters(inputId, { shape: '(64,)' });
    arch.patchNode(classOutId, { label: 'class logits' });

    const featId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 320, y: 220 });
    arch.patchNode(featId, { label: 'feature extractor' });
    arch.patchLayerParameters(featId, { in_features: 64, out_features: 128 });

    const featReluId = arch.addLayerByLibraryKey('torch.nn.ReLU', { x: 320, y: 360 });

    const classId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 80, y: 540 });
    arch.patchNode(classId, { label: 'class head' });
    arch.patchLayerParameters(classId, { in_features: 128, out_features: 10 });

    const domId = arch.addLayerByLibraryKey('torch.nn.Linear', { x: 600, y: 540 });
    arch.patchNode(domId, { label: 'domain head' });
    arch.patchLayerParameters(domId, { in_features: 128, out_features: 2 });

    const domOutId = arch.addLayerByLibraryKey('deepflow.io.Output', { x: 600, y: 740 });
    arch.patchNode(domOutId, { label: 'domain logits' });

    arch.connect({ source: inputId,    target: featId });
    arch.connect({ source: featId,     target: featReluId });
    arch.connect({ source: featReluId, target: classId });
    arch.connect({ source: classId,    target: classOutId });

    // Gradient-reversal edge: feat → domain head, with λ = 1.0
    const grlEdgeId = arch.connect({ source: featReluId, target: domId });
    arch.connect({ source: domId, target: domOutId });
    if (grlEdgeId) arch.setEdgeGradientOp(grlEdgeId, 'reverse', 1.0);

    // Training: one optimizer, two losses (class + domain), one phase combining both
    const optId = train.addOptimizer('SGD', { x: 60, y: 100 });
    train.patchHyperparameters(optId, { learning_rate: 1e-2, momentum: 0.9 });
    train.patchNode(optId, { label: 'SGD optimizer' });

    const classLossId = train.addLoss('cross_entropy', { x: 360, y: 100 }, classOutId);
    train.patchNode(classLossId, { label: 'class loss' });

    const domLossId = train.addLoss('cross_entropy', { x: 360, y: 240 }, domOutId);
    train.patchNode(domLossId, { label: 'domain loss', weight: 0.5 });

    train.connect({ source: classLossId, target: optId });
    train.connect({ source: domLossId,   target: optId });
}
