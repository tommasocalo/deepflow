import { generateArchitectureModule } from './architecture.js';
import { indent, kwArgs, toPyIdent, toPyLiteral } from './python.js';
import { validate } from '../validation/validator.js';
import { generateEduTrainingScript } from './edu/trainScript.js';

/**
 * Generate a runnable PyTorch training script that:
 *   - imports / defines the model
 *   - constructs each optimizer with its parameter group
 *   - constructs each loss
 *   - runs a training loop that respects PhaseNodes (alternation)
 *
 * The script is intentionally readable rather than minimal — the user can
 * edit it. Datasets and dataloaders are stubbed; Phase 5 (Data graph) will
 * fill them in.
 *
 * With `data` (a bound Edu dataset, docs/edu/SPEC.md §6.11.2) the script is
 * complete instead: real loaders, a validation pass and an accuracy metric
 * (codegen/edu/trainScript.js). Without it the output is the stub as before.
 *
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} archGraph
 * @param {import('../domain/training/types.js').TrainingGraph} trainGraph
 * @param {Object} opts
 * @param {Object<string, import('../domain/architecture/types.js').ArchitectureTemplate>} [opts.templates]
 * @param {{ spec: Object, options?: Object, split?: Object, dataSeed?: number, settings?: Object }} [opts.data]
 */
export function generateTrainingScript(archGraph, trainGraph, { templates = {}, data } = {}) {
    if (data?.spec) return generateEduTrainingScript(archGraph, trainGraph, { templates, data });
    // DeepFlow's own script, byte-identical to the fork point: no nn.Linear comments (architecture.js `annotate`).
    const modelSrc = generateArchitectureModule(archGraph, { templates, className: 'Model', annotate: false });
    const optimizers = Object.values(trainGraph.nodes ?? {}).filter((n) => n.kind === 'optimizer');
    const losses = Object.values(trainGraph.nodes ?? {}).filter((n) => n.kind === 'loss');
    const phases = Object.values(trainGraph.nodes ?? {}).filter((n) => n.kind === 'phase');

    const layerFields = collectLayerFields(archGraph, templates);

    const optimizerInit = optimizers.map((o) => emitOptimizer(o, layerFields)).join('\n');
    const lossInit = losses.map((l) => emitLoss(l)).join('\n');
    // Each loss reads its own label key from the batch, and cross-entropy on
    // a language model's (B, T, V) scores is flattened first; the GPT-2
    // export used to crash on its first step.
    const flattenFor = lossOutputRanks(archGraph, templates, losses);
    const term = (loss) => lossTerm(loss, { flatten: flattenFor(loss), target: batchTarget(loss) });
    const phaseDefs = phases.length
        ? emitPhases(phases, optimizers, losses, term)
        : emitDefaultStep(optimizers, losses, term);

    return `${modelSrc}\n\n# -----------------------------------------------------------------------\n# Auto-generated training script. Edit the data loaders to match your task.\n# -----------------------------------------------------------------------\n\nimport torch\nfrom torch.utils.data import DataLoader\n\n\ndef build_dataloaders():\n    """Replace this with real datasets — see Data graph in DeepFlow v2."""\n    raise NotImplementedError("Wire up your datasets / dataloaders here.")\n\n\ndef train(num_epochs: int = 10, device: str | None = None):\n    device = device or ("cuda" if torch.cuda.is_available() else "cpu")\n    model = Model().to(device)\n\n${indent(optimizerInit || '# (no optimizers)', 1)}\n${indent(lossInit || '# (no losses)', 1)}\n\n    train_loader, val_loader = build_dataloaders()\n\n    global_step = 0\n    for epoch in range(num_epochs):\n        model.train()\n        for batch in train_loader:\n            x = batch["inputs"].to(device)\n            y = batch.get("labels")\n            if y is not None:\n                y = y.to(device)\n${indent(phaseDefs, 3)}\n            global_step += 1\n        # TODO: validation pass per epoch\n        print(f"epoch {epoch} done — step={global_step}")\n\n\nif __name__ == "__main__":\n    train()\n`;
}

// -------- helpers --------

function emitOptimizer(opt, layerFields) {
    const varName = toPyIdent(opt.label || opt.type, 'opt');
    return `${varName} = ${optimizerExpression(opt, layerFields)}`;
}

/**
 * `torch.optim.<type>(<parameters>, <hyperparameters>)` for an optimizer
 * node. `lrFirst` lists `lr` first (the Edu script); the legacy script keeps
 * the node's own order with `lr` last.
 */
export function optimizerExpression(opt, layerFields, { lrFirst = false } = {}) {
    const params = renderParameterGroup(opt.parameterGroup ?? [], layerFields);
    // PyTorch optimizers take `lr=`, not `learning_rate=`. Translate.
    const renamed = renameOptimizerKwargs(opt.hyperparameters, lrFirst);
    const hp = kwArgs(renamed, allowedOptimizerArgs(opt.type));
    const torchClass = `torch.optim.${opt.type}`;
    return `${torchClass}(${params}${hp ? ', ' + hp : ''})`;
}

function renameOptimizerKwargs(hp, lrFirst = false) {
    if (!hp) return {};
    const out = { ...hp };
    if ('learning_rate' in out) {
        out.lr = out.learning_rate;
        delete out.learning_rate;
    }
    if (lrFirst && 'lr' in out) {
        const { lr, ...rest } = out;
        return { lr, ...rest };
    }
    return out;
}

function allowedOptimizerArgs(type) {
    switch (type) {
        case 'Adam':
        case 'AdamW': return ['lr', 'betas', 'eps', 'weight_decay'];
        case 'SGD':   return ['lr', 'momentum', 'weight_decay', 'nesterov', 'dampening'];
        case 'RMSprop': return ['lr', 'alpha', 'eps', 'momentum', 'weight_decay'];
        default: return null;
    }
}

function renderParameterGroup(groupIds, layerFields) {
    if (!groupIds.length) return 'model.parameters()';
    const fieldNames = groupIds
        .map((id) => layerFields[id])
        .filter(Boolean);
    if (!fieldNames.length) return 'model.parameters()';
    if (fieldNames.length === 1) return `model.${fieldNames[0]}.parameters()`;
    const list = fieldNames.map((f) => `list(model.${f}.parameters())`).join(' + ');
    return list;
}

function emitLoss(loss) {
    const varName = toPyIdent(loss.label || loss.type, 'loss');
    return `${varName} = ${lossExpression(loss)}`;
}

/** `torch.nn.<LossModule>(<hyperparameters>)` for a loss node. */
export function lossExpression(loss) {
    const moduleName = lossModuleName(loss.type);
    const hp = kwArgs(loss.hyperparameters, allowedLossArgs(loss.type));
    return `${moduleName}(${hp})`;
}

export function lossModuleName(type) {
    return type === 'cross_entropy' ? 'torch.nn.CrossEntropyLoss'
        :  type === 'bce'           ? 'torch.nn.BCEWithLogitsLoss'
        :  type === 'mse'           ? 'torch.nn.MSELoss'
        :  type === 'l1'            ? 'torch.nn.L1Loss'
        :                              'torch.nn.Module  # custom — fill in';
}

function allowedLossArgs(type) {
    return type === 'cross_entropy' ? ['weight', 'reduction', 'label_smoothing']
        :  ['reduction', 'weight'];
}

/**
 * One loss of the training step: `name(prediction, target) * weight`.
 * Cross-entropy takes (N, C) scores and (N,) class indices. When the scores
 * are on the last axis of a longer output, such as a language model's
 * (B, T, V) against (B, T) next-token targets, both are flattened to (B·T, V)
 * and (B·T,) (`flatten`). Channels-first outputs, such as a CNN's
 * (B, C, H, W), keep PyTorch's own per-pixel cross-entropy.
 * `convertTarget` (Edu loaders yield int64 labels) turns the targets of
 * BCE, MSE and L1 into floats shaped like the prediction.
 */
export function lossTerm(loss, { flatten = false, prediction = 'logits', target = 'y', convertTarget = false } = {}) {
    const lossName = toPyIdent(loss.label || loss.type, 'loss');
    let input = prediction;
    let labels = target;
    if (loss.type === 'cross_entropy' && flatten) {
        input = `${prediction}.flatten(0, -2)`;
        labels = `${target}.flatten()`;
    } else if (convertTarget && ['bce', 'mse', 'l1'].includes(loss.type)) {
        labels = `${target}.float().view_as(${prediction})`;
    }
    return `${lossName}(${input}, ${labels}) * ${toPyLiteral(loss.weight ?? 1)}`;
}

/** The batch entry a loss compares against: `y` (read from "labels") or its own label key. */
function batchTarget(loss) {
    const key = loss.labelKey;
    return !key || key === 'labels' ? 'y' : `batch[${JSON.stringify(key)}].to(device)`;
}

/**
 * The root Inputs and Outputs with their inferred shapes (batch axis first,
 * as null), in the order the generated model takes and returns them.
 */
export function rootTensors(graph, templates = {}) {
    let shapes = {};
    try {
        shapes = validate(graph, { templates }).shapes ?? {};
    } catch {
        shapes = {};
    }
    const pick = (libraryKey) => Object.values(graph?.nodes ?? {})
        .filter((node) => node.kind === 'layer' && node.libraryKey === libraryKey)
        .map((node) => ({ id: node.id, node, shape: Array.isArray(shapes[node.id]) ? shapes[node.id] : null }));
    return { inputs: pick('deepflow.io.Input'), outputs: pick('deepflow.io.Output') };
}

const PASS_THROUGH = new Set([
    'torch.nn.ReLU', 'torch.nn.LeakyReLU', 'torch.nn.GELU', 'torch.nn.Tanh', 'torch.nn.Sigmoid',
    'torch.nn.Dropout', 'torch.nn.Softmax', 'torch.nn.LogSoftmax',
]);
const LAST_AXIS_PRODUCERS = new Set(['torch.nn.Linear', 'deepflow.op.MatMul']);

/**
 * Whether an Output's scores lie on its last axis: walking back through
 * activations, Dropout and (Log)Softmax reaches a Linear (layer or the
 * editable component) or a MatMul, which act on the last axis.
 */
export function scoresOnLastAxis(graph, templates, outputId) {
    const nodes = graph?.nodes ?? {};
    let current = nodes[outputId];
    const seen = new Set();
    while (current && !seen.has(current.id)) {
        seen.add(current.id);
        const incoming = (graph.edges ?? []).filter((edge) => edge.target === current.id && nodes[edge.source]);
        if (incoming.length !== 1) return false;
        const previous = nodes[incoming[0].source];
        if (previous.kind === 'layer') {
            if (LAST_AXIS_PRODUCERS.has(previous.libraryKey)) return true;
            if (!PASS_THROUGH.has(previous.libraryKey)) return false;
            current = previous;
            continue;
        }
        const template = templates[previous.templateId];
        return (template?.standardKey ?? template?.componentStyleKey) === 'linear';
    }
    return false;
}

/** Whether cross-entropy must flatten an output: more than two axes, scores on the last one. */
export function flattenForCrossEntropy(graph, templates, output) {
    return (output?.shape?.length ?? 2) > 2 && scoresOnLastAxis(graph, templates, output.id);
}

/** loss → whether its cross-entropy flattens the output it is bound to (else the first Output). */
function lossOutputRanks(archGraph, templates, losses) {
    // Only cross-entropy depends on the output's shape, so skip shape inference otherwise.
    if (!losses.some((loss) => loss.type === 'cross_entropy')) return () => false;
    const { outputs } = rootTensors(archGraph, templates);
    return (loss) => {
        const output = outputs.find((entry) => entry.id === loss.architectureOutputId) ?? outputs[0];
        return flattenForCrossEntropy(archGraph, templates, output);
    };
}

const plainTerm = (loss) => lossTerm(loss);

export function emitDefaultStep(optimizers, losses, term = plainTerm) {
    if (!optimizers.length || !losses.length) {
        return '# Add optimizers / losses to fill in the step.\n            pass';
    }
    const optName = toPyIdent(optimizers[0].label || optimizers[0].type, 'opt');
    const lossLines = losses.map((l) => term(l));
    const totalExpr = lossLines.length === 1 ? lossLines[0] : lossLines.join(' + ');
    return `${optName}.zero_grad()\nlogits = model(x)\nloss_total = ${totalExpr}\nloss_total.backward()\n${optName}.step()`;
}

export function emitPhases(phases, optimizers, losses, term = plainTerm) {
    const lines = [];
    for (const phase of phases) {
        const condParts = [];
        if ((phase.everyNSteps ?? 1) > 1) {
            condParts.push(`global_step % ${phase.everyNSteps} == 0`);
        }
        const cond = condParts.length ? `if ${condParts.join(' and ')}:` : null;
        const phaseOpts = (phase.optimizerIds ?? [])
            .map((id) => optimizers.find((o) => o.id === id))
            .filter(Boolean);
        const phaseLosses = (phase.lossIds ?? [])
            .map((id) => losses.find((l) => l.id === id))
            .filter(Boolean);
        if (!phaseOpts.length || !phaseLosses.length) continue;

        const lossParts = phaseLosses.map((l) => term(l));
        const totalExpr = lossParts.length === 1 ? lossParts[0] : lossParts.join(' + ');

        const inner = [];
        for (const o of phaseOpts) {
            inner.push(`${toPyIdent(o.label || o.type, 'opt')}.zero_grad()`);
        }
        inner.push(`logits = model(x)`);
        inner.push(`loss_total = ${totalExpr}`);
        inner.push(`loss_total.backward()`);
        for (const o of phaseOpts) {
            inner.push(`${toPyIdent(o.label || o.type, 'opt')}.step()`);
        }

        const innerSrc = inner.join('\n');
        const phaseLabel = phase.label || 'phase';
        if (cond) {
            lines.push(`# ${phaseLabel}\n${cond}\n${indent(innerSrc, 1)}`);
        } else {
            lines.push(`# ${phaseLabel}\n${innerSrc}`);
        }
    }
    return lines.length ? lines.join('\n\n') : '# (no phases configured)\npass';
}

/**
 * Build a flat map of architecture-node-id → Python field name in the
 * generated Model. We use the same fieldCounter rule as architecture
 * codegen so the names line up with what the model emits.
 */
export function collectLayerFields(rootGraph, templates) {
    const used = {};
    /** @type {Object<string, string>} */
    const out = {};
    function nextField(label) {
        const base = toPyIdent(label, 'layer');
        const n = (used[base] = (used[base] || 0) + 1);
        return n === 1 ? base : `${base}_${n}`;
    }
    function visit(graph) {
        const ids = Object.keys(graph.nodes ?? {});
        for (const id of ids) {
            const node = graph.nodes[id];
            if (node.kind === 'layer') {
                if (
                    node.libraryKey === 'deepflow.template.Input' ||
                    node.libraryKey === 'deepflow.template.Output' ||
                    node.libraryKey === 'deepflow.io.Input' ||
                    node.libraryKey === 'deepflow.io.Output' ||
                    node.libraryKey === 'deepflow.op.Split' ||
                    node.libraryKey === 'deepflow.op.Add' ||
                    node.libraryKey === 'deepflow.op.Multiply' ||
                    node.libraryKey === 'deepflow.op.Concat'
                ) continue;
                out[id] = nextField(node.label);
            } else if (node.kind === 'supernode' || node.kind === 'repeat') {
                const tpl = templates[node.templateId];
                if (tpl) visit(tpl.graph);
            }
        }
    }
    visit(rootGraph);
    return out;
}
