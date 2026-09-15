import { generateArchitectureModule } from './architecture.js';
import { indent, kwArgs, toPyIdent, toPyLiteral } from './python.js';

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
 * @param {import('../domain/architecture/types.js').ArchitectureGraph} archGraph
 * @param {import('../domain/training/types.js').TrainingGraph} trainGraph
 * @param {Object} opts
 * @param {Object<string, import('../domain/architecture/types.js').ArchitectureTemplate>} [opts.templates]
 */
export function generateTrainingScript(archGraph, trainGraph, { templates = {} } = {}) {
    const modelSrc = generateArchitectureModule(archGraph, { templates, className: 'Model' });
    const optimizers = Object.values(trainGraph.nodes ?? {}).filter((n) => n.kind === 'optimizer');
    const losses = Object.values(trainGraph.nodes ?? {}).filter((n) => n.kind === 'loss');
    const phases = Object.values(trainGraph.nodes ?? {}).filter((n) => n.kind === 'phase');

    const layerFields = collectLayerFields(archGraph, templates);

    const optimizerInit = optimizers.map((o) => emitOptimizer(o, layerFields)).join('\n');
    const lossInit = losses.map((l) => emitLoss(l)).join('\n');
    const phaseDefs = phases.length
        ? emitPhases(phases, optimizers, losses)
        : emitDefaultStep(optimizers, losses);

    return `${modelSrc}\n\n# -----------------------------------------------------------------------\n# Auto-generated training script. Edit the data loaders to match your task.\n# -----------------------------------------------------------------------\n\nimport torch\nfrom torch.utils.data import DataLoader\n\n\ndef build_dataloaders():\n    """Replace this with real datasets — see Data graph in DeepFlow v2."""\n    raise NotImplementedError("Wire up your datasets / dataloaders here.")\n\n\ndef train(num_epochs: int = 10, device: str | None = None):\n    device = device or ("cuda" if torch.cuda.is_available() else "cpu")\n    model = Model().to(device)\n\n${indent(optimizerInit || '# (no optimizers)', 1)}\n${indent(lossInit || '# (no losses)', 1)}\n\n    train_loader, val_loader = build_dataloaders()\n\n    global_step = 0\n    for epoch in range(num_epochs):\n        model.train()\n        for batch in train_loader:\n            x = batch["inputs"].to(device)\n            y = batch.get("labels")\n            if y is not None:\n                y = y.to(device)\n${indent(phaseDefs, 3)}\n            global_step += 1\n        # TODO: validation pass per epoch\n        print(f"epoch {epoch} done — step={global_step}")\n\n\nif __name__ == "__main__":\n    train()\n`;
}

// -------- helpers --------

function emitOptimizer(opt, layerFields) {
    const varName = toPyIdent(opt.label || opt.type, 'opt');
    const params = renderParameterGroup(opt.parameterGroup ?? [], layerFields);
    // PyTorch optimizers take `lr=`, not `learning_rate=`. Translate.
    const renamed = renameOptimizerKwargs(opt.hyperparameters);
    const hp = kwArgs(renamed, allowedOptimizerArgs(opt.type));
    const torchClass = `torch.optim.${opt.type}`;
    return `${varName} = ${torchClass}(${params}${hp ? ', ' + hp : ''})`;
}

function renameOptimizerKwargs(hp) {
    if (!hp) return {};
    const out = { ...hp };
    if ('learning_rate' in out) {
        out.lr = out.learning_rate;
        delete out.learning_rate;
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
    const moduleName = lossModuleName(loss.type);
    const hp = kwArgs(loss.hyperparameters, allowedLossArgs(loss.type));
    return `${varName} = ${moduleName}(${hp})`;
}

function lossModuleName(type) {
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

function emitDefaultStep(optimizers, losses) {
    if (!optimizers.length || !losses.length) {
        return '# Add optimizers / losses to fill in the step.\n            pass';
    }
    const optName = toPyIdent(optimizers[0].label || optimizers[0].type, 'opt');
    const lossLines = losses.map((l) => {
        const lossName = toPyIdent(l.label || l.type, 'loss');
        return `${lossName}(logits, y) * ${toPyLiteral(l.weight ?? 1)}`;
    });
    const totalExpr = lossLines.length === 1 ? lossLines[0] : lossLines.join(' + ');
    return `${optName}.zero_grad()\nlogits = model(x)\nloss_total = ${totalExpr}\nloss_total.backward()\n${optName}.step()`;
}

function emitPhases(phases, optimizers, losses) {
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

        const lossParts = phaseLosses.map((l) => {
            const lossName = toPyIdent(l.label || l.type, 'loss');
            return `${lossName}(logits, y) * ${toPyLiteral(l.weight ?? 1)}`;
        });
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
function collectLayerFields(rootGraph, templates) {
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
