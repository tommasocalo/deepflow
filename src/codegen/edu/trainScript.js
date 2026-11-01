/**
 * train.py for a bound dataset (docs/edu/SPEC.md §6.11.2): one self-contained script with the model, the dataset's
 * loader (§6.11.1), the optimizer and loss of the Training tab (§6.10.5), a training loop with a validation pass
 * every epoch, an accuracy metric, one final test evaluation where the dataset has a test split, and, for
 * next-token data, a text sample. The same script runs locally (`python train.py`) and in Google Colab.
 *
 * Without a dataset, codegen/training.js keeps DeepFlow's stub script.
 */
import { PRODUCT_NAME } from '../../edu/config.js';
import { defaultObjective, effectiveObjective } from '../../edu/objective.js';
import { generateArchitectureModule } from '../architecture.js';
import { indent, toPyIdent } from '../python.js';
import {
    collectLayerFields, emitPhases, flattenForCrossEntropy, lossExpression, lossTerm, optimizerExpression, rootTensors,
} from '../training.js';

export const COLAB_NOTE = 'Runs in Google Colab: upload this file and run `!python train.py` '
    + '(GPU optional; CIFAR-10 and Tiny Shakespeare are faster on GPU).';
export const PHASES_COMMENT = "# Multi-phase training: see DeepFlow's phase graph.";

const WIDTH = 100;
const RULE = `# ${'='.repeat(WIDTH - 2)}`;
const section = (title) => `# ${'-'.repeat(WIDTH - 2)}\n# ${title}\n# ${'-'.repeat(WIDTH - 2)}`;
const MODEL_IMPORTS = new Set(['import torch', 'import torch.nn as nn', 'import torch.nn.functional as F']);
const STDLIB = /^import (math|os|random|json|urllib\.request)$/;
const DEFAULT_SETTINGS = { epochs: 50, batchSize: 32, seed: 1 };
const LOSS_NAMES = { cross_entropy: 'Cross-entropy', bce: 'Binary cross-entropy (BCE)', mse: 'Mean squared error (MSE)', l1: 'Mean absolute error (L1)' };

/** Wrap text into `# ` comment lines of at most WIDTH characters; `hang` indents continuation lines. */
function comment(text, { hang = '' } = {}) {
    const lines = [];
    let line = '#';
    for (const word of String(text).split(/\s+/).filter(Boolean)) {
        const next = `${line} ${word}`;
        if (next.length > WIDTH && line.length > 2) {
            lines.push(line);
            line = `# ${hang}${word}`;
        } else {
            line = next;
        }
    }
    lines.push(line);
    return lines.join('\n');
}

/** Text that is safe inside a Python docstring or comment. */
const plain = (value) => String(value).replace(/["\\]/g, "'").replace(/\s+/g, ' ');

/** The dimensions of a shape string, '(1, 28, 28)' → [1, 28, 28]; null when not a list of whole numbers. */
function dims(shape) {
    if (typeof shape !== 'string') return null;
    const parts = shape.replace(/[()[\]]/g, '').split(',').map((part) => part.trim()).filter(Boolean);
    const values = parts.map(Number);
    return values.length && values.every((value) => Number.isSafeInteger(value) && value > 0) ? values : null;
}
const shapeText = (values) => `(${values.join(', ')}${values.length === 1 ? ',' : ''})`;

/**
 * What differs from the browser (§6.11.2), as header lines: PyTorch's own initial weights and batch order, the
 * meaning of train_loss, and, for export datasets, their Colab presets for epochs and batch size.
 */
function comparisonNotes(spec, data, settings) {
    const preset = spec.training === 'export' ? exportSettingsNote(spec, data, settings) : null;
    return [
        spec.training === 'browser'
            ? 'Same data, split, loss and optimizer as in the browser, but PyTorch draws its own initial weights and batch order (from SEED), so the curves come out close to the browser\'s, not identical.'
            : 'PyTorch draws the initial weights and the batch order from SEED.',
        'train_loss is the running mean of each epoch\'s batch losses, measured in training mode (Dropout on) while the weights still change. val_loss is one pass after the epoch in evaluation mode, which is how the browser measures both losses.',
        ...(preset ? [preset] : []),
    ];
}

/** The Training tab's epochs and batch size, when train.py replaces them with an export dataset's presets. */
function tabSettings(data, settings) {
    const tab = data.settings ?? {};
    const differs = (name, value) => Number.isFinite(Number(tab[name])) && Math.round(Number(tab[name])) !== value;
    return { epochs: differs('epochs', settings.epochs) ? Math.round(Number(tab.epochs)) : null, batchSize: differs('batchSize', settings.batchSize) ? Math.round(Number(tab.batchSize)) : null };
}

function exportSettingsNote(spec, data, settings) {
    const tab = tabSettings(data, settings);
    const theirs = tab.epochs !== null || tab.batchSize !== null
        ? `, not the Training tab's (${tab.epochs ?? settings.epochs} and ${tab.batchSize ?? settings.batchSize})` : '';
    return `EPOCHS and BATCH_SIZE are ${spec.name}'s presets for Colab (${settings.epochs} and ${settings.batchSize})${theirs}.`;
}

/** Epochs, batch size and seed: export datasets use their Colab defaults (§6.11.2), the others the trainer's. */
export function scriptSettings(data) {
    const settings = { ...DEFAULT_SETTINGS, ...(data.settings ?? {}) };
    const defaults = data.spec.training === 'export' ? data.spec.exportDefaults : null;
    return {
        epochs: Math.max(1, Math.round(Number(defaults?.epochs ?? settings.epochs) || DEFAULT_SETTINGS.epochs)),
        batchSize: Math.max(1, Math.round(Number(defaults?.batchSize ?? settings.batchSize) || DEFAULT_SETTINGS.batchSize)),
        seed: Math.max(0, Math.round(Number(settings.seed) || 0)),
    };
}

/** An Input's shape without the batch axis: inferred (globals resolved) when known, else its shape field. */
function inputDims(input) {
    const inferred = input?.shape?.slice(1);
    if (inferred?.length && inferred.every((value) => Number.isSafeInteger(value) && value > 0)) return inferred;
    return dims(input?.node?.parameters?.shape);
}

/** The sequence length T of a (T,) Input, or null. */
function sequenceLength(input) {
    const values = inputDims(input);
    return values?.length === 1 ? values[0] : null;
}

/** Python import lines: the loader's and the script's, minus the model's own, standard library first. */
function importBlock(lists) {
    const lines = [...new Set(lists.flat())].filter((line) => !MODEL_IMPORTS.has(line));
    const stdlib = lines.filter((line) => STDLIB.test(line));
    const others = lines.filter((line) => !STDLIB.test(line));
    return [stdlib.join('\n'), others.join('\n')].filter(Boolean).join('\n\n');
}

/** What the learner should fix before running: input shape and type, output size, extra Inputs (as comments). */
function scriptChecks(spec, tensors, { blockSize, datasetBlockSize, lossType }) {
    const checks = [];
    const [input, ...extraInputs] = tensors.inputs;
    const expected = dims(spec.input.shape);
    if (!input) {
        checks.push('Your model has no Input block.');
    } else {
        const actual = inputDims(input);
        if (spec.target.kind === 'next-token') {
            if (!actual || actual.length !== 1) checks.push(`Your Input is ${input.node.parameters?.shape ?? 'unset'}, but ${spec.name} examples are windows of token ids, shaped (T,): use Data → Use this dataset.`);
        } else if (expected && (!actual || actual.join(',') !== expected.join(','))) {
            checks.push(`Your Input is ${actual ? shapeText(actual) : 'unset'}, but ${spec.name} examples are ${shapeText(expected)}: use Data → Use this dataset.`);
        }
        const dtype = input.node.parameters?.dtype ?? 'float32';
        if (spec.input.dtype === 'int64' && dtype !== 'int64') checks.push(`Your Input's dtype is ${dtype}, but ${spec.name} gives token ids (int64).`);
    }
    if (extraInputs.length) checks.push(`Your model has ${tensors.inputs.length} Input blocks, but ${spec.name} gives one input per example: train.py passes it to the first.`);
    const output = tensors.outputs[0]?.shape;
    const width = output?.at(-1);
    if (Number.isSafeInteger(width)) {
        const size = spec.target.size;
        if (spec.target.kind === 'classes') {
            const ok = size === 2 ? (lossType === 'cross_entropy' ? width === 2 : width === 1) : width === size;
            const want = size === 2 ? (lossType === 'cross_entropy' ? 2 : 1) : size;
            if (!ok) checks.push(`Your model gives ${width} output${width === 1 ? '' : 's'} per example, but ${spec.name} needs ${want} for ${size} classes: set the last Linear's Output to ${want}.`);
        } else if (spec.target.kind === 'next-token') {
            if (width !== size) checks.push(`Your model gives ${width} scores per position, but ${spec.name} has ${size} distinct characters: set the last Linear's Output to ${size}.`);
            if (output.length !== 3) checks.push(`Your model's output is ${output.length - 1}-D per example; a next-character model gives one row of scores per position, (B, T, ${size}).`);
        } else if (width !== 1) {
            checks.push(`Your model gives ${width} outputs per example, but ${spec.name} has 1 target value: set the last Linear's Output to 1.`);
        }
    }
    if (blockSize && datasetBlockSize && blockSize !== datasetBlockSize) {
        checks.push(`train.py reads windows of ${blockSize} characters, the length of your Input, instead of the dataset's ${datasetBlockSize}.`);
    }
    return checks;
}

/** The effective optimizer and loss, with the browser trainer's defaults when the Training tab has none. */
function resolveObjective(trainGraph, archGraph, spec, tensors) {
    const found = effectiveObjective(trainGraph, archGraph);
    const width = tensors.outputs[0]?.shape?.at(-1) ?? null;
    const defaults = defaultObjective({
        targetKind: spec.target.kind,
        numClasses: spec.target.kind === 'classes' ? spec.target.size : null,
        outputSize: Number.isSafeInteger(width) ? width : null,
    });
    return { optimizer: found.optimizer, loss: found.loss, defaults, output: found.output };
}

/** How train.py turns outputs into a loss and an accuracy for this task and loss. */
function lossPlan({ loader, lossType, lastAxis }) {
    const task = loader.task;
    const flat = lossType === 'cross_entropy' && (lastAxis || task === 'next-token');
    const classes = task === 'binary' || task === 'classification' || task === 'next-token';
    let accuracy = null;
    if (classes) {
        if (lossType === 'cross_entropy' || lossType === 'custom') accuracy = 'argmax';
        else if (lossType === 'bce') accuracy = 'logit';
        else if (task !== 'next-token') accuracy = 'threshold';
    }
    return { task, flat, accuracy };
}

function optimizerSource(objective, layerFields) {
    const optimizer = objective.optimizer;
    const node = optimizer ?? { type: objective.defaults.optimizer.type, hyperparameters: objective.defaults.optimizer.hyperparameters, parameterGroup: [] };
    const doc = optimizer
        ? `${node.type}, as set in the Training tab.${node.parameterGroup?.length ? ' Only its parameter group is trained; the other weights stay as they are.' : ''}`
        : `No optimizer in the Training tab: ${node.type} with learning rate 0.01, as the browser trainer uses.`;
    return `def make_optimizer(model):
    """${doc}"""
    return ${optimizerExpression(node, layerFields, { lrFirst: true })}`;
}

function lossModuleSource(objective) {
    const loss = objective.loss ?? { type: objective.defaults.loss.type, hyperparameters: objective.defaults.loss.hyperparameters };
    const note = objective.loss
        ? `# ${LOSS_NAMES[loss.type] ?? 'The loss'} from the Training tab ("${plain(objective.loss.label ?? loss.type)}").`
        : `# No loss in the Training tab: ${LOSS_NAMES[loss.type] ?? loss.type}, as the browser trainer uses.`;
    if (loss.type === 'custom') {
        const module = String(loss.hyperparameters?.module ?? '').trim();
        if (/^[A-Za-z_][\w.]*$/.test(module)) return `${note}\nloss_fn = ${module}()`;
        return `${note} Write your loss here: it gets the model's outputs and the targets.
def loss_fn(logits, y):
    raise NotImplementedError("Write your custom loss here (see the Training tab).")`;
    }
    return `${note}\nloss_fn = ${lossExpression(loss)}`;
}

function computeLossSource({ lossType, flat, reduction }) {
    const mean = reduction === 'none' ? '.mean()' : '';
    if (lossType === 'cross_entropy') {
        if (flat) {
            return `def compute_loss(logits, y):
    """Cross-entropy at every position: the scores (B, T, V) become (B*T, V) and the targets (B, T) become
    (B*T,), so each position counts as one prediction of the next token."""
    return loss_fn(logits.flatten(0, -2), y.flatten())${mean}`;
        }
        return `def compute_loss(logits, y):
    """Cross-entropy takes one score per class (the softmax is inside the loss) and the index of the right class."""
    return loss_fn(logits, y)${mean}`;
    }
    if (lossType === 'bce') {
        return `def compute_loss(logits, y):
    """BCE takes one score per example (the sigmoid is inside the loss); the 0/1 labels become floats of the same shape."""
    return loss_fn(logits, y.float().view_as(logits))${mean}`;
    }
    if (lossType === 'mse' || lossType === 'l1') {
        return `def compute_loss(logits, y):
    """The targets become floats shaped like the outputs, so the loss compares each output with its target."""
    return loss_fn(logits, y.float().view_as(logits))${mean}`;
    }
    return `def compute_loss(logits, y):
    return loss_fn(logits, y)${mean}`;
}

function accuracySource(rule, task) {
    if (rule === 'argmax') {
        return task === 'next-token'
            ? `def accuracy(logits, y):
    """Fraction of positions where the highest score is the token that really comes next."""
    return (logits.argmax(dim=-1) == y).float().mean().item()`
            : `def accuracy(logits, y):
    """Fraction of examples whose highest score is the right class."""
    return (logits.argmax(dim=-1) == y).float().mean().item()`;
    }
    if (rule === 'logit') {
        return `def accuracy(logits, y):
    """A score above 0 (a probability above 0.5 after the sigmoid) means class 1."""
    return ((logits.view_as(y) > 0).long() == y).float().mean().item()`;
    }
    if (rule === 'threshold') {
        return `def accuracy(logits, y):
    """An output above 0.5 means class 1."""
    return ((logits.view_as(y) > 0.5).long() == y).float().mean().item()`;
    }
    return null;
}

function predictSource(tensors, outputIndex) {
    if (tensors.outputs.length < 2) return null;
    const label = tensors.outputs[outputIndex]?.node?.label ?? 'Output';
    return `def predict(model, x):
    """Your model has ${tensors.outputs.length} Output blocks and returns one tensor for each. The loss and the
    accuracy use output ${outputIndex} (${plain(label)}), the one the Training tab's loss is connected to."""
    return model(x)[${outputIndex}]`;
}

function evaluateSource({ call, accuracy, sum }) {
    const weight = sum ? '' : ' * len(y)';
    if (!accuracy) {
        return `def evaluate(model, loader):
    """Mean loss on one split. model.eval() switches Dropout off (every unit is used) and makes BatchNorm use
    its running averages; torch.no_grad() skips the gradient bookkeeping, because nothing is learned here."""
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = ${call}
            total_loss += compute_loss(logits, y).item()${weight}
            count += len(y)
    return total_loss / count, None`;
    }
    return `def evaluate(model, loader):
    """Mean loss and accuracy on one split. model.eval() switches Dropout off (every unit is used) and makes
    BatchNorm use its running averages; torch.no_grad() skips the gradient bookkeeping, because nothing is
    learned here."""
    model.eval()
    total_loss, total_correct, count = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = ${call}
            total_loss += compute_loss(logits, y).item()${weight}
            total_correct += accuracy(logits, y) * len(y)
            count += len(y)
    return total_loss / count, total_correct / count`;
}

function generateSource(call) {
    const scores = call === 'model(x)' ? 'model(ids[:, -BLOCK_SIZE:])' : 'predict(model, ids[:, -BLOCK_SIZE:])';
    return `def generate(model, vocab, start="\\n", length=300, temperature=0.8):
    """Write \`length\` characters after \`start\`, one at a time. The model reads the last BLOCK_SIZE characters
    (the most it was trained on), its scores at the last position are divided by the temperature and turned
    into probabilities, and the next character is drawn from them. A temperature below 1 makes likely
    characters even likelier; above 1 it flattens the distribution."""
    model.eval()
    ids = torch.tensor([vocab.encode(start)], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        for _ in range(length):
            scores = ${scores}[:, -1, :]  # scores for the character after the last one
            probs = torch.softmax(scores / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
    return vocab.decode(ids[0].tolist())`;
}

/** The "before training" line: the validation loss of the untrained model and what guessing would give. */
function initialLine({ task, lossType, spec, accuracy }) {
    const acc = accuracy ? '  val_acc {val_acc:.3f}' : '';
    let expectation = '';
    if (task === 'next-token' && lossType === 'cross_entropy') {
        expectation = `  (guessing evenly among ${spec.target.size} tokens gives ln ${spec.target.size} = {math.log(${spec.target.size}):.4f})`;
    } else if ((task === 'classification' || task === 'binary') && lossType === 'cross_entropy') {
        expectation = `  (guessing evenly among ${spec.target.size} classes gives ln ${spec.target.size} = {math.log(${spec.target.size}):.4f})`;
    } else if (task === 'binary' && lossType === 'bce') {
        expectation = '  (a model that answers 0.5 for every example gives ln 2 = {math.log(2):.4f})';
    }
    return { text: `print(f"before training  val_loss {val_loss:.4f}${acc}${expectation}")`, usesMath: !!expectation };
}

/**
 * @param {Object} archGraph  the root architecture graph
 * @param {Object} trainGraph the training graph
 * @param {{ templates?: Object, data: { spec: Object, options?: Object, split?: Object, dataSeed?: number,
 *          settings?: { epochs?: number, batchSize?: number, seed?: number } } }} opts
 * @returns {string} Python source
 */
export function generateEduTrainingScript(archGraph, trainGraph, { templates = {}, data }) {
    const { spec } = data;
    if (typeof spec?.python !== 'function') throw new Error(`Dataset "${spec?.id ?? '?'}" has no Python loader`);
    const settings = scriptSettings(data);
    const tensors = rootTensors(archGraph, templates);
    const blockSize = spec.target.kind === 'next-token' ? sequenceLength(tensors.inputs[0]) : null;
    const options = blockSize ? { ...(data.options ?? {}), blockSize } : (data.options ?? {});
    const loader = spec.python({ options, split: data.split ?? {}, batchSize: settings.batchSize, dataSeed: data.dataSeed ?? 1, trainSeed: settings.seed });
    const modelSource = generateArchitectureModule(archGraph, { templates, className: 'Model' });

    const training = trainGraph ?? { nodes: {}, edges: [] };
    const nodes = Object.values(training.nodes ?? {});
    const phases = nodes.filter((node) => node.kind === 'phase');
    const losses = nodes.filter((node) => node.kind === 'loss');
    const optimizers = nodes.filter((node) => node.kind === 'optimizer');
    const objective = resolveObjective(training, archGraph, spec, tensors);
    const datasetBlockSize = spec.options?.blockSize ? Number(data.options?.blockSize ?? spec.options.blockSize.default) : null;
    const lossNode = objective.loss ?? objective.defaults.loss;
    const lossType = lossNode.type;
    const reduction = lossNode.hyperparameters?.reduction ?? 'mean';
    const outputIndex = Math.max(0, tensors.outputs.findIndex((entry) => entry.id === objective.output?.id));
    const lastAxis = flattenForCrossEntropy(archGraph, templates, tensors.outputs[outputIndex]);
    const plan = lossPlan({ loader, lossType, lastAxis });
    const call = tensors.outputs.length > 1 ? 'predict(model, x)' : 'model(x)';
    const layerFields = collectLayerFields(archGraph, templates);
    const indexOf = (loss) => Math.max(0, tensors.outputs.findIndex((entry) => entry.id === loss.architectureOutputId));
    const phaseCode = phases.length && optimizers.length && losses.length
        ? emitPhases(phases, optimizers, losses, (loss) => lossTerm(loss, {
            flatten: loss.type === 'cross_entropy' && (spec.target.kind === 'next-token' || flattenForCrossEntropy(archGraph, templates, tensors.outputs[indexOf(loss)])),
            prediction: tensors.outputs.length > 1 ? `logits[${indexOf(loss)}]` : 'logits',
            convertTarget: true,
        }))
        : '';
    // Phases that name no optimizer or no loss emit no step; train like a single-objective graph then.
    const usePhases = phaseCode.includes('loss_total = ');

    // 1. Header.
    const checks = scriptChecks(spec, tensors, { blockSize, datasetBlockSize, lossType });
    const licence = spec.licence ?? {};
    const header = [
        RULE,
        comment(`train.py: train your model on ${spec.name}. Written by ${PRODUCT_NAME} from your model, your dataset and the Training tab.`),
        comment(COLAB_NOTE),
        '#',
        comment(`Dataset: ${loader.comment}`, { hang: '  ' }),
        ...(String(loader.comment).includes(spec.attribution) ? [] : [comment(`Attribution: ${spec.attribution}`, { hang: '  ' })]),
        comment(`Licence: ${licence.name ?? 'see the source'}${licence.verified === false ? ' (not verified)' : ''}${licence.url ? `. ${licence.url}` : ''}`, { hang: '  ' }),
        '#',
        spec.training === 'browser' ? '# Compared with the browser:' : '# Notes:',
        ...comparisonNotes(spec, data, settings).map((line) => comment(`- ${line}`, { hang: '  ' })),
        ...(checks.length ? ['#', '# Check before running:', ...checks.map((line) => comment(`- ${line}`, { hang: '  ' }))] : []),
        RULE,
    ].join('\n');

    // 3–4. Imports and constants.
    const initial = initialLine({ task: plan.task, lossType, spec, accuracy: plan.accuracy });
    const imports = importBlock([initial.usesMath ? ['import math'] : [], loader.imports]);
    const seedNote = spec.kind === 'toy2d'
        ? "the dataset's seed: the generated points and the train/validation split, as in the browser"
        : spec.training === 'browser'
            ? "the dataset's seed: which examples go to training and which to validation, as in the browser"
            : "the seed of the train/validation split";
    // Browser datasets take EPOCHS and BATCH_SIZE from the Training tab as they are; export datasets use presets.
    const tab = tabSettings(data, settings);
    const preset = (value) => `${spec.name}'s preset for Colab${value === null ? '' : ` (the Training tab says ${value})`}`;
    const epochsNote = spec.training === 'export' ? `  # ${preset(tab.epochs)}; change it freely` : '';
    const batchNote = spec.training === 'export' ? `  # ${preset(tab.batchSize)}` : '';
    const constants = [
        `SEED = ${settings.seed}  # seeds PyTorch's initial weights, Dropout and batch order (its own draws, not the browser's)`,
        `DATA_SEED = ${Math.round(Number(data.dataSeed ?? 1)) || 0}  # ${seedNote}`,
        `EPOCHS = ${settings.epochs}${epochsNote}`,
        `BATCH_SIZE = ${settings.batchSize}${batchNote}`,
        'DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # the GPU if there is one',
    ].join('\n');

    // 6–9. Optimizer, loss, compute_loss, accuracy, evaluate.
    const extraLosses = losses.filter((loss) => loss !== objective.loss);
    const lossNotes = !usePhases && objective.loss && extraLosses.length
        ? comment(`The Training tab has ${losses.length} losses. A dataset gives one target per example, so train.py trains with "${plain(objective.loss.label ?? objective.loss.type)}", the loss connected to the model's first Output.`) + '\n'
        : '';
    const objectiveBlocks = [
        usePhases ? null : optimizerSource(objective, layerFields),
        lossNotes + lossModuleSource(objective),
        computeLossSource({ lossType, flat: plan.flat, reduction }),
        accuracySource(plan.accuracy, plan.task),
        predictSource(tensors, outputIndex),
    ].filter(Boolean);

    const evaluate = evaluateSource({ call, accuracy: plan.accuracy, sum: reduction === 'sum' });
    const text = plan.task === 'next-token' ? generateSource(call) : null;

    // 10. train().
    const unpack = plan.accuracy ? 'val_loss, val_acc' : 'val_loss, _';
    const accText = plan.accuracy ? '  val_acc {val_acc:.3f}' : '';
    const testAcc = plan.accuracy ? '  test_acc {test_acc:.3f}' : '';
    const weight = reduction === 'sum' ? '' : ' * len(y)';
    let setup;
    let step;
    if (usePhases) {
        const phaseLossNames = [...new Set(losses.map((loss) => toPyIdent(loss.label || loss.type, 'loss')))];
        setup = [
            '# Every Optimizer and Loss block of the Training tab; each phase of the loop below uses some of them.',
            ...optimizers.map((opt) => `${toPyIdent(opt.label || opt.type, 'opt')} = ${optimizerExpression(opt, layerFields, { lrFirst: true })}`),
            ...losses.map((loss) => `${toPyIdent(loss.label || loss.type, 'loss')} = ${lossExpression(loss)}`),
            'global_step = 0',
        ].join('\n');
        step = [
            PHASES_COMMENT,
            phaseCode,
            `loss = loss_total  # the last phase's loss, for the progress line (${phaseLossNames.join(', ')})`,
        ].join('\n');
    } else {
        setup = 'optimizer = make_optimizer(model)';
        step = [
            'optimizer.zero_grad()  # clear the gradients of the previous batch',
            `loss = compute_loss(${call}, y)`,
            'loss.backward()  # the gradient of the loss for every weight',
            'optimizer.step()  # move every weight a small step against its gradient',
        ].join('\n');
    }
    const counter = usePhases ? '\n            global_step += 1' : '';
    const sample = text
        ? `\n    print("\\nA sample of generated text (temperature 0.8):")\n    print(generate(model, train_loader.vocab, start="\\n", length=300, temperature=0.8))`
        : '';
    const trainSource = `def train():
    """Train the model, measure it on the validation split after every epoch, and return it with the weights
    of the epoch that had the lowest validation loss."""
    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader = build_dataloaders(BATCH_SIZE, DATA_SEED, SEED)
    model = Model().to(DEVICE)
${indent(setup, 1)}
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, training on {DEVICE}")

    ${unpack} = evaluate(model, val_loader)
    ${initial.text}

    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, EPOCHS + 1):
        model.train()  # training mode: Dropout drops units
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
${indent(step, 3)}
            total += loss.item()${weight}
            count += len(y)${counter}
        ${unpack} = evaluate(model, val_loader)
        # train_loss: the running mean of this epoch's batch losses, in training mode (Dropout on).
        # val_loss: one pass after the epoch in evaluation mode, as the browser measures both losses.
        print(f"epoch {epoch:3d}  train_loss {total / count:.4f}  val_loss {val_loss:.4f}${accText}")
        if val_loss < best_loss:  # keep a copy of the best weights so far
            best_loss, best_epoch = val_loss, epoch
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    print(f"best epoch: {best_epoch} (lowest val_loss {best_loss:.4f}); keeping its weights")
    if best_state is not None:
        model.load_state_dict(best_state)
    if test_loader is not None:
        test_loss, ${plan.accuracy ? 'test_acc' : '_'} = evaluate(model, test_loader)
        print(f"test  test_loss {test_loss:.4f}${testAcc}  (measured once, with the best epoch's weights)")${sample}
    return model`;

    const blocks = [
        header,
        modelSource.trimEnd(),
        `${section('Settings')}\n${imports}\n\n${constants}`,
        `${section(`Data: ${spec.name}`)}\n${loader.source.trimEnd()}`,
        `${section('Loss, optimizer and accuracy (from the Training tab)')}\n${objectiveBlocks.join('\n\n\n')}`,
        `${section('Training and evaluation')}\n${[evaluate, text, trainSource].filter(Boolean).join('\n\n\n')}`,
        'if __name__ == "__main__":\n    train()',
    ];
    return `${blocks.join('\n\n\n')}\n`;
}
