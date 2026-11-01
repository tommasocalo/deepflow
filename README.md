# DeepFlow Edu

DeepFlow Edu is an educational fork of DeepFlow v2, the visual editor for neural-network architectures described in the second half of this file. It adds a guided course that starts from one neuron and ends inside a small Transformer. You work in the real editor; each lesson step is checked against the live model, its validation report, the training graph, the bound dataset and your training runs. Help comes as a ladder (a nudge, a specific hint, then **Show me**, which performs the step as one undoable edit). You predict a number, a shape or an outcome before the tool reveals it. Small models train in the browser, and every model exports to a complete PyTorch `train.py` that runs in Google Colab. The product and curriculum specification, with its change log, is [docs/edu/SPEC.md](docs/edu/SPEC.md).

**How it relates to earlier work** (SPEC §0). In-application tutorial systems, such as Stencils-based tutorials (Kelleher and Pausch, CHI 2005), DocWizards (Bergman et al., UIST 2005), ToolClips (Grossman and Fitzmaurice, CHI 2010), Pause-and-Play (Pongnumkul et al., UIST 2011) and MixT (Chi et al., UIST 2012), teach inside a real application: they show the next step at the interface element that performs it, or follow the application's state. Visual ML tools used in teaching, such as Orange and KNIME, let learners build and run ML workflows visually. DeepFlow Edu differs in three ways: each step is checked against the model, validation and training state of a full neural-network editor, not against screen state or a click script; feedback is tied to named misconceptions and comes as graded help; and the browser run and the exported PyTorch script share the data, the split, the loss and the optimizer.

## Run DeepFlow Edu

Use Node.js 20.19+, 22.13+ or 24+:

```sh
npm install
npm run dev
```

Open the address Vite prints, http://localhost:5173/ by default (`npm run dev -- --port 5174` picks another port). For a production build run `npm run build`, then `npm run preview` (http://localhost:4173/).

**What a learner sees first.** The app opens on the **Learn** tab: the heading "Learn to build neural networks", a **Start · 1 Your first neuron** button, **Open the sandbox**, and one card per module with its status, datasets and time. Modules are never locked; a card says which module it is recommended after. Module 1 opens with two myth-busting warm-up questions, then its first step in the lesson panel.

**During a lesson** the panel is docked to the right of the Architecture, Data and Training tabs (360 px, 320 px below 1200 px, a bottom drawer below 900 px); it never covers the canvas. A step shows its subgoal (for example BUILD THE MODEL), up to 60 words of text, an optional "Why?", a pink ring around the control it names, a live status line, the hint ladder, and Back, Reset step, Skip and Next. Validation errors appear there rewritten in plain words, for example "Linear 1 expects 784 input features but receives 64." Prediction gates keep Train locked, or Dimensions hidden, until you answer. A prediction about a training result is only locked when you submit it; once your run passes the step, the panel compares it with what that run shows ("You said … · Your run shows …"). Options appear in an order shuffled for each learner (seeded, stable across reloads, and recorded in the learning log), and a quiz answer counts only when you press **Check**. A misconception note waits until you stop editing for a few seconds, train or ask for a hint. The status line and the navigation stay at the bottom of the panel, and its header has Undo and Redo. Each module ends with a short quiz and three takeaways. Exit (✕) saves the lesson and returns to Learn; Continue resumes where you left off, also after a reload.

**The sandbox** (Open the sandbox) is the plain DeepFlow editor with the GPT-2 example, plus the Data and Training extras. The sandbox and each module keep their own workspace, so lessons never overwrite your own model.

**Progress** stays in this browser (localStorage): nothing is sent over the network. The Learn footer has **Reset progress** (it keeps the sandbox) and **Download learning log** (`deepflow-edu-log.json`, the local event log for studies). A link such as `#/learn/m3/s4` opens a step directly; the lesson fills in the steps before it, without their answers or training runs, and says so.

## The seven modules

| # | Module | What you do | Datasets | Min |
|---|---|---|---|---|
| 1 | Your first neuron | Count a neuron's parameters, train it on two blobs, then turn it into a 3-class Iris classifier with cross-entropy, read its confusion matrix and find the train-split scaling in train.py | Two blobs, Iris | 20 |
| 2 | Going non-linear | Watch one neuron and two stacked Linear blocks fail on a circle, add ReLU, find the smallest hidden width, open Linear down to MatMul, then build an XOR network yourself | Circle, XOR | 20 |
| 3 | Shapes and parameters | Read (B, C, H, W) shapes, Flatten MNIST, build and hand-count an MLP, fix a planted shape bug, train a small version on Digits 8×8 and export it for MNIST | MNIST, Digits 8×8 | 25 |
| 4 | Convolutions | Predict Conv2d and pooling output shapes with and without padding, build two convolution blocks, connect them to a classifier and count their shared weights | Fashion-MNIST | 20 |
| 5 | Training well | Check the initial loss against ln C, sweep the learning rate, overfit on purpose, read the curves (early stopping, and why the final score needs a test split), add Dropout and compare honestly, find `model.eval()` | Spiral (3 classes) | 25 |
| 6 | Reusable blocks and residual connections | Add a skip connection, group it into a component, repeat it three times, edit the shared definition and watch all three copies follow, see separate weights and the gradient paths | CIFAR-10 | 20 |
| 7 | Attention and a tiny Transformer | Predict a character-level GPT's output shape, open its embedding table, positions, causal attention heads and head width, sample with a temperature, then scale the same blocks to GPT-2 small | Tiny Shakespeare | 25 |

Every module has 6–8 steps (52 in all) ordered read and predict, then modify, then a build of your own; modules 2–7 open with two retrieval questions on earlier modules, and every module ends by reading the exported code. Numbers in the copy come from a facts ledger: the tests recompute each one from the fixtures, except five marked with their source: two results quoted from the Fashion-MNIST README, CIFAR-10's download size, and two measured with the trainer, the Dropout comparison (SPEC §11.2) and the range of Iris's initial loss (§11.1).

## Datasets

The **Data** tab lists the datasets in two groups. Each shows a preview, its class balance, the split (validation 10–50 %, test 0, 10 or 20 %) and seeds, the scaling (statistics fitted on the training split only), its source and licence. **Use this dataset** sets the model's Input shape and the last Linear's output size in one undoable edit; in a lesson step whose exercise is choosing the output size, it sets only the Input.

**Trained in the browser** (generated or embedded; nothing is downloaded):

| Dataset | Source | Licence |
|---|---|---|
| Two blobs, Circle, XOR, Spiral (2 classes), Plane (regression) | Seeded generators, adapted from TensorFlow Playground | Generated here; algorithm Apache-2.0 |
| Spiral (3 classes) | Seeded generator, adapted from CS231n | Generated here; algorithm MIT |
| Iris (150 flowers, 4 features, 3 species) | Fisher (1936); values as in scikit-learn's `load_iris()` (not the UCI copy, which has two wrong rows) | Public-domain measurements; file BSD-3-Clause (scikit-learn) |
| Digits 8×8 (1,797 images, 10 classes) | scikit-learn's copy of the UCI optical digits (Alpaydin & Kaynak, 1998); loaded as a separate chunk | BSD-3-Clause (scikit-learn copy) |

**Export only** (shapes, classes, sizes and licence in the app; `train.py` downloads the data when it runs):

| Dataset | Downloaded by train.py from | Licence |
|---|---|---|
| MNIST (70,000 images) | torchvision `datasets.MNIST` | Not bundled; no licence stated by the authors |
| Fashion-MNIST (70,000 images) | torchvision `datasets.FashionMNIST` | MIT, © 2017 Zalando SE |
| CIFAR-10 (60,000 images, about 170 MB) | torchvision `datasets.CIFAR10` | Not bundled; no licence stated by the authors |
| Tiny Shakespeare (1,115,394 characters, 65 distinct) | karpathy/char-rnn's `input.txt` | Public-domain text; file MIT |

The exported Python regenerates the toy data and reproduces every split from the same seeds (`scripts/edu-python-parity.test.mjs` compares them with the browser's).

## The in-browser trainer

The **Training** tab keeps DeepFlow's training canvas (optimizer, loss and phase blocks) on top and adds the trainer below; lessons embed a compact card with the same controls. It uses the training canvas's optimizer and loss, shows train and validation loss and accuracy per epoch (epoch 0 is the untrained model, both measured with Dropout off), the chance level ln C, the best-validation epoch (▲), up to five earlier runs dimmed and labelled with what differed, a decision boundary for 2D data and a confusion matrix on the validation split. **Check initial loss** evaluates the untrained model once. When the model or the dataset has changed since the shown run, the card and the tab dim that result and say "This run used a different model: press Train to see results for this one". When the dataset has a test split (M5's overfitting run holds out 30 points), a run keeps its best epoch's weights and scores the test split once with them: "Test accuracy 90% (best epoch 141, used once)".

- **Scope.** Graphs with one Input and one Output built from Linear (the layer or the editable component), ReLU, LeakyReLU, Tanh, Sigmoid, GELU, Softmax, LogSoftmax, Dropout, Flatten and the primitive operations the Linear and Norm components are made of. Losses: cross-entropy, BCE, MSE and L1. Optimizers: Adam, AdamW, SGD and RMSprop. Datasets: the browser group above.
- **Limits.** No convolutions, pooling, BatchNorm, embeddings, attention or multi-phase training in the browser: the preflight says so and links to train.py, so modules 4, 6 and 7 train in Colab. At most 2,000,000 parameters and about 4,000,000 activation values per batch. Training runs on the main thread in short slices, so it suits small models: the digits MLP (9,610 parameters, 15 epochs) trains in about 2 s.
- **Determinism and honesty.** Runs are seeded: the same settings give the same curves. The trainer computes in float64 with its own seeded initialisation and batch order, so PyTorch's curves are similar, not identical, on the same data and split. Lesson thresholds and the numbers quoted in the copy were measured with this trainer or a matching simulation (SPEC §11).

## Export: train.py, the notebook and Colab

**Code** (top right) shows `model.py` and `train.py`. `model.py` writes each Linear block out as its weight, transpose, matmul and bias add, and opens it with a comment naming the everyday PyTorch layer it equals, such as `# Linear 1 = nn.Linear(2, 8)`. With a dataset bound, `train.py` is one self-contained script: the model, a loader for the bound dataset (the seeded generator and split for toy data, scikit-learn for Iris and digits, torchvision for the image presets, a download for Tiny Shakespeare), the training canvas's optimizer and loss, a validation pass after every epoch under `model.eval()` and `torch.no_grad()`, accuracy, the first loss next to ln C, the best epoch's weights, one test evaluation where a test split exists, and generated text for Tiny Shakespeare. A "Check before running" header lists anything in the model that does not fit the data. The header also says what differs from the browser: PyTorch draws its own initial weights and batch order, so the curves come out close, not identical; `train_loss` is the running mean of each epoch's batch losses in training mode (Dropout on), while the browser, like `val_loss`, measures after the epoch in evaluation mode; and the export-only datasets train with their own Colab presets for epochs and batch size (MNIST 5 and 64). **Download .ipynb** wraps the script in a two-cell notebook: in Google Colab choose File → Upload notebook and run both cells, or upload `train.py` and run `!python train.py`. A GPU is optional (faster for CIFAR-10 and Tiny Shakespeare). Without a bound dataset, `train.py` is DeepFlow's training skeleton.

## Tests

```sh
npm test        # every suite in scripts/*.test.mjs (Node's test runner; UI suites use jsdom)
npm run lint
npm run build
```

The Edu suites (`scripts/edu-*.test.mjs`) cover the lesson engine (checks, actions, targets, feedback, progress, workspaces), the shell and the lesson UI, the datasets (generators, split, Iris and digits data, presets, Use this dataset), the trainer (autograd, optimizers, compile, store, charts, the lesson thresholds) and the export. `edu-content.test.mjs` validates the seven modules (schema, text budget, facts) and walks all 52 steps headlessly through the real session, engine and trainer: each step must fail on entry, give nothing away before its prediction is answered, and pass after its Show me. `edu-step-entry.test.mjs` opens every step directly, by deep link and by its progress segment, from a fresh profile, and passes it with Show me.

Tests that need Python use `python3`, or the interpreter in `EDU_PYTHON`. With NumPy, the exported generators and splits are compared with the browser's; with PyTorch, NumPy and scikit-learn, each browser dataset's `train.py` trains one epoch and the module 7 Transformer trains a few steps and samples text. Without them these tests are skipped.

```sh
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install numpy scikit-learn
EDU_PYTHON=.venv/bin/python npm test
```

Three Playwright checks run against a dev server on port 5174 (`npx vite --host 127.0.0.1 --port 5174 --strictPort`) and are not part of `npm test`: `node scripts/edu-learn-browser-check.mjs` (Learn home and lesson panel), `node scripts/edu-data-browser-check.mjs` (Data tab, trainer and its timing budget) and `node scripts/edu-walkthrough-browser-check.mjs` (every module with Hint, Hint, Show me at 1440×900 and 390×844, about ten minutes per size).

## Edu source layout

```
src/edu/
  config.js             product name, storage prefix, locale
  bridge.js             read-only sources and workspace slices that Edu parts share
  useBridgeSource.js    React hook over a bridge source
  objective.js          the training graph's effective optimizer and loss (trainer, checks and export)
  shell/                tab and code-panel store, deep links, Save/Load of the Edu slices, layout CSS
  components/           Learn home, module cards, lesson panel, step, prediction, quiz and recap views, hints,
                        palette chips, spotlight
  lessons/engine/       schema, checks, actions (Show me), spotlight targets, friendly validation feedback,
                        session, progress, workspaces and storage, event log, outcome observers (observe.js),
                        option order (order.js), palette splices (palette.js), UI text (text/en/ui.js)
  lessons/modules/      the seven modules (m1-first-neuron.js … m7-transformer.js), fact helpers, lesson text
                        (text/en/m1.js … m7.js, common.js with the misconception catalogue)
  data/                 dataset registry, seeded generators, split and scaling, presets (toy, Iris, digits, MNIST,
                        Fashion-MNIST, CIFAR-10, Tiny Shakespeare), Python loaders, the Data tab
  trainer/              compile, tensors and autograd, losses, optimizers, runs and metrics, preflight, store,
                        charts, TrainerPanel and TrainerCard (components/)
src/codegen/edu/        train.py for a bound dataset, and the .ipynb wrapper
scripts/edu-*.test.mjs  tests; scripts/fixtures/edu-test-module.js is a small demo module for the UI tests
docs/edu/SPEC.md        product, curriculum and technical specification (§12: change log)
```

Edu also touches a few DeepFlow files: the navbar name, the tab bar and layout (`ToolbarStrip.jsx`, `Layout.jsx`), the Examples menu (it opens examples in the sandbox), the code panel (controlled tabs, the dataset's train.py, Download .ipynb, Esc to close), `codegen/training.js` (the bound-dataset script and two loss fixes), the canvas bridge in `ArchitectureWorkspace.jsx`, the training canvas and palette (fit beside the palette, fold it on narrow screens), a LogSoftmax block, and the parameter count of scalar Conv kernels. The default canvas shows no Edu text; everything Edu says lives in panels and tabs.

---

# DeepFlow v2

A local visual editor for neural network architectures. Build graphs from PyTorch layers, group connected blocks into reusable templates, expose parameters, and repeat templates with per-iteration expressions.

## Run locally

Use Node.js 20.19+, 22.13+, or 24+.

```sh
npm install
npm run dev
```

## Working in the editor

- **Ribbon:** **Components** contains editable standard components and other layer modules; **Operations** contains math, routing, sources, and outputs; **Templates** contains project definitions. Commands act on the active graph pane; dragging into another pane targets that destination.
- **Search:** each insertion menu has search. Press `/` or `Cmd/Ctrl+K` to open Operations search. Enter adds the first result, Arrow Down moves into results, and Escape closes the menu.
- **Nodes:** click a node to open its compact settings; dragging closes them. The small arrow opens a component's structure, including supported native layers that can be decomposed. Numeric fields accept literals and expressions such as `$width`. Typing `$` offers the available parameters with their values and owners; use arrows and Enter to choose, or Escape to dismiss. Operation inspectors use a stable **Operation** heading and a **Type** selector that updates the node's symbol.
- **View:** **Dimensions** shows or hides tensor sizes at the ports; named inputs such as Q/K/V stay identifiable. **Gradients** separately shows structural gradient paths. These controls apply to the active pane.
- **Undo/Redo:** use **History**, after View in the ribbon, or `Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z`. Architecture edits and shared template edits use one history across both panes; a drag, paste or multi-node deletion is one step. Text fields retain normal text undo. Loading a project, choosing an example, or resetting the workspace starts a fresh history; training edits are outside this history.
- **Templates:** select at least two connected nodes and choose Group. Reuse the definition from Templates. Open a template using its button or a double-click; edit its exposed parameters from **Component → Parameters**. Editing its definition affects every use of that template.
- **Repeat:** use **Structure → Sequential / Parallel** for selected blocks or a template instance. Click the active mode again to return to a single instance; clicking the other mode switches directly. Set the number of steps or instances in the inspector with a value or expression such as `$H`; turning repeat off and on retains that setting. The **Structure → Block 1/12** (or Head/Step) menu selects an instance, applies a selection rule, and chooses parameter and structure edit scope. Displayed instance numbers start at 1; the formula index `i` starts at 0 and is configured under **Parameter formulas**. Open levels appear as cards above each canvas; switching cards keeps the other levels open. Right-click a card or use Alt+Enter to open it to the side. The Structure instance control remains visible and disabled outside a repeated component.
- **Canvas:** Shift/Command-click for multiple selection, drag the background to select, hold Space to pan, and use Delete/Backspace to remove selected nodes. Select a connection and drag an endpoint to reconnect or release it on empty canvas to detach it.
- **Copy/Paste:** `Cmd/Ctrl+C` copies the selected nodes and their internal connections; `Cmd/Ctrl+V` pastes offset copies into the focused graph pane. Component copies reference the same template definition. Native text copy/paste continues to work inside fields.
- **Save/Load:** graphs persist in this browser. Save downloads the architecture, templates and training graph together as JSON. Load opens a V2 JSON workspace or a local `.onnx` file. An ONNX file opens a preview of the source operators and their editable component mapping before replacing the graph.
- **Import:** **Insert → Hugging Face** searches model repositories and previews their architecture. Supported configurations build editable templates; other models use a published ONNX graph from the same pinned revision as an intermediate representation. ONNX-only repositories work without `config.json`. If the repository contains several graphs, select the desired export or model part explicitly. Every operator must have a supported editable mapping before import; otherwise the preview explains the missing mappings and preserves the current graph. Source operations and provenance remain attached after import. Pretrained weights are not loaded.
- **Code:** inspect or download generated `model.py` and `train.py`.

## Standard components and copies

Insert Linear, Embedding, Attention, or Norm from Components, then **Edit definition** to inspect their ordinary operations: learned parameters, matrix products, table lookup, or normalization statistics. Open nested components to continue down to the math.

Inside a repeated component, the **Shared structure** switch is in the Structure instance menu beside the instance selector. Shared edits apply to instances still using that definition. Turning the switch off isolates the selected occurrence and its nested structure; the number in the menu identifies the selected head or step. Turning it on reuses the shared definition and keeps the local version saved; turning it off again restores it. Scope changes and graph edits support Undo/Redo. Copy/paste preserves shared definitions and makes independent copies of any local versions.

**Learned positions** generates position indices and looks them up in an Embedding component. New GPT-2 examples use this version, with the sum of token and position vectors kept explicit. **Sinusoidal positions** is a separate alternative built from position indices, frequency arithmetic, sine/cosine, stacking, and slicing; it has no learned table.

Saved opaque layers stay unchanged until you open their structure with the node arrow. Supported conversions preserve node identity, placement, bindings, and connections. These include Linear (with or without bias), ordinary Embedding (including padding rows), learned positions, causal self-attention, and LayerNorm over one final axis. Sparse/max-norm embeddings, multi-axis LayerNorm, and noncausal MultiheadAttention are not converted automatically or silently approximated.

**Gradients** shows structural differentiability, not measured gradient values. It distinguishes trainable or frozen Parameters, stopped/reversed connections, and integer index paths with no derivative. Gather can train its table while its indices remain nondifferentiable; Parameter `requires_grad` controls whether a learned value is trainable.

## Editable multi-head attention

Choose **Examples → Multi-head attention** to open its internal graph. **GPT-2 small** also uses this template inside its sequential Transformer blocks.

- **Component → Parameters** exposes `H = 12` (head count), `dropout = 0.1`, and `bias`. Attention width comes from its input tensor; each head's width comes from its actual Q input, before the scores MatMul. These dimensions are read-only and remain unresolved until the inputs are known. Change `H` to another divisor of the input width; Split and the inferred parallel count update together.
- In GPT-2, attention inherits `H` from the Transformer block's `n_heads`. The `d_model` choice sizes the surrounding projections; Attention reads the resulting tensor width. Existing saved definitions retain their original behavior; loading an example uses the new standard definition.
- Follow Q/K/V projections → Split → parallel attention heads → Concat → output projection. Open a head to edit Transpose, MatMul, Scale, Causal mask, Softmax, Dropout, and the final MatMul. These are ordinary editable nodes.
- Named Q/K/V ports route each collection to its matching template input; MatMul uses ordered `a` and `b` ports. Wire to the intended port rather than relying on edge creation order.
- Parallel previews show at most three representative copies and an ellipsis, regardless of count. Open the definition and use the **Structure** instance menu to inspect any head. Displayed instance numbers start at **1**; formula indices remain zero-based. The switch beside the instance counter chooses shared or per-instance structure editing. Local versions also participate in shape validation, gradient visualization and Python export; repeated learned layers have independent weights.

Edit parameter references directly in the field, using `$` completion. The **eye beside a field label** shows or hides that field in the enclosing component's settings; its tooltip names that component. First exposure creates a parameter reference, and subsequent visibility changes preserve the reference and value. Direct references have no repeated Linked/value/provenance row. Arithmetic expressions show a compact result, with full precision in the tooltip; formatting never changes stored values. Exposure and visibility changes support Undo/Redo. **Component → Parameters** lists the fields each parameter controls. Derived dimensions stay read-only. Shape cells have fixed widths: two axes fit on one row, and additional axes scroll horizontally. Symbolic dimensions and whole-tuple references remain supported.

Parallel **Auto** count is inferred from equally sized incoming tensor collections. For a broadcast tensor without a collection, set the count manually. Changing an Auto parallel component to Sequential materializes its current count; reconnect the tensor paths to reflect the intended sequential computation.

**Sequential** repeats feed each output into the next iteration. **Parallel** repeats receive the same tensor, or item `i` from each input collection, and return a collection. Split creates equal tensor slices; Broadcast creates repeated references to one tensor. Use Concat, Stack, Add, or Multiply to combine collections explicitly.

Opening the app preserves saved projects. Existing opaque attention blocks are not automatically converted; load a new example to explore the explicit graph.

## Checks

```sh
npm test
npm run lint
npm run build
```

The Node regression suites cover template/repeat navigation, deletion, shared validation caching, and real React interactions for numeric parameters, node expansion, library search, template insertion and keyboard access. The interaction tests use jsdom; they do not verify canvas layout, pointer dragging or visual appearance. Older Playwright scripts and historical screenshots remain in `scripts/` and `screenshots/`.

Focused component and attention checks:

```sh
node --test scripts/components-model.test.mjs scripts/components-runtime.test.mjs scripts/attention-model.test.mjs scripts/parallel-runtime.test.mjs scripts/nested-attention.test.mjs
```

If `python3` can import PyTorch, these also compare exported CPU outputs and gradients with independent references, including sequential Transformers containing parallel heads. Python-dependent tests skip automatically when PyTorch is unavailable.

## Current scope

This is a research prototype. Shape validation and PyTorch generation cover a subset of available modules; passing graph checks does not guarantee a runnable architecture. In plain DeepFlow the Data workspace was pending and generated training code required user-supplied data loaders; DeepFlow Edu adds both (above). The ResNet, DANN and GAN examples demonstrate editor concepts and are not complete reference implementations.

Repeat counts must resolve to integers from 1–256. Validation/export limit template nesting to 8 levels and expansion to 20,000 nodes. Export rejects validation errors, including unresolved counts, indivisible splits, collection-count mismatches, and invalid named input ports. The GPT-2 example includes explicit causal attention; tied embedding/output weights and a verified end-to-end GPT-2 training pipeline remain outside the current scope.

The shared ONNX converter accepts graph files up to 64 MiB and supports a subset of float32 operators, including Linear patterns, Conv2d with explicit symmetric padding, activations, pooling, Flatten, residual arithmetic, MatMul, Concat and simple transposes. It preserves branches, ordered input ports and multiple graph outputs. Unsupported operators, attributes, dtypes and control-flow graphs prevent editable import. Exposed initializer tensors become frozen Parameters with their source shapes; actual tensor values are not loaded. The original operator graph is a snapshot, not a live reverse export of subsequent edits.

Hugging Face import can consume an ONNX graph already published in the chosen repository. It does not yet run a server-side exporter for repositories that contain only framework weights, or assemble separate encoder/decoder files automatically. ONNX broadens the common conversion path but does not by itself make every model editable.

## Source map

- `src/domain/`: graph, template, training and layer-library definitions.
- `src/stores/`: state and browser persistence.
- `src/components/architecture/`: library, canvas, node cards and template editing.
- `src/components/training/`: training canvas and controls.
- `src/validation/`: topology/shape checks and shared report cache.
- `src/codegen/`: generated Python.
- `src/storyboards/`: example workspaces.
