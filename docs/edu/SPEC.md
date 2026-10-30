# DeepFlow Edu: product and technical specification (v1)

Status: build spec for stages A–E. Branch `deepflow-edu`, forked from DeepFlow `main` @ `15c7c25`.
Owner of this document: the product/tech lead (stage 0). Only Stage E may change it after stage 0, and every change
must be recorded in §12 (Change log).

The product name lives in one constant, `PRODUCT_NAME` in `src/edu/config.js` (§6.2). This document calls the product
"DeepFlow Edu".

Contents

0. Summary
1. Who it is for and what they learn
2. Teaching principles and the features that implement them
3. Product design (Learn home, lesson panel, Data tab, Training tab, code export, workspaces, visual language, accessibility)
4. Curriculum: seven modules (steps, checks, hints, show-me, quizzes, recaps)
5. Datasets
6. Architecture and interfaces (files, contract files, lesson schema, check API, actions, targets, progress, workspaces,
   dataset registry, trainer API, codegen hook, shell integration)
7. Build stages and file ownership
8. Tests
9. Evaluation plan
10. Risks, limits and open questions
11. Appendix: verified numbers and how they were obtained
12. Change log

---

## 0. Summary

DeepFlow Edu turns the DeepFlow editor into a guided course. Learners work in the real editor: architecture canvas,
components, sequential and parallel repeats, shape inference, the Gradients view, the training graph and PyTorch code
export. On top of the editor, Edu adds four things:

1. **Learn**, a new first tab. It shows the seven-module curriculum with progress, a Continue button and the way into
   the free editor (the "sandbox").
2. **A lesson panel** docked beside the editor. It runs one step at a time. Each step is checked automatically
   against the live app state (graph, validation report, training graph, dataset, trainer runs, UI state). Help comes
   as a three-level ladder: nudge, then hint, then "Show me", which performs the step as one undoable edit.
3. **The Data tab**, which DeepFlow shows as "DATA soon". It holds a dataset registry with seeded 2D toy generators,
   Iris and 8×8 digits embedded, and MNIST, Fashion-MNIST, CIFAR-10 and Tiny Shakespeare as presets. Presets carry
   shapes, classes, licence and description but are never downloaded in the browser. "Use this dataset" sets the
   model's input and output shape.
4. **An in-browser trainer** for toy data, Iris and digits. It compiles the graph into a small deterministic JS
   model, uses the training graph's optimizer and loss, and shows live loss and accuracy curves, a decision-boundary
   heatmap (2D), a confusion matrix and a train/validation split. A matching **code export** writes a `train.py`
   with real data loaders, a validation loop and an accuracy metric, so the exported file runs in Colab as is.

**Contribution (for a UIST/L@S paper).** Earlier educational NN tools let learners play with a toy library
(TensorFlow Playground), hide the architecture (Teachable Machine), explain a fixed pretrained model (CNN Explainer,
Transformer Explainer), or offer building with no pedagogy (ENNUI, Fabrik). Kaggle Learn offers check/hint/solution
feedback, but only in notebooks. DeepFlow Edu is a guided-practice layer over a full visual NN editor:

- Tutorial steps are verified against the editor's own model, validation and training state, not against
  screenshots or a scripted click path.
- Help is graded and tied to named misconceptions, and is shown in the panel, not as blocking dialogs.
- The in-browser trainer uses the same seeded datasets that the exported PyTorch script regenerates, so a learner can
  carry a browser experiment into real code and compare the curves.

| Tool | Builds real architectures | Steps checked against live state | Trains in the browser | Exports runnable training code | Feedback tied to misconceptions |
|---|---|---|---|---|---|
| TensorFlow Playground | toy MLPs only | no (free play, optional tutorial text) | yes, own small library | no | no |
| Teachable Machine | no (architecture hidden) | no | yes (transfer learning) | the trained model, not training code | no |
| CNN Explainer / Transformer Explainer | no (fixed pretrained model) | no; Transformer Explainer has guided cards | no | no | Transformer Explainer: experiments on misconceptions |
| ENNUI | yes | no | yes (TF.js) | yes (Python) | no |
| Fabrik | yes | no | no | model definitions | no |
| Kaggle Learn (learntools) | in code notebooks | yes (check / hint / solution) | n/a (notebook) | it is code | yes |
| DeepFlow (IUI'25) | yes (visual, nested, repeated) | no | no | model + training skeleton without data loaders | no |
| **DeepFlow Edu** | yes (the same editor) | **yes** (graph, validation, training graph, data, runs) | yes (MLP scale, seeded) | **yes** (loaders, validation loop, Colab) | **yes** (diagnose rules, rewritten errors) |

This answers three problems reported in the DeepFlow IUI'25 study: onboarding (mixed SUS learning-curve items),
"disruptive error messages", and the request for "the whole code, including the training procedure". It also
addresses the authors' warning that novices may over-rely on the visual interface: every module ends by reading the
exported code.

**Scope of v1.**
- 7 modules: 50 tutorial steps, 28 quiz items, 14 warm-up items.
- 11 datasets: 7 trainable in the browser (5 toy generators, Iris, digits 8×8) and 4 export-only presets, plus an
  optional toy regression set.
- A trainer for MLP-style graphs and one combined `train.py` export.
- Local progress and a local event log.
- English UI, with all text in keyed files.

**Non-goals for v1.** No accounts or server. No in-browser CNN or Transformer training. No pretrained weights
(DeepFlow does not load them). No graph-to-code line highlighting (planned, §10). No instructor dashboard. The
default canvas carries no new text (all Edu UI lives in panels and tabs, §3.8).

---

## 1. Who it is for and what they learn

**Primary learners.** Students and self-learners who can write a little Python, or none, and have never trained a
neural network. **Secondary learners.** People who know some theory and want to connect it to working PyTorch.
Advanced learners can skip steps or whole modules; nothing is locked.

**Course-level objectives** (each module refines these into 3–4 objectives with quiz items aligned to them, §4):

- C1 Describe a neural network as a parameterised function trained by minimising a loss with an optimizer, and run
  the whole workflow: data, model, loss and optimizer, training, evaluation, export.
- C2 Prepare data correctly: separate features from labels, make train/validation(/test) splits, fit
  normalisation statistics on the training split only.
- C3 Reason about tensor shapes (batch axis, `(B, features)`, `(B, C, H, W)`, `(B, T)` and `(B, T, V)`) and count
  parameters by hand.
- C4 Choose layers for a task (Linear, non-linearities, Flatten, Conv2d, pooling, residual blocks, embeddings,
  attention) and explain why they work.
- C5 Diagnose training from curves (initial loss, learning rate, overfitting) and apply fixes (early stopping,
  regularisation).
- C6 Read and run the generated PyTorch code.

Expected total time: about 2.5 hours (7 modules × 15–25 minutes).

---

## 2. Teaching principles and the features that implement them

The research brief (P1–P20) maps onto concrete features. Every row below is implemented in v1 unless marked *(v2)*.

| # | Principle (source) | DeepFlow Edu feature |
|---|---|---|
| P1 | Whole game first (fastbook, d2l) | Module 1 trains a complete model in the browser within minutes, then exports runnable PyTorch. Later modules open the parts. |
| P2 | Worked → faded → independent (Sweller; Carpentries ep.05) | Each module starts from a worked fixture, then removes parts (holes, a planted bug), and ends with a "Your turn" build on a new dataset (M1 Iris, M2 XOR, M3 digits, …). |
| P3 | Parsons-style construction (Parsons & Haden; Ericson et al.) | A step can list a **palette**: chips for the blocks it needs plus 0–1 distractor. Each distractor is tied to a named misconception (e.g. Sigmoid before BCE). |
| P4 | Labelled subgoals (Margulieux et al.) | Every step has one of 7 fixed subgoal labels (§6.3): Prepare the data · Build the model · Check shapes & size · Choose loss & optimizer · Train & observe · Evaluate · Export & read the code. |
| P5 | Read → modify → write (Xie et al.) | Step order inside each module. |
| P6 | Predict-observe-explain (Mazur; Kim et al.) | **Prediction gates.** Number, choice or shape answers are committed before the tool reveals the value. Gates lock Train (and the module turns Dimensions off) until the learner answers. The reveal shows "you said / the tool says" with an explanation. |
| P7 | Immediate, specific, non-disruptive feedback (learntools; DeepFlow IUI'25) | Checks re-run on every store change. Validation errors are rewritten into friendly messages inside the panel. Help ladder: nudge → hint → Show me. No modal error dialogs. |
| P8 | Productive failure, safely (Kapur) | M2 step 1 ("one neuron on a circle") and M2 step 2 ("two stacked Linears"). Never graded, always resettable. |
| P9 | Diagnose misconceptions (Carpentries ep.02) | `diagnose` rules on predictions and quiz distractors each map to a misconception id with a targeted explanation (§4.9 catalogue). |
| P10 | Retrieval and spacing (Roediger & Karpicke; ML-For-Beginners) | Each module 2–7 opens with a 2-item warm-up on earlier modules. Each module ends with a 3–4-item quiz. M1 opens with two myth-busting items. |
| P11 | Engagement over viewing (Naps et al.; CNN Explainer) | Every chart, boundary or shape reveal is paired with a prediction or question. |
| P12 | Structure ↔ math (CNN Explainer; Transformer Explainer) | Steps open the Linear, Embedding and Attention components down to MatMul, Gather and softmax. Parameter counts are done by hand, then checked by the tool. |
| P13 | Segment, signal, details on demand (Mayer; Playground) | One concept per step. Body text of 60 words or fewer. A spotlight on the UI element. A collapsed "Why?" section. |
| P14 | Toy → real data (CNN Explainer; Transformer Explainer) | 2D → Iris → digits/MNIST → Fashion-MNIST → CIFAR-10 → Shakespeare text. |
| P15 | Teach the training recipe and debugging (Karpathy; CS231n) | Initial-loss check, learning-rate sweep, overfit-on-purpose, early-stopping marker, planted shape bug. |
| P16 | Bridge to real code (DeepFlow IUI'25) | Every module ends with "Export & read the code". `train.py` runs in Colab with real loaders. |
| P17 | Measure learning, not satisfaction (Transformer Explainer; TTT) | Quiz items aligned to objectives; a local event log; the study plan in §9. |
| P18 | Low stakes, inclusive framing (fastbook; TTT) | Unlimited retries, Reset step, Skip, no scores on the Learn home. Myth-busting warm-up. |
| P19 | Reproducible lessons (Playground; Karpathy) | Every dataset and run shows its seed. The browser and the export produce the same dataset and split (§5.3, tested). |
| P20 | Translatable content (ML-For-Beginners) | All lesson, quiz and UI-chrome text sits in keyed text files; `PRODUCT_NAME` is a constant. |

---

## 3. Product design

### 3.1 Information architecture

Top-level tabs in `ToolbarStrip` (left to right): **LEARN · ARCHITECTURE · DATA · TRAINING**. The right-hand
action group (Code | Save | Load | Reset) is unchanged. The navbar shows `PRODUCT_NAME` next to the Brain icon.
`Examples ▾` stays in the navbar.

Two modes, held in the session store (§6.7):

- **sandbox**: the plain DeepFlow editor plus Data and Training extras. No lesson panel.
- **lesson**: a module is active. The lesson panel is docked on the right of every tab except Learn, and the
  workspace is that module's own slot (§3.7).

On first start the app opens on **Learn**. DeepFlow's GPT-2 default seeding (`createDefaultModelSeeder`) is kept and
becomes the initial sandbox.

### 3.2 Learn home (`src/edu/components/LearnHome.jsx`)

```
┌ LEARN  ARCHITECTURE  DATA  TRAINING ─────────────────────────────── Code│Save│Load│Reset ┐
│                                                                                          │
│  Learn to build neural networks                                   (h1, 28px/800)         │
│  Seven short modules. Build real models, train small ones here, export PyTorch.          │
│  [▶ Continue · 2 Going non-linear · step 3 of 8]   [Open the sandbox]                    │
│                                                                                          │
│  ┌1 Your first neuron ─┐ ┌2 Going non-linear ─┐ ┌3 Shapes and parameters┐ ┌4 Convolutions┐│
│  │ ● Done · 20 min     │ │ ◐ 3/8 · 20 min     │ │ ○ 25 min              │ │ ○ 20 min     ││
│  │ Blobs · Iris        │ │ Circle · XOR       │ │ MNIST · Digits 8×8    │ │ Fashion-MNIST││
│  │ [Review]            │ │ [Continue]         │ │ [Start]               │ │ [Start]      ││
│  └─────────────────────┘ └────────────────────┘ └───────────────────────┘ └──────────────┘│
│  ┌5 Training well ─────┐ ┌6 Reusable blocks ──┐ ┌7 Attention ───────────┐                │
│  …                                                                                        │
│  Progress is saved in this browser.   Reset progress · Download learning log (JSON)       │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

- Cards are in a responsive grid: 4 columns at ≥ 1200 px, 2 at ≥ 720 px, 1 below. Each card is a `<section>` with
  an `<h2>` holding the number and title, a status line, dataset chips, the duration and one button. The card
  "goal" line holds at most 8 words.
- Status icons: ○ not started, ◐ in progress k/n, ● completed. The icon always comes with a text label for screen
  readers. Status uses monochrome plus the DeepFlow accent `#e91e63` for the primary button focus only.
- **Continue** resumes `progress.lastModuleId` at its current step. With no progress it says "Start · 1 Your first
  neuron".
- **Open the sandbox** switches to sandbox mode and the Architecture tab.
- **Reset progress** asks for confirmation (`window.confirm`, as DeepFlow's Reset). It clears progress, lesson slots
  and the event log. The sandbox slot is kept.
- **Download learning log** saves `deepflow-edu-log.json` (§6.8). Nothing is ever sent over the network.
- Modules are never locked. A card for a module whose recommended predecessor is not completed shows "Recommended
  after 2" in secondary text.
- Deep link: `#/learn/<moduleId>/<stepId>` opens that step (§6.12). The hash is updated with `history.replaceState`
  as the learner moves between steps.

### 3.3 Lesson panel (`src/edu/components/LessonPanel.jsx`)

Docked to the right of the tab content: 360 px at ≥ 1200 px window width, 320 px between 900 and 1199 px. Below 900
px it becomes a bottom drawer (45 vh, drag handle, "Lesson" toggle button). It never covers the canvas. The canvas
area shrinks instead, so React Flow pan, zoom, drag and select are untouched.

```
┌───────────────────────────────┐
│ 2 · Going non-linear    ‒  ✕  │  header: module, collapse, exit lesson
│ Step 3 of 8  ▰▰▰▱▱▱▱▱         │  segmented progress; segments carry subgoal titles
│ BUILD THE MODEL               │  subgoal label (fixed vocabulary, §6.3)
│ Add a non-linearity           │  h2, receives focus on step change
│ Insert a ReLU between the two │  body ≤ 60 words, tiny markup (§6.3.4)
│ Linear blocks, then train.    │
│ ▸ Why?                        │  <details>, optional
│ ┌ Blocks you need ──────────┐ │  palette chips (optional)
│ │ [ReLU ×1]                 │ │
│ └───────────────────────────┘ │
│ [embed: TrainerCard]          │  optional embeds (§6.3.5)
│ ○ Not yet: Linear 2 receives  │  live status (aria-live=polite)
│   1 feature but expects 8.    │
│ [Hint]                        │  hint ladder: Hint → Another hint → Show me
│ ‹ Back   ↺ Reset step   Next ›│  Next is enabled when the check passes; "Skip" link beside it
└───────────────────────────────┘
```

Behaviour:

- **Entering a step** runs `step.setup` once, then captures the step-entry snapshot (§6.8). It switches to
  `step.tab` if one is given and moves focus to the step heading. The spotlight pulses twice on `step.target`, or not
  at all under `prefers-reduced-motion`.
- **Checks** re-evaluate on any change to the architecture, templates, training, data, trainer, UI or session state,
  debounced by 150 ms. The status line shows the first failing sub-check's `detail` (§6.4). When the root
  validation report has errors, it shows the friendly rewrite of the most relevant error instead (§6.4.4). The
  canvas's own Graph checks badge is unchanged.
- **On pass** the status turns to "✓ Done", with an optional `done` sentence such as "A single neuron can only draw a
  straight line." Next becomes the primary button, and the live region announces "Step complete". There is no
  automatic advance: the learner reads the reveal, then presses Next.
- **Hint ladder.** The first press shows `hint1` (a nudge: where to look), the second `hint2` (specific values), the
  third runs **Show me**. Show me performs `step.solution` as one undoable history entry labelled "Lesson: show
  me", and a toast offers Undo. The hint level is logged.
- **Back** revisits passed steps read-only: the check status shows, setup is not re-run, and the workspace is not
  changed. **Reset step** restores the step-entry snapshot. **Skip** marks the step `skipped` and moves on.
- **Prediction gates** (`kind: 'predict'` or a task with `predict`). The input is a number field, radio choices or a
  shape field ("B, 8, 28, 28"). Submit locks the answer, and it cannot be changed (it can be retried after Reset
  step). The reveal shows "You said X · The tool shows Y" with the explanation. A matching `diagnose` rule shows its
  misconception text. Predictions are never marked failed: wording is "Close!", "Exactly", or "Not quite — here is
  why". With `locks.gate = 'train'`, the TrainerCard's Train button stays disabled until the prediction is committed.
- **Quiz** follows the last step: one item per screen. Choosing an option reveals that option's explanation at once.
  Retry is unlimited and the first answer is logged. Warm-up items use the same component before the first step and
  can be skipped.
- **Recap**: 3 takeaways, "Next module ›" and "Back to Learn".
- **Collapse (‒)** minimises the panel to a 44 px rail showing the step number and status. **Exit (✕)** saves the
  lesson workspace, switches to sandbox mode and opens Learn.

### 3.4 Data tab (`src/edu/data/DataWorkspace.jsx`, Stage B)

```
┌ Datasets ───────────────┬ Iris — 150 flowers · 4 features · 3 classes ─────────────────────────┐
│ TRAIN IN THE BROWSER    │ [Use this dataset]   Input (4,) · Output 3 classes · trains here       │
│  ● Two blobs            │ ┌ preview ─────────────────────────────────────┐ ┌ class balance ┐  │
│  ○ Circle               │ │ scatter: petal length × petal width (choose) │ │ setosa    50  │  │
│  ○ XOR                  │ │ or first 8 rows as a table (toggle)          │ │ versicolor 50 │  │
│  ○ Spiral (2 classes)   │ └──────────────────────────────────────────────┘ │ virginica 50  │  │
│  ○ Spiral (3 classes)   │ Split  validation [20 %]  test [0 %]  seed [1]  → train 120 · val 30  │
│  ○ Iris                 │ Scaling  standardise features with the TRAINING mean and std          │
│  ○ Digits 8×8           │          mean 5.83 3.07 3.72 1.18 · std 0.84 0.44 1.79 0.78 (120 rows) │
│ EXPORT TO TRAIN (COLAB) │ Source  Fisher (1936); values as in scikit-learn load_iris (BSD-3).   │
│  ○ MNIST                │         Not the UCI copy, which has two wrong rows.                     │
│  ○ Fashion-MNIST        │                                                                        │
│  ○ CIFAR-10             │                                                                        │
│  ○ Tiny Shakespeare     │                                                                        │
└─────────────────────────┴────────────────────────────────────────────────────────────────────────┘
```

- The left list is grouped by `training: 'browser' | 'export'` (§6.9). Each row carries `data-edu-target="dataset-<id>"`.
- The detail pane shows name, one-line description, input shape and dtype, target (classes with names, or vocab size),
  sizes, licence, citation, and a badge: "Trains here" or "Export to train".
- **Preview.** For toy sets, a 2D scatter with class shapes and colours: class 0 circle `#0072B2`, class 1 triangle
  `#E69F00`, class 2 square `#009E73` (Okabe-Ito, colour-blind-safe, shapes too). Train points are filled and
  validation points hollow. Iris has a 2-feature scatter with axis selectors, and a table toggle. Digits shows a grid
  of 20 thumbnails. Export-only presets have no data preview, only metadata and one schematic sentence (e.g. "60,000
  training images, 28×28, grey").
- **Options.** Generator `n` (100–1000), `noise` (0–0.5) and `seed` for toy sets. Split `val` (0.1–0.5), `test` (0,
  0.1, 0.2) and the data seed. Scaling is shown for tabular data and is always "fit on training split". The panel
  shows the actual mean and std vectors.
- **Use this dataset** (`data-edu-target="use-dataset"`) calls `useDataset(id, { setInput: true, setOutput: true })`
  (§6.9.4) as one undoable edit. It sets the root Input node's `shape` and `dtype`. If the node feeding the root
  Output is a Linear (layer or component), reached only through shape-preserving blocks (activation, dropout), it
  sets that Linear's `out_features` to the target size. An inline result line lists what changed ("Input → (4,);
  last Linear Output → 3") or why the output was not changed. In lessons, steps pass `setOutput: false` when choosing
  the output size is part of the exercise, and the button follows the current step: when the step's own Show me or
  setup calls `useDataset` with `setOutput: false`, the button sets only the Input and its hint says so (M1 s4).
- The Data tab binds the dataset to the workspace (`useEduDataStore`, persisted per workspace slot). The trainer and
  `train.py` read it from there.

### 3.5 Training tab and the trainer UI (Stage B)

The Training tab is split vertically. The top 55 % keeps DeepFlow's training canvas and `TrainingPalette`
(optimizer, loss and phase nodes). The bottom 45 % is the **TrainerPanel**
(`src/edu/trainer/components/TrainerPanel.jsx`).

```
┌ Run ───────────────────────┬ Loss ───────────────────┬ Accuracy ───────────────┬ Decision boundary ───┐
│ Dataset  Two blobs (seed 1)│  1.0┤╲                  │ 1.0┤        ______       │ ░░░░▒▒▒▒▓▓▓▓         │
│ Model    3 params · ok     │     │ ╲___ train        │    │   ____/  val        │ ░○○░▒▒▓▓△△▓         │
│ Adam  lr [0.03]            │     │   ‾‾‾ val         │    │  /                  │ ░○░░▒▓▓△△▓▓         │
│ Epochs [30] Batch [32]     │  0.0┼──────────── 30    │ 0.5┼───────────── 30     │ (val points hollow)  │
│ Seed  [1]                  │ chance ln 2 = 0.69 ---  │ best val loss: ep 27 ▲   │  or confusion matrix │
│ [▶ Train] [■ Stop]         │ previous runs dimmed    │                         │  for > 2D inputs     │
│ [Check initial loss]       │                         │                         │                      │
└────────────────────────────┴─────────────────────────┴─────────────────────────┴──────────────────────┘
```

- **Run settings** (trainer store): epochs, batch size and training seed. Optimizer type and learning rate are shown
  as fields **bound to the training graph's optimizer node**, which stays the single source of truth
  (`patchHyperparameters`).
- **Train** (`data-edu-target="trainer-run"`) and **Stop**. **Check initial loss**
  (`data-edu-target="trainer-init-loss"`) evaluates the untrained model once and records an epoch-0 run.
- **Loss chart** (`data-edu-target="trainer-curves"`): train and validation loss per epoch, with epoch 0 = before any
  update. A dashed chance-level line marks `ln C` for C-class cross-entropy (`ln 2` for BCE). The best-validation-loss
  epoch has a marker. Up to 5 previous runs are drawn dimmed and labelled with what differed ("lr 0.0001").
- **Accuracy chart**: train and validation accuracy for classification tasks.
- **Decision boundary** (`data-edu-target="trainer-boundary"`) for 2D inputs. A 64×64 grid over the data bounds plus
  a 10 % margin, redrawn at most every 100 ms while training. Colour shows the predicted class, alpha shows
  confidence, and points are overlaid with class shapes.
- **Confusion matrix** (`data-edu-target="trainer-confusion"`) on the validation split for classification with 2–10
  classes. It is a real `<table>` with true classes in rows and predicted classes in columns, cell shade by row
  fraction, and counts as text.
- **Preflight** box: errors block Train, warnings do not (§6.10.6). Example: "Conv2d is not supported in the browser.
  Open Code → train.py to train this model in Colab."
- **TrainerCard** (`TrainerCard.jsx`) is the compact embed for the lesson panel. It shows Train/Stop, "epoch k/n", a
  mini loss sparkline (train/val), the final validation accuracy, "Open Training ↗", and a 120×120 boundary thumbnail
  for 2D data. Props: `{ locked, lockedReason }`.

All charts are dependency-free SVG with `<title>`/`<desc>` and a visually hidden data table, for example "Validation
accuracy 0.96 at epoch 30".

### 3.6 Code export (Stage C)

`Code` opens DeepFlow's CodePanel (model.py | train.py) as today. The panel is controlled: `{ open, onClose, tab,
onTabChange }` from the UI store, so lessons can open `train.py`. Its tab buttons carry `data-edu-target="code-tab-<id>"`.

When a dataset is bound, `train.py` becomes a single self-contained script: the model, the data loading for the bound
dataset, the optimizer and loss from the training graph, a training loop with a validation pass every epoch
(`model.eval()` and `torch.no_grad()`), an accuracy metric, one final test evaluation where a test split exists, and
`if __name__ == "__main__": train()`. Header comment: "Runs in Google Colab: upload this file and run `!python train.py`
(GPU optional; CIFAR-10 and Tiny Shakespeare are faster on GPU)." A **Download .ipynb** button wraps the same source
in a two-cell notebook (the script, then `train()`), ready for File → Upload notebook in Colab. With no dataset bound, `train.py` is DeepFlow's
current output, byte-identical.

### 3.7 Sandbox and lesson workspaces

Every mode owns a **workspace slot**: `sandbox`, `m1` … `m7`. A slot is a JSON snapshot of the architecture root
graph, all templates, the training graph and every registered workspace slice (data binding and trainer settings,
§6.8). Switching mode captures the current stores into the outgoing slot and restores the incoming one. Restoring
starts a fresh undo history, as loading an example does in DeepFlow. A module's first start loads its `start`
fixture. Loading an example from `Examples ▾` in lesson mode first asks "Open the example in your sandbox? Your lesson
is saved." and switches to sandbox. Exceptions are lesson actions that load an example on purpose (M7 step 7).

### 3.8 Visual language and text budget

- DeepFlow's look is reused unchanged: Inter, white panels, 1 px `#0000002e` borders, 2 px black borders with
  `4px 4px 0 #000` hard shadows for primary cards and buttons, uppercase 12 px/800 tab labels, and `.df-button` /
  `.df-button-primary`. Edu CSS lives in `src/edu/**/*.css`, with class prefix `edu-`.
- **No new text on the canvas.** The spotlight is an outline ring (3 px `#e91e63`, 4 px offset, radius 8 px). It
  never adds a label. All explanations live in the panel. This keeps within DeepFlow's text budget: the default
  canvas has exactly today's text.
- Lesson copy: body of 60 words or fewer, sentences of 20 words or fewer, second person, present tense. Terms in bold
  the first time. UI names exactly as shown in the app ("**Components** ▾ in the ribbon").

### 3.9 Accessibility

- Every Edu control is a native button, input, select or details element with a visible label or `aria-label`. Focus
  rings follow DeepFlow (`outline: 2px solid #e91e63`).
- The lesson panel is `<aside aria-label="Lesson">`. The status line is `role="status" aria-live="polite"`. Focus
  moves to the step heading on step change, and to the reveal heading after a prediction is submitted.
- Instructions name the control in words, so the spotlight is never the only cue. The spotlight element is
  `aria-hidden="true"` and `pointer-events: none`.
- Text contrast is at least 4.5:1, UI contrast at least 3:1. Class colours always have a shape as well. Charts have
  text alternatives.
- There are no new global keyboard shortcuts, so DeepFlow's `/`, ⌘K, ⌘Z, Space-pan, Delete and Alt+Enter work as
  before. Esc never exits a lesson.
- Reduced motion turns off spotlight pulses and chart transitions.

---

## 4. Curriculum

### 4.0 Module template and conventions

Every module follows the same shape:

1. **Warm-up** (modules 2–7): 2 retrieval items from earlier modules, skippable. Module 1 has 2 myth-busting items
   instead.
2. **Steps** (4–8). The order is read and predict (worked fixture), then modify (faded), then write (a "Your turn"
   build). Each step has exactly one subgoal label (§6.3.2).
3. **Quiz**: 3–4 items aligned to the module's learning objectives (LO ids). Every option has an explanation, and
   each distractor names the misconception it reveals (§4.9).
4. **Recap**: 3 takeaways and a link to the next module.

Conventions used in the step blocks below:

- **Do** is draft copy. Stage D may tighten it but must keep the facts, the UI names and the ≤ 60-word limit.
- "tab X" in a step header is the step's `tab`, switched to on entry. A step whose check contains `{ visited: X }`
  never sets `tab: X`: the learner opens that tab, and the content test enforces this.
- **Check**, **Show me** and **Setup** use the exact DSL of §6.4 and §6.5. Fixture node ids (`in`, `lin1`, …) are
  stable. Nodes the learner inserts are found by type through `NodeMatch` (§6.4.2).
- `{name}` inside copy is a **fact** interpolated from the module's facts ledger (§6.3.6). The content tests
  recompute every fact from the fixtures (§8.4), so numbers in the text cannot drift.
- Training thresholds come from the reference simulation in §11, which uses the same data generators, splits,
  initialisation and optimizer rules as the planned trainer. Each threshold sits below the worst case seen over the
  simulated seeds: 10, or 5–8 where §11 says so.
- Defaults in every fixture, unless the module says otherwise: Adam with DeepFlow's default β and ε, batch size 32,
  and a training seed equal to the module number. Datasets use the options listed per module.
- Layout: fixture nodes sit in one column at x = 400, with rows 130 px apart (DeepFlow's `ROW_GAP`). Side inputs go
  to x = 820.
- Shorthand used in the blocks:

  | Shorthand | Meaning |
  |---|---|
  | `answer(id, v)` | `{ do: 'answer', id, value: v }` |
  | `train` | `{ do: 'train' }` |
  | `setParam(lin1.out_features = 3)` | `{ do: 'setParam', node: { id: 'lin1' }, name: 'out_features', value: 3 }` |
  | `setUi(code train)` | `{ do: 'setUi', codeOpen: true, codeTab: 'train' }` |
  | `linear(a, b)` | `{ block: 'component:linear', params: { in_features: a, out_features: b } }` |
  | `relu` | `{ block: 'torch.nn.ReLU' }` |

  Choice options get short slug ids chosen by Stage D (`line`, `circle`, …), and ✓ marks the correct option.
  "Diagnose x → id" is a `PredictSpec.diagnose` entry.
- **Default Show me.** Where a block lists no Show me, the solution is built in check order:
  - `answer(<predict id>, <correct value>)`
  - for each condition in the check, the action that satisfies it: `ui` → `setUi`, `level` → `openLevel` with the
    same NodeMatch, `dataset` → `useDataset(id, { setOutput: false })`, `visited` → `setUi({ tab })`

  Stage D writes every solution out explicitly in the module files. The solvability test (§8.4) is the judge.
- M4 and M6 fixtures use the scalar `kernel_size: 3` that learners get from the palette. Their parameter facts are
  therefore correct only once Stage C's Conv counting fix lands (§6.11.4), and the content test for those facts
  depends on it.

Estimated minutes per module: M1 20, M2 20, M3 25, M4 20, M5 25, M6 20, M7 25.

### 4.1 Module 1 — Your first neuron

**Goal line:** "Train one neuron, then a 3-class classifier." **Datasets:** `blobs`, `iris`. **Requires:** none.

Learning objectives:
- `m1.lo1` Count the parameters of a Linear block (in × out + out).
- `m1.lo2` Choose the output size and loss for binary and multi-class classification.
- `m1.lo3` Explain why a validation set is held out, and why scaling statistics come from the training split only.
- `m1.lo4` Train a model in the browser and read loss, accuracy and a confusion matrix.

**Start fixture `m1.start`.** Root: `in` Input (label "features", shape `(2,)`, float32) → `neuron` Linear
component (label "neuron", in 2, out 1, bias on) → `out` Output (label "logit"). Training: `opt` Adam (lr 0.03) and
`loss` BCE bound to `out`, wired loss → optimizer. Data: `blobs` {n: 200, noise: 0.1, seed: 1}, split {val: 0.2,
test: 0}. Trainer: epochs 30, batch 32, seed 1.

**Facts:** `neuronParams = 3`, `irisParams = 15`.

**Warm-up (myths).**
- `m1.w1` "True or false: you need advanced maths or a PhD to start building neural networks." **False ✓**: "You
  need arithmetic and a little algebra to start. Each formula in this course is explained when you need it." True:
  "Many practitioners started without advanced maths; it can come later."
- `m1.w2` "True or false: training a neural network always needs a huge dataset and an expensive GPU." **False ✓**:
  "In this module you train on 200 points in about a second, in your browser. Big models need big hardware;
  learning the ideas does not." True: same explanation.

#### s1 · Meet the data — `read` · Prepare the data
- **Do:** "Each dot is one **example**: two numbers, x₁ and x₂ (its **features**), and a colour, its **label** (0 or
  1). We hide 20% of the dots as a **validation set**. The model never trains on them, so they show how well it
  works on new data."
- **Why?** "If we only tested on training examples, a model that memorised them would look perfect."
- **Target:** `tab.data`, then `data.preview`. **Check:** `{ visited: 'data' }`
- **Hints:** (1) "Open the **Data** tab in the top bar." (2) "Hollow dots are the validation set; filled dots train
  the model."
- **Show me:** `{ do: 'setUi', tab: 'data' }`

#### s2 · One neuron — `predict` · Build the model · tab Architecture
- **Do:** "This model is one **Linear** block: a single neuron. It computes z = w₁·x₁ + w₂·x₂ + b. The weights w₁, w₂
  and the bias b are its **parameters**. Training changes them."
- **Target:** `node` `{ id: 'neuron' }`.
- **Predict** `m1.params` (number): "How many learnable numbers does this Linear block have?" expected 3. Diagnose 2
  → `forgot-bias`. **Reveal:** "3: one weight per feature (2) plus one bias." Embed `paramCount`.
- **Check:** `{ answered: 'm1.params' }`
- **Hints:** (1) "Count one weight for each input feature." (2) "2 features → 2 weights, plus 1 bias."
- **Show me:** `{ do: 'answer', id: 'm1.params', value: 3 }`

#### s3 · Predict, then train — `predict` · Train & observe · tab Architecture
- **Do:** "**BCE loss** measures how wrong the predicted probability σ(z) is for each dot. **Adam** nudges w₁, w₂
  and b to lower it. Before you train: what will separate the two colours?"
- **Target:** `trainer.run`. **Locks:** `gate: 'train'`. **Embeds:** `trainer`.
- **Predict** `m1.boundary` (choice): "A straight line" ✓ · "A circle" · "A wiggly curve". **Reveal:** "The neuron
  predicts orange where w₁x₁ + w₂x₂ + b > 0. The boundary is the straight line where that sum is 0."
- **Check:** `{ all: [{ answered: 'm1.boundary' }, { run: { dataset: 'blobs', minEpochs: 20, final: { valAccuracy: { gte: 0.9 } } } }] }`
- **Done:** "Validation accuracy is the share of held-out dots on the correct side of the line."
- **Hints:** (1) "Answer the question, then press **Train**." (2) "Training 30 epochs takes about a second."
- **Show me:** `answer('m1.boundary', 'line')`, `{ do: 'train' }`

#### s4 · A real dataset: Iris — `predict` · Prepare the data · tab Data
- **Do:** "Iris has 150 flowers of 3 species. For each flower a botanist measured four lengths in cm: sepal length,
  sepal width, petal length and petal width. The species is the label. Select **Iris**, then **Use this dataset**."
- **Target:** `data.dataset:iris`, `data.use`.
- **Predict** `m1.features` (number): "How many input features does the model receive per flower?" expected 4.
  Diagnose 5 → `target-in-input`, 3 → `classes-as-features`. **Reveal:** "4: the measurements. The species is what
  the model predicts, so it is not an input."
- **Check:** `{ all: [{ answered: 'm1.features' }, { dataset: 'iris' }, { inputShape: [null, 4] }] }`
- **Hints:** (1) "Inputs are measurements, not the answer." (2) "Select Iris in the list, then press **Use this
  dataset**."
- **Show me:** `answer('m1.features', 4)`, `{ do: 'useDataset', id: 'iris', setOutput: false }`

#### s5 · One output per species — `task` · Build the model · tab Architecture
- **Do:** "A classifier gives one **score** per class, and the highest score wins. Set the Linear block to **Input
  4** (features) and **Output 3** (species)."
- **Target:** `node` `{ id: 'neuron' }`, then `field:out_features`.
- **Check:** `{ all: [{ param: { node: { type: 'Linear' }, name: 'in_features', equals: 4 } }, { param: { node: { type: 'Linear' }, name: 'out_features', equals: 3 } }, { valid: true }, { outputShape: [null, 3] }] }`
- **Misconceptions (diagnostics):** `out_features = 1` → `one-output-multiclass`; `in_features = 5` →
  `target-in-input`.
- **Hints:** (1) "Click the Linear block to open its settings." (2) "Input = 4, Output = 3."
- **Show me:** `{ do: 'setParam', node: { id: 'neuron' }, name: 'in_features', value: 4 }`, the same for
  `out_features: 3`

#### s6 · Cross-entropy for three classes — `task` · Choose loss & optimizer · tab Training
- **Do:** "BCE scores one yes/no output. With 3 classes, use **Cross-Entropy**. It turns the 3 scores into
  probabilities with softmax and penalises a low probability for the true species. Change the loss type in the
  Training tab."
- **Target:** `training.loss`. **Check:** `{ training: { loss: 'cross_entropy', lossBound: true } }`
- **Misconceptions:** a Softmax block directly before Output → `softmax-before-ce`.
- **Hints:** (1) "Open **Training** and click the loss block." (2) "Set **type** to cross_entropy."
- **Show me:** `{ do: 'setLoss', type: 'cross_entropy' }`

#### s7 · Train and read the confusion matrix — `predict` · Evaluate · tab Architecture
- **Setup:** `{ do: 'setTrainer', epochs: 50, batchSize: 16 }`. **Locks:** `gate: 'train'`. **Embeds:** `trainer`
  (with confusion matrix).
- **Do:** "Train the classifier. The **confusion matrix** counts validation flowers by true species (rows) and
  predicted species (columns). Correct answers sit on the diagonal."
- **Predict** `m1.easiest` (choice): "Which species will be easiest to recognise?" setosa ✓ · versicolor · virginica.
  **Reveal:** "Setosa: its petals are much smaller, so one straight boundary separates it from the others.
  Versicolor and virginica overlap a little."
- **Check:** `{ all: [{ answered: 'm1.easiest' }, { run: { dataset: 'iris', minEpochs: 30, final: { valAccuracy: { gte: 0.85 } } } }] }`
- **Hints:** (1) "Answer, then press **Train**." (2) "The matrix appears under the curves when training ends."
- **Show me:** `answer('m1.easiest', 'setosa')`, `{ do: 'train' }`

#### s8 · Export and read the code — `predict` · Export & read the code
- **Do:** "Open **Code → train.py**. It rebuilds your model in PyTorch, loads Iris with scikit-learn, makes the same
  split as here and trains with the same loss and optimizer."
- **Target:** `toolbar.code`, then `code.tab:train`.
- **Predict** `m1.leak` (choice): "train.py scales the features with a mean and a standard deviation. Which flowers
  are they computed from?" "Only the training split" ✓ · "All 150 flowers" · "Only the validation split".
  **Reveal:** "Only the training split. Using all flowers leaks information about the validation flowers into
  training, so the validation score looks better than it really is."
- **Check:** `{ all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm1.leak' }] }`
- **Show me:** `{ do: 'setUi', codeOpen: true, codeTab: 'train' }`, `answer('m1.leak', 'train')`

**Quiz.**
- `m1.q1` (lo1) "A Linear block with 4 inputs and 3 outputs has how many parameters?" 7 ("that adds inputs and
  outputs; each output has one weight per input") · 12 ("right number of weights, 4 × 3, but add the 3 biases") ·
  **15 ✓** ("4 × 3 weights + 3 biases").
- `m1.q2` (lo3) "What is the validation set for?" **"Measuring performance on examples the model never trained on"
  ✓** · "Giving the model more examples to learn from" ("validation examples never update the weights") · "Making
  training faster" ("it does not change training speed").
- `m1.q3` (lo2) "Your Iris model ends with Output 3. Which loss fits?" **Cross-entropy ✓** ("one score per class;
  softmax is inside the loss") · BCE ("BCE is for one yes/no output") · MSE ("MSE is for predicting numbers, such
  as a price").
- `m1.q4` (lo2) "Where is the decision boundary of a single neuron with two inputs?" **"On the line where w₁x₁ +
  w₂x₂ + b = 0" ✓** · "Around the training points" ("a neuron cannot bend around points") · "On a circle around the
  data" ("that needs hidden units; Module 2").

**Recap.** (1) A neuron computes a weighted sum plus a bias; with two inputs its decision boundary is a straight
line. (2) Training changes the parameters to lower the loss on the training split; the validation split tells you
how well that generalises. (3) Binary: one output and BCE. K classes: K outputs and cross-entropy. Fit feature
scaling on the training split only.

### 4.2 Module 2 — Going non-linear

**Goal line:** "Bend the boundary with hidden units." **Datasets:** `circle`, `xor`. **Requires:** M1.

Learning objectives:
- `m2.lo1` Explain why stacked Linear blocks without an activation collapse to one linear map.
- `m2.lo2` Add a non-linearity and predict its effect on the decision boundary.
- `m2.lo3` Relate hidden width to the regions a network can form.
- `m2.lo4` Keep layer widths consistent (the Output of one Linear equals the Input of the next).

**Fixtures.**
- `m2.start`: `in` Input (2,) → `lin1` Linear (2→1) → `out` Output. Adam lr 0.03, BCE. Data `circle` {n: 300,
  noise: 0, seed: 2}, split {val: 0.2}. Trainer epochs 50, batch 32, seed 2.
- `m2.twoLinear`: `in` → `lin1` Linear (2→8) → `lin2` Linear (8→1) → `out`. Same training, data and trainer.
- `m2.xor`: `in` Input (2,) and `out` Output, not connected. Adam lr 0.03, BCE bound to `out`. Data `xor` {n: 300,
  noise: 0, seed: 3}, split {val: 0.2}. Trainer epochs 50.

**Facts:** `twoLinearParams = 33` (2·8 + 8 + 8·1 + 1).

**Warm-up.** `m2.w1` (m1.lo2) "A model sorts photos into 5 classes. How many outputs does its last Linear need?"
**5 ✓** · 1 · 2. `m2.w2` (m1.lo3) "You scale features with the mean of all your data, before splitting. What goes
wrong?" **"Validation information leaks into training" ✓** · "Nothing" · "Training becomes slower".

#### s1 · Try first: one neuron on a circle — `predict` · Train & observe
- **Do:** "The orange dots form a disk and the blue dots a ring around it. Your model is the single neuron from
  Module 1."
- **Locks:** gate train. **Embeds:** trainer. **Target:** `trainer.run`.
- **Predict** `m2.oneNeuron` (choice): "Can one neuron separate the disk from the ring?" Yes · **No ✓**. **Reveal:**
  "No. One neuron draws one straight line, and no line separates a disk from the ring around it. Accuracy stays
  near 50%, which is chance level."
- **Check:** `{ all: [{ answered: 'm2.oneNeuron' }, { run: { dataset: 'circle', minEpochs: 30 } }] }`
- **Show me:** `answer('m2.oneNeuron', 'no')`, `train`

#### s2 · Stack two Linear blocks — `predict` · Build the model
- **Setup:** `{ do: 'loadFixture', id: 'm2.twoLinear' }`. **Locks:** gate train. **Embeds:** trainer.
- **Do:** "Now the model has two Linear blocks, 2 → 8 → 1: 8 hidden units and {twoLinearParams} parameters."
- **Predict** `m2.stacked` (choice): "Will the boundary bend now?" Yes · **No ✓**. **Reveal:** "Still a straight
  line. Two linear maps make one: W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂). More Linear blocks add parameters, not
  new shapes."
- **Check:** `{ all: [{ answered: 'm2.stacked' }, { run: { dataset: 'circle', minEpochs: 30, model: { all: [{ has: { type: 'Linear' }, count: 2 }, { not: { has: { type: 'Activation' } } }] } } }] }`
- **Show me:** `answer('m2.stacked', 'no')`, `train`

#### s3 · Add a non-linearity — `task` · Build the model
- **Palette:** `torch.nn.ReLU` ×1. **Target:** `ribbon.components`.
- **Do:** "Insert a **ReLU** between the two Linear blocks. ReLU keeps positive values and turns negative ones into
  0: relu(z) = max(0, z). It is in **Components ▾ → Activation**."
- **Check:** `{ all: [{ chain: [{ type: 'Input' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }] }, { valid: true }] }`
- **Misconceptions:** an activation directly before Output → `activation-at-output`.
- **Hints:** (1) "Use the ReLU chip above or **Components ▾ → Activation**." (2) "Select the wire between the two
  Linear blocks and press Delete, then wire Linear → ReLU → Linear."
- **Show me:** `{ do: 'insertBetween', block: 'torch.nn.ReLU', from: { id: 'lin1' }, to: { id: 'lin2' } }`

#### s4 · Train the network — `task` · Train & observe
- **Embeds:** trainer. **Target:** `trainer.run`.
- **Do:** "Train again and watch the boundary. Each hidden unit is one line; ReLU bends the output along it, and the
  second Linear combines the 8 bent pieces into a closed region."
- **Check:** `{ run: { dataset: 'circle', minEpochs: 50, final: { valAccuracy: { gte: 0.9 } }, model: { has: { type: 'Activation' } } } }`
- **Done:** "The boundary now closes around the disk."
- **Show me:** `train`

#### s5 · How many hidden units are enough? — `predict` · Build the model
- **Locks:** gate train. **Embeds:** trainer. **Target:** `node` `{ id: 'lin1' }`, `field:out_features`.
- **Do:** "Each hidden ReLU unit can add one straight edge to the boundary. After you answer, set the hidden width to 3
  and train. Change both blocks: the first Linear's **Output** and the second Linear's **Input**."
- **Predict** `m2.width` (choice): "What is the smallest number of hidden units that can enclose the disk?" 1 · 2 ·
  **3 ✓** · 8. **Reveal:** "3. A closed region needs at least 3 edges, like a triangle. With 2 units the region stays
  open and accuracy drops to about 75%. With 3, training sometimes gets stuck: change the training seed and train
  again."
- **Check:** `{ all: [{ answered: 'm2.width' }, { run: { dataset: 'circle', minEpochs: 50, model: { all: [{ param: { node: { type: 'Linear', nth: 0 }, name: 'out_features', max: 3 } }, { valid: true }] } } }] }`
- **Misconceptions:** a width mismatch between the two Linear blocks → `mismatched-widths` (via the feedback
  rewrite, §6.4.4).
- **Show me:** `setParam(lin1.out_features = 3)`, `setParam(lin2.in_features = 3)`, `train`

#### s6 · Open the box — `predict` · Check shapes & size
- **Target:** `nodeOpen` `{ id: 'lin1' }`.
- **Do:** "Linear is a component built from simpler operations. Open the first Linear with the arrow on the block.
  Inside are a **weight** table, **Transpose**, **MatMul** and **Add bias**."
- **Predict** `m2.matmul` (choice): "A batch of B points enters as (B, 2) and the weight table is (3, 2). What does
  MatMul output?" **(B, 3) ✓** · (B, 2) · (3, 2). **Reveal:** "Transpose turns the (3, 2) table into (2, 3), and (B, 2)
  × (2, 3) = (B, 3): one value per hidden unit for every point."
- **Check:** `{ all: [{ level: { open: { type: 'Linear' } } }, { answered: 'm2.matmul' }] }`
- **Show me:** `{ do: 'openLevel', path: [{ id: 'lin1' }] }`, `answer('m2.matmul', 'B3')`

#### s7 · Your turn: XOR — `task` · Build the model
- **Setup:** `{ do: 'loadFixture', id: 'm2.xor' }`. **Palette:** `component:linear` ×2, `torch.nn.ReLU` ×1,
  `torch.nn.Sigmoid` ×1 (distractor → `sigmoid-before-bce`). **Embeds:** trainer.
- **Do:** "In **XOR** a point is orange when x₁ and x₂ have the same sign. Build a network from Input to Output that
  solves it, then train it. Aim for 90% validation accuracy."
- **Check:** `{ all: [{ chain: [{ type: 'Input' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }], allow: ['Dropout'] }, { valid: true }, { run: { dataset: 'xor', minEpochs: 50, final: { valAccuracy: { gte: 0.9 } } } }] }`
- **Hints:** (1) "Same recipe as the circle: Linear → ReLU → Linear." (2) "Linear 2 → 8, ReLU, Linear 8 → 1, then Train."
- **Show me:** `{ do: 'buildChain', from: { type: 'Input' }, to: { type: 'Output' }, blocks: [{ block: 'component:linear', params: { in_features: 2, out_features: 8 } }, { block: 'torch.nn.ReLU' }, { block: 'component:linear', params: { in_features: 8, out_features: 1 } }] }`, `train`

#### s8 · Find the ReLU in the code — `predict` · Export & read the code
- **Target:** `toolbar.code`.
- **Do:** "Open **Code → model.py**. The `forward` method runs the blocks in order. Each Linear appears as a
  `torch.matmul` followed by a `torch.add`."
- **Predict** `m2.forward` (choice): "Which call comes right after the first Linear's `torch.add`?" **`self.ReLU(…)`
  ✓** · `torch.sigmoid(…)` · "Nothing: the next line is the second `torch.matmul`". **Reveal:** "`self.ReLU` applies
  max(0, z) to every hidden value. Delete it and the two Linears collapse into one again."
- **Check:** `{ all: [{ ui: { codeOpen: true, codeTab: 'model' } }, { answered: 'm2.forward' }] }`
- **Show me:** `setUi(codeOpen, codeTab 'model')`, `answer('m2.forward', 'relu')`

**Quiz.**
- `m2.q1` (lo1) "Why do two Linear blocks with nothing between them act like one?" **"Composing two linear
  functions gives another linear function" ✓** · "The second block is ignored during training" ("both train, but
  together they can only form a linear map") · "They share weights" ("they have separate weights").
- `m2.q2` (lo2) "Which of these is not a non-linearity?" ReLU · Tanh · **"Multiplying by 2" ✓** ("scaling is linear
  and cannot bend a boundary") · Sigmoid.
- `m2.q3` (lo3) "In a 2-layer ReLU network on 2D points, what does each hidden unit add to the boundary?" **"One
  straight edge" ✓** · "One circle" · "One training point".
- `m2.q4` (lo4) "Linear 2 → 8 feeds Linear 4 → 1. What happens?" **"A shape error: the second block expects 4
  inputs but receives 8" ✓** · "It trains normally" · "DeepFlow adds the missing units".

**Recap.** (1) Without a non-linearity, any stack of Linear blocks is one linear function. (2) A ReLU between
Linear blocks lets the network bend its boundary, and more hidden units give it more edges. (3) The Output of one
Linear must equal the Input of the next.

### 4.3 Module 3 — Shapes and parameters

**Goal line:** "Read tensor shapes and count parameters." **Datasets:** `mnist` (export), `digits` (browser).
**Requires:** M2.

Learning objectives:
- `m3.lo1` Read tensor shapes with a batch axis: (B, C, H, W) and (B, features).
- `m3.lo2` Use Flatten to connect image tensors to Linear blocks.
- `m3.lo3` Count the parameters of an MLP by hand.
- `m3.lo4` Predict the initial cross-entropy loss, −ln(1/C).

**Fixtures.**
- `m3.start`: `in` Input (label "images", shape `(2,)`) and `out` Output, not connected. Adam lr 0.001,
  cross-entropy bound to `out`. No dataset bound. Trainer epochs 15, batch 32, seed 3.
- `m3.bug`: the solved MLP (`in` (1, 28, 28) → `flat` Flatten → `lin1` Linear 748→128 → `relu` ReLU → `lin2` Linear
  128→10 → `out`) with the typo 748. Data `mnist`.

**Facts:** `flattenMnist = 784`, `lin1Params = 100480`, `lin2Params = 1290`, `mlpParams = 101770`,
`flattenDigits = 64`, `digitsMlpParams = 9610`, `initLoss10 = 2.30`.

**Warm-up.** `m3.w1` (m2.lo1) "A model is Linear → Linear with no activation. What boundary can it learn on 2D
data?" **"A straight line" ✓** · "A circle" · "Any curve". `m3.w2` (m2.lo4) "Linear 3 → 16 feeds Linear k → 2. What
must k be?" **16 ✓** · 3 · 2.

#### s1 · Images are tensors — `predict` · Prepare the data · tab Data
- **Setup:** `{ do: 'setUi', showDimensions: false }`. **Target:** `data.dataset:mnist`, `data.use`.
- **Do:** "MNIST has 70,000 grey 28 × 28 images of handwritten digits in 10 classes. A batch of images is a 4D
  tensor: (batch, channels, height, width). The batch size can change, so DeepFlow writes it as B. Then select
  **MNIST** and **Use this dataset**."
- **Predict** `m3.batchShape` (shape): "What is the shape of a batch of MNIST images?" expected `[null, 1, 28, 28]`.
  Diagnose `[null, 28, 28]` → `missing-channel-axis`, `[1, 28, 28]` → `missing-batch-axis`, `[null, 784]` →
  `flattened-too-early`. **Reveal:** "(B, 1, 28, 28): B images, 1 grey channel, 28 rows, 28 columns."
- **Check:** `{ all: [{ answered: 'm3.batchShape' }, { dataset: 'mnist' }, { inputShape: [null, 1, 28, 28] }] }`
- **Show me:** `answer`, `{ do: 'useDataset', id: 'mnist', setOutput: false }`

#### s2 · Flatten the image — `predict` · Check shapes & size
- **Palette:** `torch.nn.Flatten` ×1. **Target:** `ribbon.operations`, then `ribbon.dimensions`.
- **Do:** "A Linear block works on the last axis only. **Flatten** turns each (1, 28, 28) image into one long vector.
  It is in **Operations ▾ → Shape & routing**. Insert it after Input, then turn on **Dimensions** in View to check."
- **Predict** `m3.flatten` (number): "How long is each flattened vector?" expected {flattenMnist}. Diagnose 56 →
  `added-not-multiplied`, 28 → `one-axis-only`.
- **Check:** `{ all: [{ answered: 'm3.flatten' }, { chain: [{ type: 'Input' }, { type: 'Flatten' }] }, { shapeAt: { node: { type: 'Flatten' }, equals: [null, 784] } }, { ui: { showDimensions: true } }] }`
- **Show me:** `answer(784)`, `{ do: 'insertAfter', block: 'torch.nn.Flatten', after: { type: 'Input' } }`,
  `setUi(showDimensions true)`

#### s3 · Build the MLP — `task` · Build the model
- **Palette:** `component:linear` ×2, `torch.nn.ReLU` ×1.
- **Do:** "Build Flatten → Linear (784 → 128) → ReLU → Linear (128 → 10) → Output. This is a **multilayer
  perceptron** (MLP): 128 hidden units, then 10 scores, one per digit."
- **Check:** `{ all: [{ chain: [{ type: 'Input' }, { type: 'Flatten' }, { type: 'Linear' }, { type: 'Activation' }, { type: 'Linear' }, { type: 'Output' }] }, { param: { node: { type: 'Linear', nth: 0 }, name: 'out_features', equals: 128 } }, { valid: true }, { outputShape: [null, 10] }] }`
- **Show me:** `{ do: 'buildChain', from: { type: 'Flatten' }, to: { type: 'Output' }, blocks: [linear(784, 128), relu, linear(128, 10)] }`

#### s4 · Count the parameters — `predict` · Check shapes & size
- **Embeds:** `paramCount` (after reveal).
- **Do:** "Each output unit of a Linear block has one weight per input, plus one bias."
- **Predict** `m3.params1` (number): "How many parameters does the first Linear (784 → 128) have?" expected
  {lin1Params}. Diagnose 100352 → `forgot-bias`, 912 → `added-not-multiplied`. **Reveal:** "784 × 128 weights + 128
  biases = 100,480. The second Linear adds 128 × 10 + 10 = 1,290, for {mlpParams} in total. Almost all of them sit in
  the first layer, which sees every pixel."
- **Check:** `{ all: [{ answered: 'm3.params1' }, { paramCount: { equals: 101770 } }] }`
- **Show me:** `answer(100480)`

#### s5 · Fix the shape bug — `task` · Check shapes & size
- **Setup:** `{ do: 'loadFixture', id: 'm3.bug' }`. **Target:** `graph.checks`.
- **Do:** "Someone typed a wrong number into this copy of your MLP. Find the red error in **Graph checks** and fix
  it."
- **Check:** `{ all: [{ valid: true }, { param: { node: { type: 'Linear', nth: 0 }, name: 'in_features', equals: 784 } }] }`
- **Hints:** (1) "Open **Graph checks** at the top of the canvas and click the error." (2) "The first Linear must
  accept 28 × 28 = 784 features."
- **Show me:** `setParam(lin1.in_features = 784)`

#### s6 · Train a small version here — `predict` · Train & observe
- **Setup:** `{ do: 'useDataset', id: 'digits', setOutput: false }`. **Locks:** gate train. **Embeds:** trainer.
- **Do:** "Full MNIST is too big for the browser. **Digits 8×8** (1,797 tiny images) has the same structure and trains
  here in seconds. The Input is now (1, 8, 8). After you answer, update the first Linear's **Input** and train."
- **Predict** `m3.digitsFlatten` (number): "How many features does Flatten produce now?" expected {flattenDigits}.
  **Reveal:** "8 × 8 = 64. Before training the loss is close to ln 10 ≈ {initLoss10}: an untrained model gives each
  digit a probability of about 1/10."
- **Check:** `{ all: [{ answered: 'm3.digitsFlatten' }, { param: { node: { type: 'Linear', nth: 0 }, name: 'in_features', equals: 64 } }, { run: { dataset: 'digits', minEpochs: 10, final: { valAccuracy: { gte: 0.9 } } } }] }`
- **Show me:** `answer(64)`, `setParam(lin1.in_features = 64)`, `train`

#### s7 · Export for the full MNIST — `predict` · Export & read the code
- **Do:** "Switch the dataset back to **MNIST**, fix the first Linear's Input, then open **Code → train.py**. It
  downloads MNIST with torchvision and trains the same MLP."
- **Predict** `m3.normalize` (choice): "train.py applies `transforms.Normalize((0.1307,), (0.3081,))`. What does it
  do?" **"Subtracts the mean pixel value of the MNIST training images and divides by their standard deviation" ✓** ·
  "Converts the images to black and white" · "Resizes the images to 28 × 28".
- **Check:** `{ all: [{ dataset: 'mnist' }, { valid: true }, { ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm3.normalize' }] }`
- **Show me:** `useDataset('mnist', setOutput false)`, `setParam(lin1.in_features = 784)`, `setUi(code train)`,
  `answer`

**Quiz.**
- `m3.q1` (lo1) "What does B in (B, 1, 28, 28) stand for?" **"The number of examples in a batch" ✓** · "The
  brightness channel" · "The bias".
- `m3.q2` (lo2) "Flatten turns a (B, 3, 32, 32) batch into…" **(B, 3072) ✓** · (B, 96) ("multiply, don't add") ·
  (3072,) ("the batch axis is kept").
- `m3.q3` (lo3) "How many parameters does Linear 100 → 10 have?" **1,010 ✓** · 1,000 ("add the 10 biases") · 110.
- `m3.q4` (lo4) "Before training, a 10-class model's cross-entropy loss is about 2.30. Why?" **"It gives each class a
  probability of about 1/10, and −ln(1/10) ≈ 2.30" ✓** · "Loss always starts at 2.30" ("it depends on the number of
  classes") · "It already knows 23% of the answers".

**Recap.** (1) Batches are tensors: images are (B, C, H, W), and a Linear block needs (B, features), so Flatten first.
(2) Linear parameters = inputs × outputs + outputs; in an image MLP the first layer holds most of them. (3) Shape
errors are specific: read them, compute the expected size, fix the field.

### 4.4 Module 4 — Convolutions

**Goal line:** "Build a CNN and see why it is small." **Dataset:** `fashion-mnist` (export). **Requires:** M3.

Learning objectives:
- `m4.lo1` Compute Conv2d output shapes from kernel, padding and stride.
- `m4.lo2` Explain what pooling does to the shape.
- `m4.lo3` Count convolution parameters and explain weight sharing.
- `m4.lo4` Connect convolutional features to a classifier with Flatten and Linear.

**Fixtures.**
- `m4.start`: `in` Input (label "images", `(1, 28, 28)`) and `out` Output, not connected. Adam lr 0.001,
  cross-entropy. Data `fashion-mnist`. Dimensions on.
- `m4.conv1`: `in` → `conv1` Conv2d (in 1, out 8, kernel 3, padding 1, stride 1) → `out`.

**Facts:** `conv1Params = 80`, `conv2Params = 1168`, `fcParams = 7850`, `cnnParams = 9098`, `mlpParams = 101770`,
`ratio = 11`. These need Stage C's Conv2d parameter-count fix (§6.11.4).

**Warm-up.** `m4.w1` (m3.lo2) "Flatten of (B, 1, 28, 28) gives…" **(B, 784) ✓** · (B, 28) · (784,). `m4.w2` (m3.lo3)
"How many parameters does Linear 128 → 10 have?" **1,290 ✓** · 1,280 · 138.

#### s1 · Fashion-MNIST — `read` · Prepare the data
- **Do:** "Fashion-MNIST has the same format as MNIST (28 × 28 grey images, 10 classes, 60,000 training and 10,000 test
  images) but shows clothes: T-shirts, trousers, bags, sneakers. Its authors made it because MNIST is too easy:
  convolutional nets reach 99.7% on it."
- **Check:** `{ all: [{ visited: 'data' }, { dataset: 'fashion-mnist' }, { inputShape: [null, 1, 28, 28] }] }`
- **Show me:** `setUi(tab 'data')`

#### s2 · What does a convolution output? — `predict` · Check shapes & size
- **Setup:** `loadFixture('m4.conv1')`, `setUi(showDimensions false)`. **Target:** `node` `{ id: 'conv1' }`.
- **Do:** "**Conv2d** slides small 3 × 3 **filters** over the image, and each filter produces one output channel.
  This Conv2d has 8 filters and **padding 1**, a ring of zeros around the image. After you answer, turn **Dimensions**
  back on."
- **Predict** `m4.conv1` (shape): "What shape comes out of conv1?" expected `[null, 8, 28, 28]`. Diagnose `[null, 1,
  28, 28]` → `channels-unchanged`, `[null, 8, 26, 26]` → `forgot-padding`. **Reveal:** "Height = ⌊(n + 2p − k)/s⌋ + 1
  = (28 + 2 − 3)/1 + 1 = 28, and there is one channel per filter: (B, 8, 28, 28)."
- **Check:** `{ all: [{ answered: 'm4.conv1' }, { ui: { showDimensions: true } }] }`

#### s3 · Padding matters — `predict` · Check shapes & size
- **Target:** `field:padding`.
- **Do:** "Set conv1's **Padding** to 0."
- **Predict** `m4.noPad` (number): "What will the output height be?" expected 26. Diagnose 28 → `padding-no-effect`,
  25 → `minus-kernel`. **Reveal:** "(28 + 0 − 3)/1 + 1 = 26. Without padding, a 3 × 3 filter cannot be centred on the
  border pixels."
- **Check:** `{ all: [{ answered: 'm4.noPad' }, { shapeAt: { node: { id: 'conv1' }, equals: [null, 8, 26, 26] } }] }`
- **Show me:** `answer(26)`, `setParam(conv1.padding = 0)`

#### s4 · Build two convolution blocks — `task` · Build the model
- **Setup:** `setParam(conv1.padding = 1)`, with the notice "Padding is back to 1." **Palette:** `torch.nn.ReLU` ×2,
  `torch.nn.MaxPool2d` ×2, `torch.nn.Conv2d` ×1.
- **Do:** "Build conv1 → ReLU → **MaxPool2d** (Kernel 2) → Conv2d (8 → 16, Kernel 3, Padding 1) → ReLU → MaxPool2d
  (Kernel 2). Pooling keeps the largest value in each 2 × 2 window, halving height and width."
- **Check:** `{ all: [{ chain: [{ type: 'Input' }, { type: 'Conv2d' }, { type: 'Activation' }, { type: 'MaxPool2d' }, { type: 'Conv2d' }, { type: 'Activation' }, { type: 'MaxPool2d' }] }, { shapeAt: { node: { type: 'MaxPool2d', nth: 1 }, equals: [null, 16, 7, 7] } }] }`
- **Misconceptions:** conv2 `in_channels` ≠ 8 → `in-channels-mismatch`.
- **Hints:** (1) "Conv2d and pooling are in **Components ▾ → Convolution / Pooling**." (2) "conv2: In channels 8, Out
  channels 16, Kernel 3, Padding 1. Both pools: Kernel 2."
- **Show me:** `buildChain` from `{ id: 'conv1' }` to `{ type: 'Output' }` with [ReLU, MaxPool2d{kernel_size 2},
  Conv2d{in 8, out 16, kernel 3, padding 1}, ReLU, MaxPool2d{kernel_size 2}]

#### s5 · From feature maps to classes — `predict` · Check shapes & size
- **Palette:** `torch.nn.Flatten` ×1, `component:linear` ×1.
- **Do:** "The last pool outputs 16 maps of 7 × 7. Add **Flatten** and a **Linear** block that outputs the 10 class
  scores."
- **Predict** `m4.flatten` (number): "How many features does Flatten produce?" expected 784. Diagnose 49 →
  `one-map-only`, 112 → `channels-times-height`. **Reveal:** "16 × 7 × 7 = 784."
- **Check:** `{ all: [{ answered: 'm4.flatten' }, { chain: [{ type: 'MaxPool2d' }, { type: 'Flatten' }, { type: 'Linear' }, { type: 'Output' }] }, { valid: true }, { outputShape: [null, 10] }] }`
- **Misconceptions:** Linear directly after MaxPool2d → `missing-flatten` (via the feedback rewrite).
- **Show me:** `buildChain` from `{ type: 'MaxPool2d', nth: 1 }` to `{ type: 'Output' }` with [Flatten, linear(784,
  10)]

#### s6 · Convolutions share weights — `predict` · Check shapes & size
- **Embeds:** `paramCount`.
- **Do:** "A filter has one weight per input channel and kernel position, plus one bias. The same filter is used at
  every position of the image."
- **Predict** `m4.conv1Params` (number): "How many parameters does conv1 (1 → 8 channels, 3 × 3) have?" expected
  {conv1Params}. Diagnose 72 → `forgot-bias`, 6272 → `weights-per-position`. **Reveal:** "8 filters × (1 × 3 × 3) + 8
  biases = 80, however large the image. The whole CNN has {cnnParams} parameters; Module 3's MLP had {mlpParams},
  about {ratio} times more."
- **Check:** `{ all: [{ answered: 'm4.conv1Params' }, { paramCount: { equals: 9098 } }] }`

#### s7 · Export and train in Colab — `predict` · Export & read the code
- **Do:** "Open **Code → train.py** and find where the images come from. In Colab, run it to train on all 60,000
  images; a GPU makes it faster."
- **Predict** `m4.loader` (choice): "Which line loads the training images?" **`datasets.FashionMNIST(root,
  train=True, download=True, transform=transform)` ✓** · `load_iris(return_X_y=True)` · `torch.randn(60000, 1, 28, 28)`.
  **Reveal:** "torchvision downloads Fashion-MNIST the first time. In the Fashion-MNIST benchmark table, small
  two-convolution networks report about 88–93% test accuracy."
- **Check:** `{ all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm4.loader' }] }`

**Quiz.**
- `m4.q1` (lo1) "Conv2d(3 → 16, kernel 3, padding 1, stride 1) on (B, 3, 32, 32) gives…" **(B, 16, 32, 32) ✓** · (B,
  16, 30, 30) ("that is without padding") · (B, 3, 32, 32) ("channels = number of filters").
- `m4.q2` (lo2) "MaxPool2d with kernel 2 on (B, 16, 14, 14) gives…" **(B, 16, 7, 7) ✓** · (B, 8, 14, 14) ("pooling
  never changes channels") · (B, 16, 13, 13).
- `m4.q3` (lo3) "Why does a CNN need far fewer parameters than an MLP on the same image?" **"Each small filter is
  reused at every position" ✓** · "Convolutions have no weights" · "It looks at fewer pixels".
- `m4.q4` (lo3) "How many parameters does Conv2d(8 → 16, kernel 3) have?" **1,168 ✓** · 1,152 ("add the 16 biases") ·
  144.

**Recap.** (1) Conv2d output size is ⌊(n + 2p − k)/s⌋ + 1, and channels = number of filters. (2) Pooling halves
height and width, never channels. (3) Filters are reused across the image, so CNNs need far fewer parameters than
MLPs.

### 4.5 Module 5 — Training well

**Goal line:** "Read the curves, fix the training." **Dataset:** `spiral3` (CS231n 3-class spiral). **Requires:** M2.

Learning objectives:
- `m5.lo1` Predict the initial loss and use it as a sanity check.
- `m5.lo2` Recognise learning-rate problems from loss curves.
- `m5.lo3` Diagnose overfitting from training and validation curves and use early stopping.
- `m5.lo4` Explain what Dropout does in training mode and in evaluation mode.

**Fixtures.**
- `m5.start`: `in` Input (2,) → `lin1` Linear (2→64) → `relu1` ReLU → `lin2` Linear (64→3) → `out`. Adam lr 0.01,
  cross-entropy. Data `spiral3` {perClass: 100, noise: 0.2, seed: 5}, split {val: 0.2}. Trainer epochs 60, batch 32,
  seed 5.
- `m5.big`: `in` → `lin1` (2→128) → `relu1` → `lin2` (128→128) → `relu2` → `lin3` (128→3) → `out`. Adam lr 0.01,
  cross-entropy. Data `spiral3` {perClass: 50, noise: 0.3, seed: 5}, split {val: 0.5}, so 75 training points.
  Trainer epochs 300, batch 16.

**Facts:** `initLoss3 = 1.10`, `trainPoints = 75`, `dropoutSeeds = "8 of 8"` (measured with the in-browser trainer,
§11.2; not recomputed by tests, marked `source: 'simulation'`).

**Warm-up.** `m5.w1` (m4.lo1) "Conv2d(1 → 4, kernel 5, padding 0) on a 28 × 28 image gives height…" **24 ✓** · 28 ·
23. `m5.w2` (m3.lo4) "What is the initial loss for 10 equally likely classes?" **≈ 2.30 ✓** · 10 · 0.1.

#### s1 · Check the loss before training — `predict` · Train & observe
- **Embeds:** trainer. **Target:** `trainer.initLoss`.
- **Do:** "Before training, a network only guesses. Predict its loss, then press **Check initial loss**. Karpathy's
  training recipe checks this early, because it catches label and loss bugs before you waste a run."
- **Predict** `m5.initLoss` (number, tolerance 0.15): "With 3 equally likely classes, what cross-entropy loss do you
  expect before training?" expected 1.10. Diagnose 0 → `zero-initial-loss`, 0.33 or 0.67 → `loss-is-error-rate`, 3
  → `loss-is-count`. **Reveal (live):** "−ln(1/3) = ln 3 ≈ 1.10. Your untrained network measured {measured}. A first
  loss far from ln C usually means a bug in the labels, the loss or the initialisation."
- **Check:** `{ all: [{ answered: 'm5.initLoss' }, { run: { dataset: 'spiral3', initOnly: true } }] }`
- **Show me:** `answer(1.1)`, `{ do: 'evaluateInit' }`

#### s2 · Too small, just right, too large — `predict` · Train & observe
- **Embeds:** trainer. **Target:** `trainer.lr`.
- **Do:** "Train three times with learning rate 0.0001, 0.01 and 1.0, using the lr field in the run panel. Earlier
  runs stay on the chart, dimmed and labelled."
- **Predict** `m5.bestLr` (choice): "Which learning rate will give the lowest validation loss after 60 epochs?" 0.0001
  · **0.01 ✓** · 1.0. **Reveal:** "0.0001 is too small: the loss barely moves. 1.0 is too large: the loss jumps
  around and ends worse. 0.01 trains fast and stays stable. Adam's common default, 0.001, is a safe first try."
- **Check:** `{ all: [{ answered: 'm5.bestLr' }, { runs: { count: 3, lr: [0.0001, 0.01, 1], dataset: 'spiral3', minEpochs: 60 } }] }`
- **Show me:** `answer`, then for each lr in [0.0001, 0.01, 1]: `{ do: 'setOptimizer', hyperparameters: { learning_rate: lr } }`,
  `train`. Actions are awaited in order (§6.5). Finish with `setOptimizer(learning_rate 0.01)`.

#### s3 · Make overfitting happen — `predict` · Train & observe
- **Setup:** `loadFixture('m5.big')`. **Locks:** gate train. **Embeds:** trainer. **Tag run:** `m5.overfit` (the
  run that satisfies the check).
- **Do:** "Now the network is much bigger (two hidden layers of 128) and sees only {trainPoints} noisy training
  points for 300 epochs. Karpathy's recipe first makes a model big enough to overfit, and only then regularises it."
- **Predict** `m5.whichRises` (choice): "Which curve will turn upward while the other keeps falling?" **Validation
  loss ✓** · Training loss · Neither. **Reveal:** "Validation loss. The network memorises the training points,
  noise included, and becomes over-confident on the validation points it gets wrong."
- **Check:** `{ all: [{ answered: 'm5.whichRises' }, { run: { dataset: 'spiral3', minEpochs: 300, overfit: { trainAccuracyGte: 0.97, valLossRiseGte: 1.25 } } }] }`
  (`valLossRiseGte` is the final validation loss divided by its minimum over epochs.)
- **Show me:** `answer`, `train`

#### s4 · Read your curves — `predict` · Evaluate
- **Target:** `trainer.curves`.
- **Do:** "Look at the run you just made. The ▲ marks the epoch with the lowest validation loss."
- **Predict** `m5.diagnosis` (choice): "Training accuracy is near 100% and validation loss rises after its best epoch.
  The model is…" **overfitting ✓** · underfitting · training well. **Reveal:** "Overfitting. Keeping the weights from
  the ▲ epoch is **early stopping**. Validation *accuracy* barely changed, though: loss also measures confidence,
  accuracy only counts right and wrong."
- **Check:** `{ answered: 'm5.diagnosis' }`

#### s5 · Add Dropout — `task` · Build the model
- **Palette:** `torch.nn.Dropout` ×2. **Embeds:** trainer.
- **Do:** "Insert **Dropout** with p = 0.5 after each ReLU and train again. During training, Dropout sets a random
  half of the hidden values to 0 at every step, so the network cannot rely on any single unit."
- **Check:** `{ all: [{ chain: [{ type: 'Activation' }, { type: 'Dropout' }] }, { param: { node: { type: 'Dropout' }, name: 'p', min: 0.2, max: 0.6, every: true } }, { run: { dataset: 'spiral3', minEpochs: 300, model: { has: { type: 'Dropout' } } } }] }`
- **Reveal (live comparison with tag `m5.overfit`, metric `gap` = final validation loss − final training loss):**
  if smaller: "Your gap between validation and training loss went from {before} to {after}: Dropout reduced
  overfitting." Otherwise: "Your gap went from {before} to {after}. On a tiny dataset one run can mislead. Change
  the training seed and compare again: in our tests p = 0.5 shrank the gap on {dropoutSeeds} seeds."
- **Show me:** `insertAfter(Dropout{p 0.5}, after relu1)`, `insertAfter(Dropout{p 0.5}, after relu2)`, `train`
- Note: this step passes on any completed Dropout run. The research says to regularise only after diagnosing
  overfitting, and to measure the effect rather than assume it. A per-seed improvement is not guaranteed (§11).

#### s6 · Training mode and evaluation mode — `predict` · Export & read the code
- **Do:** "Open **Code → train.py** and find the validation loop."
- **Predict** `m5.eval` (choice): "The validation loop calls `model.eval()`. Why?" **"It turns Dropout off, so
  validation uses the whole network" ✓** · "It stops the weights from changing" ("`torch.no_grad()` and not calling
  `optimizer.step()` do that") · "It makes the model smaller".
- **Check:** `{ all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm5.eval' }] }`

**Quiz.**
- `m5.q1` (lo1) "A 10-class classifier starts training at loss 0.05. What should you suspect?" **"A bug, for
  example labels leaking into the inputs: an untrained model should start near ln 10 ≈ 2.30" ✓** · "A great
  initialisation" · "Nothing; the first loss is arbitrary".
- `m5.q2` (lo2) "The loss jumps up and down and ends higher than it started. What do you try first?" **"Lower the
  learning rate" ✓** · "Train for more epochs" · "Add more layers".
- `m5.q3` (lo3) "Which epoch's weights should you keep?" **"The one with the lowest validation loss" ✓** · "The
  last one" · "The one with the lowest training loss" ("that rewards memorising").
- `m5.q4` (lo4) "What does Dropout do in evaluation mode (`model.eval()`)?" **"Nothing: every unit is used" ✓** ·
  "It drops the same units as in training" · "It drops half of the inputs". (PyTorch scales kept values by
  1/(1−p) during training, so no rescaling is needed at evaluation.)

**Recap.** (1) Check the loss before training: about −ln(1/C) for C balanced classes. (2) Too small a learning rate
is slow; too large is unstable. (3) Overfitting shows as validation loss rising while training loss falls. Stop
early or regularise (Dropout, weight decay), then measure again.

### 4.6 Module 6 — Reusable blocks and residual connections

**Goal line:** "Group, repeat, and add skip connections." **Dataset:** `cifar10` (export). **Requires:** M4.

Learning objectives:
- `m6.lo1` Group blocks into a reusable component and edit its shared definition.
- `m6.lo2` Repeat a component and reason about shared structure and separate weights.
- `m6.lo3` Build a residual connection with matching shapes.
- `m6.lo4` Explain why residual paths help gradients flow, and what DeepFlow's Gradients view shows.

**Fixture `m6.start`.** `in` Input (label "images", `(3, 32, 32)`) → `stem` Conv2d (3→32, kernel 3, padding 1) →
`bn0` BatchNorm2d (32) → `relu0` ReLU → `c1` Conv2d (32→32, kernel 3, padding 1, bias off) → `b1` BatchNorm2d (32) →
`r1` ReLU → `c2` Conv2d (32→32, kernel 3, padding 1, bias off) → `b2` BatchNorm2d (32) → `r2` ReLU → `pool`
AdaptiveAvgPool2d (output 1) → `flat` Flatten → `fc` Linear (32→10) → `out`. Adam lr 0.001, cross-entropy. Data
`cifar10`. Dimensions on.

**Facts:** `blockParams = 18560`, `stemParams = 896`, `oneBlockModel = 19850`, `threeBlockModel = 56970`.

**Warm-up.** `m6.w1` (m5.lo3) "Training loss keeps falling but validation loss rises. What is happening?"
**Overfitting ✓** · Underfitting · Normal training. `m6.w2` (m4.lo1) "Conv2d(32 → 32, kernel 3, padding 1) on (B, 32, 16,
16) gives…" **(B, 32, 16, 16) ✓** · (B, 32, 14, 14) · (B, 64, 16, 16).

#### s1 · CIFAR-10 — `read` · Prepare the data
- **Do:** "CIFAR-10 has 60,000 colour photos of 32 × 32 pixels in 10 classes, such as airplane, cat and truck. Colour
  means 3 channels: (B, 3, 32, 32). It is too big to train here, so you build the model now and train it in Colab."
- **Check:** `{ all: [{ visited: 'data' }, { dataset: 'cifar10' }, { inputShape: [null, 3, 32, 32] }] }`

#### s2 · Add a skip connection — `predict` · Build the model
- **Palette:** `deepflow.op.Add` ×1. **Target:** `ribbon.operations`.
- **Do:** "A **residual block** adds its input to its output: y = relu(x + F(x)), where F is conv → BN → ReLU → conv
  → BN. Insert **Add** (Operations ▾ → Math) between the second BatchNorm and the last ReLU. Then also wire the
  block's input, the first ReLU, into Add."
- **Predict** `m6.addShapes` (choice): "Can Add combine (B, 32, 32, 32) with (B, 16, 32, 32)?" Yes · **No ✓**.
  **Reveal:** "No: Add works element by element, so both inputs need the same shape. That is why every conv in this
  block keeps 32 channels and padding 1."
- **Check:** `{ all: [{ answered: 'm6.addShapes' }, { custom: 'residualAdd' }, { valid: true }] }`.
  `residualAdd`: a root Add node has exactly two inputs, one from `relu0` and one from the BatchNorm2d at the end of
  the chain `c1 → b1 → r1 → c2 → b2`, and Add feeds `r2`.
- **Misconceptions:** an Add whose inputs differ in shape → `add-shapes` (feedback rewrite).
- **Show me:** `{ do: 'insertBetween', block: 'deepflow.op.Add', from: { id: 'b2' }, to: { id: 'r2' } }`,
  `{ do: 'connect', from: { id: 'relu0' }, to: { type: 'Add' } }`

#### s3 · Group the block into a component — `task` · Build the model
- **Target:** `ribbon.group`.
- **Do:** "Select the 7 blocks of the residual block, from the first Conv2d after the stem to the ReLU after Add. Press
  **Group** in the ribbon and name it ResBlock."
- **Check:** `{ component: { contains: [{ type: 'Conv2d', count: 2 }, { type: 'BatchNorm2d', count: 2 }, { type: 'Add', count: 1 }] } }`
- **Hints:** (1) "Shift-click each block, or drag a box around them on empty canvas." (2) "Select c1, b1, r1, c2, b2,
  Add and r2, then press Structure → Group."
- **Show me:** `{ do: 'group', nodes: [{ id: 'c1' }, { id: 'b1' }, { id: 'r1' }, { id: 'c2' }, { id: 'b2' }, { type: 'Add' }, { id: 'r2' }], name: 'ResBlock' }`

#### s4 · Repeat it three times — `predict` · Check shapes & size
- **Target:** `ribbon.sequential`. **Embeds:** `paramCount`.
- **Do:** "Select ResBlock, press **Structure → Sequential**, then set the count to 3 in its settings. Each copy's
  output feeds the next."
- **Predict** `m6.blockParams` (number): "One ResBlock has two Conv2d (32 → 32, 3 × 3, bias off) and two
  BatchNorm2d(32), each with a scale and a shift per channel. How many parameters?" expected {blockParams}. Diagnose
  18432 → `forgot-batchnorm`, 18624 → `conv-bias-when-disabled`. **Reveal:** "2 × 32 × 32 × 9 = 18,432 conv weights +
  2 × 64 BatchNorm parameters = 18,560. Three blocks bring the model to {threeBlockModel}. BatchNorm's running mean and
  variance are buffers, not parameters: gradient descent does not train them."
- **Check:** `{ all: [{ answered: 'm6.blockParams' }, { repeat: { node: { type: 'Component' }, mode: 'sequential', count: 3 } }, { valid: true }, { paramCount: { equals: 56970 } }] }`
- **Show me:** `answer`, `{ do: 'repeat', node: { type: 'Component' }, count: 3, mode: 'sequential' }`

#### s5 · Shared structure, separate weights — `predict` · Build the model
- **Target:** `nodeOpen` `{ type: 'Component' }`.
- **Do:** "Open the repeated ResBlock. Its definition is shared: change a block inside and all 3 copies change. The
  **Instances** button in Structure shows each copy."
- **Predict** `m6.sharedWeights` (choice): "Do the three copies share the same weights?" Yes · **No ✓**. **Reveal:**
  "No. The structure is shared, but every copy has its own weights. model.py creates separate layers (Conv2d_2,
  Conv2d_3, …) for each copy."
- **Check:** `{ all: [{ answered: 'm6.sharedWeights' }, { level: { open: { type: 'Component' } } }] }`

#### s6 · Why skip connections help — `predict` · Evaluate
- **Target:** `ribbon.gradients`.
- **Do:** "Turn on **Gradients** in View. It shows which paths can carry gradients back to each parameter: the
  structure only, not how large the gradients are."
- **Predict** `m6.residualGrad` (choice): "For y = x + F(x), what does the skip path give the gradient flowing back
  to x?" **"A direct identity path, so it does not vanish even if F's gradient is tiny" ✓** · "Twice the gradient of
  F" · "Nothing; gradients only flow through F". **Reveal:** "∂y/∂x = I + ∂F/∂x. The identity term passes the
  gradient straight through, which is why very deep residual networks still train."
- **Check:** `{ all: [{ answered: 'm6.residualGrad' }, { ui: { showGradientFlow: true } }] }`

#### s7 · Find the skip connection in the code — `predict` · Export & read the code
- **Do:** "Open **Code → model.py**. Each repeated block is unrolled into its own layers. In Colab, train.py downloads
  CIFAR-10 (about 170 MB); choose a GPU runtime."
- **Predict** `m6.codeAdd` (choice): "Which kind of line is a skip connection?" **`tN = torch.add(tA, tB)` ✓** ·
  `tN = self.Conv2d_2(tA)` · `tN = self.ReLU(tA)`. **Reveal:** "torch.add sums the block's input and the second
  BatchNorm's output: the wire you added."
- **Check:** `{ all: [{ ui: { codeOpen: true, codeTab: 'model' } }, { answered: 'm6.codeAdd' }] }`

**Quiz.**
- `m6.q1` (lo1) "What does Group do?" **"Turns the selected blocks into a reusable component" ✓** · "Trains those
  blocks separately" · "Merges them into one layer".
- `m6.q2` (lo2) "Repeating a block of 18,560 parameters 3 times adds how many parameters in total?" **55,680 ✓** ·
  18,560 ("each copy has its own weights") · 6,187.
- `m6.q3` (lo3) "Why must the two inputs of Add have the same shape?" **"Add sums matching elements" ✓** · "Otherwise
  the gradient is reversed" · "They don't; Add concatenates" ("that is Concat").
- `m6.q4` (lo4) "DeepFlow's Gradients view shows…" **"which paths are differentiable and which parameters are
  trainable or frozen" ✓** · "the size of each gradient during training" · "the learning rate".

**Recap.** (1) Group makes a component; Repeat stacks copies with shared structure but separate weights. (2) A skip
connection adds a block's input to its output, and both must have the same shape. (3) y = x + F(x) gives gradients an
identity path, so deep residual networks still train.

### 4.7 Module 7 — Attention and a tiny Transformer

**Goal line:** "Build a character-level GPT, then open GPT-2." **Dataset:** `tiny-shakespeare` (export). **Requires:**
M3, M6.

Learning objectives:
- `m7.lo1` Explain tokens and embeddings as learned lookup tables and count their parameters.
- `m7.lo2` Relate d_model, the number of heads and head width.
- `m7.lo3` Explain causal masking and why self-attention needs positions.
- `m7.lo4` Explain what sampling temperature does.

**Fixture `m7.start`.** Run `gpt2Storyboard` with `configuration: { vocabSize: 65, context: 64, dModel: 64, nHeads: 4,
nLayers: 2, mlpWidth: 256 }` and `includeTraining: true`, then `setOptimizer(AdamW, learning_rate 0.001, weight_decay
0.1)`. Data `tiny-shakespeare` {blockSize: 64}. Dimensions on. This is DeepFlow's GPT-2 block template (pre-LN,
explicit causal attention, learned positions, untied `lm_head` without bias) at a small size.

**Facts:** `tinyParams = 112512`, `embParams = 4160`, `dHead4 = 16`, `initLoss65 = 4.17`, `gpt2Params = 163037184`,
`gpt2TiedParams = 124439808`, `gpt2Heads = 12`.

**Warm-up.** `m7.w1` (m6.lo2) "A block is repeated 3 times. Do the copies share weights?" Yes · **No ✓**. `m7.w2`
(m3.lo4) "What is the initial loss with 65 equally likely characters?" **≈ 4.17 ✓** · 65 · 0.

#### s1 · Text as numbers — `predict` · Prepare the data · tab Data
- **Do:** "Tiny Shakespeare is 1,115,394 characters of plays, written with 65 distinct characters. Each character
  becomes a **token** id from 0 to 64. The model reads 64 ids and, at every position, predicts the next character."
- **Predict** `m7.outShape` (shape): "What shape are the model's outputs for a batch?" expected `[null, 64, 65]`.
  Diagnose `[null, 65]` → `one-prediction-per-sequence`, `[null, 64]` → `ids-not-scores`. **Reveal:** "(B, 64, 65): 65
  scores at each of the 64 positions, so one next-character guess per position."
- **Check:** `{ all: [{ answered: 'm7.outShape' }, { dataset: 'tiny-shakespeare' }, { outputShape: [null, 64, 65] }] }`

#### s2 · An embedding is a lookup table — `predict` · Build the model
- **Target:** `nodeOpen` `{ label: 'token embedding' }`.
- **Do:** "The **token embedding** stores one learned vector of 64 numbers for each of the 65 characters. Open it to
  see the table and a **Gather** that picks rows by token id."
- **Predict** `m7.embParams` (number): "How many learnable numbers are in the token embedding?" expected {embParams}.
  Diagnose 129 → `added-not-multiplied`.
- **Check:** `{ all: [{ answered: 'm7.embParams' }, { level: { open: { type: 'Embedding' } } }] }`

#### s3 · Why positions? — `predict` · Build the model
- **Target:** `node` `{ label: 'positional embedding' }`.
- **Do:** "Attention compares every token with every other token, but it has no built-in notion of order. The
  **positional embedding** adds a learned vector for each position, 0 to 63, to the token vectors."
- **Predict** `m7.positions` (choice): "Without positions, could the model tell 'dog bites man' from 'man bites dog'?"
  Yes · **No ✓**. **Reveal:** "No. Without positions, self-attention treats its inputs as a set: shuffling the
  tokens only shuffles the outputs. Position vectors break that symmetry."
- **Check:** `{ answered: 'm7.positions' }`

#### s4 · Causal self-attention — `predict` · Build the model
- **Target:** `nodeOpen` `{ label: 'Transformer blocks' }`.
- **Do:** "Open **Transformer blocks → causal self-attention → Attention heads**. A head computes Q × Kᵀ, scales it by
  1/√d_head, applies the **causal mask**, takes softmax and mixes V."
- **Predict** `m7.causal` (choice): "When predicting the 6th character, which characters can a head look at?"
  **"Characters 1 to 6" ✓** · "All 64" · "Only the 6th". **Reveal:** "The causal mask sets the scores of later
  positions to −∞ before softmax, so their weights are 0. Otherwise the model could peek at the character it must
  predict."
- **Check:** `{ all: [{ answered: 'm7.causal' }, { level: { open: { type: 'AttentionHead' } } }] }`
- **Show me:** `{ do: 'openLevel', path: [{ label: 'Transformer blocks' }, { type: 'Attention' }, { type: 'AttentionHead' }] }`, `answer`

#### s5 · Heads must divide the width — `predict` · Check shapes & size
- **Target:** `ribbon.parameters`.
- **Do:** "d_model = 64 is split equally among the heads. Open **Parameters ▾ → Global**, set n_heads to 3 and read
  the error, then set n_heads to 8."
- **Predict** `m7.dHead` (number): "With d_model 64 and 4 heads, how wide is each head?" expected {dHead4}.
  **Reveal:** "64 / 4 = 16. With 3 heads, 64 / 3 is not a whole number, so Split reports an error. With 8 heads each
  head is 8 wide, and the parameter count does not change: the Q, K and V projections stay 64 × 64."
- **Check:** `{ all: [{ answered: 'm7.dHead' }, { global: { name: 'n_heads', equals: 8 } }, { valid: true }] }`
- **Show me:** `answer(16)`, `{ do: 'setGlobal', name: 'n_heads', value: 8 }`

#### s6 · Export, train in Colab, and sample — `predict` · Export & read the code
- **Embeds:** `softmaxTemperature`.
- **Do:** "Open **Code → train.py**. It downloads Tiny Shakespeare, trains your {tinyParams}-parameter Transformer and
  prints generated text. Try the temperature slider below on a fixed set of 5 scores."
- **Predict** `m7.initLoss` (number, tolerance 0.2): "Before training, what loss do you expect with 65 equally likely
  characters?" expected {initLoss65}. Diagnose 65 → `loss-is-count`. **Reveal:** "−ln(1/65) ≈ 4.17. Temperature
  divides the scores before softmax: below 1 it sharpens the distribution, above 1 it flattens it. It rescales
  probabilities; it is not a 'creativity' setting."
- **Check:** `{ all: [{ ui: { codeOpen: true, codeTab: 'train' } }, { answered: 'm7.initLoss' }] }`

#### s7 · Same blocks, bigger numbers: GPT-2 — `predict` · Evaluate
- **Action button in the step:** "Load GPT-2 small" → `{ do: 'loadStoryboard', id: 'gpt2' }` in the lesson
  workspace. **Reset step** brings your tiny model back.
- **Do:** "GPT-2 small uses the same blocks: 12 Transformer blocks, d_model 768, a 50,257-token vocabulary and 1,024
  positions. Load it and open one block's attention."
- **Predict** `m7.gpt2Heads` (number): "How many heads does each GPT-2 small block use?" expected {gpt2Heads}.
  **Reveal:** "12 heads of width 64 (768 / 12). As built here, GPT-2 small has {gpt2Params} parameters because it
  keeps a separate output layer (lm_head). The original ties lm_head to the token embedding, which gives
  {gpt2TiedParams}."
- **Check:** `{ all: [{ global: { name: 'd_model', equals: 768 } }, { global: { name: 'n_layers', equals: 12 } }, { answered: 'm7.gpt2Heads' }] }`
- **Show me:** `loadStoryboard('gpt2')`, `answer(12)`

**Quiz.**
- `m7.q1` (lo3) "What does the causal mask prevent?" **"Attending to later positions" ✓** · "Attending to the same
  position" · "Using the positional embedding".
- `m7.q2` (lo4) "Lowering the sampling temperature from 1.0 to 0.5…" **"makes the most likely characters even more
  likely" ✓** · "makes the model more creative" ("it only rescales the scores") · "changes the model's weights".
- `m7.q3` (lo1) "An Embedding with 65 rows and 64 columns has how many parameters?" **4,160 ✓** · 129 · 65.
- `m7.q4` (lo2) "d_model = 96. Which head count works?" **8 ✓** (96 / 8 = 12) · 5 · 7.

**Recap.** (1) Text becomes token ids, and an embedding table turns each id into a learned vector. (2) Self-attention
mixes information between positions; the causal mask hides the future and position vectors give order. (3) GPT-2 is
the same recipe at a larger scale: more blocks, wider vectors, more heads.

### 4.8 Totals

| Module | Steps | Warm-up | Quiz | In-browser runs | Export-only |
|---|---|---|---|---|---|
| M1 | 8 | 2 | 4 | blobs, iris | — |
| M2 | 8 | 2 | 4 | circle, xor | — |
| M3 | 7 | 2 | 4 | digits | mnist |
| M4 | 7 | 2 | 4 | — | fashion-mnist |
| M5 | 6 | 2 | 4 | spiral3 | — |
| M6 | 7 | 2 | 4 | — | cifar10 |
| M7 | 7 | 2 | 4 | — | tiny-shakespeare |
| **Total** | **50** | **14** | **28** | | |

### 4.9 Misconception catalogue

Ids are stable. Text lives in `src/edu/lessons/modules/text/en/common.js` under `mis.<id>`. Each entry below is the
draft text.

| id | Draft explanation |
|---|---|
| `forgot-bias` | Each output unit also has one bias, so add the number of outputs. |
| `added-not-multiplied` | Sizes multiply: a 28 × 28 image has 784 pixels; a Linear has inputs × outputs weights. |
| `one-axis-only` | Flatten combines every axis after the batch axis: 1 × 28 × 28. |
| `target-in-input` | The label (the species) is what the model predicts; it is not an input feature. |
| `classes-as-features` | 3 is the number of species, the outputs. The inputs are the 4 measurements. |
| `one-output-multiclass` | With K classes the model needs K outputs, one score per class. |
| `softmax-before-ce` | Cross-entropy already applies softmax. A Softmax block before it squashes the scores twice and slows learning: output raw scores (logits). |
| `sigmoid-before-bce` | DeepFlow's BCE is BCEWithLogitsLoss, which applies the sigmoid itself. Output the raw score. |
| `activation-at-output` | A ReLU right before the output clips negative scores to 0. Put activations between Linear blocks. |
| `mismatched-widths` | The second Linear's Input must equal the first Linear's Output. |
| `missing-channel-axis` | Grey images still have a channel axis of size 1: (B, 1, 28, 28). |
| `missing-batch-axis` | Models process batches: the first axis is the batch size B. |
| `flattened-too-early` | Images arrive as (B, 1, 28, 28). Flatten is a block you add; it is not part of the data. |
| `channels-unchanged` | A convolution outputs one channel per filter. |
| `forgot-padding` | That would be right without padding. Padding 1 keeps 28. |
| `padding-no-effect` | Without padding, the filter cannot be centred on the border pixels, so the map shrinks. |
| `minus-kernel` | For stride 1 the size is n − k + 1 = 28 − 3 + 1 = 26. |
| `in-channels-mismatch` | A conv's In channels must equal the previous conv's Out channels. |
| `one-map-only` / `channels-times-height` | Flatten multiplies all three axes: channels × height × width. |
| `missing-flatten` | Linear acts on the last axis only. Add Flatten after the last pooling block. |
| `weights-per-position` | A filter's weights are shared across positions, so they do not grow with image size. |
| `forgot-batchnorm` | BatchNorm2d(32) learns a scale and a shift per channel: 64 parameters. |
| `conv-bias-when-disabled` | These convolutions have bias off; BatchNorm's shift does that job. |
| `add-shapes` | Both inputs of Add must have the same shape. |
| `zero-initial-loss` | An untrained model guesses, so its loss starts high, near −ln(1/C). |
| `loss-is-error-rate` | Loss is not the error rate. For uniform guesses it is −ln(1/C). |
| `loss-is-count` | Loss is −ln(probability of the right answer), here −ln(1/C). |
| `one-prediction-per-sequence` | A language model predicts the next token at every position, not once per sequence. |
| `ids-not-scores` | The output holds a score for every token in the vocabulary at each position. |

---

## 5. Datasets

### 5.1 Registry contents

`training: 'browser'` sets train in the in-browser trainer. `'export'` sets are metadata only: shape, classes,
licence and a Python loader. No dataset file is downloaded in the browser, ever.

| id | Name | Kind | Training | Input `shape` · dtype | Target | Default options | Size |
|---|---|---|---|---|---|---|---|
| `blobs` | Two blobs | toy2d | browser | `(2,)` · float32 | 2 classes: "blue", "orange" | n 200, noise 0.1, seed 1 | n |
| `circle` | Circle | toy2d | browser | `(2,)` | 2: "ring", "disk" | n 300, noise 0, seed 1 | n |
| `xor` | XOR | toy2d | browser | `(2,)` | 2: "different signs", "same sign" | n 300, noise 0, seed 1 | n |
| `spiral2` | Spiral (2 classes) | toy2d | browser | `(2,)` | 2: "arm A", "arm B" | n 300, noise 0, seed 1 | n |
| `spiral3` | Spiral (3 classes) | toy2d | browser | `(2,)` | 3: "arm A", "arm B", "arm C" | perClass 100, noise 0.2, seed 1 | 3 × perClass |
| `plane` *(optional)* | Plane (regression) | toy2d | browser | `(2,)` | regression, 1 value | n 300, noise 0, seed 1 | n |
| `iris` | Iris | tabular | browser | `(4,)` | 3: setosa, versicolor, virginica | seed 1, scaling `standardize` | 150 |
| `digits` | Digits 8×8 | image | browser | `(1, 8, 8)` | 10: "0"…"9" | seed 1, pixels ÷ 16 | 1,797 |
| `mnist` | MNIST | image | export | `(1, 28, 28)` | 10: "0"…"9" | valSize 5,000 | 60,000 train + 10,000 test |
| `fashion-mnist` | Fashion-MNIST | image | export | `(1, 28, 28)` | 10: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot | valSize 5,000 | 60,000 + 10,000 |
| `cifar10` | CIFAR-10 | image | export | `(3, 32, 32)` | 10: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck | valSize 5,000 | 50,000 + 10,000 |
| `tiny-shakespeare` | Tiny Shakespeare | text | export | `(64,)` · int64 (`blockSize`) | next token, vocabulary 65 | blockSize 64 | 1,115,394 characters; 90/10 split = 1,003,854 / 111,540 |

The default split for browser datasets is {val: 0.2, test: 0}. The Data tab allows val 0.1–0.5 and test 0, 0.1 or
0.2. Export datasets use the official train/test split and carve `valSize` validation images out of the training set
with a seeded `random_split`.

Normalisation constants for export presets (standard values; Stage C states them in the code comments):
MNIST mean 0.1307, std 0.3081. Fashion-MNIST mean 0.2860, std 0.3530. CIFAR-10 mean (0.4914, 0.4822, 0.4465), std
(0.2470, 0.2435, 0.2616). The CIFAR std is the per-channel std over all training pixels; the also-common (0.2023,
0.1994, 0.2010) is a mean of per-image stds and is not used.

### 5.2 Sources and licences

| id | Source | Licence / status | Shown attribution |
|---|---|---|---|
| toy2d | Generated in the browser by Edu code | Our code. The algorithms are adapted from TensorFlow Playground `dataset.ts` (Apache-2.0) and the CS231n spiral (MIT); credit both | "Generator adapted from TensorFlow Playground (Apache-2.0)" or "from CS231n (MIT)" |
| `iris` | Values identical to scikit-learn `sklearn/datasets/data/iris.csv` (sha256 `f13ffa8f…1449`), which follows Fisher (1936), as R's `iris` does | Fisher's 1936 measurements are treated as public domain; the file comes from scikit-learn (BSD-3-Clause), whose notice is kept in the data file header | "Fisher, R.A. (1936) The use of multiple measurements in taxonomic problems. Values as in scikit-learn's load_iris()." |
| `digits` | scikit-learn `sklearn/datasets/data/digits.csv.gz`: a copy of the test set of the UCI Optical Recognition of Handwritten Digits data (Alpaydin & Kaynak, 1998) | Distributed with scikit-learn (BSD-3-Clause). The UCI page could not be checked from this sandbox; Stage C records `verified: false` for the UCI terms | "E. Alpaydin, C. Kaynak (1998), UCI ML Repository; via scikit-learn" |
| `mnist` | torchvision `datasets.MNIST`, downloaded at run time | Not bundled. Sources disagree on the licence and it was not verified | "LeCun, Cortes & Burges, The MNIST database" |
| `fashion-mnist` | torchvision `datasets.FashionMNIST`, downloaded at run time | MIT, Copyright © 2017 Zalando SE; not bundled | "Xiao, Rasul & Vollgraf (2017), Fashion-MNIST (MIT)" |
| `cifar10` | torchvision `datasets.CIFAR10`, downloaded at run time (about 170 MB) | No licence stated by the authors; not bundled | "Krizhevsky (2009), Learning Multiple Layers of Features from Tiny Images" |
| `tiny-shakespeare` | `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`, downloaded at run time | The text is Shakespeare (public domain by age); the file is from the karpathy/char-rnn repository (MIT) | "Tiny Shakespeare, from Karpathy's char-rnn" |

Two facts about the Iris file:

- **sklearn copy, not UCI.** The UCI copy has two wrong rows: sample 35 should be 4.9, 3.1, 1.5, 0.2 and sample 38
  should be 4.9, 3.6, 1.4, 0.1. Edu embeds the corrected values (sklearn's `iris.csv`) so the browser matches
  `load_iris()` exactly. A test pins rows 35 and 38 (1-based) and the class counts 50/50/50.
- **No tfjs copy.** The tfjs-examples `iris/data.js` copies UCI and must not be used.

### 5.3 Seeded generation, split and scaling (shared by browser and Python)

Everything that decides which examples land in which split is defined bit for bit. The exported `train.py` can then
regenerate exactly the browser's data. This is tested in §8.2.

**PRNG.** mulberry32 with a 32-bit state:

```js
export function mulberry32(seed) {
    let a = seed >>> 0;
    return function next() {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
```

The Python port masks every intermediate with `& 0xFFFFFFFF`, and `imul(a, b) = (a * b) & 0xFFFFFFFF`. It returns
the same float sequence. (Reference port: §11.4.)

**Streams.** `deriveSeed(seed, label)` combines the user seed with a stream label, so data, split, init, batch order
and dropout use independent streams:

```js
const fnv1a = (s) => { let h = 0x811c9dc5; for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193) >>> 0; } return h; };
export function deriveSeed(seed, label) {
    let x = (Math.imul(seed >>> 0, 0x9E3779B1) ^ fnv1a(label)) >>> 0;
    x ^= x >>> 16; x = Math.imul(x, 0x85EBCA6B) >>> 0; x ^= x >>> 13; x = Math.imul(x, 0xC2B2AE35) >>> 0; x ^= x >>> 16;
    return x >>> 0;
}
```

Labels: `'data'` (generator), `'split'`, `'init'`, `'batches'`, `'dropout'`. Data uses the dataset seed. Init,
batches and dropout use the training seed.

**Normal samples.** Marsaglia polar method; one value per call, the second value discarded:

```js
function normal(next, mean = 0, std = 1) {
    let v1, v2, s;
    do { v1 = 2 * next() - 1; v2 = 2 * next() - 1; s = v1 * v1 + v2 * v2; } while (s >= 1 || s === 0);
    return mean + std * v1 * Math.sqrt(-2 * Math.log(s) / s);
}
const uniform = (next, a, b) => next() * (b - a) + a;
```

**Generators.** Each returns `{ x: Float64Array(n·2), y: Int32Array(n) }` (regression: `Float64Array`) in generation
order, drawing from `mulberry32(deriveSeed(seed, 'data'))`.

- `blobs(n, noise)`. TensorFlow Playground two-Gaussian: variance = 0.5 + 3.5 · (noise / 0.5) and std =
  √variance. First ⌊n/2⌋ points with label 1 around (2, 2), then ⌊n/2⌋ with label 0 around (−2, −2). For each
  point, x₁ = normal(c, std), then x₂ = normal(c, std).
- `circle(n, noise)`. Playground, radius R = 5. First ⌊n/2⌋ points: r = uniform(0, 2.5), angle = uniform(0, 2π),
  x = (r·sin angle, r·cos angle). Then ⌊n/2⌋ points with r = uniform(3.5, 5). For each point, after x, draw
  noise_x = uniform(−5, 5)·noise and noise_y = uniform(−5, 5)·noise. The label is 1 if ‖x + noise‖ < 2.5, else 0.
  Coordinates are stored without the noise, as in Playground.
- `xor(n, noise)`. Playground. x₁ = uniform(−5, 5) ± 0.3 (pushed away from 0 by its sign), then x₂ the same way.
  Then noise_x and noise_y as for circle, and label 1 if (x₁ + noise_x)(x₂ + noise_y) ≥ 0.
- `spiral2(n, noise)`. Playground. For label 1 (Δ = 0), then label 0 (Δ = π), i = 0 … ⌊n/2⌋ − 1: r = 5i/(n/2),
  t = 1.75 · 2π · i/(n/2) + Δ, x = (r sin t + uniform(−1, 1)·noise, r cos t + uniform(−1, 1)·noise).
- `spiral3(perClass, noise)`. CS231n, K = 3. For class j = 0, 1, 2 and i = 0 … N − 1 (N = perClass): r = i/(N − 1),
  t = 4j + 4i/(N − 1) + normal(0, 1)·noise, x = (r sin t, r cos t).
- `plane(n, noise)` *(optional)*. Playground regressPlane: x₁ = uniform(−6, 6), x₂ = uniform(−6, 6), then
  noise_x = uniform(−6, 6)·noise and noise_y = uniform(−6, 6)·noise. target = (x₁ + noise_x + x₂ + noise_y)/10, which
  is Playground's linear scale from [−10, 10] to [−1, 1]. It is not clamped, so |target| can reach 1.2.

**Split.** Shuffle the indices 0 … n − 1 with Fisher–Yates using `mulberry32(deriveSeed(dataSeed, 'split'))`: for i
from n − 1 down to 1, j = ⌊next() · (i + 1)⌋, swap(i, j). Then:

- `nVal = Math.floor(n · val + 0.5)` and `nTest = Math.floor(n · test + 0.5)`. Python must use the same expression,
  not `round()`, which rounds half to even.
- val = the first nVal shuffled indices, test = the next nTest, train = the rest. Each split keeps shuffled order.

**Scaling.** Iris uses `standardize`: per-feature mean and population std (ddof 0) computed on the train split only,
then applied to every split, with std floored at 1e-12. Digits: pixel / 16. Toy sets are not scaled, as in
Playground.

**Mini-batches.** Each epoch shuffles the train indices with Fisher–Yates on the `'batches'` stream. The stream is
created once per run and advances across epochs. Batches are consecutive slices of size `batchSize`, and the last
one may be smaller. The PyTorch export uses `DataLoader(shuffle=True, generator=torch.Generator().manual_seed(seed))`.
Batch order therefore differs from the browser; data and split do not (§10).

---

## 6. Architecture and interfaces

### 6.1 File layout and owners

`[A]`–`[E]` mark the owning stage (§7). Each file has exactly one owner.

```
src/edu/
  config.js                         [A] contract file, content fixed in §6.2
  bridge.js                         [A] contract file, content fixed in §6.2
  objective.js                      [A] contract file, content fixed in §6.2 (effective optimizer and loss)
  shell/
    uiStore.js                      [A] active tab, code panel, panel collapse (§6.12)
    deepLink.js                     [A] #/learn/<module>/<step>
    shell.css                       [A]
  lessons/
    engine/                         [A]
      schema.js  nodes.js  checks.js  feedback.js  actions.js  context.js  session.js  storage.js
      progress.js  workspace.js  fixtures.js  i18n.js  log.js  targets.js  canvasBridge.js
      text/en/ui.js                 [A] UI chrome strings (buttons, labels, subgoal names)
    modules/                        [D]
      index.js                      [D] module registry and text bundles (A may create the placeholder, §6.2)
      m1-first-neuron.js  m2-non-linear.js  m3-shapes.js  m4-convolutions.js
      m5-training-well.js  m6-residual.js  m7-transformer.js
      text/en/index.js  common.js  m1.js … m7.js
  components/                       [A]
    LearnHome.jsx  ModuleCard.jsx  LessonPanel.jsx  StepView.jsx  PredictGate.jsx  QuizView.jsx  RecapView.jsx
    HintLadder.jsx  PaletteChips.jsx  ParamCountReadout.jsx  Spotlight.jsx  RichText.jsx  SoftmaxTemperature.jsx
    learn.css  lesson-panel.css
  data/                             [B] except the files marked [C]
    registry.js  store.js  useDataset.js  rng.js  generators.js  split.js  prepare.js
    presets/index.js  toy.js  tabular.js  irisData.js  digits.js  digitsData.js
    presets/images.js               [C] mnist, fashion-mnist, cifar10 (B may create the placeholder)
    presets/text.js                 [C] tiny-shakespeare (B may create the placeholder)
    python/common.js  python/toy.js  python/iris.js  python/digits.js
    DataWorkspace.jsx               [B] mount point (A may create the placeholder)
    DatasetList.jsx  DatasetDetail.jsx  ScatterPreview.jsx  data.css
  trainer/                          [B]
    tensor.js  ops.js  compile.js  init.js  losses.js  optimizers.js  metrics.js  boundary.js  run.js  store.js
    components/TrainerPanel.jsx     [B] mount point (A may create the placeholder)
    components/TrainerCard.jsx      [B] mount point (A may create the placeholder)
    components/LossChart.jsx  AccuracyChart.jsx  BoundaryView.jsx  ConfusionMatrix.jsx  RunSettings.jsx  trainer.css
src/codegen/training.js             [C] adds the `data` option; legacy output unchanged
src/codegen/edu/trainScript.js      [C]
src/codegen/edu/notebook.js         [C]
src/components/CodePanel.jsx        [C] controlled tabs, Edu script, .ipynb download
src/domain/optimization/parameterCount.js   [C] Conv kernel fix + countParametersByNode (§6.11.4)
src/domain/library/pytorchBlocks.js         [B] add torch.nn.LogSoftmax only
src/domain/library/nodeFamilies.js          [B] add the LogSoftmax activation variant only
src/validation/shapes.js                    [B] add the LogSoftmax passthrough rule only
src/App.jsx  src/main.jsx  index.html       [A]
src/components/Layout.jsx  Navbar.jsx  ToolbarStrip.jsx  StoryboardMenu.jsx      [A]
src/components/architecture/ArchitectureWorkspace.jsx                            [A] canvas event bridge only
scripts/edu-*.test.mjs, scripts/edu-*-browser-check.mjs   owner = the stage named in §8
docs/edu/SPEC.md [stage 0, then E]   docs/edu/QA.md [E]   README.md [E]
```

**Node-importable rule.** Plain Node must be able to import every non-component module: `src/edu/lessons/engine/**`,
`src/edu/lessons/modules/**`, `src/edu/data/**` except `*.jsx`, `src/edu/trainer/**` except `components/`, the
contract files and `src/codegen/**`. Content tests and the headless Show me rely on it. These modules contain no
JSX and no CSS imports, never use `import.meta.glob`, and read `import.meta.env` only as `import.meta.env?.DEV`.

`package.json` and `package-lock.json` do not change. v1 adds **zero runtime dependencies**: charts, the tensor engine,
the PRNG and markup rendering are small in-repo code. A stage that believes it needs a dependency stops and justifies
it in its report. Only Stage E may add one.

### 6.2 Contract files and placeholders

**Contract files** are the only files more than one stage needs from day one. Stage A creates them in its first
commit. A stage that starts before they exist creates them **byte-identical** to the content below and commits them.
No stage edits them afterwards, except Stage E, which then also updates this section.

`src/edu/config.js`:

```js
/** Product identity. Rename the product here and nowhere else (see docs/edu/SPEC.md §6.2). */
export const PRODUCT_NAME = 'DeepFlow Edu';
export const PRODUCT_TAGLINE = 'Learn to build neural networks';
/** Prefix for every localStorage key written by Edu code. DeepFlow's own keys ('deepflow:*') are unchanged. */
export const STORAGE_PREFIX = 'deepflow-edu';
export const DEFAULT_LOCALE = 'en';
export const CURRICULUM_VERSION = 1;
```

`src/edu/bridge.js`:

```js
/**
 * Cross-stage registry (docs/edu/SPEC.md §6.2). Stores register read-only sources and workspace slices here so the
 * lesson engine can check and snapshot app state without importing other stages' modules. Do not edit; Stage E only.
 */
const sources = new Map();
const slices = new Map();
const listeners = new Set();

/** api: { read(): object, subscribe?(listener): () => void, actions?: Object<string, Function> } */
export function registerSource(name, api) {
    sources.set(name, api);
    for (const listener of listeners) listener(name);
    return () => { if (sources.get(name) === api) sources.delete(name); };
}
export function readSource(name) {
    const api = sources.get(name);
    try { return api ? api.read() : null; } catch { return null; }
}
export function subscribeSource(name, listener) {
    const api = sources.get(name);
    return api?.subscribe ? api.subscribe(listener) : () => {};
}
export function callSource(name, action, ...args) {
    const fn = sources.get(name)?.actions?.[action];
    if (!fn) throw new Error(`Source "${name}" has no action "${action}"`);
    return fn(...args);
}
export function onSourceRegistered(listener) {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

/** slice: { capture(): json-serialisable, restore(value | undefined): void } */
export function registerSlice(name, slice) {
    slices.set(name, slice);
    return () => { if (slices.get(name) === slice) slices.delete(name); };
}
export function workspaceSlices() {
    return [...slices.entries()];
}

/** Development-only handles for browser checks: window.__deepflow.edu.<name>. */
export const eduDevHandles = {};
export function exposeEduHandle(name, value) {
    eduDevHandles[name] = value;
    return value;
}
```

`src/edu/objective.js` (the rule of §6.10.5; used by the trainer, lesson checks and actions, `useDataset`, and
codegen):

```js
import { OPTIMIZER_DEFAULTS } from '../domain/training/index.js';

/**
 * Which optimizer and loss train the model (docs/edu/SPEC.md §6.10.5). Pure; shared by the trainer, lesson checks
 * and codegen. Do not edit; Stage E only.
 */
export function effectiveObjective(training, rootGraph) {
    const nodes = Object.values(training?.nodes ?? {});
    const edges = training?.edges ?? [];
    const output = Object.values(rootGraph?.nodes ?? {})
        .find((node) => node.kind === 'layer' && node.libraryKey === 'deepflow.io.Output') ?? null;
    const issues = [];
    if (nodes.some((node) => node.kind === 'phase')) issues.push({ severity: 'error', code: 'multi-phase' });
    const lossNodes = nodes.filter((node) => node.kind === 'loss');
    let losses = output ? lossNodes.filter((node) => node.architectureOutputId === output.id) : [];
    if (losses.length > 1) issues.push({ severity: 'error', code: 'multi-loss' });
    if (!losses.length && lossNodes.length === 1) {
        losses = lossNodes;
        issues.push({ severity: 'warning', code: 'loss-unbound' });
    }
    const loss = losses[0] ?? null;
    const optimizers = nodes.filter((node) => node.kind === 'optimizer');
    const linked = loss ? optimizers.find((node) => edges.some((edge) => edge.source === loss.id && edge.target === node.id)) : null;
    const optimizer = linked ?? (optimizers.length === 1 ? optimizers[0] : null);
    if (!optimizer && optimizers.length > 1) issues.push({ severity: 'warning', code: 'optimizer-ambiguous' });
    return { optimizer, loss, lossBound: !!loss && !!output && loss.architectureOutputId === output.id, output, issues };
}

/** Defaults when the training graph has no optimizer or no loss (§6.10.5). */
export function defaultObjective({ targetKind, numClasses = null, outputSize = null }) {
    const lossType = targetKind === 'regression' ? 'mse'
        : numClasses === 2 && outputSize === 1 ? 'bce' : 'cross_entropy';
    return {
        optimizer: { type: 'Adam', hyperparameters: { ...OPTIMIZER_DEFAULTS.Adam, learning_rate: 0.01 } },
        loss: { type: lossType, hyperparameters: {} },
    };
}
```

Registered names: sources `'data'` and `'trainer'` (Stage B). Slices `'data'` and `'trainer'` (Stage B). Dev
handles: `'session'`, `'progress'`, `'ui'`, `'engine'` (A); `'data'`, `'trainer'`, `'registry'` (B); `'codegen'` (C).

**Placeholders.** A stage that imports another stage's mount point may create it **only if the file does not exist**,
with exactly this content. The owner then replaces it.

| File | Owner | May be created by | Placeholder content |
|---|---|---|---|
| `src/edu/data/DataWorkspace.jsx` | B | A | `export default function DataWorkspace() { return <div className="edu-placeholder" role="status">Data workspace is being built.</div>; }` |
| `src/edu/trainer/components/TrainerPanel.jsx` | B | A | `export default function TrainerPanel() { return <div className="edu-placeholder" role="status">Trainer is being built.</div>; }` |
| `src/edu/trainer/components/TrainerCard.jsx` | B | A | `export default function TrainerCard() { return <div className="edu-placeholder" role="status">Trainer is being built.</div>; }` |
| `src/edu/lessons/modules/index.js` | D | A | `export const MODULES = [];` and `export const TEXT = { en: {} };` |
| `src/edu/data/presets/images.js` | C | B | `export default [];` |
| `src/edu/data/presets/text.js` | C | B | `export default [];` |

Mount-point props are part of the contract: `DataWorkspace()` and `TrainerPanel()` take none. `TrainerCard({ locked
= false, lockedReason = '' })`.

### 6.3 Lesson schema

Module files (Stage D) export plain objects. `src/edu/lessons/engine/schema.js` (Stage A) holds the JSDoc typedefs
and `validateModule(module, text) → string[]` (an empty array means valid).

#### 6.3.1 Types

```js
/**
 * @typedef {'m1'|'m2'|'m3'|'m4'|'m5'|'m6'|'m7'} ModuleId
 * @typedef {'data'|'model'|'shapes'|'objective'|'train'|'evaluate'|'export'} SubgoalId
 * @typedef {'learn'|'architecture'|'data'|'training'} TabId
 *
 * @typedef {Object} LessonModule
 * @property {ModuleId} id
 * @property {number} version              bump when step ids or meaning change (§6.8.2)
 * @property {number} order                1..7
 * @property {number} minutes
 * @property {string[]} datasets           dataset ids used (must exist in the registry)
 * @property {ModuleId[]} [recommendedAfter]
 * @property {Object<string, Fixture>} fixtures
 * @property {string} start                fixture id loaded when the module starts the first time
 * @property {Object<string, Fact>} facts  numbers used in text (§6.3.6)
 * @property {Object<string, (ctx: CheckContext) => CheckResult>} [checks]   custom checks used by { custom: name }
 * @property {QuizItem[]} warmup           0–3 items
 * @property {LessonStep[]} steps          4–8 items
 * @property {QuizItem[]} quiz             3–4 items
 * @property {string[]} objectives         LO ids, e.g. ['m1.lo1', …]; text key `<lo>` holds the sentence
 *
 * @typedef {Object} LessonStep
 * @property {string} id                   's1'…; unique in the module
 * @property {'read'|'task'|'predict'} kind
 * @property {SubgoalId} subgoal
 * @property {TabId} [tab]                 tab shown when the step opens
 * @property {TargetRef[]} [target]        spotlight, in order (the first visible one is ringed)
 * @property {ActionSpec[]} [setup]        run once on first entry, before the entry snapshot
 * @property {CheckSpec} check
 * @property {Diagnostic[]} [diagnose]     evaluated in order while the check fails
 * @property {PredictSpec} [predict]       required when kind === 'predict'
 * @property {ActionSpec[]} solution       Show me; must make check pass from the entry state (tested, §8.4)
 * @property {PaletteItem[]} [palette]
 * @property {{ gate?: 'train' }} [locks]
 * @property {EmbedId[]} [embeds]
 * @property {StepButton[]} [buttons]      extra action buttons, e.g. M7 s7 "Load GPT-2 small"
 * @property {string} [tagRun]             tag the run that satisfied a `run` check when the step passes
 * @property {RevealSpec} [reveal]         live values shown after the prediction or the pass
 *
 * @typedef {Object} PredictSpec
 * @property {string} id                   'm1.params'; stored in progress answers
 * @property {'number'|'choice'|'shape'} kind
 * @property {number} [expected]  @property {number} [tolerance]    absolute; default 0 (numbers compare after rounding to 1e-9)
 * @property {string[]} [options] @property {string} [correct]      option ids; text `<m>.<s>.predict.opt.<id>`
 * @property {Array<number|null>} [expectedShape]                    null = batch axis; the input accepts "B, 1, 28, 28"
 * @property {{ equals: number|string|Array<number|null>, misconception: string }[]} [diagnose]
 *
 * @typedef {Object} QuizItem
 * @property {string} id                   'm1.q1' or 'm2.w1'
 * @property {string} objective            LO id
 * @property {string[]} options            option ids; text `<id>.opt.<option>` and `<id>.why.<option>`
 * @property {string} correct
 * @property {Object<string, string>} [misconceptions]   option id → misconception id
 *
 * @typedef {{ block: BlockRef, count?: number, distractor?: string }} PaletteItem   distractor = misconception id
 * @typedef {string} BlockRef        'component:linear' | a library key such as 'torch.nn.ReLU' | 'deepflow.op.Add'
 * @typedef {'trainer'|'paramCount'|'softmaxTemperature'} EmbedId
 * @typedef {{ id: string, actions: ActionSpec[] }} StepButton        label text `<m>.<s>.button.<id>`
 * @typedef {{ run?: 'initLoss'|'finalValAccuracy', compare?: { tag: string, metric: 'gap'|'valLoss'|'valAccuracy' } }} RevealSpec
 * @typedef {{ when: CheckSpec, misconception: string }} Diagnostic
 *
 * @typedef {Object} Fixture
 * @property {'snapshot'|'storyboard'} kind
 * @property {(helpers: FixtureHelpers) => WorkspaceSnapshot} [build]     kind 'snapshot': pure; returns a snapshot (§6.8.3)
 * @property {string} [storyboard]  @property {Object} [configuration]   kind 'storyboard': run a STORYBOARDS entry with options
 * @property {ActionSpec[]} [after]          applied after a storyboard (e.g. setOptimizer, useDataset)
 *
 * @typedef {Object} Fact
 * @property {number|string} value
 * @property {(helpers: FactHelpers) => number|string} [compute]   recomputed by the content test (§8.4)
 * @property {'simulation'|'literature'} [source]                   facts that tests cannot recompute
 */
```

#### 6.3.2 Subgoal vocabulary

| SubgoalId | Label (`subgoal.<id>` in `text/en/ui.js`) |
|---|---|
| `data` | Prepare the data |
| `model` | Build the model |
| `shapes` | Check shapes & size |
| `objective` | Choose loss & optimizer |
| `train` | Train & observe |
| `evaluate` | Evaluate |
| `export` | Export & read the code |

#### 6.3.3 Text keys

All learner-visible strings come from text bundles: `TEXT[locale][key]` in `modules/index.js`, merged with Stage A's
`text/en/ui.js`. `t(key, params)` falls back to `en`, then to the key itself; in development it warns once per
missing key. Required keys per module (checked by `validateModule`):

- `<m>.title`, `<m>.goal`, `<m>.recap.1` … `<m>.recap.3`, and every LO id in `objectives`
- per step: `<m>.<s>.title`, `<m>.<s>.body`, `<m>.<s>.hint1`, `<m>.<s>.hint2`; optional `<m>.<s>.why`, `<m>.<s>.done`
- per prediction: `<m>.<s>.predict.prompt`, `<m>.<s>.predict.reveal`, and `<m>.<s>.predict.opt.<id>` for choices
- per quiz or warm-up item: `<id>.prompt`, `<id>.opt.<option>`, `<id>.why.<option>`
- per misconception id used: `mis.<id>` (in `common.js`)
- per step button: `<m>.<s>.button.<id>`; per reveal variant: `<m>.<s>.reveal.better`, `<m>.<s>.reveal.notBetter`

No text key may be unused (§8.4).

#### 6.3.4 Markup

`RichText` renders a safe subset into React elements and never uses `dangerouslySetInnerHTML`:

- `**bold**` and `` `code` ``
- a blank line starts a paragraph; lines starting with `- ` are bullets
- `{name}` interpolates facts and live parameters; numbers get locale grouping (101,770)

#### 6.3.5 Embeds

| EmbedId | Component | Shows |
|---|---|---|
| `trainer` | Stage B `TrainerCard` with `locked`/`lockedReason` from the step's gate | compact training UI |
| `paramCount` | `ParamCountReadout` [A] | total parameters from `countArchitectureParameters` (root). When available, a per-root-node breakdown from `countParametersByNode` (§6.11.4), found by feature detection through a namespace import, so it works before Stage C lands |
| `softmaxTemperature` | `SoftmaxTemperature` [A] | 5 fixed scores (2.0, 1.0, 0.5, 0.0, −1.0) for the characters `e t a o z`; a slider T ∈ [0.2, 2.0], step 0.1; bars show softmax(scores / T) |

#### 6.3.6 Facts ledger

Every number in lesson copy that comes from the app (parameter counts, flatten sizes) is a fact. Text uses
`{factName}`. Facts with `compute` are recomputed by the content test from the module's own fixtures, after applying
the solutions up to the step that uses them. Facts from simulation or literature carry `source` and are listed in
§11.

### 6.4 Check API (`src/edu/lessons/engine/checks.js`, Stage A)

#### 6.4.1 Context

```js
/**
 * @typedef {Object} CheckContext
 * @property {ArchitectureGraph} graph      main pane root graph (useArchitectureStore.getState().graph)
 * @property {Object<string, ArchitectureTemplate>} templates
 * @property {TrainingGraph} training
 * @property {Object} scope                 withGlobalParameters(graph)
 * @property {ValidationReport} report      cachedValidate(graph, templates, scope) from domain/architecture/levelContext.js
 * @property {{ tab: TabId, showDimensions: boolean, showGradientFlow: boolean, codeOpen: boolean, codeTab: 'model'|'train',
 *              currentPath: string[], expandedNodeId: string|null }} ui
 * @property {DataSnapshot|null} data       readSource('data')
 * @property {TrainerSnapshot|null} trainer readSource('trainer')
 * @property {{ answers: Object<string, *>, enteredAtSeq: number, visitedTabs: TabId[], tags: Object<string, string> }} step
 * @property {Object<string, number|string>} facts
 */
```

`buildCheckContext()` reads all stores synchronously. `subscribeCheckContext(listener)` subscribes to the
architecture, templates and training stores, the UI store, the session store and every bridge source (also sources
registered later, via `onSourceRegistered`). Checks are **pure functions** of the context. The panel re-evaluates
150 ms after the last change.

**Check result:** `{ pass: boolean, detail?: { key: string, params?: object }, misconception?: string }`. `detail`
explains the first failing sub-check, e.g. `{ key: 'check.param', params: { label: 'neuron', name: 'Output', actual:
1, expected: 3 } }` → "neuron: Output is 1; this step needs 3."

#### 6.4.2 Nodes: types and matches (`engine/nodes.js`)

`nodeType(node, templates)` returns a normalised type:

- **Layer nodes** use the last segment of `libraryKey` (`torch.nn.Conv2d` → `Conv2d`, `deepflow.op.Add` → `Add`).
  Special cases: `deepflow.io.Input` → `Input`, `deepflow.io.Output` → `Output`, `deepflow.nn.CausalSelfAttention` →
  `Attention`, `deepflow.nn.PositionalEmbedding` → `LearnedPositions`, `torch.nn.LayerNorm` → `LayerNorm`.
- **Supernode and repeat nodes** use the template's `standardKey ?? componentStyleKey`: `linear` → `Linear`,
  `embedding` → `Embedding`, `attention` → `Attention`, `attention-head` → `AttentionHead`, `layernorm` →
  `LayerNorm`, `learned-position` → `LearnedPositions`, `sinusoidal-position` → `SinusoidalPositions`. Any other
  template → `Component`.
- **Groups** for matching: `Activation` = ReLU | LeakyReLU | GELU | Tanh | Sigmoid.

So Components → Linear (an editable supernode) and a raw `torch.nn.Linear` layer both match `{ type: 'Linear' }`.

```js
/** @typedef {Object} NodeMatch
 * @property {string|string[]} [type]   normalised type or group
 * @property {string} [id]              fixture-stable id
 * @property {string} [label]           exact node label
 * @property {string} [name]            template name (supernode/repeat)
 * @property {number} [nth]             0-based index among the matches, in topological order of the root graph
 * @property {Object<string, *>} [params]   every listed parameter must equal (after resolution)
 */
```

`paramValue(node, name, { templates, scope })`:

- layer: `resolveParameterValue(node.parameters[name] ?? libraryDefault, scope)`
- supernode: `resolveParameterValue(node.parameterValues?.[name] ?? template freeParameter default, scope)`
- repeat: the binding for iteration 0

Numbers compare with `Object.is` after `Number()` for numeric strings, so `'3'` equals `3`. Shapes compare
element-wise, with `null` equal only to `null`.

#### 6.4.3 Predicates

A `CheckSpec` is an object with exactly one key. All graph predicates look at the **root graph** unless stated.

| Predicate | Passes when |
|---|---|
| `{ all: [C…] }` / `{ any: [C…] }` / `{ not: C }` | combinators |
| `{ visited: TabId }` | that tab was active at some point since step entry |
| `{ tab: TabId }` | that tab is active now |
| `{ answered: id }` / `{ answeredCorrectly: id }` | prediction or quiz answer committed (correct) |
| `{ dataset: id }` | `data.datasetId === id` |
| `{ split: { val?, test? } }` | the split fractions match |
| `{ inputShape: Shape }` | the inferred shape of the first root Input (`report.shapes`) equals Shape |
| `{ outputShape: Shape }` | the shape reaching the first root Output equals Shape |
| `{ valid: true }` | `report.ok` (nested template errors included) |
| `{ has: NodeMatch, count?: n \| { min?, max? }, deep?: bool }` | the number of matching nodes satisfies `count` (default min 1). `deep` also searches templates in use |
| `{ param: { node: NodeMatch, name, equals? \| oneOf? \| min? \| max?, every?: bool } }` | the first matching node satisfies the condition (all matching nodes when `every`) |
| `{ global: { name, equals? \| oneOf? \| min? \| max? } }` | the root global parameter's `default` satisfies it |
| `{ chain: NodeMatch[], allow?: TypeOrGroup[] }` | a directed root path visits nodes matching each element in order. Consecutive elements are adjacent, except that nodes whose type is in `allow` may sit between them |
| `{ edge: { from: NodeMatch, to: NodeMatch } }` | a direct root edge exists |
| `{ shapeAt: { node: NodeMatch, equals: Shape } }` | `report.shapes[node.id]` equals Shape |
| `{ paramCount: { equals? \| min? \| max? } }` | `countArchitectureParameters(graph, templates)` is `complete` and its total satisfies the condition |
| `{ training: { optimizer?, loss?, lossBound?: true, lr?: { min?, max? } } }` | the training graph's effective optimizer and loss (§6.10.5) match; `lossBound` requires `architectureOutputId` to be the root Output |
| `{ run: RunQuery }` | at least one trainer run with `seq > step.enteredAtSeq` satisfies the query |
| `{ runs: { count, lr: number[], dataset?, minEpochs? } }` | completed runs since entry cover every listed learning rate (±1 %) |
| `{ level: { open: NodeMatch } }` | some node on the main pane's `currentPath` matches (resolved level by level) |
| `{ ui: { showDimensions?, showGradientFlow?, codeOpen?, codeTab? } }` | the UI flags match |
| `{ repeat: { node: NodeMatch, mode?, count? } }` | a matching root repeat node has that mode and resolved count |
| `{ component: { contains: (NodeMatch & { count?: number })[], kind?: 'supernode'\|'repeat' } }` | a root supernode or repeat has a template graph containing at least `count` (default 1) nodes of each listed type |
| `{ custom: name }` | `module.checks[name](ctx)` passes |

```js
/** @typedef {Object} RunQuery
 * @property {string} [dataset]
 * @property {number} [minEpochs]            completed epochs (history length − 1)
 * @property {boolean} [initOnly]            an evaluateInit record (0 epochs)
 * @property {{ [metric in 'valAccuracy'|'trainAccuracy'|'valLoss'|'trainLoss']?: { gte?: number, lte?: number } }} [final]
 * @property {{ trainAccuracyGte: number, valLossRiseGte: number }} [overfit]   final train acc ≥ x and final valLoss / min valLoss ≥ r
 * @property {CheckSpec} [model]             evaluated against the run's captured graph, templates and training
 * @property {{ min?: number, max?: number }} [lr]
 */
```

**Global diagnostics** run on every failing task step in addition to `step.diagnose`:

- A Softmax directly before the root Output with cross-entropy loss → `softmax-before-ce`.
- A Sigmoid directly before the root Output with BCE loss → `sigmoid-before-bce`.

#### 6.4.4 Friendly validation feedback (`engine/feedback.js`)

`friendlyIssues(report, graph, templates)` takes the root `report.errors` (including the joined "Template … has N
internal errors" strings) and returns `[{ nodeId, key, params }]`, sorted with the node nearest the Input first. The
panel shows the first one; GraphStatusPanel is not changed. Rules (the regexes match `validator.js` and `shapes.js`
messages as they are today):

| Pattern | Key | Example output |
|---|---|---|
| `Linear in_features=(\d+) but input has (\d+) features`, or `MatMul inner dimensions mismatch: (\d+) vs (\d+)` inside a node of type `Linear` | `fb.linear-in` | "{label} expects {expected} input features but receives {actual}." When the input rank is ≥ 3 and `expected` equals the product of the non-batch axes, use `fb.missing-flatten` instead |
| `Conv2d in_channels=(\d+) but input has (\d+) channels` | `fb.conv-in` | "{label} expects {expected} channels but receives {actual}." |
| `Conv2d expects 4D \(B, C, H, W\); got rank (\d+)` | `fb.conv-rank` | "Conv2d needs images shaped (B, C, H, W). Did Flatten come too early?" |
| `(\w+) is required` | `fb.required` | "Set {param} on {label}." |
| `Split dimension (\d+) must be divisible by (\d+) chunks` | `fb.heads-divide` | "The width {dim} must be divisible by the number of heads ({heads})." |
| `Element-wise dim .* mismatch` | `fb.add-shapes` | "Both inputs of {label} must have the same shape." |
| `(BatchNorm2d\|LayerNorm) .* but input has` | `fb.norm-width` | "{label} expects {expected} features but receives {actual}." |
| anything else | `fb.generic` | the original message |

The rewritten line carries a "Show on canvas" link. It selects the node, opens its inspector and fits the view
(canvas bridge, §6.12).

### 6.5 Actions API (`engine/actions.js`, Stage A)

`runActions(actions, ctx) → Promise<{ ok: boolean, error?: string }>` runs actions in order and awaits each one. It
stops at the first error. Graph-changing actions run inside one `recordArchitectureEdit('Lesson: <label>', …)`, so a
whole Show me is a single Undo step. Actions that replace the workspace (`loadFixture`, `loadStoryboard`) start a
fresh history, as loading an example does. Graph actions always target the main pane's **root** graph: they call
`goToLevel(0)` first, except `openLevel`.

| Action | Arguments | Effect |
|---|---|---|
| `loadFixture` | `id` | replace the workspace with `module.fixtures[id]` (graph, templates, training, data slice, trainer settings, UI flags) |
| `loadStoryboard` | `id`, `configuration?` | run `STORYBOARDS[id].run` on the live stores |
| `useDataset` | `id`, `setInput = true`, `setOutput = true`, `options?`, `split?` | `callSource('data', 'useDataset', …)` |
| `setDatasetOptions` | `options?`, `split?` | `callSource('data', 'setOptions' / 'setSplit')` |
| `setParam` | `node: NodeMatch`, `name`, `value` | layer → `patchLayerParameters`; supernode → `patchNode({ parameterValues })` (merging) |
| `setGlobal` | `name`, `value` | `patchGlobalParameter(name, { default: value })` |
| `insertBetween` | `block: BlockRef`, `from`, `to`, `params?` | remove the edge from→to, insert the block between, connect from→new→to |
| `insertAfter` | `block`, `after`, `params?` | insert the block after `after`; the outgoing edges of `after` now start at the new block |
| `buildChain` | `from`, `to`, `blocks: { block, params? }[]` | insert the blocks in a column and connect `from → b1 → … → bn → to`, replacing any edge from→to |
| `connect` / `disconnect` | `from`, `to`, `targetHandle?` | `connect` / `removeEdge` |
| `remove` | `node` | `removeNode` |
| `group` | `nodes: NodeMatch[]`, `name` | `groupSelection(ids, name)` at the root |
| `repeat` | `node`, `count`, `mode` | `convertInstanceToRepeat(id, count, mode)`, or `patchRepeatNode` if it is already a repeat |
| `openLevel` | `path: NodeMatch[]` | resolve each match in the current level and `openLevel(path)` |
| `setOptimizer` | `type?`, `hyperparameters?` | patch the effective optimizer node (§6.10.5) |
| `setLoss` | `type` | patch the effective loss node |
| `setTrainer` | `epochs?`, `batchSize?`, `seed?` | `callSource('trainer', 'setSettings', …)` |
| `train` | — | `await callSource('trainer', 'start', { sync: ctx.headless })` |
| `evaluateInit` | — | `await callSource('trainer', 'evaluateInit', { sync: ctx.headless })` |
| `answer` | `id`, `value` | commit a prediction answer |
| `setUi` | `tab?`, `showDimensions?`, `showGradientFlow?`, `codeOpen?`, `codeTab?` | UI store and main architecture store flags |

New blocks are positioned in the root column: x = the median x of the root nodes, y = 130 px below the `from`/`after`
node. Nodes below are pushed down 130 px with `moveNodes`, inside the same history entry. A palette chip inserts
through the canvas bridge (`deepflow-edu:canvas` → `insert`), which uses DeepFlow's own non-overlapping placement in
the visible canvas, then selects the new node. `headless: true` is set by tests (§8.4) and makes training run
synchronously.

Show me completes what the learner started. It passes `ctx.entryNodeIds`, the root node ids of the step-entry
snapshot. `insertBetween`, `insertAfter` and `buildChain` then first adopt blocks of the same BlockRef that were added
since the step began and are unwired, or wired only to the action's own blocks (`from`, `to`, `after` and its next
blocks, or other adopted blocks). An adopted block is unwired, moved into the column slot, given the action's
`params` and `label`, and wired like a new one. Blocks of the entry state and blocks wired elsewhere are never
adopted, and setup and fixture actions pass no entry ids. So a palette chip followed by Show me leaves one block,
not a stranded chip block beside a second one.

### 6.6 Spotlight targets (`engine/targets.js`, Stage A)

A `TargetRef` is a string. The registry resolves it to an element at render time. Node targets filter
`.react-flow__node` elements in JavaScript by `dataset.id`, as GraphStatusPanel does, and never interpolate ids into
selectors. The ring goes to the first target that is visible and not yet used (`isTargetDone`): a tab target is used
once the learner visited that tab in this step, `toolbar.code` while the code panel is open, `code.tab:<id>` while
it shows that tab, `data.dataset:<id>` while that row is shown, and a `nodeOpen` target while the main pane shows the
inside of that block at any depth (`reachedOpenTarget`). So M3 s7 rings Data, then the MNIST row, Use this dataset,
Code and train.py in turn. A target counts as visible only with its centre inside the window and inside every
scrolling or clipping box around it, so a block panned off its canvas is missing. When an unused target is missing
for a second, the panel offers "Where is it?", which switches to the step's tab, goes to the root level and centres
a node target, opens the code panel for `code.*` targets, fits the Training canvas beside its palette for
`training.*` targets, and scrolls a list or panel to any other target. It is not offered when every target is used
(M7 s4 at the attention-head level only waits for the answer).

| TargetRef | Element |
|---|---|
| `tab.learn` · `tab.architecture` · `tab.data` · `tab.training` | `.df-toolbar-tabs [data-tab="<id>"]` (A adds `data-tab`) |
| `toolbar.code` | `.df-action-segment[aria-label="Code"]` |
| `ribbon.components` · `ribbon.operations` · `ribbon.group` · `ribbon.sequential` · `ribbon.parameters` · `ribbon.dimensions` · `ribbon.gradients` · `ribbon.instances` | DeepFlow aria-labels: "Browse components", "Insert operations", "Group selection into component", "Repeat selection sequentially", "Edit parameters", "Show dimensions", "Show gradient flow", `.df-ribbon-instance-button` |
| `graph.checks` | `[data-graph-pane="main"] [aria-label^="Graph checks:"]` |
| `node` + NodeMatch | the main pane's node at the current level |
| `nodeOpen` + NodeMatch | that node's `[aria-label="Open <label>"]` button |
| `field:<param>` | `.df-inspector [data-parameter="<param>"]` |
| `data.dataset:<id>` · `data.use` · `data.preview` · `data.options` | `[data-edu-target="dataset-<id>"]`, `"use-dataset"`, `"dataset-preview"`, `"dataset-options"` (B) |
| `trainer.run` · `trainer.lr` · `trainer.curves` · `trainer.boundary` · `trainer.confusion` · `trainer.initLoss` | `[data-edu-target="trainer-run"]`, `"trainer-lr"`, `"trainer-curves"`, `"trainer-boundary"`, `"trainer-confusion"`, `"trainer-init-loss"` (B; present in both TrainerPanel and TrainerCard, the visible one wins) |
| `training.loss` · `training.optimizer` | the training canvas node of the effective loss or optimizer |
| `code.tab:model` · `code.tab:train` | `[data-edu-target="code-tab-model"]`, `"code-tab-train"` (C) |
| `code.notebook` · `code.colabNote` | `[data-edu-target="code-notebook"]` (Download .ipynb), `"code-colab-note"` (the Run in Colab note); both on train.py with a bound dataset (C) |
| `lesson.palette` | the palette box in the lesson panel |

In module files, a node target is written as `{ node: NodeMatch }` or `{ nodeOpen: NodeMatch }`, and the others as
strings.

### 6.7 Session and lesson flow (`engine/session.js`, Stage A)

`useLessonSession` (zustand) holds the mode and the position in the current module:

```js
{
  mode: 'sandbox' | 'lesson',
  moduleId: ModuleId | null,
  phase: 'warmup' | 'steps' | 'quiz' | 'recap',
  stepIndex: number, quizIndex: number,
  collapsed: boolean,
  runtime: { enteredAtSeq, visitedTabs: TabId[], revealShown: boolean },   // per current step; not persisted
  // actions
  startModule(id, { stepId? }), exitLesson(), goToStep(index), next(), back(), skip(), resetStep(),
  useHint(), showMe(), submitPrediction(id, value), answerQuiz(id, optionId), finishModule(),
}
```

- **startModule** captures the current workspace into its slot (`sandbox`, or the previous module), then restores
  the module's slot. If the module has no slot, it builds the `start` fixture. It jumps to the saved `stepId` (or the
  first step not yet passed) and logs `module-start`. A module with warm-up items and no recorded warm-up answers
  starts in `phase: 'warmup'`. It never leaves the learner on Learn: it switches to the step's `tab`, or to
  Architecture when the step has none.
- **Entering a step:** if the step was never entered, run `setup`. Then capture the step-entry snapshot (§6.8.3),
  set `runtime.enteredAtSeq = trainer.seq`, reset `visitedTabs` to the current tab, and log `step-enter`.
- **next()** is allowed when the check passes. It records `passed`; at the last step it moves to `quiz`, after the
  quiz to `recap`. **skip()** records `skipped` and moves on. **back()** goes to the previous step without running
  setup and without touching the workspace.
- **finishModule()** marks the module completed and logs `module-complete`. The module counts as completed when every
  step is `passed` or `skipped` and every quiz item has an answer.
- **exitLesson()** saves the slot, switches to sandbox (restoring its slot), and sets the tab to `learn`.

### 6.8 Persistence (Stage A unless stated)

Every `localStorage` access goes through `engine/storage.js`: `readJSON(key, fallback)` and `writeJSON(key, value)`
wrap `getItem`/`setItem` in try/catch. They fall back to an in-memory map when storage is missing, throws, or is
full. On the first failed write they set `storageDegraded`. The Learn home then shows: "Your browser is not saving
progress right now; it is kept until you close this tab." Edu code never calls `localStorage` directly.

#### 6.8.1 Keys

| Key | Content | Owner |
|---|---|---|
| `deepflow-edu:progress:v1` | progress (§6.8.2) | A |
| `deepflow-edu:session:v1` | `{ mode, moduleId, phase, stepIndex, quizIndex, collapsed }` | A |
| `deepflow-edu:ui:v1` | `{ activeTab }` | A |
| `deepflow-edu:slot:<sandbox\|m1…m7>:v1` | workspace snapshot (§6.8.3) | A |
| `deepflow-edu:step-entry:v1` | `{ moduleId, stepId, snapshot }` for the current step only | A |
| `deepflow-edu:log:v1` | event log ring buffer (§6.8.4) | A |
| `deepflow-edu:data:v1` | live data binding of the active workspace | B |
| `deepflow-edu:trainer:v1` | trainer settings `{ epochs, batchSize, seed }` | B |

DeepFlow's own keys (`deepflow:architecture-graph`, `deepflow:templates`, `deepflow:training-graph`,
`deepflow:seeded` and the viewports) are unchanged. They always hold the **active** workspace.

#### 6.8.2 Progress

```js
{
  version: 1, curriculumVersion: 1, lastModuleId: 'm2' | null,
  modules: {
    m2: {
      moduleVersion: 1, status: 'not-started' | 'in-progress' | 'completed',
      currentStepId: 's3',
      steps: { s1: { status: 'passed' | 'skipped' | 'visited', hintLevel: 0 | 1 | 2 | 3, enteredAt, passedAt } },
      answers: { 'm2.oneNeuron': { value: 'no', correct: true, at } },
      quiz: { 'm2.q1': { first: 'a', correct: true, attempts: 1 } },
      warmup: { 'm2.w1': { first: 'b', correct: true } },
      completedAt?: number,
    },
  },
}
```

Pure reducers in `engine/progress.js` (unit-tested): `enterStep`, `passStep`, `skipStep`, `raiseHint`,
`recordAnswer`, `recordQuiz`, `recordWarmup`, `completeModule`, `moduleStatus`, and `migrate(progress, modules)`. When
`moduleVersion` changes, `migrate` keeps step records whose ids still exist and drops `currentStepId` if that step is
gone.

#### 6.8.3 Workspace snapshots and slots (`engine/workspace.js`)

```js
/** @typedef {{ v: 1, savedAt: number, graph: ArchitectureGraph, templates: Object, training: TrainingGraph,
 *              slices: Object<string, *>, ui?: { showDimensions?: boolean, showGradientFlow?: boolean } }} WorkspaceSnapshot */
captureWorkspace() → WorkspaceSnapshot          // root graph, templates, training graph, every registered slice
restoreWorkspace(snapshot)                      // goToLevel(0); set stores (as ToolbarStrip's Load does); restore slices; clear history
saveSlot(id, snapshot) / loadSlot(id) / deleteSlot(id)
switchToLesson(moduleId) / switchToSandbox()
```

`restoreWorkspace` sets each store with `setState`, then calls `useArchitectureHistory.getState().clear()`, so the
undo stack never crosses workspaces. Fixture helpers (`engine/fixtures.js`) build snapshots without touching the
stores:

- `input(id, shape, opts)`, `output(id, opts)`, `layer(id, libraryKey, params, label)`, `linear(id, in, out, opts)`
  (uses the standard Linear component through `ensureStandardTemplate` on an in-memory templates store), `chain(...)`
- `optimizer(type, hp)`, `loss(type, outputId)`, `data(datasetId, options, split)`, `trainer(settings)`
- `snapshot({ nodes, edges, templates, training, data, trainer, ui })`

Nodes are placed at x = 400, y = row × 130.

#### 6.8.4 Event log (`engine/log.js`)

Ring buffer of the last 2,000 events: `{ t: epochMs, type, module, step?, data? }`. Types: `module-start`,
`step-enter`, `step-pass`, `step-skip`, `hint` {level}, `show-me`, `reset-step`, `predict` {id, value, correct},
`warmup` / `quiz` {id, option, correct, attempt}, `run` {dataset, epochs, finalValAccuracy, finalValLoss},
`code-open` {tab}, `module-complete`. "Download learning log" exports `{ product: PRODUCT_NAME, curriculumVersion,
exportedAt, progress, events }`. Nothing leaves the browser otherwise.

### 6.9 Dataset registry (Stage B; image and text presets Stage C)

#### 6.9.1 Spec

```js
/** @typedef {Object} DatasetSpec
 * @property {string} id
 * @property {string} name                   English UI name; translatable later through `i18n?: { [locale]: { name, description } }`
 * @property {string} description            one sentence
 * @property {'toy2d'|'tabular'|'image'|'text'} kind
 * @property {'browser'|'export'} training
 * @property {{ shape: string, dtype: 'float32'|'int64' }} input        Input node values, e.g. '(2,)', '(1, 28, 28)', '(64,)'
 * @property {{ kind: 'classes'|'regression'|'next-token', classes?: string[], size: number }} target   size = number of classes, 1, or the vocabulary size
 * @property {{ train?: number, test?: number, total?: number, note?: string }} size
 * @property {{ name: string, url?: string, verified: boolean, note?: string }} licence
 * @property {string} attribution
 * @property {string[]} citation
 * @property {Object<string, { default: number, min?: number, max?: number, step?: number, label: string }>} [options]
 * @property {'none'|'standardize'|'divide16'} [scaling]                 browser datasets only
 * @property {(options) => Promise<RawDataset>} [load]                   browser datasets only; may dynamic-import data
 * @property {(ctx: PythonLoaderContext) => PythonLoader} python         codegen hook (§6.11.1)
 * @property {{ mean: number[], std: number[] }} [normalization]         export image sets (§5.1)
 * @property {{ epochs: number, batchSize: number, stepsPerEpoch?: number }} [exportDefaults]   export sets: used by train.py instead of the browser trainer settings
 */
/** @typedef {{ x: Float64Array, y: Int32Array|Float64Array, n: number, featureShape: number[], featureNames?: string[] }} RawDataset */
/** @typedef {{ spec: DatasetSpec, train: Split, val: Split, test: Split, featureShape: number[], numClasses: number|null,
 *              classNames: string[]|null, scaling: { mean: number[], std: number[] } | null,
 *              bounds: [[number, number], [number, number]] | null }} PreparedDataset */
/** @typedef {{ x: Float64Array, y: Int32Array|Float64Array, n: number, indices: Int32Array }} Split */
```

`registry.js` exports `DATASETS` (in display order: blobs, circle, xor, spiral2, spiral3, plane?, iris, digits,
mnist, fashion-mnist, cifar10, tiny-shakespeare), `getDataset(id)` and `browserDatasets()`. It is built from
`presets/index.js`, which concatenates `toy.js`, `tabular.js`, `digits.js`, `images.js` (C) and `text.js` (C).

`prepare.js` exports `prepareDataset(spec, options, split, dataSeed) → Promise<PreparedDataset>`. It loads, splits
(§5.3) and scales (fitting on train only), and caches by `(id, options, split, seed)`. `bounds` is set for 2D inputs
from all points plus a 10 % margin, made square.

#### 6.9.2 Data store (`data/store.js`)

```js
useEduDataStore: {
  datasetId: string | null,
  options: { [datasetId]: object },     // last-used options per dataset (defaults from the spec)
  split: { val: 0.2, test: 0 },
  setDataset(id), setOptions(patch), setSplit(patch),
  current(): { spec, options, split, dataSeed },   // dataSeed = options.seed ?? 1
}
```

It registers bridge source `'data'`: `read() → { datasetId, spec: { id, name, kind, training, input, target } | null,
options, split }`, with `subscribe` and actions `useDataset`, `setDataset`, `setOptions` and `setSplit`. It also
registers slice `'data'` (`capture → { datasetId, options, split }`).

#### 6.9.3 Toy and tabular presets

`toy.js` holds the generators of §5.3 with their option schemas: `n` 100–1000 step 50, `noise` 0–0.5 step 0.05,
`seed` 0–9999, `perClass` 30–300 for `spiral3`. `tabular.js` is `iris` (`irisData.js`: 150 × 4 numbers + labels,
with the sklearn and Fisher citation header). `digits.js` is `digits`: `digitsData.js` stores 1,797 × 64 values
0–16, encoded as base-36 characters (1 char per pixel, `'0'`–`'g'`), plus labels. It is loaded with a dynamic
`import()`, so it stays out of the main bundle (< 120 KB raw).

#### 6.9.4 `useDataset(id, { setInput = true, setOutput = true, options, split }) → { changed: string[], warnings: string[] }`

1. `setDataset(id)`, and apply `options` and `split` if given.
2. Inside `recordArchitectureEdit('Use dataset <name>', …)` on the **root** graph (`useArchitectureStore`, pure
   `patchLayerParameters` / `patchNode` on `state.graph`, never the current level):
   - `setInput`: set the first root `deepflow.io.Input` node's `shape` to `spec.input.shape` and `dtype` to
     `spec.input.dtype`.
   - `setOutput`: walk back from the first root Output through nodes of type Activation, Dropout, Softmax and
     LogSoftmax. If the walk reaches a Linear (layer or component), set its `out_features` to `spec.target.size`,
     except 2-class datasets: set 1 if the effective loss is BCE, else 2. Otherwise add the warning
     "Could not set the output size: the last block is not a Linear."
3. Return the human-readable list of changes, which the Data tab shows inline.

### 6.10 In-browser trainer (Stage B)

#### 6.10.1 Pipeline

`store.start()` → `preflight()` → `compileModel({ graph, templates })` → `prepareDataset(...)` →
`createRun({ program, dataset, objective, settings })` → `runEpochs()`, sliced, until done or stopped. The pure parts
(`compile`, `createRun`, `runEpochs`) take no stores and run in Node for tests.

#### 6.10.2 Compile (`compile.js`)

`compileModel({ graph, templates }) → { ok, program, issues, inputShape, outputShape, paramCount }`.

- Requires `validate(graph, { templates }).ok`. Otherwise the result is an `invalid-graph` error carrying the friendly
  messages.
- It expands the graph the same way `src/codegen/architecture.js` does. Supernodes are inlined with
  `buildParameterScope(template.freeParameters, node.parameterValues, scope, undefined, childInputShapes)`; derived
  parameters get their input shapes from a per-level `validate` call. Sequential repeats are unrolled, with an
  iteration scope `{ ...scope, [iterationVariable || 'i']: index }` and `effectiveTemplateId(node, index)` for local
  versions. Edge gradient operators apply: `detach` / `stop` → stop-gradient; `reverse` → gradient × (−λ).
- Exactly one root Input and one root Output are required (else `multi-input` / `multi-output`). The batch axis is
  implicit.

Supported operations (anything else → error `unsupported-op`: "<label> (<Type>) is not supported by the browser
trainer. Use Code → train.py to train this model in PyTorch."):

| Library key | Forward | Notes |
|---|---|---|
| `deepflow.io.Input/Output`, `deepflow.template.Input/Output` | identity | |
| `torch.nn.Linear` | x Wᵀ + b | input rank 2 only. Init W ~ U(−1/√in, 1/√in) and b ~ U(−1/√in, 1/√in), PyTorch's default (`kaiming_uniform_(a=√5)`) |
| `torch.nn.ReLU`, `LeakyReLU(negative_slope)`, `Tanh`, `Sigmoid` | element-wise | |
| `torch.nn.GELU(approximate)` | `'none'`: x·Φ(x) with erf from Abramowitz & Stegun 7.1.26 (abs. error ≤ 1.5·10⁻⁷); `'tanh'`: the tanh formula | |
| `torch.nn.Softmax(dim)`, `torch.nn.LogSoftmax(dim)` | over the last axis | only dim −1 or the last axis |
| `torch.nn.Dropout(p)` | train mode: inverted dropout (keep with prob 1−p, scale 1/(1−p)); eval mode: identity | `'dropout'` stream |
| `torch.nn.Flatten(start_dim, end_dim)` | reshape | only start 1 and end −1 |
| `deepflow.op.Parameter` | a learned tensor | `init`: `normal(0, scale)`, `uniform(−scale, scale)`, `kaiming_uniform` (bound 1/√fan_in, fan_in = shape[1]), `zeros`, `ones`. `requires_grad: false` → frozen. `enabled: false` → constant `disabled_init` |
| `deepflow.op.Transpose(dim0, dim1)` | swap the last two axes | rank 2, dims {−2, −1} or {0, 1} |
| `deepflow.op.MatMul(a, b)` | (n×k)·(k×m) | rank 2 only |
| `deepflow.op.Add`, `Multiply` (n inputs), `Subtract(a, b)` | NumPy broadcasting | |
| `deepflow.op.Scale(factor)`, `AddScalar(value)`, `Pow(exponent)`, `Exp`, `Rsqrt` | element-wise | |
| `deepflow.op.Mean(dim −1, keepdim)` | last axis only | |

With these, the standard **Linear** and **Norm (LayerNorm)** components train as their editable op graphs, so the
browser trains the same graph the learner sees and exports. Not supported in v1 (error `unsupported-op`): Conv*,
pooling, BatchNorm*, GroupNorm, `torch.nn.LayerNorm` (layer form), Embedding / Gather, attention, recurrent,
TransformerEncoderLayer, Split / Concat / Stack / Broadcast, parallel repeats.

`program` is `{ inputShape, outputShape, ops: [{ id, kind, inputs: string[], attrs, origin: { rootNodeId, path } }],
params: [{ id, shape, init, scale, requiresGrad, enabled, disabledInit, origin }] }`. Limits: total parameters
≤ 2,000,000 and activations ≤ 4·10⁶ floats per batch, else error `too-large`.

#### 6.10.3 Tensor engine (`tensor.js`, `ops.js`)

Dense `Float64Array` tensors `{ shape: number[], data }`, row-major. The tape records each forward op with a
`backward(gradOut)` closure. `backward(loss)` walks the tape in reverse and accumulates `.grad`. The tape covers only
the ops in the table above, each with a hand-written gradient checked by finite differences (§8.2). Float64 is used
for determinism and simplicity; differences from PyTorch's float32 are far below the metric thresholds used in
lessons.

#### 6.10.4 Losses (`losses.js`, reduction `mean`, PyTorch semantics)

- `cross_entropy`: −log softmax(logits)[y], averaged over the batch, computed with log-sum-exp. `label_smoothing` ε:
  (1−ε)·NLL + ε·mean over classes of −log p. Logits (B, C), targets int in [0, C).
- `bce` (BCEWithLogitsLoss): mean of max(x, 0) − x·y + log(1 + e^(−|x|)). Logits (B, 1), targets {0, 1}.
- `mse`: mean((x − y)²). `l1`: mean(|x − y|).
- `custom`: error `custom-loss`. A non-null `weight` hyperparameter is ignored with warning `loss-weight-ignored`.
- Target checks: `cross_entropy` needs output size = number of classes ≥ 2; `bce` needs output size 1 and a binary
  dataset; `mse` / `l1` need a regression dataset or output size 1. Otherwise error `loss-target-mismatch`, with a
  text that names the fix ("Iris has 3 classes: set the last Linear's Output to 3 or use the Data tab's Use this
  dataset").

#### 6.10.5 Objective from the training graph (shared rule, used by trainer, checks and codegen)

`effectiveObjective(trainingGraph, rootGraph) → { optimizer: OptimizerNode | null, loss: LossNode | null, issues }`:

1. Losses are the loss nodes whose `architectureOutputId` is the root Output id. Several losses → error
   `multi-loss`. None → if the graph has exactly one loss node, use it with warning `loss-unbound`; else the
   default.
2. The optimizer is the one connected from that loss by an edge. Else, if the graph has exactly one optimizer, that
   one; else the default.
3. Any phase node → error `multi-phase` ("Phases train only in the exported code").
4. Defaults when missing: Adam (lr 0.01, other values from `OPTIMIZER_DEFAULTS`). The loss is cross-entropy for
   datasets with ≥ 3 classes or 2 classes with output size 2, BCE for binary with output size 1, and MSE for
   regression. The trainer notes "No optimizer/loss in the Training tab: using Adam (lr 0.01) and <loss>."

This rule is the contract file `src/edu/objective.js` (§6.2), so the trainer (B), the checks and actions (A) and
codegen (C) all agree.
`optimizer-ambiguous` (several optimizers, none linked to the loss) falls back to the default optimizer, with a
warning.

Optimizers (`optimizers.js`). Hyperparameter names follow DeepFlow (`learning_rate`, `weight_decay`, `betas` as the
string "(0.9, 0.999)", `eps`, `momentum`, `nesterov`, `alpha`). Update rules follow PyTorch:

- **SGD**: g ← g + λp; if μ ≠ 0: b ← g on the first step, else b ← μb + g; g ← (nesterov ? g + μb : b); p ← p − lr·g.
- **Adam**: g ← g + λp; m ← β₁m + (1−β₁)g; v ← β₂v + (1−β₂)g²; p ← p − lr·(m/(1−β₁ᵗ)) / (√(v/(1−β₂ᵗ)) + ε).
- **AdamW**: p ← p(1 − lr·λ), then the Adam step without the λp term.
- **RMSprop**: g ← g + λp; v ← αv + (1−α)g²; if μ > 0: b ← μb + g/(√v + ε), p ← p − lr·b; else p ← p − lr·g/(√v + ε).
  ε defaults to 1e-8.
- A non-empty `parameterGroup` trains only the parameters whose `origin.rootNodeId` is in the group. The others are
  frozen for the run.

#### 6.10.6 Preflight codes

Errors block Train; warnings do not. `no-dataset`, `dataset-export-only` ("MNIST is too large to train in the
browser. Open Code → train.py and run it in Colab."), `invalid-graph`, `unsupported-op`, `multi-input`,
`multi-output`, `input-mismatch` ("Your Input is (2,), but Iris has 4 features: use Data → Use this dataset"),
`loss-target-mismatch`, `multi-loss`, `multi-phase`, `custom-loss`, `too-large` (errors). `softmax-before-ce`,
`sigmoid-before-bce`, `loss-unbound`, `loss-weight-ignored`, `defaults-used` (warnings).

#### 6.10.7 Runs, metrics and determinism (`run.js`, `metrics.js`)

- **Epoch 0** evaluates the untrained model. After each epoch, all metrics are computed in **eval mode** (dropout
  off) over the full train and validation splits: `trainLoss`, `valLoss`, `trainAccuracy`, `valAccuracy`
  (classification), plus `ms`. Loss values include no regularisation term. Weight decay acts in the optimizer, as in
  PyTorch.
- `best = { epoch, valLoss }` is the epoch with the lowest validation loss. The final `confusion` matrix is
  computed on the validation split (train split if val is empty). `boundary` probabilities on the 64 × 64 grid are
  computed at most every 100 ms during a run and at the end.
- **Determinism:** the same graph, templates, training graph, dataset options, split and settings give bit-identical
  histories in the same JS engine. Streams: init = `deriveSeed(seed, 'init')` with parameters initialised in
  program order; batches and dropout use their own streams.
- **Time slicing:** the store runs epochs in slices of ≤ 12 ms, yielding with `setTimeout(0)`, and pushes UI updates
  at most 10 times per second. `start({ sync: true })` runs to completion in one call (tests, headless Show me).
  Budget on a 2020 laptop: circle MLP (2-8-1, 300 points, 50 epochs) < 0.5 s; spiral3 overfit (2-128-128-3, 75
  points, 300 epochs) < 4 s; digits MLP (64-128-10, 1,437 points, 15 epochs) < 4 s.

`RunRecord`:

```js
{ id, seq, tag?, status: 'completed' | 'stopped' | 'failed', error?, startedAt,
  datasetId, dataOptions, split, settings: { epochs, batchSize, seed },
  optimizer: { type, hyperparameters }, loss: { type, hyperparameters },
  graph, templates, training,                     // immutable references captured at start
  paramCount, history: EpochMetrics[], best: { epoch, valLoss } | null,
  confusion: number[][] | null, classNames: string[] | null }
EpochMetrics = { epoch, trainLoss, valLoss, trainAccuracy, valAccuracy, ms }   // accuracies null for regression
```

#### 6.10.8 Store (`store.js`)

```js
useTrainerStore: {
  settings: { epochs: 50, batchSize: 32, seed: 1 },   // persisted (deepflow-edu:trainer:v1)
  status: 'idle' | 'running' | 'stopping' | 'error',
  seq: 0,                         // +1 whenever a run (or init evaluation) starts
  current: RunRecord | null,
  runs: RunRecord[],              // completed and stopped runs, most recent last, at most 8
  preflight: { ok, errors: Issue[], warnings: Issue[] },   // recomputed 200 ms after graph, data or training changes
  setSettings(patch), start({ sync = false } = {}) → Promise<RunRecord>, stop(), evaluateInit({ sync }) → Promise<RunRecord>,
  tagRun(runId, tag), clearRuns(),
}
```

It registers bridge source `'trainer'` (`read → { status, seq, settings, current, runs, preflight }`, `subscribe`,
actions `start`, `stop`, `evaluateInit`, `setSettings`, `tagRun` and `clearRuns`) and slice `'trainer'` (`capture →
{ settings }`; `restore` also clears `runs`).

### 6.11 Code export (Stage C)

#### 6.11.1 Python loader hook (every DatasetSpec)

```js
/** @typedef {{ options: object, split: { val: number, test: number }, batchSize: number, dataSeed: number, trainSeed: number }} PythonLoaderContext */
/** @typedef {{ imports: string[], source: string, task: 'binary'|'classification'|'regression'|'next-token',
 *              numClasses?: number, classNames?: string[], vocabNote?: string, comment: string }} PythonLoader */
```

`source` defines `build_dataloaders(batch_size, data_seed, train_seed)`, which returns `(train_loader, val_loader,
test_loader_or_None)`. Loaders yield `(x, y)`:

- x is float32 shaped `(B, *input)`, or int64 `(B, T)` for text.
- y is int64 class indices for binary and classification, float32 `(B, 1)` for regression, int64 `(B, T)` for
  next-token.

Shared Python helpers (Stage B, `data/python/common.js`) export source strings:

- `PY_PRNG`: `mulberry32`, `derive_seed`, `normal`, `uniform`, bit-identical to §5.3
- `PY_SPLIT`: `split_indices(n, val, test, seed)`
- `PY_LOADERS`: `make_loaders(X, y, splits, batch_size, train_seed)`, using `TensorDataset` + `DataLoader(shuffle=True,
  generator=torch.Generator().manual_seed(train_seed))` for train and `shuffle=False` for val/test

Per-dataset loaders:

- **toy (B)**: the generator ported line by line from §5.3, then the same split.
- **iris (B)**: `from sklearn.datasets import load_iris`, then `split_indices`, then standardise with the train mean
  and std (`X_train.mean(0)`, `X_train.std(0)`; ddof 0).
- **digits (B)**: `load_digits()`, `X / 16.0` reshaped to (N, 1, 8, 8), then the same split.
- **mnist / fashion-mnist / cifar10 (C)**: `torchvision.datasets.<Name>(root='data', train=True/False,
  download=True, transform=Compose([ToTensor(), Normalize(mean, std)]))`, then
  `random_split(train, [n − valSize, valSize], generator=torch.Generator().manual_seed(data_seed))`. The test loader
  is the official test set.
- **tiny-shakespeare (C)**: download `input.txt` with `urllib.request` (cached in `data/`). `chars =
  sorted(set(text))`, `stoi` and `itos`, encode as a `torch.long` tensor, 90/10 split as nanoGPT. Datasets yield
  random windows (`x = d[i:i+T]`, `y = d[i+1:i+T+1]`) with 200 batches per "epoch" for train and 50 for val, sampled
  with a seeded `torch.Generator`.

#### 6.11.2 `train.py` structure (`codegen/edu/trainScript.js`)

`generateTrainingScript(archGraph, trainGraph, { templates, data })` where `data = { spec, options, split,
dataSeed, settings: { epochs, batchSize, seed } }`. For export datasets, `settings.epochs` and `batchSize` come from
`spec.exportDefaults`: MNIST and Fashion-MNIST 5 epochs × batch 64; CIFAR-10 10 × 128; Tiny Shakespeare 10 × 32,
with 200 steps per epoch. Browser datasets use the trainer settings, so browser and export train for the same
number of epochs:

- `data` absent → today's output, byte-identical (a regression test pins it).
- `data` present, in this order:
  1. header comment (dataset, licence, attribution, "Runs in Colab")
  2. the model source (`generateArchitectureModule`)
  3. imports
  4. constants `SEED`, `DATA_SEED`, `EPOCHS`, `BATCH_SIZE`, `DEVICE`
  5. the loader source
  6. `make_optimizer(model)` from the effective optimizer (§6.10.5; `parameterGroup` as today's codegen); `loss_fn`
     from the effective loss
  7. `compute_loss(logits, y)`: CE `loss_fn(logits, y)`; BCE `loss_fn(logits, y.float().view_as(logits))`; MSE/L1
     `loss_fn(logits, y.view_as(logits))`; next-token `loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))`
  8. `accuracy(logits, y)`: argmax for CE, `logits > 0` for BCE, per-position argmax for next-token, none for
     regression
  9. `evaluate(model, loader)` under `model.eval()` and `torch.no_grad()`
  10. `train()`: `torch.manual_seed(SEED)`. Before the loop it prints the initial validation loss and the expected
      `ln C` for classification. Each epoch runs `model.train()` batches, then prints
      `epoch {e:3d}  train_loss {..:.4f}  val_loss {..:.4f}  val_acc {..:.3f}`. It reports the best-validation epoch,
      then evaluates the test split once if present.
  11. For next-token, `generate(model, start='\n', length=300, temperature=0.8)` prints a sample. Its context is
      cropped to the model's T.
  12. `if __name__ == "__main__": train()`
- **Phases** (multi-optimizer) → the script keeps today's phase step logic inside the new loop and adds the comment
  "# Multi-phase training: see DeepFlow's phase graph."
- Every generated script must pass `python3 -c "import ast,sys; ast.parse(sys.stdin.read())"`.

#### 6.11.3 CodePanel (`src/components/CodePanel.jsx`)

- Props: `{ open, onClose, tab, onTabChange }`. Tab buttons carry `data-edu-target="code-tab-model"` and
  `"code-tab-train"`.
- It reads the data binding with `readSource('data')` (Stage B's store via the bridge) and the trainer settings with
  `readSource('trainer')`, subscribing to both, then builds `data` for the codegen. Without a bound dataset, `data`
  is `undefined`.
- Buttons: copy, download (`model.py` / `train.py`), and **Download .ipynb** (`codegen/edu/notebook.js`, nbformat
  4.5):
  - cell 1 is `train.py` without its final `if __name__ == "__main__":` block. In Jupyter `__name__` is
    `"__main__"`, so keeping the guard would train twice.
  - cell 2 is `model = train()`.
- `useEduUiStore.codeTab` mirrors `tab`, so checks can read it.

#### 6.11.4 Parameter counting fixes (`src/domain/optimization/parameterCount.js`)

- **Bug** (verified on `15c7c25`): `Conv{1,2,3}d` with a scalar `kernel_size` multiplies by k instead of k^d.
  Conv2d(1→8, k=3) counts 32 instead of 80, and the M4 CNN totals 8,282 instead of 9,098. **Fix:** expand a scalar
  kernel to d axes (1, 2 or 3 from the key). Tuples are unchanged.
- **New export** `countParametersByNode(graph, templates, outerScope = {}) → { [rootNodeId]: { total, trainable,
  complete } }`. It sums each root node's expanded parameters, using `countArchitectureParameters`' rules.
- The existing GPT-2 assertions (163,037,184, and 7,087,872 per block) must keep passing.

### 6.12 Shell integration (Stage A)

- **`useEduUiStore`** (`shell/uiStore.js`): `{ activeTab: 'learn', codeOpen: false, codeTab: 'model', setTab,
  openCode(tab), closeCode, setCodeTab }`. Only `activeTab` is persisted. Visiting a tab appends it to the session's
  `runtime.visitedTabs`.
- **`Layout.jsx`**:
  - `activeTab` and code state come from the UI store, and the default model seeder is kept.
  - Structure: `.df-app` → Navbar → ToolbarStrip → `.df-app-main.edu-main` (flex row) → `.edu-stage` (flex 1,
    `position: relative`; holds the tab content, `TrainingPalette` and `CodePanel`) + `LessonPanel` (when `mode ===
    'lesson' && activeTab !== 'learn'`).
  - Tabs: `learn` → `LearnHome`; `architecture` → `ArchitectureWorkspace`; `data` → `DataWorkspace`; `training` →
    `.edu-training-split` = `TrainingCanvas` (top) + `TrainerPanel` (bottom).
  - `document.title = \`${PRODUCT_NAME} — ${PRODUCT_TAGLINE}\`` on mount.
- **`ToolbarStrip.jsx`**:
  - Tabs `learn`, `architecture`, `data` (enabled), `training`, each `<button data-tab=…>` and labelled from
    `text/en/ui.js`.
  - Save adds `edu: { version: 1, slices: Object.fromEntries(workspaceSlices().map(([k, s]) => [k, s.capture()])) }`,
    and Load restores it when present. A file without `edu` loads exactly as today.
  - Reset also restores every slice to `undefined` (unbound dataset, default trainer settings). In lesson mode its
    confirm text says it resets this lesson's workspace, and progress is kept.
- **`Navbar.jsx`**: the brand text is `PRODUCT_NAME` and the Brain icon stays.
- **`StoryboardMenu.jsx`**: in lesson mode, choosing an example asks "Open the example in your sandbox? Your lesson
  is saved." On OK it calls `switchToSandbox()`, then runs the storyboard.
- **`ArchitectureWorkspace.jsx`** (canvas bridge, ≤ 15 lines): it listens for
  `window.dispatchEvent(new CustomEvent('deepflow-edu:canvas', { detail }))` with:
  - `detail.command === 'fit'` → `commands.current.main?.fitView()`
  - `'insert'` → `detail.result = commands.current.main?.insert(detail.entry, detail.isTemplate)`
  - `'focusNode'` → select the node, expand it, and center the view on it

  `engine/canvasBridge.js` wraps these as `requestFit()`, `requestInsert(entry)` and `focusNode(id)`. Dispatch is
  synchronous, so `result` is available right after the call.
- **`main.jsx`**: `window.__deepflow = { ...window.__deepflow, architectureStore, …, edu: eduDevHandles }`, still
  only in development.
- **`index.html`**: `<title>DeepFlow Edu — Learn to build neural networks</title>`. A test asserts that it equals
  `PRODUCT_NAME + ' — ' + PRODUCT_TAGLINE`, because static HTML cannot import the constant.
- **Deep links** (`shell/deepLink.js`): `#/learn` opens Learn, `#/learn/m3` starts or resumes module 3, and
  `#/learn/m3/s4` opens step 4 (marking earlier unseen steps `visited`, not passed). The engine writes the hash with
  `history.replaceState` on step change.

---

## 7. Build stages and file ownership

### 7.1 Rules for every stage

1. **One owner per file.** A stage edits only files it owns (§6.1). The only exceptions are the contract files and
   placeholders of §6.2, under the rules stated there. If a stage needs a change in another stage's file, it records
   the request in its report. It does not make the change.
2. **No regressions.** Before every commit run `npm test`, `npm run lint` and `npm run build`; all three must be
   green. The fork baseline is 281 pass and 21 skipped. A stage may update an existing (non-`edu-`) test only when
   its owned core change requires it, and it must say why in the commit message.
3. **Commits.** Commit locally on `deepflow-edu`, adding only your own paths:
   `git add <paths> && git commit -m "…" -- <paths>`. Retry if `.git/index.lock` is busy. Never push. Every commit
   message ends with:
   ```
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_016y4tDWPLbkRh3nF1vrF99V
   ```
4. **Browser work** uses this worktree's dev server on port 5174 (`npx vite --host 127.0.0.1 --port 5174
   --strictPort`). Playwright runs with `chromium.launch({ executablePath: '/opt/pw-browsers/chromium' })`. Scratch
   scripts and screenshots go to `/tmp/claude-0/-home-user-deepflow/7a8350aa-ff30-52a8-8452-b7a05647a64e/scratchpad/edu/`,
   not the repo. Only real tests go in `scripts/`.
5. **Zero new dependencies** (§6.1).
6. **Missing dependencies on other stages.** Code against this spec. A test that needs another stage's module checks
   for the file with `existsSync` and uses `test(name, { skip: 'needs Stage X: <file>' }, …)`. Stage E makes sure no
   such skip remains.
7. **Report.** Each stage ends with a short report: files, tests added, deviations from the spec with reasons, and
   open requests to other stages.

### 7.2 Stages

| Stage | Scope | Owns | Depends on |
|---|---|---|---|
| **A** Lesson engine, Learn UI, shell | contract files; engine (schema, nodes, checks, feedback, actions, context, session, progress, storage, workspace, fixtures, i18n, log, targets, canvas bridge); Learn home; lesson panel and its widgets; UI store; deep links; shell edits | `src/edu/config.js`, `src/edu/bridge.js`, `src/edu/objective.js`, `src/edu/shell/**`, `src/edu/lessons/engine/**`, `src/edu/components/**`, `src/App.jsx`, `src/main.jsx`, `index.html`, `src/components/{Layout,Navbar,ToolbarStrip,StoryboardMenu}.jsx`, the canvas-bridge lines of `src/components/architecture/ArchitectureWorkspace.jsx`, `scripts/edu-engine*.test.mjs`, `scripts/edu-shell*.test.mjs`, `scripts/edu-learn-ui*.test.mjs`, `scripts/fixtures/edu-test-module.js`, `scripts/edu-learn-browser-check.mjs` | spec only. Uses B and D through the bridge and placeholders |
| **B** Trainer and Data tab | dataset registry core; seeded generators, split, scaling; toy, Iris and digits presets with their Python loaders; data store; `useDataset`; Data tab UI; trainer (compile, tensor engine, losses, optimizers, metrics, boundary, runs, store); TrainerPanel and TrainerCard; LogSoftmax library entry | `src/edu/data/**` except `presets/images.js` and `presets/text.js`; `src/edu/trainer/**`; the LogSoftmax lines in `src/domain/library/{pytorchBlocks,nodeFamilies}.js` and `src/validation/shapes.js`; `scripts/edu-data*.test.mjs`, `scripts/edu-trainer*.test.mjs`, `scripts/edu-python-parity.test.mjs`, `scripts/edu-data-browser-check.mjs` | spec only |
| **C** Export presets and codegen | MNIST, Fashion-MNIST, CIFAR-10 and Tiny Shakespeare presets with Python loaders; `train.py` assembly; `.ipynb`; CodePanel; parameter-count fix and per-node counts | `src/edu/data/presets/{images,text}.js`, `src/codegen/training.js`, `src/codegen/edu/**`, `src/components/CodePanel.jsx`, `src/domain/optimization/parameterCount.js`, `scripts/edu-codegen*.test.mjs`, `scripts/edu-presets*.test.mjs`, `scripts/edu-param-count.test.mjs` | B's `DatasetSpec` shape (§6.9.1) and `data/python/common.js` (§6.11.1) |
| **D** Lesson content | 7 module definitions, fixtures, facts, custom checks, English text, the misconception catalogue, the module registry | `src/edu/lessons/modules/**`, `scripts/edu-content*.test.mjs` | schema and DSL (§6.3–6.6). Solvability tests need A's engine and B's trainer; the M4/M6 parameter facts need C's Conv fix |
| **E** Integration, QA, README | wire-up fixes, the full browser walkthrough of all 7 modules with Show me, accessibility and responsive checks, performance, QA report, README section, spec change log | any file, for minimal integration fixes (each listed in `docs/edu/QA.md`); `README.md`; `docs/edu/**`; `scripts/edu-walkthrough-browser-check.mjs` | A, B, C, D |

**Order.** A and B start in parallel. C starts once B has committed `registry.js`, `presets/index.js` and
`python/common.js`, or earlier, coding against the spec, with skips. D starts in parallel and writes content first,
then turns its solvability tests on when A and B land. E runs last.

**Shared-file rule (summary).** Stage A mounts the Data tab by importing `src/edu/data/DataWorkspace.jsx`, and the
trainer by importing `TrainerPanel.jsx` / `TrainerCard.jsx`. It may create a placeholder only if the file does not
exist (§6.2). Stage B never edits Layout, Navbar, ToolbarStrip or any shell file. It exposes everything through its
mount points, the bridge sources `'data'` and `'trainer'`, and `data-edu-target` attributes.

### 7.3 Definition of done per stage

- **A:**
  - On a fresh profile the app opens on Learn with 7 cards (or the placeholder registry's empty state).
  - A test module in `scripts/fixtures/edu-test-module.js` runs end to end in jsdom: warm-up, steps with each kind,
    hints, Show me, Reset step, quiz, recap.
  - Progress survives a reload. Storage that throws does not crash the app.
  - `PRODUCT_NAME` is the only place the name appears in `src/` (§8.1).
  - The canvas has no new text.
- **B:**
  - Every browser dataset generates deterministically and matches its Python loader (parity test).
  - The trainer reaches the §11 thresholds for every lesson configuration, in Node.
  - Use this dataset is a single undo step.
  - The Data and Training tabs render, with every `data-edu-target` listed in §6.6.
  - Unsupported graphs give the specified messages.
- **C:**
  - `train.py` for every registered dataset (11, or 12 with `plane`) parses as Python.
  - With PyTorch installed, the toy, Iris and digits scripts run one epoch end to end (skipped otherwise).
  - The legacy output is byte-identical without `data`.
  - The Conv count fix is covered by tests.
  - CodePanel works in both modes.
- **D:**
  - All 7 modules pass `validateModule`.
  - Every step is solvable by its Show me from its entry state, in sequence, headless.
  - Every fact is recomputed and matches. No text key is missing or unused.
  - The copy is within the length limits (§3.8).
- **E:**
  - All of the above hold together, with zero skips in `edu-*` tests.
  - Browser walkthrough of M1–M7 at 1440×900 and 390×844, no console errors, screenshots in the scratch folder.
  - README section and QA report written.

---

## 8. Tests

All `scripts/edu-*.test.mjs` files run under `npm test` (`node --test`, no browser). Node tests install an in-memory
`localStorage` before importing stores, as the existing tests do (see `scripts/startup-regressions.test.mjs`). React
tests follow `scripts/ui-regressions.test.mjs`:

- bundle the components with esbuild (`packages: 'external'`, `jsx: 'automatic'`, `loader: { '.css': 'empty' }`)
  into a temporary `.edu-ui-*` directory in the project root, and delete it in `after()`
- render with jsdom

Browser checks (`*-browser-check.mjs`) are run by hand or by Stage E against port 5174 and are not part of
`npm test`.

### 8.1 Stage A

- `edu-shell.test.mjs`
  - `PRODUCT_NAME` single source: scan `src/**/*.{js,jsx}` except `src/edu/config.js`; the literal `DeepFlow Edu`
    must not occur. `index.html`'s `<title>` equals `PRODUCT_NAME + ' — ' + PRODUCT_TAGLINE`.
  - Tabs render in the order Learn, Architecture, Data, Training. Data is enabled. Each has `data-tab`.
  - The Save payload includes `edu.slices`. Loading a payload without `edu` behaves as today.
  - The canvas bridge `insert` returns the new node id. `fit` calls `fitView`.
- `edu-engine-checks.test.mjs`. Every predicate of §6.4.3 passes and fails on hand-built contexts:
  - `nodeType` for standard components versus layers (Components → Linear vs `torch.nn.Linear`).
  - `chain` with and without `allow`, and `shapeAt` / `inputShape` / `outputShape` on validated graphs.
  - `paramCount`, `run` queries with fake trainer sources (`seq` filtering, `final`, `overfit`, `model`), `runs`
    coverage of learning rates, `level`, `component`, `repeat`.
  - Global diagnostics (Softmax before cross-entropy).
- `edu-engine-feedback.test.mjs`: each rewrite rule against the real validator messages, produced by building
  graphs with those errors (including the Linear-component MatMul message and the missing-Flatten case).
- `edu-engine-actions.test.mjs`
  - Each action against in-memory stores: insertBetween rewires correctly, buildChain, group, repeat, openLevel,
    setParam on a layer and on a component.
  - Show me is a single undo entry. loadFixture clears history.
- `edu-engine-progress.test.mjs`
  - Reducers, `migrate` after a module version bump, and module status.
  - Storage that throws, or a full quota, falls back to memory and sets `storageDegraded`.
  - Event log ring buffer limit.
- `edu-engine-workspace.test.mjs`
  - Capture and restore round trip (graph, templates, training, slices).
  - sandbox ↔ lesson switching keeps both workspaces.
  - The step-entry snapshot restores on Reset step.
- `edu-learn-ui.test.mjs` (jsdom, test module fixture)
  - LearnHome statuses from progress, and Continue resumes.
  - Lesson panel: focus moves to the step heading; aria-live announces "Step complete"; the hint ladder goes 1 → 2
    → Show me; a prediction gate locks Train until answered and then shows the reveal; quiz explanations appear per
    option; Back does not rerun setup.
  - Deep link opens the right step.
- `edu-learn-browser-check.mjs` (manual): screenshots of Learn, the lesson panel docked at 1440 px, and the drawer at
  390 px. Checks that no horizontal page scroll appears and that the spotlight finds `ribbon.components`.

### 8.2 Stage B

- `edu-data-rng.test.mjs`
  - mulberry32 vectors: the first 5 outputs for seeds 0, 1 and 42, pinned from this spec's reference port
    (§11.4).
  - `deriveSeed` vectors and the polar normal: mean ≈ 0, std ≈ 1 over 10⁵ samples within 0.01.
- `edu-data-generators.test.mjs`
  - Each generator: count, label balance, value ranges, and determinism for the same seed (different seed →
    different data).
  - circle labels match the radius rule. spiral3 matches the CS231n formula on its noiseless points.
- `edu-data-split.test.mjs`
  - Split sizes use the `floor(n·f + 0.5)` rule. Splits are disjoint, cover all indices and are deterministic.
  - Standardisation uses train statistics only: changing a val row leaves the mean and std unchanged.
- `edu-data-iris.test.mjs`
  - 150 rows, 50/50/50, rows 35 and 38 (1-based) equal the corrected values of §5.2.
  - The canonical CSV re-serialisation hashes to sklearn's `iris.csv` sha256.
- `edu-data-digits.test.mjs`
  - 1,797 × 64 values in 0–16.
  - Class counts are pinned from the source file (expected [178, 182, 177, 183, 181, 182, 181, 179, 174, 180]).
  - Dynamic import works.
- `edu-data-usedataset.test.mjs`
  - Sets the root Input even when the store is inside a component.
  - Sets the last Linear's Output through Activation/Dropout; reports when it cannot.
  - Binary + BCE → 1 output, binary + CE → 2 outputs.
  - One undo entry restores everything.
- `edu-trainer-autograd.test.mjs`: finite-difference gradient checks (relative error < 1e-6 in float64) for every
  op and loss of §6.10.2 and §6.10.4, including broadcasting Add and Mean keepdim.
- `edu-trainer-optim.test.mjs`
  - Each optimizer's first 3 steps on a 2-parameter quadratic against hand-computed PyTorch formulas (SGD
    momentum and nesterov, Adam bias correction, AdamW decoupled decay, RMSprop with and without momentum).
  - `parameterGroup` freezing.
- `edu-trainer-compile.test.mjs`
  - The standard Linear component and a `torch.nn.Linear` layer with the same weights give identical outputs.
  - The LayerNorm component matches a direct computation.
  - Unsupported ops give `unsupported-op` naming the node.
  - Parallel repeats rejected; `too-large`; input and output mismatches.
  - LogSoftmax + cross-entropy equals cross-entropy on logits.
- `edu-trainer-lessons.test.mjs`: every lesson configuration of §11.1, with the fixture's seeds, reaches its
  threshold (Node, sync).
  - blobs linear ≥ 0.9 val acc in 20 epochs; circle linear ≤ 0.75 at 30 epochs; circle 2-8-1 ≥ 0.9 at 50.
  - xor 2-8-1 ≥ 0.9 at 50; iris 4→3 ≥ 0.85 at 30; digits 64-128-10 ≥ 0.9 at 10.
  - spiral3 init loss within 0.15 of ln 3; spiral3 learning-rate ordering (0.01 has the lowest final val loss).
  - spiral3 overfit signature (train acc ≥ 0.97, rise ≥ 1.25).
  - Two runs with identical inputs give deep-equal histories. Each case logs its time; the total must stay
    < 30 s.
- `edu-python-parity.test.mjs`: skips when `python3` with numpy is unavailable. For each toy dataset (2 seeds) and
  Iris/digits split indices, generate the Python loader source, run it with `python3 -I` to print JSON, and compare
  with JS: coordinates |Δ| ≤ 1e-12, labels and indices equal.
- `edu-data-ui.test.mjs` (jsdom)
  - Data list groups and `data-edu-target` attributes; Use this dataset updates the Input.
  - TrainerPanel: Train → runs grow, confusion `<table>` for Iris, Check initial loss records an epoch-0 run, a
    locked TrainerCard disables Train.

### 8.3 Stage C

- `edu-param-count.test.mjs`
  - Conv2d(1→8, k=3) = 80, also with `(3, 3)`; Conv1d(k=3) unchanged.
  - The M4 CNN = 9,098, the M6 ResNet = 56,970, GPT-2 = 163,037,184 (unchanged).
  - `countParametersByNode` sums to the total.
- `edu-presets.test.mjs`: export presets have input shapes, class lists of the right length, normalisation
  constants, `training: 'export'` and no `load`.
- `edu-codegen.test.mjs`
  - Legacy `generateTrainingScript` output without `data` is byte-identical to the fork for the GPT-2, ResNet, DANN
    and GAN storyboards (pinned snapshots generated from `15c7c25`).
  - For every dataset, the script contains the loader, `model.eval()`, `torch.no_grad()`, the accuracy metric where
    applicable, the right `compute_loss` target handling for BCE, CE, MSE and next-token, the constants, and the
    header attribution.
  - Each script passes `ast.parse` (skips without `python3`).
  - With torch importable (normally skipped here), it runs the blobs, Iris and digits scripts for 1 epoch.
  - The `.ipynb` is valid JSON with nbformat 4.
- `edu-codepanel.test.mjs` (jsdom): controlled tabs, `data-edu-target` tab buttons, Edu script when a dataset is
  bound, legacy script otherwise.

### 8.4 Stage D

`edu-content.test.mjs`:

1. **Schema:** `validateModule` returns no errors for each module. Sizes: 4–8 steps, 3–4 quiz items, ≤ 3 warm-up
   items. Every quiz option has an explanation. Every `PredictSpec` has a reveal. Every dataset id, target ref,
   BlockRef, embed, LO id and misconception id exists. No step with `{ visited: X }` in its check sets `tab: X`.
2. **Text:** every required key exists in `en`, no key is unused, and bodies are ≤ 60 words (measured after
   interpolation).
3. **Facts:** every fact with `compute` equals its `value`, computed on the fixture state where it is used.
4. **Solvability, the key test.** For each module in order, headless: build the `start` fixture into in-memory
   stores, then for each step apply `setup`, capture the entry state, run `solution` with `headless: true`
   (synchronous training through Stage B's real trainer), and assert `check` passes. It also asserts the check
   **fails** at entry for task and predict steps, so no step passes trivially. Total runtime < 60 s.
5. **Misconceptions:** each `diagnose` rule fires on a hand-made wrong state.

### 8.5 Stage E

- `edu-walkthrough-browser-check.mjs` (playwright, port 5174)
  - Fresh profile → Learn. For each module: Start, then per step press Hint, Hint, Show me, wait for "✓ Done",
    then Next; answer the quiz; reach the recap.
  - Screenshots at 1440×900 and 390×844 per module start and per training step.
  - Asserts: no console errors; no horizontal scroll; every spotlight target resolves at least once.
  - Reload mid-module resumes at the same step; Exit lesson restores the GPT-2 sandbox.
- **Accessibility pass** (manual; results in §12, Stage E)
  - Keyboard-only run of M1: Tab order, Enter and Space on all controls, focus visible.
  - Screen-reader labels via `getByRole` queries.
  - Contrast spot checks; reduced motion.
- **Performance** (§12, Stage E): the trainer timings of §6.10.7 measured in Chromium
  (`scripts/edu-data-browser-check.mjs`), and bundle size. The digits data is a separate chunk.
- **Regression:** `npm test` (baseline 281 + all Edu tests, 0 failures, no Edu skips except the PyTorch-dependent
  ones), `npm run lint`, `npm run build`.

---

## 9. Evaluation plan

**Research questions.**
- RQ1: Do novices learn more with DeepFlow Edu than with a baseline?
- RQ2: Which Edu mechanisms (prediction gates, graded hints, export-and-read) are associated with learning gains?
- RQ3: Is DeepFlow Edu usable for novices? DeepFlow scored SUS 79.5 with mostly experienced users.

**Design.** Between-subjects, two arms, about 60 minutes each, modules M1–M3:

- **Edu arm:** DeepFlow Edu.
- **Baseline arm:** the same DeepFlow build with the lesson panel replaced by a static page. The page has the same
  step text, figures and fixtures (loaded by buttons), but no automatic checks, hint ladder, prediction gates or Show
  me. The in-browser trainer and Data tab stay available.

This isolates the guided-practice layer, which is the paper's contribution. An optional third arm, the same text
with TensorFlow Playground, adds the field's common reference point if the participant pool allows (26 more).

**Participants.** 26 per arm (power 0.8, α 0.05, two-sided, for d = 0.8). Students with at most one ML course and
no neural-network training experience. Screened with a 5-item prior-knowledge test.

**Measures.**
1. **Pre/post test:** 14 items aligned to the learning objectives of M1–M3, drawn from a bank built from the quiz
   and warm-up items plus isomorphic variants. Forms A and B are counterbalanced. 4 items target the named "broken
   models": stacked Linears, initial loss, input size including the label, normalising before the split.
2. **Transfer task** (10 min, both arms in the same build with the lesson UI hidden): build and
   train a classifier for a new tabular dataset (Palmer Penguins numeric columns). Scored by a rubric: correct
   input size, output size, loss, a valid graph, validation accuracy ≥ 0.85.
3. **Code reading:** 3 questions on an exported `train.py`.
4. **Self-report:** SUS, Paas mental-effort item per module, and perceived understanding. The gap between perceived
   understanding and the post-test probes "illusory understanding" (Transformer Explainer).
5. **Logs** (Edu arm, local and exported with consent): hint levels, Show me use, time per step, prediction
   accuracy, number of runs.

**Analysis.** ANCOVA on the post-test with the pre-test as covariate. Mann–Whitney on transfer rubric scores. Within
the Edu arm, mixed-effects logistic models of item correctness by hint level and prediction correctness (RQ2). SUS
is reported with a 95 % CI against DeepFlow's 79.5.

**Pilot.** 5 participants, to calibrate timing and wording and to check that no step needs Show me for more than
30 % of pilot users. Steps above that threshold get rewritten.

**Study build.** The baseline arm needs a static lesson view, behind a `#/learn?view=static` flag. It is planned for
v1.1 (Stage E or later) and is not part of the v1 stages.

**Technical evaluation** (reported with the system paper):
- Solvability (all 50 steps by Show me, §8.4).
- Browser–PyTorch dataset parity (§8.2).
- Trainer timings (§6.10.7).
- Python validity of every dataset's export.
- Where PyTorch is available, the curve agreement between the browser and the exported run for blobs, circle, Iris
  and digits: final validation accuracy within 3 percentage points, mean of 5 seeds.

**Ethics.** Local-only logging, explicit consent before downloading a log, no personal data in the log.

---

## 10. Risks, limits and open questions

| Risk or limit | Mitigation |
|---|---|
| The browser trainer and PyTorch are not bit-identical (float64 vs float32, different weight-init RNG, batch order, GELU erf approximation) | Data and split are identical and tested. Init follows the same distributions. The text says "the same data and split; curves will be similar, not identical". Agreement is measured as part of §9 |
| Dropout's benefit on tiny toy data varies by seed (§11.2) | M5 s5 passes on any Dropout run and shows the learner's own comparison honestly, with a seed-retry suggestion. No step claims a guaranteed improvement |
| Learners lean on Show me | The hint ladder puts two text hints first, Show me is logged, and copy says "try first". The pilot threshold is in §9 |
| Some editor gestures need a pointer (drag to connect) | Show me and palette chips offer keyboard-reachable paths. DeepFlow's own keyboard support is unchanged |
| localStorage limits (GPT-2 slot snapshots can be about 150–250 KB) | At most 8 slots plus 1 step snapshot. Writes degrade to memory with a notice (§6.8) |
| MNIST and CIFAR-10 licences are not verified | Never bundled; downloaded by torchvision at run time with attribution (§5.2) |
| Colab or network unavailable for export-only modules | M3–M7 keep all in-app steps meaningful without running the export. Only the "Export & read" steps mention Colab, and their checks need only opening the code |
| Parameter count bug in DeepFlow (Conv scalar kernel) | Fixed by Stage C (§6.11.4) before M4 and M6 facts are trusted |
| Content correctness drift | Facts ledger and solvability tests (§8.4). Thresholds come from §11 |
| Scope | v1 excludes in-browser CNN and Transformer training, graph-to-code highlighting, accounts, translations other than English, and the Penguins dataset (used only in the study transfer task, loaded from its CC0 CSV by the study notebook) |

**Open questions for the product owner** (none blocks v1):

1. Should Learn also show an optional "capstone": improve a weak Fashion-MNIST baseline with the training recipe
   (research curriculum M13)? Proposed for v1.1.
2. Should the sandbox default stay GPT-2, as in DeepFlow, or become a small MLP for Edu users? v1 keeps GPT-2, so
   existing tests and the M7 comparison are untouched.
3. Should graph-to-code highlighting be built in v1.1? It needs codegen source maps (node id → line ranges).

---

## 11. Appendix: verified numbers and how they were obtained

### 11.1 Training thresholds (reference simulation)

A NumPy re-implementation of the planned trainer produced these numbers. It used the same mulberry32 PRNG,
Playground and CS231n generators, Fisher–Yates split, PyTorch-default Linear initialisation (U(±1/√fan_in) for
weights and biases), Adam with β = (0.9, 0.999), ε = 1e-8, and BCE/CE as in §6.10.4. Mean and [min, max] over
seeds 1–10 (digits 1–5). Validation fraction 0.2 unless stated. Scripts:
`/tmp/claude-0/-home-user-deepflow/7a8350aa-ff30-52a8-8452-b7a05647a64e/scratchpad/edu/spec-sim/` (`sim.py`,
`exp1.py`, `exp2.py`, `exp5.py`).

| Lesson configuration | Final val accuracy | Initial val loss | Threshold used |
|---|---|---|---|
| M1 blobs (n 200, noise 0.1), Linear 2→1, Adam 0.03, batch 32, 20 / 30 epochs | 0.997 [0.975, 1.000] (both) | 0.91 [0.26, 2.37] (varies: inputs not scaled) | ≥ 0.9 at ≥ 20 epochs |
| M1 Iris standardised, Linear 4→3, Adam 0.03, batch 16, 50 epochs | 0.950 [0.900, 1.000] | 1.28 [0.92, 1.80] | ≥ 0.85 at ≥ 30 epochs |
| M2 circle (n 300, noise 0), Linear 2→1, 50 epochs | 0.477 [0.383, 0.683] | — | run only (no threshold) |
| M2 circle, Linear 2→8 → Linear 8→1, no activation | 0.447 [0.283, 0.617] | — | run only |
| M2 circle 2-8-1 ReLU, 50 epochs | 1.000 [1.000, 1.000] | — | ≥ 0.9 at ≥ 50 epochs |
| M2 circle 2-3-1 ReLU, 50 epochs | 0.932 [0.767, 1.000] | — | run only; copy says "sometimes gets stuck" |
| M2 circle 2-2-1 ReLU, 100 epochs | 0.750 [0.567, 0.867] | — | copy "about 75%" |
| M2 xor (n 300, noise 0) 2-8-1 ReLU, 50 epochs | 0.998 [0.983, 1.000] | — | ≥ 0.9 |
| M2 xor, Linear 2→1 | 0.537 [0.283, 0.667] | — | — |
| M3 digits 64-128-10, Adam 0.001, batch 32, 15 epochs (5 seeds) | 0.964 [0.955, 0.972] | 2.298 [2.285, 2.310] ≈ ln 10 | ≥ 0.9 at ≥ 10 epochs |
| M5 spiral3 (100/class, noise 0.2) linear | 0.495 [0.350, 0.600] (train 0.532; CS231n reports 49 %) | 1.17 | — |
| M5 spiral3 2-64-3, Adam 0.01, 200 epochs | 0.977 [0.950, 1.000] (CS231n reports 98 % train) | 1.102 [1.077, 1.151] ≈ ln 3 | init within 0.15 of 1.10 |
| M5 spiral3 2-64-3, 60 epochs, lr 0.0001 / 0.01 / 1.0 | 0.545 / 0.967 / 0.745 [0.317, 0.917] (final val loss 0.978 / 0.137 / 0.678) | — | 0.01 best |

**The in-browser trainer on the M2 fixtures** (Stage E, `stageE/m2-one-neuron.mjs` and `m2-widths.mjs`, training
seeds 1–10, data as in the fixtures, 50 epochs). The lesson copy reports the learner's own `{measured}` value; these
are the values learners see:

| Lesson configuration | Final val accuracy, mean [min, max] | Copy |
|---|---|---|
| M2 s1 circle, Linear 2→1 | 0.595 [0.55, 0.65] (fixture seed 2: 0.583) | "not much better than the 50% you get by guessing" |
| M2 s2 circle, Linear 2→8 → Linear 8→1, no activation | 0.587 [0.53, 0.65] | "about the same as the single neuron" |
| M2 s5 circle 2-2-1 ReLU | 0.742 [0.63, 0.82] | "about 75%" |
| M2 s5 circle 2-3-1 ReLU | 0.883 [0.52, 1.00] (seed 5 gets stuck at 0.52) | "training sometimes gets stuck" |
| M2 s4 circle 2-8-1 ReLU | 1.000 [1.00, 1.00] | check ≥ 0.9 |

The NumPy rows above for one neuron and for two stacked Linears are lower (0.477 and 0.447): both sit near chance,
the browser runs a little above it.

**The in-browser trainer on the M5 fixtures** (Stage E, `stageE/m5-claims.mjs`). s2, `m5.start`, 60 epochs, training
seeds 5 (the fixture), 1, 2, 3: lr 0.0001 moves the validation loss only from 1.10 to 0.97–0.99; lr 0.01 ends at
0.117–0.122 (accuracy 0.967); lr 1.0 jumps up by more than 1.2 between epochs and ends at 0.45–0.75. s3, `m5.big`,
seeds 1–8: training accuracy 1.000 on 7 of 8 seeds (0.947 on seed 3, which then fails the step's 0.97 bar), final
training loss ≤ 0.004 on those 7, final/min validation loss 2.4–8.3; validation accuracy falls from 0.947–0.987 at the
best epoch to 0.81–0.95 at the end (seed 5: 0.960 → 0.920), the "few points" of s4.

Overfitting (M5 s3; spiral3 50/class, noise 0.3, val 0.5 → 75 train points; 2-128-128-3, Adam 0.01, batch 16, 300
epochs; 8 seeds): train accuracy 0.990 [0.987, 1.000]. Final/min validation-loss ratio
[2.51, 3.09, 6.56, 2.00, 1.39, 4.01, 2.29, 2.68] (threshold 1.25). Best epochs 11–106. Validation accuracy stays
0.937 [0.907, 0.960]. That is why M5 s4 says that validation accuracy "barely changed".

### 11.2 Regularisation on the overfitting set (8 seeds; `exp4.py`, `exp6.py`)

Same data as M5 s3 (spiral3, 50 per class, noise 0.3, val 0.5), 2-128-128-3, batch 16, 300 epochs. "Gap" is final
validation loss minus final training loss, both measured in eval mode. Dropout masks and batch order in this
experiment used NumPy's RNG; only the statistics matter here.

| Variant | lr | Final val loss (mean) | Seeds with lower final val loss than baseline | Seeds with a smaller gap |
|---|---|---|---|---|
| baseline | 0.01 (as in M5) | 0.888 | — | — |
| Dropout p = 0.5 after each ReLU | 0.01 | — | 5 / 8 | **7 / 8** |
| Dropout p = 0.3 | 0.01 | — | 6 / 8 | 6 / 8 |
| AdamW weight decay 0.1 | 0.01 | — | 6 / 8 | 7 / 8 |
| baseline | 0.003 | 0.869 | — | — |
| Dropout p = 0.5 | 0.003 | 0.448 | 6 / 8 | 7 / 8 |
| Dropout p = 0.3 | 0.003 | 0.471 | 6 / 8 | 7 / 8 |
| AdamW weight decay 0.1 | 0.003 | 0.519 | 7 / 8 | 8 / 8 |

Validation accuracy changed little in every variant (0.93–0.94). Conclusion for content: regularisation helps on
average but not on every seed. M5 s5 therefore reports the learner's own comparison.

**The in-browser trainer** (Stage D probe, re-measured in Stage E with `stageE/dropout-seeds.mjs`): the `m5.big`
fixture itself (spiral3 50/class, noise 0.3, data seed 5, val 0.5, 2-128-128-3, Adam lr 0.01, batch 16, 300 epochs),
with and without Dropout p = 0.5 after each ReLU, training seeds 1–8, all metrics in evaluation mode:

| | Mean final gap (val − train loss) | Mean final val loss | Mean final val accuracy |
|---|---|---|---|
| baseline | 0.359 | 0.378 | 0.913 |
| Dropout p = 0.5 | 0.160 | 0.214 | 0.910 |

The gap shrank on **8 of 8** seeds and the final validation loss fell on 8 of 8; validation accuracy barely moved.
M5 s5 quotes "8 of 8" (`dropoutSeeds`), the number of the trainer the learner uses. The NumPy simulation above, with
its own dropout masks and batch order, gave 7 of 8.

### 11.3 Parameter counts (DeepFlow's own `countArchitectureParameters` on `15c7c25`, with the Conv fix applied by hand where noted)

| Model | Count | Check |
|---|---|---|
| M3 MLP 784-128-10 | 101,770 | 784·128 + 128 + 128·10 + 10 |
| M3 digits MLP 64-128-10 | 9,610 | 64·128 + 128 + 128·10 + 10 |
| M4 CNN (8 and 16 filters, 3×3, pad 1, two 2×2 pools, Linear 784→10) | 9,098 with kernel `(3, 3)`; **8,282 with scalar `3` before the fix** | 80 + 1,168 + 7,850 |
| M6 one ResBlock (2 × Conv 32→32 3×3 no bias + 2 × BN 32) | 18,560 | 2·9,216 + 2·64 |
| M6 model with 3 blocks | 56,970 | 896 + 64 + 3·18,560 + 330 (validated through `groupSelection` + `convertInstanceToRepeat`) |
| M7 tiny GPT (vocab 65, ctx 64, d 64, 4 heads, 2 blocks, MLP 256) | 112,512 | 4,160 + 4,096 + 2·49,984 + 128 + 4,160; output shape (B, 64, 65) |
| GPT-2 small as built (untied lm_head) | 163,037,184 | existing test |
| GPT-2 small with tied lm_head | 124,439,808 | 163,037,184 − 50,257·768 |

Other constants: ln 2 = 0.693, ln 3 = 1.099, ln 10 = 2.303, ln 65 = 4.174. Tiny Shakespeare: 1,115,394 characters,
65 unique, 90/10 split 1,003,854 / 111,540 (nanoGPT `prepare.py`). Digits class counts [178, 182, 177, 183, 181,
182, 181, 179, 174, 180] (from sklearn's `digits.csv.gz`). sklearn `iris.csv` sha256
`f13ffa8fdd56fd8e6c8d16d4081a3fbd3114bcd0aae4256c43205169cd9d1449`.

### 11.4 Reference Python port of the PRNG (used by the simulation and by `PY_PRNG`)

```python
M32 = 0xFFFFFFFF
def imul(a, b):
    return (a * b) & M32
def mulberry32(seed):
    a = seed & M32
    def next_():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = imul(a ^ (a >> 15), a | 1)
        t = (t ^ ((t + imul(t ^ (t >> 7), t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296.0
    return next_
def fnv1a(s):
    h = 0x811C9DC5
    for ch in s:
        h ^= ord(ch)
        h = imul(h, 0x01000193)
    return h
def derive_seed(seed, label):
    x = (imul(seed & M32, 0x9E3779B1) ^ fnv1a(label)) & M32
    x ^= x >> 16; x = imul(x, 0x85EBCA6B); x ^= x >> 13; x = imul(x, 0xC2B2AE35); x ^= x >> 16
    return x
```

Test vectors, identical in this Python port and in the JS code of §5.3 (checked with Node 22 and Python 3.13). Stage B
pins them in `edu-data-rng.test.mjs`:

| Call | Values |
|---|---|
| `mulberry32(0)` first 5 | 0.26642920868471265, 0.0003297457005828619, 0.2232720274478197, 0.1462021479383111, 0.46732782293111086 |
| `mulberry32(1)` first 5 | 0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849 |
| `mulberry32(42)` first 5 | 0.6011037519201636, 0.44829055899754167, 0.8524657934904099, 0.6697340414393693, 0.17481389874592423 |
| `deriveSeed(1, 'data')` | 2212395750 |
| `deriveSeed(1, 'split')` | 3534800360 |
| `deriveSeed(5, 'init')` | 2825648146 |

With these definitions, the Iris split for data seed 1 and val 0.2 gives 120 training rows with mean (5.83, 3.07,
3.72, 1.18) and std (0.84, 0.44, 1.79, 0.78). These are the values shown in the §3.4 wireframe.

---

## 12. Change log

| Date | Author | Change |
|---|---|---|
| 2026-10-29 | Stage 0 (product/tech lead) | v1 of the specification |
| 2026-10-30 | Stage C (recorded by Stage E) | **Export.** Without a bound dataset `train.py` stays DeepFlow's stub, byte-identical for ResNet, DANN and GAN, but with two fixes that change the GPT-2 output in one line (§3.6 said byte-identical for all): cross-entropy flattens `(B, T, V)` scores and `(B, T)` targets when the scores sit on the last axis (a Linear or MatMul output; the GPT-2 export crashed on its first step), and each loss reads its own `labelKey` from the batch. Channels-first scores such as ResNet's `(B, C, H, W)` keep PyTorch's per-pixel cross-entropy. |
| 2026-10-30 | Stage C (recorded by Stage E) | **Export.** With a bound dataset, `train.py` also keeps the best-validation weights and restores them before the test evaluation (§6.11.2 only reported the best epoch), prints the first loss next to `ln C`, and opens with a "Check before running" list when the model does not fit the data (input shape, output size, several Inputs). Tiny Shakespeare takes its window length from the model's Input. The loaders of the export-only presets live in `data/presets/images.js` and `text.js`, next to their specs, not in `data/python/`. The code panel generates code only while it is open. |
| 2026-10-30 | Stage D (recorded by Stage E) | **Quizzes.** `m1.q4` and `m4.q4` were replaced so that every learning objective has a quiz item: `m1.q4` (lo4) "A confusion matrix has true classes in its rows and predicted classes in its columns. Where are the correct predictions?" **On the diagonal** · In the first row · In the last column (it was a second lo2 item on the neuron's boundary); `m4.q4` (lo4) "The last pooling block outputs (B, 32, 4, 4). What must the Input of the Linear block after Flatten be?" **512** · 16 · 40 (it was a second lo3 parameter count). `m3.q2`'s distractors are (B, 67) (added, not multiplied), (B, 1024) (one channel map only) and (3072,) (no batch axis), instead of (B, 96) and (3072,). |
| 2026-10-30 | Stage D (recorded by Stage E) | **Module 7 copy.** s3 asks about attention "without positions and without a causal mask", because a causal mask alone leaks some order; the reveal says so and that GPT models still add positions. s4 asks "The output at position 6 predicts character 7. Which characters can a head at position 6 look at?" (**Characters 1 to 6** · All 64 · Only character 6 · Characters 1 to 7), which removes the ambiguity of "when predicting the 6th character". |
| 2026-10-30 | Stage D (recorded by Stage E) | **Gates and tabs.** Beyond §4: M5 s2 locks Train until its learning-rate prediction is answered (`locks.gate: 'train'`). M2 s6, M3 s6, M4 s3 and s5, and M7 s1 (through its fixture's `ui`) turn Dimensions off until the shape prediction is answered; M2 s6, M4 s3 and s5 then ask to turn them back on, and M7 s2 turns them on. M5 s4 opens the Training tab, where the curves are. |
| 2026-10-30 | Stage D (recorded by Stage E) | **Wording and numbers.** M6 uses the app's own labels: "set **Steps** to 3" for the repeat count and the "**Step 1/3**" button for stepping through copies (§4.6 said "count" and "Instances"). M2 s1's reveal no longer says "Accuracy stays near 50%"; its done line shows the learner's measured accuracy, about 60% in the browser (§11.1), as "not much better than the 50% you get by guessing". `dropoutSeeds` is "8 of 8", measured with the in-browser trainer and marked `source: 'simulation'` (§11.2). |
| 2026-10-30 | Stage D (recorded by Stage E) | **Metadata the engine ignores**, read only by the content tests (§8.4): facts carry `at` (the step whose state the fact is computed on) and `note` (how a `simulation` or `literature` fact was obtained); export steps carry `codeQuotes` (lines of model.py or train.py the copy points at, checked against the generated code); `diagnose` rules carry an `example` action list that builds the wrong state; M6's custom check reports its sub-steps with the text keys `m6.check.*`. |
| 2026-10-30 | Stage E | **Show me adopts the learner's blocks** (§6.5): insert actions reuse same-kind blocks added since the step began that are unwired or wired only to the action's blocks, so a palette chip followed by Show me no longer leaves a stray block (M3 s2). |
| 2026-10-30 | Stage E | **Trainer text** (§3.5): invalid-graph preflight issues carry the friendly rewrite of §6.4.4 as plain text (the raw validator text stays in `validatorMessage`), so the TrainerCard and the Training tab no longer show "Template "Linear" has 1 internal error". While a prediction gate is open, the card's problem line and the Training tab's preflight list are hidden: in M3 s6 they named the answer. The content test asserts that no status line or ungated card shows an expected answer before it is given. |
| 2026-10-30 | Stage E | **One accuracy format**: one decimal at most, no trailing .0, never rounded up to 100% ("97.5%", "93.3%", "100%"), used by the trainer, the reveals and the check messages (the card said 98% where the reveal said 97.5%). |
| 2026-10-30 | Stage E | **Use this dataset** follows the step's `setOutput: false` (§3.4); M1 s5 now asks for Input 4 and Output 3 (it said "make sure Output is 3" because the button had set it). The lesson bridge source gains `datasetUse`. |
| 2026-10-30 | Stage E | **Training canvas**: it fits beside the TrainingPalette (right 40%, at most 100% zoom; the Controls' fit button too), and in a lesson it fits when it opens instead of restoring a viewport saved in another workspace, so the loss block of M1 s6 is clickable at 1440×900 and 1024×768. Event `deepflow-edu:training-canvas` (`fit`) for "Where is it?". |
| 2026-10-30 | Stage E | **Targets** (§6.6): `code.notebook` and `code.colabNote` added; "Where is it?" is not offered inside a level a `nodeOpen` target opened, opens the code panel for code targets and fits the Training canvas for training targets. §11.1 and §11.2 gained the in-browser trainer's measurements. |
| 2026-10-30 | Stage E | **Spotlight follows the learner** (§6.6): used targets are skipped (`isTargetDone`), targets off screen count as missing, and "Where is it?" also scrolls lists and panels. M3 s7's targets add the MNIST row and Use this dataset. |
| 2026-10-30 | Stage E | **QA record.** `scripts/edu-walkthrough-browser-check.mjs` walks every module with Hint, Hint, Show me at 1440×900 and 390×844: each step with an unused target rings it at entry or, below the fold on the phone (the drawer's trainer card, the ribbon's edge), offers "Where is it?", which brings it into view; no console errors, no horizontal scroll; a reload resumes the step and Exit restores the sandbox. By hand in Chromium: first run → Learn → M1 by mouse and keyboard → quiz → recap → M2; sandbox and back; reload mid-step; Reset progress; tab switches mid-lesson; Undo of a Show me; 1440×900, 1024×768 and 390×844. Keyboard: Tab reaches every Edu control, Enter submits answers and runs chips, Space chooses options, Esc closes menus and the code panel (never the lesson), focus rings are 2 px #e91e63, focus moves to the step heading and to the reveal. Text contrast: every Edu text element met WCAG AA in the audited states (disabled controls excepted). Trainer timings in Chromium (§6.10.7 budgets): circle 2-8-1 about 30 ms, spiral3 overfit about 3.5–4 s, digits MLP about 1.1–1.5 s; the UI keeps about 40 frames per second while training. |
| 2026-10-30 | Stage E | **QA fixes from the browser walk.** Show me keeps the column compact (lower blocks move only as far as the new rows need), so M3 and M4 stay readable after a fit. Check messages use articles ("connect an activation (such as ReLU)") and no "block(s)"; M6 s3's entry status asks to group the blocks (a standard Linear component is no grouping attempt). The code panel closes with Esc and explains an invalid model in plain words ("No code yet: …") instead of a stack trace. Below 900 px the TrainingPalette starts folded to its header. The Training tab's 10-class confusion matrix uses compact cells, and its result toggle wraps. M2 s7's done line no longer says "You built that network yourself", which was untrue after Show me. Dead exports and duplicate CSS in `src/edu` were removed. |
