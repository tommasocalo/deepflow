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
- **Model table:** **View → Table** docks a table beside the active pane, with one row per node at the chosen depth: **Root**, **Blocks** (repeats and custom components open; standard modules such as Linear or LayerNorm stay one row) or **Ops**. Each row shows its occurrences per forward pass (×N), output shape, defining settings, parameters, MACs, and its share of the whole model by parameters, MACs or activations. Totals assume B = 1, sizes from the graph input, fp32 and a forward pass; MACs count MatMul, Linear and Conv, and weights are never tied, so GPT-2 small counts 163,037,184 parameters. Unknown sizes give partial totals (≥). Click a row to select its node; double-click anywhere on it, or press Enter, to open its level. To edit a setting in place, click it again once its cell is selected, press Enter or F2, or start typing: **Symbols** edits the expression where it is written (a shared definition edits every use), and **Values** sets the value it reads, such as a global. Each edit is one Undo step.
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

This is a research prototype. Shape validation and PyTorch generation cover a subset of available modules; passing graph checks does not guarantee a runnable architecture. The Data workspace is pending, and generated training code requires user-supplied data loaders. The ResNet, DANN and GAN examples demonstrate editor concepts and are not complete reference implementations.

Repeat counts must resolve to integers from 1–256. Validation/export limit template nesting to 8 levels and expansion to 20,000 nodes. Export rejects validation errors, including unresolved counts, indivisible splits, collection-count mismatches, and invalid named input ports. The GPT-2 example includes explicit causal attention; tied embedding/output weights and a verified end-to-end GPT-2 training pipeline remain outside the current scope.

The shared ONNX converter accepts graph files up to 64 MiB and supports a subset of float32 operators, including Linear patterns, Conv2d with explicit symmetric padding, activations, pooling, Flatten, residual arithmetic, MatMul, Concat and simple transposes. It preserves branches, ordered input ports and multiple graph outputs. Unsupported operators, attributes, dtypes and control-flow graphs prevent editable import. Exposed initializer tensors become frozen Parameters with their source shapes; actual tensor values are not loaded. The original operator graph is a snapshot, not a live reverse export of subsequent edits.

Hugging Face import can consume an ONNX graph already published in the chosen repository. It does not yet run a server-side exporter for repositories that contain only framework weights, or assemble separate encoder/decoder files automatically. ONNX broadens the common conversion path but does not by itself make every model editable.

## Source map

- `src/domain/`: graph, template, training and layer-library definitions.
- `src/domain/metrics/`: per-node parameters, MACs and activation sizes; `src/domain/modelTable/`: Model table rows and edit targets.
- `src/stores/`: state and browser persistence.
- `src/components/architecture/`: library, canvas, node cards and template editing.
- `src/components/training/`: training canvas and controls.
- `src/validation/`: topology/shape checks and shared report cache.
- `src/codegen/`: generated Python.
- `src/storyboards/`: example workspaces.
