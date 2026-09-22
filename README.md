# DeepFlow v2

A local visual editor for neural network architectures. Build graphs from PyTorch layers, group connected blocks into reusable templates, expose parameters, and repeat templates with per-iteration expressions.

## Run locally

Use Node.js 20.19+, 22.13+, or 24+.

```sh
npm install
npm run dev
```

## Working in the editor

- **Ribbon:** **Components** contains editable standard components and other layer modules; **Operations** contains math, routing, sources, and outputs; **Templates** contains project definitions. Use **Components → Customize pinned components** for up to six shortcuts. Commands act on the active graph pane; dragging into another pane targets that destination.
- **Search:** each insertion menu has search. Press `/` or `Cmd/Ctrl+K` to open Operations search. Enter adds the first result, Arrow Down moves into results, and Escape closes the menu.
- **Nodes:** click a node to open its compact settings; dragging closes them. The small arrow opens a component's structure, including supported native layers that can be decomposed. Numeric fields accept literals and expressions such as `$width`.
- **View:** **Dimensions** shows or hides tensor sizes at the ports; named inputs such as Q/K/V stay identifiable. **Gradients** separately shows structural gradient paths. These controls apply to the active pane.
- **Undo/Redo:** use the toolbar or `Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z`. Architecture edits and shared template edits use one history across both panes; a drag or multi-node deletion is one step. Text fields retain normal text undo. Loading a project, choosing an example, or resetting the workspace starts a fresh history; training edits are outside this history.
- **Templates:** select connected nodes and choose Group. Reuse the definition from Templates. Open a template using its button or a double-click; edit its exposed parameters from **Component → Parameters**. Editing its definition affects every use of that template.
- **Repeat:** use **Structure → Sequential / Parallel** for selected blocks or a template instance. Click the active mode again to return to a single instance; clicking the other mode switches directly. Set the number of steps or instances in the inspector with a value or expression such as `$H`; turning repeat off and on retains that setting. The instance viewer counts from 1; the formula index `i` still starts at 0 and is configured under **Parameter formulas**. Open the shared body and use the level tabs to navigate.
- **Canvas:** Shift/Command-click for multiple selection, drag the background to select, hold Space to pan, and use Delete/Backspace to remove selected nodes. Select a connection and drag an endpoint to reconnect or release it on empty canvas to detach it.
- **Save/Load:** graphs persist in this browser. Save downloads the architecture, templates and training graph together as JSON. Load opens that workspace. These files use the V2 schema.
- **Code:** inspect or download generated `model.py` and `train.py`.

## Standard components and copies

Insert Linear, Embedding, Attention, or Norm from Components, then **Edit definition** to inspect their ordinary operations: learned parameters, matrix products, table lookup, or normalization statistics. Open nested components to continue down to the math.

**Edit definition** changes the shared template. **Make copy** creates a reusable independent definition for this instance, including nested definitions. When copying inside a component, its containing definitions are also copied so other root instances retain their original behavior. Parameters and connections are preserved; the copy appears in Templates.

**Learned positions** generates position indices and looks them up in an Embedding component. New GPT-2 examples use this version, with the sum of token and position vectors kept explicit. **Sinusoidal positions** is a separate alternative built from position indices, frequency arithmetic, sine/cosine, stacking, and slicing; it has no learned table.

Saved opaque layers stay unchanged until you open their structure with the node arrow. Supported conversions preserve node identity, placement, bindings, and connections. These include Linear (with or without bias), ordinary Embedding (including padding rows), learned positions, causal self-attention, and LayerNorm over one final axis. Sparse/max-norm embeddings, multi-axis LayerNorm, and noncausal MultiheadAttention are not converted automatically or silently approximated.

**Gradients** shows structural differentiability, not measured gradient values. It distinguishes trainable or frozen Parameters, stopped/reversed connections, and integer index paths with no derivative. Gather can train its table while its indices remain nondifferentiable; Parameter `requires_grad` controls whether a learned value is trainable.

## Editable multi-head attention

Choose **Examples → Multi-head attention** to open its internal graph. **GPT-2 small** also uses this template inside its sequential Transformer blocks.

- **Component → Parameters** exposes `D = 768` (model width), `H = 12` (head count), `d_head = D / H`, and `dropout = 0.1`. Change `H` to another divisor of `D`; the splits, parallel head count, and derived head width update together.
- In GPT-2, attention inherits `D` and `H` from the Transformer block's `d_model` and `n_heads` bindings. Change those bindings on the Transformer repeat to configure that instance; a bound value takes precedence over a template default.
- Follow Q/K/V projections → Split → parallel attention heads → Concat → output projection. Open a head to edit Transpose, MatMul, Scale, Causal mask, Softmax, Dropout, and the final MatMul. These are ordinary editable nodes.
- Named Q/K/V ports route each collection to its matching template input; MatMul uses ordered `a` and `b` ports. Wire to the intended port rather than relying on edge creation order.
- Head inspection uses **zero-based indices**: index `0` is the first head. The preview shows that instance's resolved parameters and shapes. Body edits affect all instances of the shared template; repeated learned layers have independent weights in exported Python. Inspecting another head does not create a separate definition.

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

## Source map

- `src/domain/`: graph, template, training and layer-library definitions.
- `src/stores/`: state and browser persistence.
- `src/components/architecture/`: library, canvas, node cards and template editing.
- `src/components/training/`: training canvas and controls.
- `src/validation/`: topology/shape checks and shared report cache.
- `src/codegen/`: generated Python.
- `src/storyboards/`: example workspaces.
