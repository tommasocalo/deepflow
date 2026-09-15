# DeepFlow v2

A local visual editor for neural network architectures. Build graphs from PyTorch layers, group connected blocks into reusable templates, expose parameters, and repeat templates with per-iteration expressions.

## Run locally

Use Node.js 20.19+, 22.13+, or 24+.

```sh
npm install
npm run dev
```

## Working in the editor

- **Block library:** expand a category to browse its exact modules. Click a layer to add it in the visible canvas, or drag it to a position. Recently used layers appear at the top.
- **Search:** press `/` or `Cmd/Ctrl+K`. Search by name, description or category; `layer norm` finds `LayerNorm`. Enter adds the first result and Arrow Down moves focus into the results. Escape clears the query.
- **Library tabs:** Layers and Templates support arrow-key navigation. Collapse the library for more canvas space.
- **Nodes:** click to select; use the chevron to open parameters. Required values, output shapes and issues remain visible in the compact card. Numeric fields accept literals and expressions such as `$width`.
- **Templates:** select connected nodes and choose Group. Reuse the definition from the Templates tab. Open a template using its button or a double-click; expose its parameters from the sidebar. Editing its definition affects every use of that template.
- **Repeat:** turn a template instance into a Repeat, set an iteration count, and bind parameters to expressions such as `64 * 2 ** i`. Open the shared template and navigate back with the breadcrumb.
- **Canvas:** Shift/Command-click for multiple selection, drag the background to select, hold Space to pan, and use Delete/Backspace to remove selected nodes. Click an edge to configure gradient flow.
- **Save/Load:** graphs persist in this browser. Save downloads the architecture, templates and training graph together as JSON. Load opens that workspace. These files use the V2 schema.
- **Code:** inspect or download generated `model.py` and `train.py`.

## Checks

```sh
npm test
npm run lint
npm run build
```

The Node regression suites cover template/repeat navigation, deletion, shared validation caching, and real React interactions for numeric parameters, node expansion, library search, template insertion and keyboard access. The interaction tests use jsdom; they do not verify canvas layout, pointer dragging or visual appearance. Older Playwright scripts and historical screenshots remain in `scripts/` and `screenshots/`.

## Current scope

This is a research prototype. Shape validation and PyTorch generation cover a subset of available modules; passing graph checks does not guarantee a runnable architecture. The Data workspace is pending, and generated training code requires user-supplied data loaders. The ResNet, DANN and GAN examples demonstrate editor concepts and are not complete reference implementations.

GPT-2 is the next architecture target. Causal attention, shared embedding/output weights and a verified complete decoder remain future work; the current editor does not yet implement GPT-2 end to end.

## Source map

- `src/domain/`: graph, template, training and layer-library definitions.
- `src/stores/`: state and browser persistence.
- `src/components/architecture/`: library, canvas, node cards and template editing.
- `src/components/training/`: training canvas and controls.
- `src/validation/`: topology/shape checks and shared report cache.
- `src/codegen/`: generated Python.
- `src/storyboards/`: example workspaces.
