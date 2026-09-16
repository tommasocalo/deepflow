# Compact node and inspector review

Reviewed in isolated Chromium sessions at 1440×960 and 1050×760.

- `01-node-width-before.png`: first visual check, original fixed 190×56 nodes; the first inspector revision is already applied here. The complete previous design is retained in `../operations-mlp-inspector.png`.
- `02-mlp-first-pass.png`: content-sized cards; this check exposed the output port label overlapping the floating inspector.
- `final-mlp-inspector.png` and `final-attention-inspector.png`: placement now accounts for the visible port labels, with paired fields and one-row header.
- `03-positional-embedding.png`: long name, alternate variant, Context/Width on one row, panel placed above the node to clear the palette.
- `04-domain-head.png`: custom name, numeric Input/Output on one row, panel placed left of the node.
- `05-smaller-viewport.png`: same layout at 1050×760; neither inspector nor port labels cover the node or palette.

Measured unscaled sizes:

| Element | Previous | Current |
| --- | --- | --- |
| `mlp fc` node |190×56|87×36|
| `ln_1` node |190×56|71×36|
| `causal self-attention` node |190×56|172×36|
| Linear inspector |272×258|260×152|
| Positional embedding inspector | — |260×122|

Cards contain only the shared operation mark and assigned name. Input/output tensor shapes appear by the ports on hover, keyboard focus, selection or inspection. Ports preserve their full tensor shapes; equal multiple inputs are summarized with a count. Connecting and reconnecting do not trigger the inspector.

Reproduce final checks with `node scripts/operations-visual-check.mjs`, `node scripts/compact-layout-review.mjs` and `node scripts/connectors-browser-check.mjs`. The layout review also checks long-name truncation, keyboard opening, parameter editing, shared icons, and label/panel separation.
