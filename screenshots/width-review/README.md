# Inspector width review

Actual inspector crops captured in Chromium at 2× device scale; pixel sizes below are CSS pixels.

| Case | Before | After |
| --- | --- | --- |
| Numeric Linear / domain head |260px|168px|
| GPT-2 MLP expressions |260px|200px|
| Linear type selector |189px|67px|
| Numeric input fields |116px|70px|

`03-inspector-before.png` and `04-domain-before.png` show the previous panel. Images 05–08 show the final positional embedding, MLP, attention and domain-head panels. `09-domain-context.png` shows the smaller panel on the canvas.

Verified actual rendered text widths, including the dropdown arrow, against available control space. Long expressions use a full row; editing retains focus while the panel changes width. All screenshots were opened and visually reviewed.
