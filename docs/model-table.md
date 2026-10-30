# Model table

The Model table is an optional view in DeepFlow's architecture editor. It
summarises the whole model as one table beside the canvas. You can edit
defining settings in place, and every number states how complete it is.

## What it is and how to use it

**View → Table** docks the table beside the active pane. In narrow panes it
becomes a sheet below the canvas, unless the pane is too short for one. Each
row is a node of the model at the chosen depth:

- **Root** shows the nodes of the main graph.
- **Blocks** opens repeats and custom components. Standard modules such as
  Linear or LayerNorm stay one row.
- **Ops** shows every layer and operation.

Each row shows:

- how often the node occurs per forward pass (×N);
- its output shape, B being the batch;
- its defining settings;
- its parameters and its MACs (or its activation bytes);
- its share of the whole model.

Totals assume B = 1, sizes taken from the graph input, fp32 and one forward
pass. MACs count MatMul, Linear and Conv products, plus the products inside
attention and recurrent modules. Activations are the sum of forward op outputs.
Weights are never tied. Under these assumptions, GPT-2 small counts 163,037,184
parameters and 145.8G MACs at T = 1024.

- **Select and navigate.** Click a row to select its node on the canvas.
  Double-click a row, or press Enter in any cell except its settings, to open
  its level. A selection on the canvas marks its row in the table.
- **Edit.** To edit a setting, click it again once its cell is selected, press
  Enter or F2, or start typing. **Symbols** shows and edits each value as
  written, such as `4 * d_model`, at the place where it is written. **Values**
  shows resolved numbers. An edit there sets the value that number is read
  from, for example the global `d_model`. Before anything is written, the editor
  says which value it will change. Each edit is one Undo step.
- **Read partial results.** When some sizes cannot be determined (an
  unresolved count, a shape error, a module whose weights are not modelled),
  totals become lower bounds marked ≥. Shares become upper bounds marked ≤, and
  unknown values read ?. Hovering, or the status line under the table, says
  what was not counted and why.

## Positioning

Several kinds of tools already show or change what a model costs. We describe
each only at the level we rely on here:

- **torchinfo** [1] prints a static text summary of a model, layer by layer.
- **Netscope / CNN Analyzer** [2] and **Sony Neural Network Console** [3]
  report per-layer cost and offer table and graph views of a network.
- **Netron** [4], **Google Model Explorer** [5] and **TensorBoard Graph** [6]
  show a model's hierarchy read-only, with overlays.
- **Skyline** [7] lets users change a performance driver by direct
  manipulation.
- **Spreadsheet-style editors** and **prodirect manipulation** [8] let people
  edit values in a result and have the change applied to the formulas or
  program behind them.

DeepFlow models are built from shared definitions (templates), repeats, local
versions of single instances, and settings bound to globals or to the
container's parameters. Against that background, the Model table makes three
changes:

1. **Aggregation across shared definitions and repeats.** A row merges one node
   across every instance of every enclosing repeat or component. ×N is the
   number of occurrences per forward pass, and the hover shows how nested repeats
   multiply (12 blocks × 12 heads = 144). When instances differ, for example
   after a local version changes a width, the cells show per-instance ranges
   such as `B×1024×1536…3072` or `1.18–2.36M`. The hover lists which instances
   hold which value, and which instances lack the node.
2. **Value-directed edits with a stated write target.** An edit is aimed at a
   displayed value, not at a storage location. The table finds where that value
   is really written:
   - a global parameter;
   - a repeat or component binding;
   - the default in a shared definition;
   - a single instance.

   It names that target in the editor and the status line before anything is
   written. It refuses edits that would silently replace a formula or reach only
   some local versions. Each commit is one Undo step, labelled with the value it
   changed.
3. **Honest partial aggregates.** A total that leaves something out reads ≥ and
   names what it left out. Shares of a partial total read ≤, or ? when the row
   is partial too. A per-instance figure appears only when every instance is
   known. An output shape known in only some occurrences says so (`1/12`).
   After a shape error that leaves a shape open, nothing downstream is sized
   from validation's guess, so a ≥ total remains a true lower bound.

None of these is new on its own. The contribution is to combine them in one
editable view that stays in step with the canvas.

## Evaluation plan

**Design.** A within-subjects study with about 12 ML practitioners. All have
written PyTorch models in the past year. There are two conditions:

- canvas plus the existing node inspector;
- canvas plus the Model table.

In both conditions a torchinfo printout of the same model is available as a
reference. Each participant works on GPT-2 small in one condition and the ResNet
example in the other. The order of conditions and models is counterbalanced.
A session lasts about 60 minutes, with a short tutorial before each condition.

**Tasks** (one set per model):

1. *Locate.* Find where parameters, MACs and activations concentrate: name the
   heaviest rows or components and estimate their shares.
2. *Predict, then apply.* Predict the parameter and MAC totals after a change
   (`d_model` or `mlp_width` for GPT-2, a stage width for ResNet), then make
   the change in the editor.
3. *Diagnose.* In a prepared variant with one local version and one shape
   error, find which instances differ or fail, and the root cause.

**Hypotheses**, each against the inspector condition and named by the change
it tests:

- *H1* (aggregation). With the table, participants complete Locate tasks
  faster and more accurately.
- *H2* (stated write targets). With the table, participants make fewer
  wrong-target edits.
- *H3* (per-instance and partial results). With the table, participants more
  often name the instances that differ or fail, and the root cause, in
  Diagnose tasks.

**Measures.**

- task time;
- accuracy, meaning a prediction within 5 % of the true value or the correct
  set of instances;
- wrong-target edits, meaning an edit that changed a different value than the
  participant intended, such as one instance instead of the shared definition
  (see *Recording intent* below);
- uses of Undo;
- SUS after each condition;
- think-aloud, coded for strategy (for example, how often participants consult
  the torchinfo printout or navigate levels).

**Recording intent.** Before each edit, the participant first names the value
they intend to change and its scope, for example "the MLP width of every
block" or "the width of the second stage only". The experimenter logs it
before the edit is committed. Comparing the model before and after the edit
gives the value that changed; the edit is wrong-target when that value, or
its scope, differs from the logged intent.

Times and SUS are compared with paired tests. Accuracy and wrong-target edits
are compared with mixed-effects models that include participant as a random
effect.

**Threats to validity.** Two familiar models, one prepared error and a small
sample limit how far the results generalise, and recall of GPT-2 or ResNet
could stand in for reading the tool. Novelty, and our role as the tool's
builders, may favour the table; counterbalancing, a scripted protocol and
measures fixed in advance reduce these risks but do not remove them.

## References

1. torchinfo. Open-source library, github.com/TylerYep/torchinfo.
2. Netscope (ethereon, 2015), and CNN Analyzer (dgschwend), a fork of
   Netscope.
3. Sony Neural Network Console (2017).
4. Netron (Lutz Roeder).
5. Google Model Explorer (2024).
6. TensorBoard Graph: K. Wongsuphasawat et al. "Visualizing Dataflow Graphs
   of Deep Learning Models in TensorFlow". IEEE TVCG (VAST 2017).
7. Skyline: G. X. Yu, T. Grossman, G. Pekhimenko. "Skyline: Interactive
   In-Editor Computational Performance Profiling for Deep Neural Network
   Training". UIST 2020.
8. Prodirect manipulation: R. Chugh, B. Hempel, M. Spradlin, J. Albers.
   "Programmatic and Direct Manipulation, Together at Last". PLDI 2016.
