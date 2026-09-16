# Operation marks

Generated with the user-provided `../figure_gen.py` script, using its configured model. `prompt.txt` contains the exact generation specification; `figuregen-operations01.png` is the original sheet and the accompanying `.txt` is FigureGen's prompt record. No credentials are stored here.

Run `python3 assets/operation-icons/extract.py` from the app root to recreate the eight 192px runtime PNGs in `public/assets/operations/` (requires Pillow). The script only splits, crops and scales the sheet.

Canvas and palette share `OperationIcon.jsx`. Each learned-layer family has one mark; variants are chosen in the inspector. Add, multiply, concat, split, flatten and I/O use deterministic SVG symbols so their meaning remains exact at small sizes.
