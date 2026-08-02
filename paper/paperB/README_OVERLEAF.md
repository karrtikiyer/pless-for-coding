# Paper 2 — Overleaf project

**Paper:** *When a Hyperparameter-Free Decoder Fails: Diagnosing and Repairing p-less Sampling Loops
in Reasoning-Model Code Generation.*

## Files
- `main.tex` — the manuscript (compiles standalone with `pdfLaTeX` + `BibTeX`).
- `refs.bib` — bibliography (16 entries, all cited).
- `figures/` — `fig_failure_t1.png`, `fig_taxonomy.png`, `fig_apps_alpha.png` (the 3 figures used).

## Open in Overleaf — two ways

### A. Upload the zip (simplest)
1. In Overleaf: **New Project → Upload Project**.
2. Select `paperB_overleaf.zip`.
3. Overleaf auto-detects `main.tex` as the main document. Menu → **Compiler: pdfLaTeX** (BibTeX runs automatically for `\bibliography`).
4. **Recompile.** First compile runs LaTeX → BibTeX → LaTeX → LaTeX; the references resolve on the 2nd–3rd pass (Overleaf does this automatically).

### B. GitHub import (Overleaf premium)
1. Push this repo to GitHub.
2. Overleaf: **New Project → Import from GitHub**, pick the repo, set the root to `paper/paperB/`.
3. Same compiler settings as above. Changes sync both ways.

## Switching to the ICLR 2027 style (when submitting)
`main.tex` currently uses the generic `article` class so it compiles anywhere. Lines marked
`% [ICLR-SWAP]` are the only ones to change:
1. Download `iclr2027_conference.sty` (and `.bst` if provided) from the ICLR 2027 author kit; drag both
   into the Overleaf project.
2. Replace the two `[ICLR-SWAP]` lines (the `\documentclass` and the `\usepackage[margin=1in]{geometry}`)
   with the ICLR preamble from the author kit (`\documentclass{article}` + `\usepackage{iclr2027_conference}`,
   per their template).
3. Fill in `\author{...}` (currently blank) — or keep the anonymized form for the double-blind version.

## Notes
- No custom packages beyond standard ones (`amsmath, booktabs, graphicx, natbib, hyperref, xcolor`), all
  on Overleaf by default.
- The `\td{...}` macro (red TODO notes) is defined in the preamble; there are no `\td` notes left in the
  body, but the macro is kept for drafting.
