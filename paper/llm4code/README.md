# LLM4Code 2027 — p-less Sampling on Code Generation

Working directory for the LLM4Code 2027 workshop submission.

- **Submission deadline**: 13 Nov 2026 AOE (tentative per https://llm4code.github.io/dates/)
- **Notification**: 11 Dec 2026 · **Camera-ready**: 29 Jan 2027 · **Venue**: ICSE 2027, Dublin
- **CFP page**: not yet posted for 2027. **Template inferred from ICSE 2027 + LLM4Code 2025/2026 CFPs**:
  - **IEEE conference proceedings** — `\documentclass[10pt,conference]{IEEEtran}` (ICSE 2027 switched back to IEEE from ACM; verified via the ICSE 2027 research-track CFP)
  - **Page limit**: 4-8 pg research / 1-4 pg position, **including references** (LLM4Code 2025 + 2026 identical; expect same)
  - **Review**: double-blind (both prior years)
  - **Recheck against the LLM4Code 2027 CFP the moment it posts.**
- **Overleaf sync target**: `github.com/karrtikiyer/pless-for-coding-paper` (private, exists, cloned as sibling at `../pless-for-coding-paper/`)
- **`gh` account note**: paper repo is under `karrtikiyer` account, not `karrtikiyer-tw`. Run `gh auth switch --user karrtikiyer` before any `gh` calls that touch it.
- **Plan of record**: `~/.claude/plans/look-at-docs-pasted-content-tranquil-ritchie.md`
- **User narrative sketch**: `docs/paper_idea/LLM4CodeConference.pdf`

## Files in this directory

- `README.md` — this file (submission logistics)
- `SKELETON.md` — section-by-section outline mapped to the paper arc + evidence sources
- `MODEL_ROSTER.md` — actual model coverage on MBPP + HumanEval + APPS-252 as observed on disk 2026-09-22, with the shortlist decision
- `TABLE_INVENTORY.md` — which tables the paper will render, source-of-truth files, extractor script
- `FIGURE_INVENTORY.md` — which figures the paper will use, existing PNGs to reuse, regenerations needed
- `main.tex` — LaTeX source (placeholder until CFP posts and template is known)
- `refs.bib` — starter bibliography

## Workflow

1. All numerical claims are extracted from source JSON/CSV via `scripts/paper/make_llm4code_tables.py` — never typed by hand.
2. Literature citations are verified against the source (WebFetch) before landing in `refs.bib` or prose.
3. Draft LaTeX directly in `../pless-for-coding-paper/main.tex` (IEEEtran skeleton is in place with section stubs and \\input for auto-generated tables).
4. Paper source (LaTeX, refs.bib, generated tables/figures) lives in the **sibling repo** at `../pless-for-coding-paper/`. Planning notes (this dir) stay in the code repo. `scripts/paper/make_llm4code_tables.py` emits tables to `paper/llm4code/tables/` (source of truth in code repo); to publish, copy or symlink into the paper repo's `tables/` dir.

## Framing rules (per plan)

- Do **not** phrase G_k gain on DeepSeek as "very competitive" — absolute pass@1 = 0.485 doesn't back that. Use "recovers the loss the α=2 collision-entropy threshold imposed on the DeepSeek p-less baseline."
- Attribute reasoning-model peakedness to RL-style training as a **hypothesis with citations**, not our own demonstrated result.
- No cross-paper cell-by-cell comparison with Shi et al. (arXiv:2402.06925) — cite in related work only.
- Small-model callout vs Hasan et al. (arXiv:2507.03160) is scoped to the 3 shared models (Qwen2.5-Coder-1.5B/3B, OCI-DS-1.3B) and stays on MBPP unless we add HE runs.

## Out of scope (this paper)

- τ_α family (paper A) — cite once, don't re-litigate.
- Instruct-model G_k Rényi sweeps.
- Theory work on majorization (todos B1/B2) — theory paper.
- Cross-paper method-vs-method comparison with Shi et al. baselines we didn't run.
