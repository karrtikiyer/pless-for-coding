"""Re-cluster the algosim paper's pre-generated APPS outputs as reference baselines.

The paper (Lee et al., EMNLP 2025 Findings — arXiv:2503.00691) released the
100-sample-per-problem generations behind their Table 2 on HuggingFace at
[`sh0416/outputs-apps`](https://huggingface.co/datasets/sh0416/outputs-apps).
This module:

  1. Loads a slice of that dataset by (model, source, difficulty).
  2. Filters to ``status == "Passed"`` samples (the paper's own protocol — they
     cluster only functionally-correct solutions).
  3. Groups by ``problem_id``, taking the post-processed ``code`` field when
     populated and falling back to a markdown-codeblock extract from
     ``completion`` otherwise.
  4. Writes one parquet per (model, source, difficulty) under
     ``algosim_data/apps_paper_baselines/requests/`` in the schema
     algosim's clustering script expects: ``problem_id`` (str, starting with
     "ATCODER" or "CODEFORCES" to pass the prefix filter), ``question``,
     ``solutions``.

These parquets are then shipped to the GPU pod and run through
``run_algosim_judge_qwen3.sh`` for clustering. The resulting NAUADC values
serve **two** purposes:

  - **Pipeline validation** — our re-clustered deepseek-6.7B-base NAUADC on
    AtCoder introductory should land within ~0.1 nat of the paper's Table 2
    figure for that model and bucket.
  - **Reference rows** in our final APPS comparison (``algosim_apps_findings.md``).

Models available in the dataset and recommended for re-clustering:

  - ``deepseek-ai/deepseek-coder-6.7b-base`` — closest in size to our Qwen3-8B
  - ``deepseek-ai/deepseek-coder-6.7b-instruct``
  - ``TheBloke/deepseek-coder-33B-instruct-AWQ``
  - ``gpt-4o-2024-08-06`` (20 samples/problem — proprietary)
  - ``gpt-4o-mini-2024-07-18`` (20 samples/problem)

Note: ``Qwen2.5-Coder-32B-Instruct-AWQ`` is referenced in the paper but **not**
in the public HF dataset; we rely on the paper's published Table 2 for that
model in the final report (no re-clustering possible).

Usage::

    uv run python -m bench.eval.algosim_paper_replicate \\
        --difficulty competition \\
        --models deepseek-ai/deepseek-coder-6.7b-base,deepseek-ai/deepseek-coder-6.7b-instruct,TheBloke/deepseek-coder-33B-instruct-AWQ \\
        --sources ATCODER,CODEFORCES \\
        --output-dir algosim_data/apps_paper_baselines/requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

# Short slug for each model — used both in filenames and (with the source
# prefix) in the algosim ``problem_id`` so we can disambiguate when many
# (model, bucket) combinations are clustered.
MODEL_SLUG: dict[str, str] = {
    "deepseek-ai/deepseek-coder-6.7b-base": "ds6.7B-base",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "ds6.7B-instruct",
    "TheBloke/deepseek-coder-33B-instruct-AWQ": "ds33B-instruct-AWQ",
    "gpt-4o-2024-08-06": "gpt4o",
    "gpt-4o-mini-2024-07-18": "gpt4o-mini",
}


def _extract_code(row: dict) -> str:
    """Return the executable code for one row.

    Prefer the dataset's post-processed ``code`` field; fall back to a
    markdown codeblock extract from ``completion``. If neither yields a
    non-empty string, return ''.
    """
    code = row.get("code")
    if isinstance(code, str) and code.strip():
        return code.strip()
    completion = row.get("completion") or ""
    m = _CODE_FENCE_RE.search(completion)
    if m:
        return m.group(1).strip()
    return ""


def _iter_rows(difficulty: str):
    """Yield rows from sh0416/outputs-apps for a given difficulty subset."""
    from datasets import load_dataset
    ds = load_dataset("sh0416/outputs-apps", name=difficulty, split="test")
    yield from ds


def build_parquets(
    *,
    difficulty: str,
    models: Iterable[str],
    sources: Iterable[str],
    output_dir: Path,
) -> list[dict]:
    """Write one parquet per (model, source) combination.

    Returns a manifest list of per-output stats.
    """
    import pandas as pd

    wanted_models = {m for m in models}
    wanted_sources = {s for s in sources}
    for m in wanted_models:
        if m not in MODEL_SLUG:
            raise SystemExit(f"Unknown paper model {m!r}; known: {sorted(MODEL_SLUG)}")

    # Bucket rows by (model, source, problem_id) → list[code]
    buckets: dict[tuple[str, str], dict[int, dict]] = defaultdict(lambda: defaultdict(lambda: {
        "solutions": [], "question": None,
    }))
    counts = {"total": 0, "wrong_model": 0, "wrong_source": 0,
              "not_passed": 0, "empty_code": 0, "kept": 0}

    for row in _iter_rows(difficulty):
        counts["total"] += 1
        if row["model"] not in wanted_models:
            counts["wrong_model"] += 1
            continue
        if row["source"] not in wanted_sources:
            counts["wrong_source"] += 1
            continue
        if row["status"] != "Passed":
            counts["not_passed"] += 1
            continue
        code = _extract_code(row)
        if not code:
            counts["empty_code"] += 1
            continue
        key = (row["model"], row["source"])
        bucket = buckets[key][int(row["problem_id"])]
        bucket["solutions"].append(code)
        if bucket["question"] is None:
            bucket["question"] = row["prompt"]
        counts["kept"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for (model, source), per_problem in buckets.items():
        slug = MODEL_SLUG[model]
        rows = []
        for pid, body in per_problem.items():
            sols = body["solutions"]
            if not sols:
                continue
            rows.append({
                # ATCODER/CODEFORCES prefix satisfies algosim's clustering filter;
                # the slug disambiguates (model, difficulty, problem) across all
                # request parquets ending up in the same response directory.
                "problem_id": f"{source}_{slug}_{difficulty}_{pid}",
                "question": body["question"] or "",
                "solutions": sols,
            })
        df = pd.DataFrame(rows)
        out_name = f"{slug}_{source}_{difficulty}.parquet"
        out_path = output_dir / out_name
        df.to_parquet(out_path, index=False)
        manifest.append({
            "model": model,
            "model_slug": slug,
            "source": source,
            "difficulty": difficulty,
            "parquet": str(out_path),
            "n_problems": len(df),
            "n_solutions_total": int(df["solutions"].map(len).sum()) if len(df) else 0,
        })

    return manifest, counts


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--difficulty",
        required=True,
        choices=["competition", "introductory", "interview"],
        help="APPS difficulty subset of the paper's dataset.",
    )
    p.add_argument(
        "--models",
        default=",".join([
            "deepseek-ai/deepseek-coder-6.7b-base",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "TheBloke/deepseek-coder-33B-instruct-AWQ",
        ]),
        help="Comma-separated paper model ids to re-cluster. "
             "Default: the 3 deepseek variants.",
    )
    p.add_argument(
        "--sources",
        default="ATCODER,CODEFORCES",
        help="Comma-separated APPS sources to keep. Default: both ATCODER and "
             "CODEFORCES (OPENKATTIS is excluded to match paper scope).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("algosim_data/apps_paper_baselines/requests"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    print(f"[paper_replicate] difficulty={args.difficulty}")
    print(f"[paper_replicate] models  ={models}")
    print(f"[paper_replicate] sources ={sources}")
    print(f"[paper_replicate] writing → {args.output_dir}")

    manifest, counts = build_parquets(
        difficulty=args.difficulty,
        models=models,
        sources=sources,
        output_dir=args.output_dir,
    )

    print()
    print("Row counters:")
    for k, v in counts.items():
        print(f"  {k:>13}: {v:>10,}")
    print()
    print("Per-(model, source) parquet manifest:")
    print(f"  {'model':<48} {'source':<10} {'n_problems':>10} {'n_solutions':>11}")
    for m in sorted(manifest, key=lambda r: (r["model"], r["source"])):
        print(f"  {m['model']:<48} {m['source']:<10} {m['n_problems']:>10} "
              f"{m['n_solutions_total']:>11}")

    manifest_path = args.output_dir.parent / f"manifest_{args.difficulty}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({
        "difficulty": args.difficulty,
        "models": models,
        "sources": sources,
        "counters": counts,
        "parquets": manifest,
    }, indent=2))
    print(f"\n[paper_replicate] wrote manifest → {manifest_path}")


if __name__ == "__main__":
    main()
