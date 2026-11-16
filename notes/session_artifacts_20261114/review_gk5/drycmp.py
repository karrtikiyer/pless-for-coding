import sys, os, re, subprocess, tempfile, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("t", "/tmp/wt_topk/tests/test_gpu2_launch_ministral_gk_batches.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
def run(script, extra):
    tmp = Path(tempfile.mkdtemp(dir="/tmp/review_gk5"))
    gen = t.env.__wrapped__(tmp) if hasattr(t.env, "__wrapped__") else None
    env = next(gen)
    r = subprocess.run(["bash", script], env={**env["env"], "DRY_RUN": "1", **extra}, capture_output=True, text=True, timeout=180)
    out = [l for l in t._norm_dry(env, t._qlog(env["repo"])).copy()]
    full = t._qlog(env["repo"]).replace(str(tmp), "<TMP>")
    try: next(gen)
    except StopIteration: pass
    return r.returncode, out, full
rc0, head, hfull = run("/tmp/review_gk5/head_launcher.sh", {})
rc1, new, nfull = run("/tmp/wt_topk/scripts/gpu2_launch_ministral_gk_batches.sh", {})
print("rc", rc0, rc1)
import difflib
print("\n".join(difflib.unified_diff(head, new, lineterm="", n=0)))
exp = Path("/tmp/wt_topk/tests/data/gpu2_launch_gk_default_dry_expected.txt").read_text().splitlines()
print("HEAD == recorded expectation:", head == exp)
# non-DRY lines diff (gates etc.), strip timestamps
strip = lambda s: [re.sub(r"^\[[^\]]+\] ", "", re.sub(r"\d{8}_\d{6}", "TS", re.sub(r"pid=\d+", "pid=P", l))) for l in s.splitlines() if "DRY:" not in l]
print("---- non-DRY log diff")
print("\n".join(difflib.unified_diff(strip(hfull), strip(nfull), lineterm="", n=0)))
for extra in ({"GK_CONFIGS": "gk_0_5"},):
    rc, out, full = run("/tmp/wt_topk/scripts/gpu2_launch_ministral_gk_batches.sh", extra)
    print("---- GK_CONFIGS=gk_0_5 rc", rc); print("\n".join(strip(full)[-6:]))
