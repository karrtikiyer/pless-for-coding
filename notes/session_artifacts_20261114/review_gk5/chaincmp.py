import sys, re, subprocess, tempfile, time, importlib.util, difflib
from pathlib import Path
spec = importlib.util.spec_from_file_location("t", "/tmp/wt_topk/tests/test_gpu2_launch_ministral_gk_batches.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
def run(script, extra, rows=None):
    tmp = Path(tempfile.mkdtemp(dir="/tmp/review_gk5"))
    gen = t.env.__wrapped__(tmp); env = next(gen)
    e = env["env"]
    if rows: e = t._add_rows(env, rows)
    r = subprocess.run(["bash", script], env={**e, "STUB_SLEEP": "1", **extra}, capture_output=True, text=True, timeout=180)
    time.sleep(8)
    ld = sorted((env["repo"]/"logs").glob("*_launch_*"))[0]
    files = {p.relative_to(ld).as_posix(): p.read_text().replace(str(tmp), "<TMP>") for p in ld.rglob("*") if p.is_file()}
    logs = {p.name: p.read_text().replace(str(tmp), "<TMP>") for p in (env["repo"]/"logs").glob("ministral_*.log")}
    try: next(gen)
    except StopIteration: pass
    return r.returncode, ld.name, files, logs
norm = lambda s: re.sub(r"HEAD=\w+", "HEAD=H", re.sub(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d IST", "T", re.sub(r"\d{8}_\d{6}", "TS", s)))
a = run("/tmp/review_gk5/head_launcher.sh", {})
b = run("/tmp/wt_topk/scripts/gpu2_launch_ministral_gk_batches.sh", {})
print("rc", a[0], b[0], a[1][:14], b[1][:14])
for f in ("batch1/chain_batch1.sh", "batch1/resume_batch1.sh", "batch1/cmdline.txt", "check_step_complete.py", "integrity_check.sh"):
    d = list(difflib.unified_diff(norm(a[2][f]).splitlines(), norm(b[2][f]).splitlines(), lineterm="", n=0))
    print("=====", f, "difflines", len(d)); print("\n".join(l[:250] for l in d[2:60]))
rows = '    ("gk_0_5",           "pless_renyi",      1.0, 0.5),\n    ("gk_0_05",          "pless_renyi",      1.0, 0.05),\n'
c = run("/tmp/wt_topk/scripts/gpu2_launch_ministral_gk_batches.sh", {"GK_CONFIGS": "gk_0_5"}, rows)
print("gk_0_5 rc", c[0], c[1], sorted(c[3]))
for n, txt in c[3].items(): print(n, [l[:160] for l in txt.splitlines() if "gk-chain" in l])
print(c[2]["batch1/chain_batch1.sh"])
