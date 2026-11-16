import subprocess, tempfile, importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location("t", "/tmp/wt_topk/tests/test_gpu2_launch_ministral_gk_batches.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
for val in ("", "gk_0_35\ngk_0_45"):
    tmp = Path(tempfile.mkdtemp(dir="/tmp/review_gk5")); gen = t.env.__wrapped__(tmp); env = next(gen)
    r = subprocess.run(["bash", str(t.SCRIPT)], env={**env["env"], "DRY_RUN": "1", "GK_CONFIGS": val}, capture_output=True, text=True)
    print(repr(val), "rc", r.returncode, [l.split("] ",1)[1][:110] for l in t._qlog(env["repo"]).splitlines() if "GK_CONFIGS" in l or "STOP" in l][:2])
