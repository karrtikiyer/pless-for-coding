import csv, json, sys
csv_path, batch, done_cfg, next_cfg = sys.argv[1:5]
step, nsteps = 1, 2
files = []
for a in sys.argv[5:]:
    if a.startswith("--step="):
        step, nsteps = (int(x) for x in a[len("--step="):].split("/", 1))
    else:
        files.append(a.split("=", 1))
final = next_cfg == "-"
want = {plat: set() for plat, _ in files}
with open(csv_path, newline="") as fh:          # csv module handles CRLF
    for row in csv.DictReader(fh):
        row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
        if row.get("batch") == batch.strip() and row.get("platform") in want:
            want[row["platform"]].add(row["task_id"])
tot_missing = tot_dup = tot_bad = 0
present = []
for plat, path in files:
    seen, dups, bad, other_dups = set(), set(), 0, 0
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:                       # stream; never print sample text
                if not line.strip():
                    continue
                try:
                    tid = str(json.loads(line)["task_id"])
                except Exception:
                    bad += 1
                    continue
                if tid in seen:
                    if tid in want[plat]:
                        dups.add(tid)
                    else:
                        other_dups += 1
                seen.add(tid)
    except FileNotFoundError:
        print(f"{plat}: FILE MISSING {path}")
    missing = sorted(want[plat] - seen)
    if not want[plat]:
        print(f"{plat}: WARNING batch {batch} has 0 {plat} tasks in {csv_path}")
        tot_missing += 1
    have = len(want[plat] & seen)
    present.append(have)
    print(f"{plat}: {done_cfg} batch{batch}_expected={len(want[plat])} present={have} "
          f"missing={len(missing)} dup={len(dups)} bad={bad}"
          + (f" (WARNING other-batch duplicate lines={other_dups})" if other_dups else ""))
    if missing:
        print(f"{plat}: missing (<=10): {missing[:10]}")
    if dups:
        print(f"{plat}: duplicated (<=10): {sorted(dups)[:10]}")
    tot_missing += len(missing); tot_dup += len(dups); tot_bad += bad
label = "FINAL" if final else f"STEP {step}"
if tot_missing or tot_dup or tot_bad:
    tail = (f" -- {done_cfg} batch {batch} NOT complete" if final
            else f" -- step {step + 1}/{nsteps} ({next_cfg}) NOT started")
    print(f"{label} INCOMPLETE: missing={tot_missing} dup={tot_dup} bad={tot_bad}{tail}")
    sys.exit(3)
print(f"{label} COMPLETE verified (" + "/".join(str(n) for n in present) + ")")
sys.exit(0)
