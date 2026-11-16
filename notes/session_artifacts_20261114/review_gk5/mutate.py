import json, sys
src, dst, mode, batch_ids_csv, batch = sys.argv[1:6]
import csv
ids = {r["task_id"] for r in csv.DictReader(open(batch_ids_csv)) if r["batch"] == batch}
lines = open(src, encoding="utf-8").readlines()
idx = next(i for i, l in enumerate(lines) if json.loads(l)["task_id"] in ids)
if mode == "remove": del lines[idx]
elif mode == "dup": lines.insert(idx + 1, lines[idx])
elif mode == "corrupt": lines[idx] = lines[idx][: len(lines[idx]) // 2] + "\n"
open(dst, "w", encoding="utf-8").writelines(lines)
print(mode, "affected task_id", json.loads(open(src).readlines()[idx])["task_id"])
