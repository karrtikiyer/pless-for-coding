import json,hashlib,sys
D="results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/"
def H(p):
    h={};tc={}
    for raw in open(p,"rb"):
        r=json.loads(raw); s=r["samples_with_thinking"]
        h[r["task_id"]]={hashlib.sha256(x.encode()).hexdigest() for x in s}; tc[r["task_id"]]=sum("[/THINK]" in x for x in s)
    return h,tc
for P in ("atcoder","leetcode"):
    a,at=H(f"/tmp/gk05_validate/gk_0_5_{P}_think_t1.0.jsonl")
    for nm in ("gk_0_4","gk_0_35","gk_0_45","top_p_0_95"):
        b,bt=H(D+f"{nm}_{P}_think_t1.0.jsonl")
        print(P,nm,"shared sample hashes:",sum(len(a[t]&b[t]) for t in a if t in b),"tasks w/ equal per-task term count:",sum(at[t]==bt[t] for t in a if t in b),"/",len(a))
