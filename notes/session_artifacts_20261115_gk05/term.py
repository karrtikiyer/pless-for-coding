import json
tot={}
for p in ("atcoder","leetcode"):
    n=c=nr=0; ids=set()
    with open(f"/tmp/gk05_validate/gk_0_5_{p}_think_t1.0.jsonl") as f:
        for line in f:
            r=json.loads(line); nr+=1; ids.add(r["task_id"])
            for s in r["samples_with_thinking"]:
                n+=1; c+= "[/THINK]" in s
    tot[p]=(n,c); print(p,"records",nr,"unique",len(ids),"samples",n,"closed",c,"term%%=%.2f"%(100*c/n))
N=sum(v[0] for v in tot.values()); C=sum(v[1] for v in tot.values()); print("pooled %d/%d = %.2f"%(C,N,100*C/N))
