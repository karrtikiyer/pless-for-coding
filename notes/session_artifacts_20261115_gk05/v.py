import json, numpy as np, sys
M="/Users/karrtikiyer/projects/airesearch/pless-for-coding/results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/metrics/"
cfgs=["gk_0_5","gk_0_35","gk_0_4","gk_0_45","top_p_0_9","top_p_0_95","ministral_recipe"]
data={}
for c in cfgs:
    t="t0.7" if c=="ministral_recipe" else "t1.0"
    keys=[];cs=[];meta={}
    for p,n in (("atcoder",242),("leetcode",178)):
        d=json.load(open(f"{M}{c}_{p}_think_{t}_metrics.json"))
        pt=d["per_task"]; ids=[x["task_id"] for x in pt]
        ok = len(pt)==n==d["num_tasks"] and len(set(ids))==n and all(len(x["pass_results"])==10 and x["num_correct"]==sum(x["pass_results"]) and len(x["statuses"])==10 for x in pt)
        c_=np.array([x["num_correct"] for x in pt])
        to=sum(s=="Timeout" for x in pt for s in x["statuses"])
        meta[p]=dict(ok=ok,corr=int(c_.sum()),p1=100*c_.mean()/10,p10=100*(c_>0).mean(),
             jp1=100*d["pass_at_k"]["1"],jp10=100*d["pass_at_k"]["10"],term=d["termination_pct"],tok=d["mean_token_size"],
             to=100*to/(10*n),jto=d["timeout_pct"])
        keys+= [f"{p}::{i}" for i in ids]; cs.append(c_)
    c_=np.concatenate(cs); data[c]=(keys,c_)
    print(c,{p:{k:(round(v,3) if isinstance(v,float) else v) for k,v in m.items()} for p,m in meta.items()})
    print("  pooled correct",c_.sum(),"/4200 p1=%.3f p10=%.3f"%(100*c_.mean()/10,100*(c_>0).mean()))
    if c=="gk_0_5":
        a,l=meta["atcoder"],meta["leetcode"]
        print("  pooled tok (sample-weighted)=%.1f term pooled from json=%.2f"%((a["tok"]*2420+l["tok"]*1780)/4200,(a["term"]*2420+l["term"]*1780)/4200))
k0=data["gk_0_5"][0]
print("alignment same order:",{c:data[c][0]==k0 for c in cfgs}, "sets:",{c:set(data[c][0])==set(k0) for c in cfgs})
A=data["gk_0_5"][1]
for B in cfgs[1:]:
    b=data[B][1]
    obs1=10*(A.mean()-b.mean()); obs10=100*((A>0).mean()-(b>0).mean())
    print(f"gk_0_5 - {B}: dcount={A.sum()-b.sum()} obs p1={obs1:.3f} p10={obs10:.3f}")
    for R in (2000,5000):
        rng=np.random.default_rng(42); idx=rng.integers(0,420,(R,420))
        d1=10*(A[idx].mean(1)-b[idx].mean(1)); d10=100*((A[idx]>0).mean(1)-(b[idx]>0).mean(1))
        print(f"   R={R} p1 CI[{np.percentile(d1,2.5):.2f},{np.percentile(d1,97.5):.2f}] <=0:{(d1<=0).mean():.3f} | p10 CI[{np.percentile(d10,2.5):.2f},{np.percentile(d10,97.5):.2f}] <=0:{(d10<=0).mean():.3f}")
