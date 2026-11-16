import json, csv, hashlib, sys, statistics
from collections import Counter, defaultdict
F="/tmp/gk045_validate/gk_0_45_leetcode_think_t1.0.jsonl"
R="/Users/karrtikiyer/projects/airesearch/pless-for-coding/"
D=R+"results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/"
out=[]
def p(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
rows=list(csv.DictReader(open(R+"results/lcb_v6_full1046/batch_assignment.csv",newline="")))
rows=[{k.strip():(v or "").strip() for k,v in r.items()} for r in rows]
p("CSV batch x platform:",Counter((r["platform"],r["batch"]) for r in rows))
csvd={r["task_id"]:r for r in rows}
exp={r["task_id"] for r in rows if r["platform"]=="leetcode" and r["batch"] in ("1","2")}
p("expected leetcode b1+b2:",len(exp))
def summarize(path, ids=None, full=False):
    recs={}; info=Counter(); dup=[]; keysets=Counter()
    with open(path,"rb") as fh:
        for i,raw in enumerate(fh):
            info["lines"]+=1
            if not raw.endswith(b"\n"): info["no_newline"]+=1
            s=raw.strip()
            if not s: info["blank"]+=1; continue
            try: r=json.loads(s)
            except Exception as e:
                info["unparseable"]+=1
                # test multiple objects
                dec=json.JSONDecoder(); t=s.decode("utf-8","replace"); pos=0; n=0
                try:
                    while pos<len(t):
                        _,pos=dec.raw_decode(t,pos); n+=1
                        while pos<len(t) and t[pos].isspace(): pos+=1
                except Exception: pass
                p(f"  line {i+1} unparseable, objects decodable prefix={n}, err={str(e)[:80]}")
                continue
            if s.count(b'{"task_id"')>1: info["multi_taskid_marker"]+=1
            tid=r.get("task_id")
            if ids is not None and tid not in ids: continue
            if tid in recs: dup.append(tid)
            keysets[tuple(sorted(r.keys()))]+=1
            swt=r.get("samples_with_thinking") or []
            sm=r.get("samples") or []
            recs[tid]=dict(
              cfg=(r.get("method"),r.get("sampler_name"),r.get("renyi_k"),r.get("temperature"),r.get("top_p"),r.get("top_k"),r.get("backend"),r.get("enable_thinking"),len(sm),len(swt)),
              diff=r.get("difficulty"),src=r.get("source"),platform=r.get("platform"),ts=r.get("timestamp"),
              ph=hashlib.sha256(str(r.get("prompt_text")).encode()).hexdigest(),
              term=sum("[/THINK]" in x for x in swt), think=sum("</think>" in x for x in swt),
              empty=sum(1 for x in swt if not x.strip())+0, empty_s=sum(1 for x in sm if not (x or "").strip()),
              dupe=len(swt)-len(set(swt)), lens=[len(x) for x in swt], nsw=len(swt))
    return recs,info,dup,keysets
a,ai,ad,ak=summarize(F)
p("== gk_0_45 line info:",dict(ai),"dups:",ad[:10],len(ad))
ids=set(a)
p("records",len(a),"missing",sorted(exp-ids)[:10],len(exp-ids),"extra",sorted(ids-exp)[:10],len(ids-exp))
p("difficulty mismatch:",sum(1 for t,v in a.items() if t in csvd and v["diff"]!=csvd[t]["difficulty"]))
p("source/platform Counter:",Counter((v["src"],v["platform"]) for v in a.values()))
p("cfg Counter:",Counter(v["cfg"] for v in a.values()))
p("keysets:",len(ak))
ks=list(ak)[0]
b,bi,bd,bk=summarize(D+"gk_0_4_leetcode_think_t1.0.jsonl")
p("== gk_0_4 line info:",dict(bi),"records",len(b),"dups",len(bd),"common with 0.45:",len(ids&set(b)))
bks=list(bk)
p("gk_0_4 keysets n=",len(bks),"; 0.45-only keys:",sorted(set(ks)-set(bks[0])),"; 0.4-only keys:",sorted(set(bks[0])-set(ks)))
p("gk_0_4 cfg:",Counter(v["cfg"] for v in b.values()))
c,ci,cd,ck=summarize(D+"top_p_0_95_leetcode_think_t1.0.jsonl")
p("== top_p line info:",dict(ci),"records",len(c),"dups",len(cd),"common:",len(ids&set(c)))
p("top_p cfg:",Counter(v["cfg"] for v in c.values()))
ts=sorted(v["ts"] for v in a.values() if v["ts"])
p("ts missing:",len(a)-len(ts),"min",ts[0],"max",ts[-1])
p("ts types sample:",ts[0][:30])
p("before 2026-11-11T14:11:",sum(1 for t in ts if t<"2026-11-11T14:11"))
p("per-hour:",sorted(Counter(t[:13] for t in ts).items()))
def agg(name,recs,sel):
    rs=[recs[t] for t in sel if t in recs]
    ns=sum(r["nsw"] for r in rs); L=[l for r in rs for l in r["lens"]]
    p(f"{name}: tasks={len(rs)} samples={ns} term[/THINK]={sum(r['term'] for r in rs)/ns:.4f} </think>={sum(r['think'] for r in rs)} empty_swt={sum(r['empty'] for r in rs)} empty_samples={sum(r['empty_s'] for r in rs)} dup_within={sum(r['dupe'] for r in rs)} recs_with_dup={sum(1 for r in rs if r['dupe'])} medlen={statistics.median(L)} maxlen={max(L)}")
com=ids&set(b)&set(c)
agg("gk_0_45 all",a,ids)
for nm,rr in (("gk_0_45",a),("gk_0_4",b),("top_p_0_95",c)): agg(nm+" common3",rr,com)
p("per-record term distribution 0.45:",sorted(Counter(v["term"] for v in a.values()).items()))
pd=[t for t in ids&set(b) if a[t]["ph"]!=b[t]["ph"]]
p("prompt hash mismatch vs gk_0_4:",len(pd),pd[:10]," vs top_p:",sum(1 for t in ids&set(c) if a[t]["ph"]!=c[t]["ph"]))
m=json.load(open(D+"metrics/gk_0_4_leetcode_think_t1.0_metrics.json"))
p("metrics keys:",list(m.keys())[:20])
pt=m.get("per_task")
if pt is not None:
    items=pt.items() if isinstance(pt,dict) else [(x.get("task_id"),x) for x in pt]
    pt=dict(items)
    p("metrics per_task n=",len(pt),"common w/0.45:",len(ids&set(pt)),"sample entry keys:",list(next(iter(pt.values())).keys())[:12] if isinstance(next(iter(pt.values())),dict) else type(next(iter(pt.values()))))
    nn=Counter((v.get("n") if isinstance(v,dict) else None) for t,v in pt.items() if t in ids)
    p("metrics n Counter on common:",nn)
p("expected per batch:",Counter(csvd[t]["batch"] for t in exp),"got per batch:",Counter(csvd[t]["batch"] for t in ids if t in csvd))
p("all keysets 0.45:",len(ak),"gk_0_4:",len(bk),"top_p:",len(ck))
p("ids with leading zero/non-digit in file:",sum(1 for t in ids if not str(t).isdigit() or (len(str(t))>1 and str(t)[0]=="0")),"in csv leetcode:",sum(1 for t in exp if not t.isdigit() or (len(t)>1 and t[0]=="0")))
import json as _j
tt=Counter()
with open(F,"rb") as fh:
    for raw in fh:
        if raw.strip(): tt[type(_j.loads(raw)["task_id"]).__name__]+=1
p("task_id types:",tt)
p("gk_0_4 records:",len(b),"top_p records:",len(c),"common3:",len(com))
open("/tmp/validate_gk045_leetcode/log.txt","w").write("\n".join(out)+"\n")
