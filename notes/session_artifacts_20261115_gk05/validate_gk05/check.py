# adapted from /tmp/validate_gk045_{atcoder,leetcode}/check.py (+extra.py) for gk_0_35
import json, csv, hashlib, sys, statistics
from collections import Counter
PLAT=sys.argv[1]
F=f"/tmp/gk05_validate/gk_0_5_{PLAT}_think_t1.0.jsonl"
R="/Users/karrtikiyer/projects/airesearch/pless-for-coding/"
D=R+"results/pless_variants_cuda/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046/"
out=[]
def p(*a):
    s=" ".join(str(x) for x in a); print(s); out.append(s)
rows=list(csv.DictReader(open(R+"results/lcb_v6_full1046/batch_assignment.csv",newline="")))
rows=[{k.strip():(v or "").strip() for k,v in r.items()} for r in rows]
csvd={r["task_id"]:r for r in rows}
exp={r["task_id"] for r in rows if r["platform"]==PLAT and r["batch"] in ("1","2")}
p("PLATFORM",PLAT,"expected b1+b2:",len(exp),Counter(csvd[t]["batch"] for t in exp))
def summarize(path):
    recs={}; info=Counter(); dup=[]; keysets=Counter(); tt=Counter()
    with open(path,"rb") as fh:
        for i,raw in enumerate(fh):
            info["lines"]+=1
            if not raw.endswith(b"\n"): info["no_newline"]+=1
            s=raw.strip()
            if not s: info["blank"]+=1; continue
            if s.count(b'{"task_id"')>1: info["multi_taskid_marker"]+=1
            dec=json.JSONDecoder(); t=s.decode("utf-8","replace")
            try:
                r,pos=dec.raw_decode(t,0)
                if t[pos:].strip(): info["trailing_after_obj"]+=1
                r=json.loads(s)
            except Exception as e:
                info["unparseable"]+=1; p(f"  line {i+1} unparseable err={str(e)[:80]}"); continue
            tid=r.get("task_id"); tt[type(tid).__name__]+=1
            if tid in recs: dup.append(tid)
            keysets[tuple(sorted(r.keys()))]+=1
            swt=r.get("samples_with_thinking") or []; sm=r.get("samples") or []
            recs[tid]=dict(
              cfg=(r.get("method"),r.get("sampler_name"),r.get("renyi_k"),r.get("temperature"),r.get("top_p"),r.get("top_k"),r.get("backend"),r.get("enable_thinking"),len(sm),len(swt)),
              diff=r.get("difficulty"),src=r.get("source"),hasplat="platform" in r,ts=r.get("timestamp"),
              ph=hashlib.sha256(str(r.get("prompt_text")).encode()).hexdigest(),
              term=sum("[/THINK]" in x for x in swt), think=sum("</think>" in x for x in swt),
              empty=sum(1 for x in swt if not x.strip()), empty_s=sum(1 for x in sm if not (x or "").strip()),
              dupe=len(swt)-len(set(swt)), lens=[len(x) for x in swt], nsw=len(swt))
    with open(path,"rb") as fh:
        fh.seek(-1,2); info["last_byte_hex_"+fh.read(1).hex()]+=1
    return recs,info,dup,keysets,tt
a,ai,ad,ak,att=summarize(F)
ids=set(a)
p("sha256 file:",hashlib.sha256(open(F,"rb").read()).hexdigest())
p("leetcode-style id oddities (non-digit/leading zero):",sum(1 for t in a if not str(t).isdigit() or (len(str(t))>1 and str(t)[0]=="0")) if PLAT=="leetcode" else "n/a")
p("(1) line info:",dict(ai),"dups:",len(ad),ad[:10],"task_id types:",dict(att))
p("(2) records",len(a),"missing",len(exp-ids),sorted(exp-ids)[:10],"extra",len(ids-exp),sorted(ids-exp)[:10])
p("    difficulty mismatch:",sum(1 for t,v in a.items() if t in csvd and v["diff"]!=csvd[t]["difficulty"]),"source:",Counter(v["src"] for v in a.values()),"has platform key:",sum(v["hasplat"] for v in a.values()))
p("    per batch:",Counter(csvd[t]["batch"] for t in ids if t in csvd))
p("(3) cfg:",Counter(v["cfg"] for v in a.values()))
p("    keysets 0.5:",len(ak))
refs={}
for nm in ("gk_0_4","gk_0_35","gk_0_45","top_p_0_95"):
    refs[nm]=summarize(D+f"{nm}_{PLAT}_think_t1.0.jsonl")
    r_,ri,rd,rk,_=refs[nm]
    p(f"    ref {nm}: lines={ri['lines']} recs={len(r_)} dups={len(rd)} keysets={len(rk)} common={len(ids&set(r_))} cfg={Counter(v['cfg'] for v in r_.values()).most_common(2)}")
ks=set(list(ak)[0]); bks=set(list(refs["gk_0_4"][3])[0])
p("    keyset vs gk_0_4: 0.5-only",sorted(ks-bks),"0.4-only",sorted(bks-ks))
ts=sorted(v["ts"] for v in a.values() if v["ts"])
p("(4) ts missing:",len(a)-len(ts),"min",ts[0],"max",ts[-1],"before 2026-11-14T08:26:",sum(1 for t in ts if t<"2026-11-14T08:26"))
p("    per-hour:",sorted(Counter(t[:13] for t in ts).items()))
def agg(name,recs,sel):
    rs=[recs[t] for t in sel if t in recs]
    ns=sum(r["nsw"] for r in rs); L=[l for r in rs for l in r["lens"]]
    p(f"    {name}: tasks={len(rs)} samples={ns} term[/THINK]={sum(r['term'] for r in rs)/ns:.4f} </think>={sum(r['think'] for r in rs)} empty_swt={sum(r['empty'] for r in rs)} empty_samples={sum(r['empty_s'] for r in rs)} dup_within={sum(r['dupe'] for r in rs)} recs_with_dup={sum(1 for r in rs if r['dupe'])} medlen={statistics.median(L)} maxlen={max(L)}")
p("(5)")
agg("gk_0_5 all",a,ids)
c2=ids&set(refs["gk_0_4"][0])&set(refs["gk_0_45"][0])
c2=c2&set(refs["gk_0_35"][0])
for nm,rr in (("gk_0_5",a),("gk_0_4",refs["gk_0_4"][0]),("gk_0_35",refs["gk_0_35"][0]),("gk_0_45",refs["gk_0_45"][0])): agg(nm+" common(0.5,0.4,0.35,0.45)",rr,c2)
c3=c2&set(refs["top_p_0_95"][0])
for nm,rr in (("gk_0_5",a),("gk_0_4",refs["gk_0_4"][0]),("gk_0_35",refs["gk_0_35"][0]),("gk_0_45",refs["gk_0_45"][0]),("top_p_0_95",refs["top_p_0_95"][0])): agg(nm+" common5",rr,c3)
b=refs["gk_0_4"][0]
pd=[t for t in ids&set(b) if a[t]["ph"]!=b[t]["ph"]]
p("(6) prompt hash mismatch vs gk_0_4:",len(pd),"of",len(ids&set(b)),pd[:10],"; vs gk_0_45:",sum(1 for t in ids&set(refs['gk_0_45'][0]) if a[t]['ph']!=refs['gk_0_45'][0][t]['ph']),"; vs gk_0_35:",sum(1 for t in ids&set(refs['gk_0_35'][0]) if a[t]['ph']!=refs['gk_0_35'][0][t]['ph']))
g35=set(refs["gk_0_35"][0])
p("(7) ids 0.5 vs 0.35: equal=",ids==g35,"0.5-only",len(ids-g35),sorted(ids-g35)[:10],"0.35-only",len(g35-ids),sorted(g35-ids)[:10])
g45=set(refs["gk_0_45"][0])
p("(7) ids 0.5 vs 0.45: equal=",ids==g45,"0.5-only",len(ids-g45),sorted(ids-g45)[:10],"0.45-only",len(g45-ids),sorted(g45-ids)[:10])
open(f"/tmp/validate_gk05/log_{PLAT}.txt","w").write("\n".join(out)+"\n")
