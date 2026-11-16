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
