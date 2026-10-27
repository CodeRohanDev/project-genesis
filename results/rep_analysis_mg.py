# Exp 28 analysis (MiniGrid, seeds 1-20 pooled; pre-repo groups use _v2 for seeds 1-10). PYTHONPATH=. .venv/bin/python results/rep_analysis_mg.py
import sys
sys.argv = sys.argv[:1]
exec(open(__file__.replace("rep_analysis_mg.py", "rep_analysis.py")).read().split('print("MinAtar")')[0])
import os
def grp(p):  # pooled 1-20 on current code: old pre-repo groups use _v2 for seeds 1-10
    v2 = os.path.exists(R/f"{p}_v2_seed1.json") or p in V2
    a = mg(p+"_v2", A) if v2 else mg(p, A)
    return a, mg(p, B)
V2 = {"L_distill","L_none","L_joint","C_distill","C_none","C_joint"}
for ours, others in [("L_blind_all", ["L_distill","L_clear","L_none","L_joint","L_rd","L_rdres"]),
                     ("O_full_all", ["C_distill","C_clear","C_none","C_joint","C_rd","C_rdres"])]:
    oa, ob = grp(ours); print(f"\n{ours:12s} 1-10 {s(oa)}  11-20 {s(ob)}  1-20 {s({**oa, **{k+100:v for k,v in ob.items()}})}  min {min(list(oa.values())+list(ob.values())):.0f}")
    for p in others:
        a, b = grp(p)
        if not a and not b: continue
        al = {**a, **{k+100:v for k,v in b.items()}}; ol = {**oa, **{k+100:v for k,v in ob.items()}}
        lo = sum(v < 60 for v in al.values())
        print(f"{p:12s} 1-10 {s(a)}  11-20 {s(b)}  1-20 {s(al)}  <60:{lo}")
        print(f"   ours-{p:8s} 1-10 {pair(oa,a) if a else '-'}  11-20 {pair(ob,b) if b else '-'}  1-20 {pair(ol,al) if a and b else '-'}")
