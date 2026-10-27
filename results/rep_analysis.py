# Exp 28 analysis (MinAtar + gradual MinAtar). Run from repo root: PYTHONPATH=. .venv/bin/python results/rep_analysis.py
import json, sys, numpy as np
from pathlib import Path
from summarize import ci
R = Path("results/runs")
def load(p, seeds): return [json.loads((R/f"{p}_seed{s}.json").read_text()) for s in seeds if (R/f"{p}_seed{s}.json").exists()]
A, B, ALL = range(1,11), range(11,21), range(1,21)
games = ["breakout","space_invaders","freeway","asterix","seaquest"]
scratch = np.array([np.mean([r["matrix"][-1][i] for r in load(f"MAS_{g}", A)]) for i,g in enumerate(games)])
def ma(p, seeds): return {r["config"]["seed"]: 100*float(np.mean(np.array(r["matrix"][-1])/scratch)) for r in load(p, seeds)}
def mg(p, seeds):
    out = {}
    for r in load(p, seeds):
        row = r["matrix"][-1]; out[r["config"]["seed"]] = 100*float(np.mean(row[:11] if p.startswith(("L_","G_L")) or "_L_" in p else row))
    return out
def s(d):
    if not d: return "-"
    m,h = ci(list(d.values())); return f"{m:5.1f}±{h:4.1f} (n={len(d)})"
def pair(a, b):
    k = sorted(set(a)&set(b)); d = [a[i]-b[i] for i in k]; m,h = ci(d); return f"{m:+5.1f}±{h:4.1f} ({sum(x>0 for x in d)}/{len(k)})"
print("MinAtar")
for p in ["MA_blind_all","MA_clear","MA_rd","MA_joint","MA_none","GM_ours"]:
    print(f"{p:14s} 1-10 {s(ma(p,A))}  11-20 {s(ma(p,B))}  1-20 {s(ma(p,ALL))}")
o = {k: ma("MA_blind_all", k) for k in (A,B,ALL)}
for p in ["MA_clear","MA_rd","MA_joint","MA_none"]:
    print(f"ours-{p:10s} 1-10 {pair(o[A],ma(p,A))}  11-20 {pair(o[B],ma(p,B))}  1-20 {pair(o[ALL],ma(p,ALL))}")
print("GM_ours - abrupt(seeds1-10)", pair(ma("GM_ours",A), o[A]))
g = load("GM_ours", A)
print("GM per-game final % scratch:", np.round(100*np.mean([np.array(r["matrix"][-1])/scratch for r in g],0),1), " abrupt:", np.round(100*np.mean([np.array(r["matrix"][-1])/scratch for r in load("MA_blind_all",A)],0),1))
for p in ["GM_ours", "MA_blind_all"]:  # surprises within 2M steps after each game switch
    hit, d = 0, []
    for r in load(p, A):
        ev = [t for t, k in r["life_events"] if k == "surprise"]
        for b in [2e6, 4e6, 6e6, 8e6]:
            f = [t - b for t in ev if b <= t < b + 2e6]
            if f: hit += 1; d.append(min(f))
    print(p, f"{hit}/40 switches with a surprise, first after median {np.median(d)/1e3:.0f}k steps")
