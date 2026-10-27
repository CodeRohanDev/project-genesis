"""Summarize runs: mean ± 95% CI over seeds.

    python summarize.py B1_      # all runs whose name starts with B1_
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

T95 = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31, 10: 2.26,
       11: 2.23, 12: 2.20, 13: 2.18, 14: 2.16, 15: 2.14, 16: 2.13, 17: 2.12, 18: 2.11, 19: 2.10, 20: 2.09}


def ci(xs):
    """mean, half-width of 95% CI (t-dist)."""
    m = statistics.mean(xs)
    return m, (T95.get(len(xs), 1.96) * statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0)


def fmt(xs, pct=False):
    m, h = ci(xs)
    return f"{m:.0%} ± {h:.0%}" if pct else f"{m:,.0f} ± {h:,.0f}"


if __name__ == "__main__":
    groups = defaultdict(list)
    for f in Path("results/runs").glob(f"{sys.argv[1] if len(sys.argv) > 1 else ''}*.json"):
        groups[f.stem.rsplit("_seed", 1)[0]].append(json.loads(f.read_text()))

    for name, runs in sorted(groups.items()):
        tasks, evals = runs[0]["tasks"], runs[0]["eval_tasks"]
        print(f"\n## {name}  ({len(runs)} seeds, {statistics.mean(r['wall_seconds'] for r in runs):.0f}s/run)")
        print("Eval success after each stage (rows = trained up to, cols = tested on):")
        print("| after \\ eval | " + " | ".join(e.removeprefix("MiniGrid-") for e in evals) + " |")
        print("|---" * (len(evals) + 1) + "|")
        for i, t in enumerate(tasks):
            cells = [fmt([r["matrix"][i][j] for r in runs], pct=True) for j in range(len(evals))]
            print(f"| {t.removeprefix('MiniGrid-')} | " + " | ".join(cells) + " |")
        print("Steps to 80% train success (seeds that never reached it are listed separately):")
        for i, t in enumerate(tasks):
            hit = [r["train"][i]["steps_to_80"] for r in runs if r["train"][i]["steps_to_80"]]
            miss = len(runs) - len(hit)
            print(f"- {t.removeprefix('MiniGrid-')}: {fmt(hit) if hit else 'never'}" + (f"  ({miss} seeds never)" if miss else ""))
