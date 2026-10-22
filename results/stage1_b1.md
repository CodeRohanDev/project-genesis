# Stage 1 / B1: PPO from scratch, one task each (2026-10-22)

Setup: ppo.py, CPU, 16 envs, 5 seeds, eval 100 episodes on held-out seed range. Mean ± 95% CI.

| Task | Budget | Final eval success | Seeds reaching 80% | Steps to 80% (hit seeds) |
|---|---|---|---|---|
| Empty-8x8 | 200k | 100% ± 0% | 5/5 | ~27k |
| DoorKey-8x8 | 1M | 40% ± 68% | 2/5 | ~430k |
| DoorKey-8x8 | 3M | 40% ± 68% | 2/5 | same seeds, same speed |
| LavaCrossingS9N1 | 1M | 14% ± 32% | 0/5 | - |
| LavaCrossingS9N1 | 3M | 39% ± 60% | 1/5 (+1 at 79%) | ~2M |

Observation: outcome is bimodal. A seed that finds reward early masters the task fast; others stay near 0%
for the whole run. 3x budget added nothing on DoorKey and helped only slowly on Lava.

Conclusion: main bottleneck is exploration (finding sparse reward), not learning capacity or budget.
Per PLAN decision rule: Stage 2 starts with curiosity (RND).

Caveat: 5 seeds, huge CI. Eval layouts drawn from finite 8x8 layout space, so not a true held-out
generalisation test yet (larger-grid test planned).

---

# Update: diagnosis + entropy fix (2026-10-22)

## Curiosity (RND) attempt: no effect
DoorKey, tuning seeds 101-103, 1M steps. Seeds solved: RND 0 -> 1/3, 0.005 -> 0/3, 0.02 -> 0/3, 0.1 -> 1/3.
Simple RND (single critic, egocentric view) gave no gain. H1 "RND fixes failed seeds" NOT supported for this implementation.

## Revised diagnosis: premature policy collapse
Several runs found reward (10-26% success) and then fell back to 0%. That is instability, not only exploration.
Stability variants, DoorKey tuning seeds:
| Variant | Solved |
|---|---|
| baseline (ent 0.01) | 1/3 |
| lr 2.5e-4 | 0/3 |
| no adv norm | 0/3 (slow ~10%) |
| no adv norm + RND 0.1 | 0/3 |
| ent 0.05 | 6/9 |
| ent 0.1 | 0/6 (max ~50%, never settles) |
-> entropy 0.05 adopted as new default ("B1v2").

## Final test, untouched seeds 1-10, 1M steps
| Task | B1 (ent 0.01) | B1v2 (ent 0.05) |
|---|---|---|
| DoorKey solved (eval >= 96%) | 3/10 | 7/10 |
| DoorKey mean eval success | 30% ± 34% | 71% ± 33% |
| DoorKey steps to 80% (solved seeds) | ~460k | ~710k (slower) |
| Lava solved | 1/10 | 1/10 |
| Lava mean eval success | 24% ± 22% | 21% ± 22% |

Per-seed DoorKey eval: B1  0 100 100 0 0 0 0 0 99 0 | B1v2  5 100 2 100 100 99 100 3 100 100

Statistics: DoorKey 3/10 vs 7/10, Fisher exact two-sided p ~ 0.18. Suggestive, NOT significant at 0.05.
Lava: no change; still unsolved at 1M by either.

Conclusions:
- Entropy 0.05 likely helps DoorKey (more seeds solved) at the cost of slower learning; not yet statistically proven.
- LavaCrossingS9N1 remains unsolved by plain PPO at 1M; cause not yet known.
- Lesson: baseline hyperparameters matter as much as the "new mechanism". Any Stage 2 claim must beat tuned baselines.
