# Generalisation test: bigger unseen grids, no extra training (2026-10-22)

Saved final agents (seeds 1-10) evaluated, 50 episodes each, on larger versions of the trained tasks.
Agents were trained only on 5x5 / S5. Same 7x7 egocentric view. Script: gen_eval.py, raw: generalisation.json.

| Agent | Empty 5 | Empty 6 | Empty 8 | DoorKey 5 | DoorKey 6 | DoorKey 8 | Lava S5 | Lava S6 | Lava S7 |
|---|---|---|---|---|---|---|---|---|---|
| untrained (random init) | 42% | 31% | 20% | 8% | 4% | 1% | 6% | 4% | 2% |
| scratch, one task each (B1s) | 90% | 59% | 58% | 100% | 58% | 14% | 100% | 35% | 15% |
| sequential, no protection | 67% | 16% | 10% | 28% | 14% | 3% | 25% | 19% | 6% |
| heads reset | 46% | 45% | 27% | 10% | 4% | 1% | 98% | 40% | 10% |
| heads + EWC | 55% | 28% | 20% | 45% | 22% | 7% | 81% | 27% | 6% |
| heads + distill | 90% | 24% | **1%** | 100% | 26% | 3% | 96% | 35% | 7% |
(CI ±10-33% at 10 seeds; see script output for exact intervals.)

## Findings
1. Skills do not generalise. Every agent collapses as the grid grows. Best case DoorKey: 100% -> 58% -> 14%.
2. The distill agent, best at retention, generalises worst: Empty-8x8 1%, below the untrained agent (20%).
   It copies its exact 5x5 behaviour; in a bigger room that behaviour gets stuck, while a random policy at least wanders.
3. Retention != understanding. Distillation preserved *behaviour on remembered observations*, not a transferable skill.

## Caveat (confound)
In 5x5 the 7x7 view covers the whole room: the goal is always visible. In bigger rooms the goal is often out of view,
so the agent needs search + memory of where it has been, a skill the 5x5 tasks never required.
So this measures "transfer to a harder version", not pure layout memorisation. Both matter for the AGI goal.

## Implication
Continual-learning result from stage1_small.md holds only inside the training distribution.
Next experiment should target generalisation directly: train on varied sizes, test on held-out sizes,
and test whether working memory (recurrent policy) is what is missing for search.

---

# G1: variety x working memory (2026-10-23)

DoorKey. 1.5M steps, seeds 1-10, entropy 0.05, 100 eval episodes per size. Script: run_gen.sh.
Train "5" = 5x5 only. Train "mix" = 16 parallel envs split across 5x5, 6x6, 8x8.
Sizes 7, 10, 12 never seen by any agent. 6 and 8 unseen only for the "5" agents.

| Agent | 5 | 6 | 7 (unseen) | 8 | 10 (unseen) | 12 (unseen) | steps to 80% train |
|---|---|---|---|---|---|---|---|
| CNN, train 5 | 100% | 42% ± 24% | 18% ± 20% | 8% ± 11% | 4% ± 5% | 3% ± 5% | ~60k |
| LSTM, train 5 | 100% | 69% ± 17% | 34% ± 21% | 17% ± 14% | 7% ± 8% | 4% ± 4% | ~190k |
| **CNN, train mix** | **100%** | **100%** | **100%** | **100%** | **96% ± 3%** | **82% ± 10%** | ~134k |
| LSTM, train mix | 97% ± 7% | 94% ± 13% | 84% ± 16% | 69% ± 21% | 56% ± 25% | 44% ± 24% | ~650k (1 seed never) |

Per seed CNN mix (5/6/7/8/10/12): every seed 100% up to size 8; size 12 between 60% and 97%.

## Findings
1. Variety of experience is what produced generalisation. Same network, same budget: training on three sizes instead
   of one moves unseen size 10 from 4% to 96% and unseen size 12 from 3% to 82%. Seeds agree (no seed below 60% at 12).
2. Side effect: mixing sizes solved DoorKey-8x8, which plain PPO failed on in Stage 1 (30-71%). Small rooms act as a
   curriculum: the agent learns key -> door -> goal where reward is easy to find, then the skill carries to big rooms.
   This also answers the "exploration" side track for DoorKey: curriculum beat curiosity here.
3. Working memory (LSTM) did not help in this test. It helped a little when training on 5x5 only
   (size 6: 69% vs 42%), but with variety it learned ~5x slower and generalised worse.
   Caveat: LSTM used the CNN's hyperparameters, untuned, 1.5M steps. Conclusion is only
   "naive LSTM, same budget, did not help", not "memory is useless".
4. Generalisation still decays with distance from training sizes (12: 82%). Interpolation (7) is perfect,
   extrapolation (10, 12) is partial.

## Meaning for the AGI goal
Child-like learning claim that this supports: varied experience -> skill that transfers to new situations.
What it does not show: transfer to a different kind of task (all of this is DoorKey).
