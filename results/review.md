# Experiment 24: answers to likely reviewer questions (2026-10-26)

Code b08aa57, script run_review.sh, 85 runs, 6h10m on 12 parallel runs (CPU). Nothing here was selected or tuned
on final seeds; all comparisons were fixed before running. Stats: mean ± 95% CI, paired over seeds,
"ours - run" with the number of seeds where ours is higher.

## 1. Is CLEAR handicapped on MinAtar by its MiniGrid coef? (final seeds 1-10)
CLEAR coef 0.1 = its best MinAtar coef on tuning seeds (exp 21).
| Run | final, % of scratch | just learned | ours - run | raw final BRK/SI/FW/AST/SEA |
|---|---|---|---|---|
| CLEAR coef 1.0 (MA_clear) | 16.5 ± 1.0 | 22.0 ± 1.3 | +26.2 ± 2.3, 10/10 | 4.0 / 21.4 / 0.2 / 0.9 / 0.9 |
| CLEAR coef 0.1 (MA_clear_0.1) | 21.4 ± 2.0 | 42.5 ± 5.9 | +21.3 ± 3.5, 10/10 | 6.6 / 18.0 / 3.1 / 1.2 / 1.3 |
| Ours (MA_blind_all) | 42.7 ± 2.2 | 91.1 ± 3.7 | - | 3.1 / 15.8 / 25.4 / 3.0 / 10.2 |
Reading: the better coef helps CLEAR (+4.9) but it still learns new games at under half of scratch; the gap stays
large on all seeds. CLEAR keeps Breakout better than ours (6.6 vs 3.1 raw).

## 2. Reset & Distill (Ahn et al. 2024), adapted to PPO, TOLD the task boundaries
At each given task end: store the last rollout (2048 observations) with the current policy's logits, then reset the
whole network + Adam, and distill (KL, coef 0.3) toward all stored tasks while learning the next one. It is our
mechanism with the boundary given and a different memory (last rollout per task vs competence-gated reservoir).
| Setup | Reset & Distill (told) | Ours (blind) | ours - R&D | seeds |
|---|---|---|---|---|
| MinAtar, final % of scratch | 45.5 ± 4.2 (learned 93.7 ± 6.7) | 42.7 ± 2.2 (learned 91.1 ± 3.7) | -2.8 ± 4.6 | ours higher 3/10 |
| MiniGrid 8-stage, mean of 11 | 83.6 ± 10.9 | 92.5 ± 1.4 | +8.9 ± 11.6 | 7/10 |
| MiniGrid 3-task, mean of 12 | 84.8 ± 10.8 | 96.1 ± 1.7 | +11.4 ± 11.4 | 8/10 |
R&D raw final MinAtar: 5.9 / 17.4 / 21.7 / 3.0 / 10.3.
Per seed: R&D loses the first task COMPLETELY on 2/10 seeds in each MiniGrid setup (L_rd seeds 1, 6: Empty 3% / 0%,
Obstacles 0%; C_rd seeds 3, 4: Empty 0% at all sizes). Other seeds are close to ours.
Reading:
- MinAtar: no detectable difference (CI includes 0). The told version gets no clear benefit from knowing the
  boundaries; the blind version matches it without being told.
- MiniGrid: ours is more reliable (CI 4x smaller); R&D has occasional total collapse of an early skill. Mean gap is
  not significant (CIs include 0), so claim "matches told R&D on average, without collapses", not "beats R&D".
- Why R&D collapses is not tested. Hypothesis: one stored rollout per task is a thin sample (e.g. the Empty stage is
  three grid sizes); our reservoir keeps samples from the whole competent period.

## 3. Sensitivity of our method (TUNING seeds 101-105, MiniGrid 8-stage, one setting changed at a time)
| Setting (default) | value | final mean of 11 | base - run | surprises per run |
|---|---|---|---|---|
| default | - | 92.2 ± 0.9 | - | 2-3 |
| mastery (0.8) | 0.7 | 92.8 ± 3.8 | -0.6 ± 4.1 | 2-3 |
| | 0.9 | 95.3 ± 1.4 | -3.0 ± 1.4, base lower 5/5 | 2-3 |
| surprise (0.3) | 0.2 | 92.9 ± 2.1 | -0.7 ± 1.5 | 2-3 |
| | 0.4 | 91.4 ± 2.0 | +0.8 ± 2.0 | 2-4 |
| distill coef (0.3) | 0.1 | 92.1 ± 4.3 | +0.2 ± 4.1 | 2-4 |
| | 1.0 | 87.0 ± 1.9 | +5.3 ± 1.7, 5/5 | 2-3 |
| reservoir (8192) | 2048 | 87.0 ± 5.6 | +5.3 ± 4.9, 5/5 | 2-3 |
| | 32768 | 93.0 ± 1.1 | -0.7 ± 1.7 | 2-3 |
TS_base repeats TL_blind_all (exp 21, same seeds/config): 92.2 both, so runs are reproducible.
Reading:
- Within 0.7-0.9 mastery, 0.2-0.4 surprise, 0.1-0.3 coef, 8k-32k reservoir the score stays 91-95: no knife-edge.
- Two settings hurt by ~5 points: distill coef 1.0 (too strong, blocks new learning) and a 4x smaller reservoir.
- Mastery 0.9 is a little better (+3.0) on these tuning seeds. Not adopted: the setting was fixed before final runs
  and was not tested on MinAtar. Reported as is.
- Limits: MiniGrid only, 5 seeds; MinAtar sensitivity not run (2.5 h per run).

## Honest claims for the paper
- "A fairly tuned CLEAR (best MinAtar coef) still ends 21 points lower, 10/10 seeds."
- "Without being told task boundaries, our learner matches a told Reset & Distill on MinAtar (-2.8 ± 4.6) and avoids
  the early-skill collapses R&D shows on 2/10 MiniGrid seeds."
- "Results are stable across a 2-4x range of each setting (MiniGrid, tuning seeds); too strong distillation or a
  much smaller memory costs ~5 points."
