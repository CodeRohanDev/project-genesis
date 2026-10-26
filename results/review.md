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

# Experiment 25: sensitivity on MinAtar (2026-10-26)
Code 297dd7a, script run_sens_ma.sh, 35 runs, 7h20m on 12 parallel. TUNING seeds 101-105; reference
TMA_blind_all (exp 21, same seeds and flags). One setting changed at a time, same values as exp 24 section 3.
| Setting (default) | value | final, % of scratch | just learned | base - run | surprises |
|---|---|---|---|---|---|
| default | - | 43.4 ± 4.0 | 90.0 ± 6.7 | - | 4-6 |
| mastery (0.8) | 0.7 | 45.2 ± 7.0 | 93.7 ± 3.4 | -1.7 ± 6.6 | 4-6 |
| | 0.9 | 42.2 ± 3.2 | 87.2 ± 9.0 | +1.3 ± 6.3 | 4-6 |
| surprise (0.3) | 0.2 | 46.5 ± 3.8 | 92.2 ± 5.0 | -3.1 ± 5.0 | 4-5 |
| | 0.4 | 42.4 ± 4.9 | 89.7 ± 5.2 | +1.0 ± 1.9 | 5-6 |
| distill coef (0.3) | 0.1 | 37.4 ± 7.5 | 98.1 ± 11.8 | +6.0 ± 11.3, 4/5 | 4-5 |
| | 1.0 | 37.5 ± 4.0 | 64.4 ± 4.3 | +5.9 ± 7.7, 4/5 | 3-5 |
| reservoir (8192) | 2048 | 34.5 ± 7.4 | 91.6 ± 10.4 | +9.0 ± 5.5, 5/5 | 4-6 |
Reading:
- Mastery 0.7-0.9 and surprise 0.2-0.4: 42-47, no detectable change. Surprise detection is robust (4-6 per run).
- Distill coef has a trade-off, as expected: 0.1 learns new games fully (98%) but keeps less; 1.0 keeps old games
  but learns new ones at 64%. Both ~6 below default (CIs include 0). 0.3 sits between.
- A 4x smaller reservoir costs 9 points (5/5 seeds): memory size matters most, as on MiniGrid.
- Mastery 0.9 (+3 on MiniGrid, exp 24) gives -1.3 here: no reason to change the default.
- Every variant still beats every baseline's final-seed mean (CLEAR 21.4, heads reset 28.1, joint 31.8), but
  tuning vs final seeds are not paired, so this is indicative only.

# Experiment 26: why does told Reset & Distill collapse? (2026-10-26)
Code b3fbc1d, script run_rdres.sh, 20 runs, 28 min on 12 parallel. Final seeds 1-10, fixed before running.
Control "told reset + our memory" (L_rdres, C_rdres): full reset + fresh Adam at every GIVEN task boundary, memory =
our competence-gated reservoir (8192, distill 0.3), no surprise reset. Differs from R&D only in memory, and from
ours only in what triggers the reset.
| Setup | R&D (told, last-rollout memory) | told reset + our memory | ours (blind, surprise reset) |
|---|---|---|---|
| MiniGrid 8-stage, mean of 11 | 83.6 ± 10.9 (worst seed 55) | 87.0 ± 1.8 (worst 83) | 92.5 ± 1.4 (worst 89) |
| MiniGrid 3-task, mean of 12 | 84.8 ± 10.8 (worst 55) | 96.5 ± 1.9 (worst 92) | 96.1 ± 1.7 (worst 91) |
Paired: 3-task, our memory vs R&D memory +11.8 ± 10.7 (9/10); told vs blind trigger +0.4 ± 2.8 (5/10).
8-stage, blind trigger vs told reset (same memory) +5.5 ± 1.9, ours higher on 10/10 seeds.
Reading:
1. The memory explains the collapse: with our reservoir, no seed loses an early skill (0/20 vs 4/20 for R&D).
2. On the 8-stage sequence, being TOLD the boundaries is worse than detecting them (10/10 seeds). The told version
   resets at all 7 boundaries; ours resets only when its own success collapses (2-4 times, exp 21), so it keeps
   useful features when the next stage is similar (repeated DoorKey/LavaGap stages, related grids). The loss is on
   the later stages (Obstacles 70 vs 82, Crossing 72 vs 92). Not tested directly: that fewer resets is the cause.
3. On 3 different tasks (2 boundaries, both surprise every time) told and blind are equal, as expected.
Claim for the paper: "Detecting change beats being told about it when not every change needs a reset (+5.5 ± 1.9,
10/10 seeds); the competence-gated memory, not the boundary signal, prevents the collapses of Reset & Distill."

# Experiment 27: gradual task change (2026-10-26)
Code a7b1bd5, script run_drift.sh, 30 runs, 24 min on 12 parallel. Final seeds 1-10, fixed before running.
MiniGrid 3-task (Empty -> DoorKey -> LavaGap, 600k steps each). Each new stage starts as a mix: an episode is the
new task with probability rising linearly 0 -> 1 over the first 200k steps (--drift 200000).
| Run | gradual (mean of 12) | abrupt counterpart | gradual - abrupt |
|---|---|---|---|
| Ours (blind, surprise reset) G_ours | 97.0 ± 1.4 (worst seed 93) | O_full_all 96.1 ± 1.7 | +0.8 ± 2.6 |
| Told reset + our memory G_told | 96.2 ± 1.3 | C_rdres 96.5 ± 1.9 | - |
| No protection G_none | 53.9 ± 7.7 | C_none 35.3 ± 10.9 | - |
Ours - none (gradual): +43.0 ± 7.2, 10/10. Ours - told (gradual): +0.7 ± 1.4 (n.s.).
Surprises (stage 2 / stage 3 transitions): gradual 0/10 and 2/10 seeds; abrupt 4/10 and 10/10.
Reading:
1. As expected, a gradual change mostly does NOT trigger the surprise (2 of 20 transitions): success never
   collapses because old episodes are still mixed in while the new task is learned.
2. It does no harm here: old skills are kept (97.0, same as abrupt) because the competence-gated reservoir keeps
   collecting and distilling whatever the agent masters; no reset is needed on MiniGrid (exp 21: heads vs full
   reset equal). Retention comes from memory, not from detecting the change.
3. Not tested: MinAtar, where the reset is what restores learning (+15, exp 21). Under gradual change there the
   reset would likely not fire, so the plasticity gain would likely be lost. Open limitation.
4. Gradual change also helps the unprotected learner (53.9 vs 35.3): less abrupt forgetting.
(steps_to_80 is not meaningful under drift: early episodes are mostly the old, already mastered task.)
