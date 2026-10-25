# Experiment 21: fixing late-game plasticity loss (tuning, 2026-10-25)

Question (from results/minatar.md, point 4): every sequential learner learns the later MinAtar games far worse
than from scratch (blind: Asterix 5.8 vs 16.4, Seaquest 2.1 vs 13.4). Can a cheap fix restore learning without
hurting MiniGrid?

## Setup
- TUNING seeds 101-105 only; final seeds 1-10 untouched. Code commit 8cebd4e, script run_fix_tune.sh.
- Candidates (all otherwise identical to MA_blind / L_blind):
  - `--reset-scope all` (A): a surprise resets the WHOLE network + Adam, not only the heads. Old skills come back
    only through reservoir distillation. Related to resets in [Nikishin 2022], but triggered by our own surprise,
    not on a schedule.
  - `--redo-every 10` (B): ReDo [Sokar 2023] every 10 rollouts, tau 0.1, heads reset on surprise as before.
  - TMA_none_redo: ReDo with no protection (diagnostic).
  - CLEAR coef 0.3 / 0.1 on MinAtar (coef 1.0 came from MiniGrid and blocked new games).
- Dormant fraction (tau 0.1) is now logged every 20 rollouts in every run (measurement only, no change).
- Selection rule, written in run_fix_tune.sh BEFORE running: a fix is adopted only if it beats TMA_blind on MinAtar
  (mean final score / final-seed scratch mean) AND is not worse than TL_blind on MiniGrid 8-stage by > 5 points.

## Results (mean ± 95% CI, 5 tuning seeds)
MinAtar, normalised by scratch (2M steps, final seeds):
| Run | final, % of scratch | just learned, % of scratch | final - TMA_blind (paired) | raw final BRK/SI/FW/AST/SEA |
|---|---|---|---|---|
| TMA_blind (current) | 28.4 ± 2.7 | 58.1 ± 8.4 | - | 3.9 / 14.7 / 22.8 / 2.2 / 1.8 |
| **TMA_blind_all (A)** | **43.4 ± 4.0** | **90.0 ± 6.7** | **+15.0 ± 5.4, 5/5 seeds** | 3.6 / 15.2 / 26.1 / 2.7 / 10.3 |
| TMA_blind_redo (B) | 25.6 ± 5.5 | 59.0 ± 6.6 | -2.8 ± 5.6, 2/5 | 3.8 / 15.0 / 18.7 / 1.6 / 1.8 |
| TMA_none_redo | 6.8 ± 1.8 | 70.0 ± 7.3 | -21.6 ± 2.2, 0/5 | 0.3 / 2.4 / 0.3 / 1.4 / 2.4 |
| TMA_clear_0.3 | 21.0 ± 3.6 | 36.8 ± 9.0 | -7.4 ± 4.3, 0/5 | 6.3 / 19.5 / 2.9 / 1.0 / 1.2 |
| TMA_clear_0.1 | 24.0 ± 4.6 | 47.2 ± 13.6 | -4.4 ± 7.1, 2/5 | 7.7 / 17.8 / 5.7 / 1.1 / 1.2 |

Raw score right after training on each game, blind vs blind_all: Asterix 5.1 vs 16.2, Seaquest 1.8 vs 10.3
(scratch 16.4, 13.4). Surprises per run: blind 3-6, blind_all 4-6.

MiniGrid 8-stage (final mean over the 11 trained tests):
| Run | final | - TL_blind (paired) |
|---|---|---|
| TL_blind | 92.6 ± 1.5 | - |
| TL_blind_all (A) | 92.2 ± 0.9 | -0.4 ± 2.0 |
| TL_blind_redo (B) | 94.6 ± 1.5 | +2.0 ± 0.9 |

Dormant units (tau 0.1): under 2% in every MinAtar run, ~0.4-0.7% for blind.

## Decision (by the pre-fixed rule)
- **A (reset whole network on surprise) is adopted**: +15 points on MinAtar (5/5 seeds), MiniGrid unchanged.
- B (ReDo) is not adopted: no gain on MinAtar. Its +2 on MiniGrid is noted but the rule needs a MinAtar gain.
- CLEAR with lower coef learns new games a bit better but still ends below blind; CLEAR in the paper stays at the
  MiniGrid-chosen coef 1.0, with this tuning reported as a sensitivity check (not tuned per domain for anyone).

## What it means
1. The late-game loss is NOT dormant units: they are under 2%, and ReDo does not help (even with no protection,
   learning stays at ~70% of scratch). The trouble lives in the weights of the shared body, not in dead units.
2. A full reset on surprise brings learning back to ~90% of scratch, and retention of earlier games is unchanged
   (Breakout/SI/Freeway equal or better), because old skills are kept in the reservoir and distilled back.
   This is the child-like picture we want: memory lives in stored experience, not in frozen weights.
3. On MiniGrid a full reset costs nothing measurable, so one setting works on both domains.
4. Limits: 5 tuning seeds only; final claim needs seeds 1-10 (run_fix_final.sh).

## Final (seeds 1-10, code 38c900c, run_fix_final.sh, 2h20m on 8 parallel runs)
MinAtar (% of scratch, mean ± 95% CI; paired = full reset minus other, same seed):
| Method | final | just learned | full reset - method | raw final BRK/SI/FW/AST/SEA |
|---|---|---|---|---|
| No protection | 5.2 ± 1.0 | 68.1 ± 3.3 | +37.5 ± 3.0, 10/10 | 0.4 / 2.0 / 0.0 / 1.0 / 1.8 |
| CLEAR (coef 1.0) | 16.5 ± 1.0 | 22.0 ± 1.3 | +26.2 ± 2.3, 10/10 | 4.0 / 21.4 / 0.2 / 0.9 / 0.9 |
| Blind, heads reset (exp 20) | 28.1 ± 1.0 | 59.4 ± 3.2 | +14.6 ± 2.5, 10/10 | 4.2 / 15.5 / 20.1 / 2.1 / 2.1 |
| Joint (10M mixed) | 31.8 ± 3.7 | - | +10.9 ± 4.3, 10/10 | 11.9 / 34.5 / 0.1 / 0.9 / 0.2 |
| **Blind, full reset** | **42.7 ± 2.2** | **91.1 ± 3.7** | - | 3.1 / 15.8 / 25.4 / 3.0 / 10.2 |
Just-learned raw: 11.7 / 43.5 / 37.8 / 18.5 / 10.2 (scratch 14.1 / 51.5 / 38.3 / 16.4 / 13.4). Surprises 4-6 per run.

MiniGrid (same final seeds, setups unchanged except --reset-scope all):
| Setup | heads reset (old) | full reset | paired diff |
|---|---|---|---|
| 8-stage, final mean of 11 trained tests | 92.5 ± 1.6 (L_blind) | 92.5 ± 1.4 (L_blind_all) | 0.0 ± 2.2 |
| 3-task, mean of 12 tests | 97.1 ± 1.5 (O_full) | 96.1 ± 1.7 (O_full_all) | -0.9 ± 2.4 |
| 3-task, unseen sizes | 94.3 ± 3.0 | 92.6 ± 3.2 | - |
8-stage per test (full reset): E8 100, E12u 100, D8 100, D12u 92, L7 99, L10u 86, Obst8 82, Obst10u 79, Cross9 92,
Cross11u 88, Unlock 100, KeyCorridor 0 (old: ... L10u 69, Obst8 88, Obst10u 85, Cross9 96, Cross11u 89).

Reading:
1. Tuning result holds on final seeds: +14.6 over heads reset, 10/10 seeds; later games learned at ~91% of scratch.
2. Now above joint training on MinAtar, 10/10 seeds (+10.9 ± 4.3). Caveat: joint here is NOT an upper bound - it
   never learns Freeway/Asterix/Seaquest in 10M mixed steps (interference between games); a longer or better-tuned
   joint run could do better. Claim "beats our joint baseline", not "beats joint training in general".
3. Cost: Breakout drops (4.2 -> 3.1 raw) - the reservoir only partly restores what the full reset erased.
4. MiniGrid: no measurable change (CIs overlap fully). One method setting now works on both domains.
5. Other MiniGrid experiments (exp 3 plasticity, generalisation, combined C_*) still use heads reset; the paper must
   say which variant each table uses, or rerun them. Figure: fig_minatar now includes the full-reset bars.
