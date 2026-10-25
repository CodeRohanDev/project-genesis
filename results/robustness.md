# Experiment 23: robustness of the MinAtar result (2026-10-25)

Checks on the final method (Life, full reset on surprise, exp 21). Fixed ablations, nothing selected or tuned.
Seeds 1-10, code 1054ac7, script run_robust.sh, 7.7 h on 7 parallel runs. Scores in % of 2M-step scratch;
"ours" = MA_blind_all (normal order) or MAR_blind_all (reversed order); paired over seeds.

## 1. What does memory add? (full reset, NO reservoir/distill)
| Run | final | just learned | ours - run | raw final BRK/SI/FW/AST/SEA |
|---|---|---|---|---|
| MA_reset_all (no memory) | 25.3 ± 3.9 | 105.2 ± 7.5 | +17.4 ± 5.3, 10/10 | 0.5 / 0.7 / 1.3 / 0.7 / 15.3 |
| MA_blind_all (ours) | 42.7 ± 2.2 | 91.1 ± 3.7 | - | 3.1 / 15.8 / 25.4 / 3.0 / 10.2 |
MiniGrid 3-task (mean of 12 tests): O_reset_all 39.0 ± 8.8 vs O_full_all 96.1 ± 1.7, +57.1 ± 9.7 (old heads-reset
variant: O_reset 44.4 ± 11.7).
Reading: without memory the reset learns each new game fully (105% of scratch) but forgets all old ones (final
score is almost only the last game). The competence-gated reservoir is what keeps old skills; reset alone does not.
Memory costs ~14 points of new-game learning (105 -> 91).

## 2. Reversed game order (Seaquest -> Asterix -> Freeway -> Space Invaders -> Breakout)
| Run | final | just learned | ours - run |
|---|---|---|---|
| MAR_none | 20.0 ± 1.9 | 89.6 ± 6.6 | +17.2 ± 3.1, 10/10 |
| MAR_clear (coef 1.0) | 24.6 ± 1.9 | 32.9 ± 3.6 | +12.7 ± 3.3, 10/10 |
| MAR_blind_all (ours) | 37.2 ± 2.1 | 80.7 ± 3.6 | - |
Raw final ours: BRK 14.0, SI 14.5, FW 17.2, AST 1.4, SEA 0.6 (first-learned games mostly forgotten).
Reading: ranking ours > CLEAR > none holds in both orders on all seeds. Order matters for the size of the score
(37.2 vs 42.7, -5.5 ± 2.0): early Seaquest/Asterix are retained poorly.

## 3. Joint with 2x budget (20M mixed steps)
| Run | final | ours - run | raw final |
|---|---|---|---|
| MA_joint (10M) | 31.8 ± 3.7 | +10.9 ± 4.3, 10/10 | 11.9 / 34.5 / 0.1 / 0.9 / 0.2 |
| MA_joint20M | 36.2 ± 3.7 | +6.5 ± 4.2, 8/10 | 15.4 / 33.1 / 0.0 / 0.9 / 0.2 |
Reading: doubling joint's budget helps it (+4.4) but it still never learns Freeway/Asterix/Seaquest; ours still
ahead (CI 2.3 to 10.7, 8/10 seeds). Joint here is a weak baseline on MinAtar because of interference between games,
not an upper bound. Ours uses 10M steps, joint20M twice that.

## Honest claims for the paper
- "Memory, not reset, keeps old skills: removing the reservoir drops the final score by 17 (MinAtar) and 57
  (MiniGrid) points on all seeds."
- "The ranking over CLEAR and no protection holds in both game orders on all 10 seeds."
- "Beats our joint baseline even with 2x its budget (+6.5 ± 4.2, 8/10 seeds)." Not: "beats joint training".
- Still open: retention of the earliest games (esp. when hard games come first); one learning-rate/PPO setting.
