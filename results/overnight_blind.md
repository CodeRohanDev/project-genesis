# O1: blind continual learning, no task-boundary signal (2026-10-23, overnight)

Same setup as C1 (results/combined.md): Empty -> DoorKey -> Lava, mixed sizes, 600k steps/task, seeds 1-10,
12 eval tests incl. unseen sizes. But the agent is NEVER told the task changed. It only sees its own success.

Mechanisms (class Life in ppo.py):
- reservoir: fixed 8192 slots. While own success >= 80% ("mastered"), 256 observations per update + current action
  probabilities are offered to the reservoir (reservoir sampling = uniform over all mastered history). Memory does not grow.
- replay-distillation from the reservoir on every PPO minibatch (KL, coef 0.3, same as C_distill).
- surprise: after mastery, if own success falls below 30% -> event; with --surprise-reset, reset heads + Adam state.
Thresholds (80% / 30%) fixed before the run, not tuned.

Final eval after all three tasks (mean ± 95% CI):
| Variant | Empty 5 | E 8 | E 10u | E 12u | DK 5 | DK 8 | DK 10u | DK 12u | Lava 5 | L 7 | L 8u | L 10u | worst test per seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_none (no protection) | 51 | 33 | 33 | 35 | 43 | 25 | 23 | 20 | 59 | 45 | 35 | 23 | 0-14 |
| C_distill (TOLD boundaries) | 100 | 100 | 100 | 100 | 100 | 99 | 94±4 | 83±10 | 96±8 | 93±12 | 90±11 | 76±16 | 18-92 |
| C_joint (all at once) | 100 | 100 | 100 | 100 | 100 | 100 | 95±6 | 90±5 | 100 | 100 | 93±5 | 76±11 | 46-96 |
| O_reset (blind, surprise reset only) | 41 | 41 | 42 | 43 | 22 | 6 | 7 | 5 | 92±18 | 89±21 | 83±19 | 62±16 | 0-12 |
| **O_replay (blind, reservoir only)** | 100 | 100 | 100 | 100 | 100 | 100 | 99±2 | 95±6 | 99±1 | 99±2 | 96±3 | 75±14 | 34-94 |
| **O_full (blind, reservoir + surprise reset)** | 100 | 100 | 100 | 100 | 100 | 100 | 99±1 | 97±3 | 100 | 99±1 | 96±4 | 74±13 | 34-98 |

Self-detected task change (hidden switches at 600k and 1.2M steps). Surprise events, in thousand steps:
- O_full: every seed detected the 1.2M switch at ~1202k (2k steps late). 4/10 also at ~630k (others never collapsed on
  DoorKey: Empty skill + replay carried over, so there was nothing to detect). No false alarms.
- O_reset: also caught both switches, but 4 extra events (143k, 333k, 550k, 1800k) = false alarms from unstable learning.

Lava learned to 80%: O_full 10/10 at ~73k steps, O_replay 10/10 at ~107k, C_distill 9/10 at ~75k.

## Findings
1. The agent does not need to be told when the task changes. Blind reservoir replay matches or slightly beats the
   boundary-given method and matches the all-at-once learner, within CI, on all 12 tests including unseen sizes.
2. Memory no longer grows with tasks (fixed 8192), which removes a second limitation from combined.md.
3. Self-detection works: the "surprise" signal (own success collapsing) fires within ~2-30k steps of a real switch and,
   with replay present, gave no false alarms.
4. Surprise reset mostly buys speed (Lava ~73k vs ~107k steps), not final quality.
5. Unexpected: replay alone, without any reset, avoided the plasticity loss seen in B2 (Lava 0/10 there, 10/10 here).
   Hypothesis, NOT tested: distilling toward stored, earlier, less-certain policies keeps the policy from collapsing
   to near-deterministic. Policy entropy logs exist to check this.
6. Reset without replay (O_reset) = learns new task, forgets old: same as C_heads. Replay is the essential part.

## Limits
- Detection relies on success collapsing. A new task that is easy from old skills never triggers it (fine here,
  but a subtly different task with a wrong old habit could go undetected).
- Thresholds 80/30 picked by hand. Still one gridworld, 3 tasks. Long-sequence test running next (L1).

## Check of hypothesis 5 (from existing entropy logs, no new runs)
| Variant | entropy at end of DoorKey | Lava 1st rollout | Lava first 10 rollouts mean |
|---|---|---|---|
| C_none | 0.21 | 0.06 | 0.36 |
| O_replay | 0.55 | 0.29 | 0.77 |
| O_full | 0.53 | 0.25 | 1.02 |
| C_heads / C_distill (explicit reset) | 0.22 | 1.86 | 1.83 / 1.59 |
Reservoir replay keeps the policy ~2.5x less deterministic at the switch, and it reopens faster on the new task.
Consistent with the hypothesis (correlation only; mechanism not isolated).
