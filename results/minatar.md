# Experiment 20: second environment, MinAtar (2026-10-24)

Question (paper step 3): does the blind continual learner work outside MiniGrid, on a different kind of task
(arcade games, dense scores, 6 actions)?

## Setup
- MinAtar [Young & Tian 2019]: 10x10 versions of 5 Atari games, sticky actions 0.1, difficulty ramping on.
- Sequence: Breakout -> Space Invaders -> Freeway -> Asterix -> Seaquest, 2M steps each (10M total).
- Same PPO and method settings as MiniGrid; nothing tuned on MinAtar except checking in a pilot (tuning seeds
  101-102, results/runs/pilotMA_*) that every game is learnable from scratch.
  - MA_none: sequential, no protection. MA_clear: CLEAR, reservoir 8192, coef 1.0 (chosen on MiniGrid).
  - MA_blind (ours): no task signal; competence measured relative to its own best score since the last surprise
    (mastery 0.8), reservoir 8192, distill coef 0.3, surprise (own score < 0.3 of best) -> heads + Adam reset.
  - MA_joint: all 5 games mixed, 10M steps. MAS_<game>: one game from scratch, 2M steps (normaliser).
- Final seeds 1-10; eval 30 episodes per game after each stage. Score normalised = score / mean scratch score.
- Code: commit 0eed4ec (runs restarted after an out-of-memory stop at commit 66519e7; no .py/.sh change between
  them, checked with git diff). Script run_minatar.sh. Wall time ~8 h on 12 CPU cores.

## Results (mean ± 95% CI, 10 seeds)
Scratch, 2M steps: Breakout 14.1±1.1, Space Invaders 51.5±5.7, Freeway 38.3±0.6, Asterix 16.4±2.7, Seaquest 13.4±3.3.

Score after all 5 games (raw):
| Method | Breakout | Space Inv. | Freeway | Asterix | Seaquest | mean % of scratch |
|---|---|---|---|---|---|---|
| No protection | 0.4±0.2 | 2.0±0.8 | 0.0±0.0 | 1.0±0.2 | 1.8±0.4 | 5±1% |
| CLEAR | 4.0±0.4 | 21.4±2.3 | 0.2±0.2 | 0.9±0.2 | 0.9±0.1 | 16±1% |
| **Blind (ours)** | 4.2±0.5 | 15.5±1.4 | **20.1±1.1** | 2.1±0.4 | 2.1±0.4 | **28±1%** |
| Joint (10M, mixed) | **11.9±1.9** | **34.5±5.2** | 0.1±0.1 | 0.9±0.4 | 0.2±0.1 | **32±4%** |

Score on each game right after training on it (learning ability, sequential methods):
| Method | Breakout | Space Inv. | Freeway | Asterix | Seaquest | mean % of scratch |
|---|---|---|---|---|---|---|
| No protection | 14.1 | 45.9 | 36.3 | 7.1 | 1.8 | 68±3% |
| CLEAR | 5.1 | 30.6 | 0.4 | 1.1 | 0.9 | 22±1% |
| Blind (ours) | 11.7 | 40.8 | 31.9 | 5.8 | 2.1 | 59±3% |

Paired over seeds (mean normalised score after all games):
- Blind - CLEAR: +12 ± 1 points, blind better on 10/10 seeds.
- Blind - none: +23 ± 1 points, 10/10.
- Blind - joint: -4 ± 4 points (CI from -8 to 0), blind better on 4/10. Close to joint, not shown equal or better.

Surprise detection (blind): a surprise right after every switch to Space Invaders, Freeway and Asterix in 10/10
seeds, after the switch to Seaquest in 7/10; also repeated surprises inside the Freeway stage (6/10 seeds).
Figure: paper/figs/fig_minatar.pdf.

## What it means
1. The MiniGrid ranking holds on a second, different domain: blind > CLEAR > no protection, all 10/10 seeds.
2. CLEAR keeps old games but almost cannot learn new ones here (Freeway 0.4 vs 38 from scratch): with replay
   coef 1.0 the replay loss dominates. On MiniGrid the same coef was fine. Its weakness is plasticity, not memory.
3. Joint training is not an upper bound here: it is best on Breakout / Space Invaders but never learns Freeway,
   Asterix or Seaquest in 10M mixed steps. Different games interfere; sequential training avoids that.
4. [FIXED in exp 21, results/plasticity_fix.md: full reset on surprise] Weakness of ours (and of every sequential method): the last two games are learned far worse than from scratch
   (Asterix 5.8 vs 16.4, Seaquest 2.1 vs 13.4), even with no protection at all (7.1, 1.8). Policy entropy is NOT
   collapsed (1.2-1.4 nats at stage start, max 1.79), so this is a different loss of plasticity than in MiniGrid
   (exp. 3); head reset alone does not fix it. Candidate fixes from the literature: reset more layers [Nikishin
   2022], continual backprop / ReDo [Dohare 2024, Sokar 2023]. Not tested yet.
5. Retention of old games is partial: blind keeps 30-52% of scratch score on games 1-3, far from MiniGrid's
   near-full retention.

## Honest claim for the paper
"On MinAtar, with no task signal and settings tuned only on MiniGrid, the blind learner beats CLEAR and no
protection on all 10 seeds and comes close to joint training (28% vs 32% of scratch, difference not significant at
95% only by a hair: CI -8 to 0). All sequential learners lose the ability to learn later games, a limitation we
report." Do not claim "matches joint" for MinAtar.
