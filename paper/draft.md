# Paper draft 1 (2026-10-25) - IJCNN 2027, IEEE format, 6 pages

Status: full first draft in Markdown; numbers taken from results/*.md (each table cites its source). Convert to
IEEEtran LaTeX after review. [TODO] marks open items. References [n] = paper/related_work.md.

---

## Title
**Told Nothing, Forgetting Little: Boundary-Blind Continual Reinforcement Learning with Competence-Gated Memory
and Surprise Resets**

## Abstract (~200 words)
Continual reinforcement learning agents are usually told when a task ends, and use that signal to consolidate
knowledge, store data or reset parts of the network. Children and animals get no such signal. We study an agent
that is never told about task changes and uses only what it can observe about itself: its own success. It (i) stores
experience in a fixed-size reservoir only while it is competent, (ii) distils its policy toward that memory on every
update, and (iii) when its success collapses after mastery, treats this surprise as a change of the world and
re-initialises its network, so that old skills survive only in memory. With PPO and no task signal, on an 8-stage
MiniGrid sequence with six kinds of task the agent keeps 92.5% success on the trained tests, equal to the same
learner given task boundaries and 25.9 points above CLEAR. On a sequence of five MinAtar games it reaches 42.7% of
single-game scores after all games, against 16.5% for CLEAR, 5.2% for no protection and 31.8% (36.2% with twice the
budget) for training on all games at once, on all or most of 10 seeds and in both game orders. Ablations show that
memory, not the reset, retains old skills, and that the reset, not dormant-unit repair, restores learning.

## I. Introduction
1. A child is not told "task 3 starts now". It notices that what used to work fails and adapts, and keeps what it
   once mastered. Most continual RL methods assume the opposite: task identities or boundaries are given and drive
   consolidation (EWC [3], Progress & Compress [6]), per-task buffers or resets (Reset & Distill [25]).
2. Task-free methods exist [9]-[12], [26]; in RL, CLEAR [8] replays all past experience with behavioural cloning and
   needs no labels. Separately, agents that keep training lose plasticity [15]-[19].
3. Question: how far can an agent get when it must decide by itself *what* to remember and *when* the world
   changed, with a single fixed-size network and memory?
4. Contributions:
   - A boundary-blind learner built from three simple, self-triggered parts: competence-gated reservoir memory,
     replay distillation, and surprise-triggered full reset (Sec. III).
   - Evidence on two domains, 10 seeds each, against CLEAR, no protection, told-boundary distillation and joint
     training: MiniGrid 3-task and 8-stage sequences, and five MinAtar games (Sec. V).
   - Ablations and diagnostics: memory vs. reset, game order, a 2x joint budget, entropy and dormant units as
     plasticity diagnostics; the plasticity loss differs between domains (Sec. VI).
5. Honesty statement: tuning only on separate tuning seeds (101+), final seeds 1-10, selection rules fixed before
   running; all code and run logs released. [TODO: repo link]

## II. Related work
Use paper/related_work.md draft 2 text (~0.6 column). Key differences to state:
- vs Reset & Distill [25]: same parts (reset + distill), but it is told when tasks end; we detect it.
- vs Aljundi et al. [9]: consolidate at loss plateaus (supervised); our analogue is "store while competent" in RL.
- vs primacy-bias resets [19]: fixed schedule, single task; ours triggered by surprise, continual.
- vs CLEAR [8]: also label-free; it stores everything and clones it; we store only mastered behaviour.
- 3RL [12]: task-agnostic can beat multi-task due to interference - we see the same with MinAtar joint.

## III. Method
**Setting.** One agent faces tasks T1, T2, ... in sequence, each for a fixed number of environment steps. It
receives observations and rewards only; it is never given task identity, boundaries or the number of tasks.

**Base learner.** PPO [21] (CleanRL-style [22]): 16 parallel environments, 128-step rollouts, 4 epochs, 8 minibatches,
Adam lr 1e-3, gamma 0.99, GAE 0.95, clip 0.2, entropy 0.05, value 0.5, grad-norm 0.5. Small CNN body with a 64-unit
tanh feature layer, linear actor and critic heads (MiniGrid: 7x7x3 egocentric view, 7 actions; MinAtar: 10x10x10,
6 actions). The same settings are used for all methods and both domains.

**Competence signal.** c = mean outcome of the agent's last 100 finished episodes. MiniGrid: outcome = success (1 if
reward > 0). MinAtar (scores, no success): c = (mean score of last 100 episodes) / (best such mean since the last
surprise), so competence is measured relative to the agent's own best, never to an external target.

**(i) Competence-gated memory.** While c >= m (mastery, m = 0.8), after each rollout 256 random observations of
that rollout are offered, with the current policy's logits, to a reservoir of fixed size N = 8192 using reservoir
sampling [13]. The memory therefore holds a uniform sample of all *mastered* behaviour, never grows, and does not
store rewards or actions for RL updates.

**(ii) Replay distillation.** Every PPO minibatch adds lambda * KL(pi_stored || pi_theta) over a batch of reservoir
observations, lambda = 0.3. Old behaviour is kept as a function of observations, not as frozen weights.

**(iii) Surprise reset.** If the agent had mastered (c >= m) and c then drops below 0.3, it logs a surprise, clears
the mastered flag, and re-initialises all network parameters and the Adam state. It keeps the reservoir; from the
next update, distillation pulls the fresh network back toward stored behaviour while PPO learns the new task.
[Fig. 1: diagram of the loop - TODO draw]

**Why a full reset.** On MiniGrid the plasticity loss is a near-deterministic policy (entropy 0.01 of 1.95 at the
start of a new task, Sec. VI), and resetting only the output heads suffices. On MinAtar entropy stays high, yet later
games are learned at ~59% of scratch; only a full reset restores this (~91%). We use the full reset everywhere; on
MiniGrid it gives the same results as the head reset (Sec. VI-C).

## IV. Experimental setup
**MiniGrid [23].**
- C1, 3 tasks: Empty -> DoorKey -> LavaGap, each a mix of 3 sizes, 600k steps per task; 12 tests incl. unseen sizes,
  50 episodes each.
- L1, 8 stages, 6 task kinds: Empty, DoorKey, LavaGap, Dynamic-Obstacles, SimpleCrossing, Unlock, DoorKey (revisit),
  LavaGap (revisit), 500k steps per stage; 12 tests (11 trained kinds/sizes + KeyCorridor, never trained), 30 episodes.
**MinAtar [24].** Breakout -> Space Invaders -> Freeway -> Asterix -> Seaquest, 2M steps each (sticky actions 0.1);
reversed order as a robustness check. Score normalised by the mean score of a single-game learner trained for 2M
steps from scratch (10 seeds); 30 evaluation episodes per game.
**Baselines.**
- No protection: plain sequential PPO.
- CLEAR [8], adapted to PPO: reservoir of all past experience (same size 8192), off-policy clipped importance-weighted
  policy gradient on replay plus coef x (policy KL + value MSE) cloning; coef 1.0 chosen on C1 tuning seeds and kept.
  Not the original IMPALA/V-trace implementation.
- Told-boundary distillation (MiniGrid): told each task end; then stores the finished task's observations, resets
  heads + Adam and distils toward them (KL, coef 0.3); an upper reference for "blind".
- Joint: all tasks mixed, same total steps (MinAtar also 2x).
**Protocol.** Tuning seeds 101-105; final seeds 1-10. Mean +- 95% CI (t), and paired differences over seeds with the
number of seeds where ours is better. Final runs: code commits recorded per result file.

## V. Results
### A. MiniGrid, 3 tasks (C1) [source: baselines.md, plasticity_fix.md]
| Method | mean of 12 tests | unseen sizes |
|---|---|---|
| No protection | 35.3 +- 10.9 | 28.1 +- 10.2 |
| CLEAR | 92.1 +- 2.9 | 86.9 +- 4.1 |
| Told-boundary distill | 94.3 +- 3.9 | 90.6 +- 5.0 |
| Joint | 96.1 +- 1.4 | 92.3 +- 2.8 |
| **Ours (blind)** | **96.1 +- 1.7** | **92.6 +- 3.2** |
Paired: ours - CLEAR +4.0 +- 3.4 (7/10 seeds); ours - joint 0.0 +- 2.6. Small setting; differences to the strong
baselines are within noise.

### B. MiniGrid, 8 stages (L1) [source: baselines.md, plasticity_fix.md]
| Method | final mean of 11 trained tests | ours - method (paired) |
|---|---|---|
| No protection | 32.4 +- 9.6 | +60.2 +- 8.6, 10/10 |
| CLEAR | 66.6 +- 2.8 | +25.9 +- 3.6, 10/10 |
| Joint (4M mixed) | 80.7 +- 10.1 | +11.8 +- 10.3, 9/10 |
| Told-boundary distill | 92.3 +- 1.7 | +0.2 +- 1.4, 6/10 |
| **Ours (blind)** | **92.5 +- 1.4** | - |
- Without being told, the agent equals the learner that is told the boundaries.
- CLEAR keeps early tasks but learns the new kinds (Obstacles, Crossing) poorly: replay of everything holds the
  policy to old behaviour. [Fig. 2: success on each test over the 8 stages - fig_long_sequence, TODO regenerate
  with full-reset runs]
- KeyCorridor (never trained) stays at 0% for all methods: no zero-shot composition.

### C. MinAtar, 5 games [source: minatar.md, plasticity_fix.md, robustness.md]
| Method | after all games, % of scratch | just learned, % of scratch | ours - method |
|---|---|---|---|
| No protection | 5.2 +- 1.0 | 68.1 +- 3.3 | +37.5 +- 3.0, 10/10 |
| CLEAR | 16.5 +- 1.0 | 22.0 +- 1.3 | +26.2 +- 2.3, 10/10 |
| CLEAR, best MinAtar coef 0.1 | 21.4 +- 2.0 | 42.5 +- 5.9 | +21.3 +- 3.5, 10/10 |
| Reset & Distill (told) | 45.5 +- 4.2 | 93.7 +- 6.7 | -2.8 +- 4.6, 3/10 |
| Joint, 10M | 31.8 +- 3.7 | - | +10.9 +- 4.3, 10/10 |
| Joint, 20M (2x) | 36.2 +- 3.7 | - | +6.5 +- 4.2, 8/10 |
| **Ours (blind)** | **42.7 +- 2.2** | **91.1 +- 3.7** | - |
- Settings come from MiniGrid; nothing method-specific was tuned on MinAtar except choosing the full reset by a rule
  fixed in advance on tuning seeds (Sec. VI-C).
- Surprise fires within 500k steps of every game switch (all 4 switches, 10/10 seeds); 4-6 surprises per run.
- Joint is not an upper bound here: even with 20M steps it never learns Freeway, Asterix or Seaquest (interference,
  as also reported for multi-task SAC in [12]). We claim "above our joint baseline", not "above joint training".
- CLEAR's weakness on MinAtar is plasticity (new games at 22% of scratch), not memory.
[Fig. 3: fig_minatar, per-game bars]

## VI. Analysis
### A. Memory, not reset, keeps old skills [robustness.md]
Full reset without reservoir: new games are learned at 105% of scratch but final score falls to 25.3% (MinAtar,
-17.4 +- 5.3, 10/10) and 39.0% on MiniGrid C1 (-57.1 +- 9.7).

### B. Game order [robustness.md]
Reversed order: ours 37.2 +- 2.1, CLEAR 24.6 +- 1.9, none 20.0 +- 1.9; ours better on 10/10 seeds against both.
Order changes the level (-5.5 +- 2.0): games learned first are retained worst.

### C. Two kinds of plasticity loss [stage1_small.md, minatar.md, plasticity_fix.md]
- MiniGrid (3 tasks, no protection): learning of the third task tracks policy entropy at its start: 0/10 seeds learn
  it with entropy 0.01; head reset restores entropy 1.83 and 10/10. [fig_plasticity]
- MinAtar: entropy is not collapsed (1.2-1.4 of 1.79), dormant units < 2%, ReDo [18] every 10 rollouts gives no gain
  (25.6 vs 28.4, tuning seeds); full reset gives +15.0 +- 5.4 (5/5 tuning seeds) and +14.6 +- 2.5 (10/10 final).
- Choice rule (fixed before running): adopt a fix only if it beats the head reset on MinAtar and is not > 5 points
  worse on MiniGrid L1. Full reset: MiniGrid L1 92.5 vs 92.5, C1 96.1 vs 97.1 (n.s.).
- Lower CLEAR coefs (0.3, 0.1) on MinAtar tuning seeds: 21.0 and 24.0. Its best (0.1) on final seeds: 21.4 +- 2.0,
  still 21.3 +- 3.5 below ours (10/10) [review.md].

### D. Cost of the full reset: forward transfer [composition.md 2c]
Skills (Unlock, Pickup) then the composite UnlockPickup, blind: final skills equal to the head reset (Pickup 43 vs
45, n.s.), but the composite is learned ~2x slower (median 258k vs 126k steps): the reset discards reusable features.

### E. Being told the boundaries does not help [review.md]
Reset & Distill [25] adapted to PPO and told each task end (full reset + distill to the last rollout of every
finished task) vs ours (blind): MinAtar 45.5 vs 42.7 (-2.8 +- 4.6, n.s.); MiniGrid L1 83.6 vs 92.5 (+8.9 +- 11.6),
C1 84.8 vs 96.1 (+11.4 +- 11.4). R&D loses an early skill completely on 2/10 seeds in each MiniGrid setup (e.g.
Empty 0%); ours never does. Claim: matches the told method on average, more reliably.
Control (told reset + our reservoir): no collapses (worst seed 83 / 92 vs 55), so the memory, not the boundary
signal, prevents them. Same memory, reset trigger told vs detected: 8-stage 87.0 vs 92.5 (+5.5 +- 1.9 for ours,
10/10); 3-task 96.5 vs 96.1 (n.s.). Told resets fire at all 7 boundaries, ours only 2-4 times, keeping features
when the next stage is similar (likely cause, not tested directly).

### F. Sensitivity [review.md, tuning seeds 101-105, one setting changed at a time]
- MiniGrid L1: mastery 0.7/0.9, surprise 0.2/0.4, coef 0.1, reservoir 32k all give 91-95 (default 92.2). Distill
  coef 1.0 and a 2k reservoir cost ~5 points (5/5 seeds).
- MinAtar: mastery 0.7/0.9 and surprise 0.2/0.4 give 42-47 (default 43.4). Coef 0.1 or 1.0 ~-6 (trade-off: 0.1
  learns new games at 98% but keeps less, 1.0 keeps more but learns at 64%). Reservoir 2048: -9.0 +- 5.5 (5/5).
- Memory size matters most on both domains; thresholds hardly matter.

## VII. Limitations
- Small networks and small worlds (MiniGrid, MinAtar); one RL algorithm (PPO); one set of hyperparameters.
- Thresholds (0.8 mastery, 0.3 surprise) and lambda were set once on MiniGrid (sensitivity: Sec. VI-F).
- A gradual task change (new task mixed in over 200k steps) rarely triggers a surprise (2/20 transitions). On
  MiniGrid this costs nothing (97.0 vs 96.1 abrupt; memory alone keeps old skills), but on MinAtar, where the reset
  restores learning, the gain would likely be lost (not tested).
- A full reset discards useful features: a composite task built from known skills is learned ~2x slower than with
  a head reset (Sec. VI-D). Choosing what to reset is open.
- Retention on MinAtar is partial (games learned first keep 20-60% of scratch).
- CLEAR is our PPO adaptation, not the original.
- Joint is weak on MinAtar due to interference; it is not an upper bound there.

## VIII. Conclusion
An agent told nothing about task changes can, with a small self-triggered loop - remember what you master, distil
toward it, restart when the world surprises you - match a learner told the boundaries on MiniGrid and beat CLEAR on
two domains. Memory carries the past; the reset carries the future. Next: grounding language in the same agent.

---
## TODOs before LaTeX
1. Fig. 1 method diagram. 2. Regenerate fig_long_sequence / fig_combined with full-reset runs (L_blind_all,
O_full_all). 3. (done) 4. (done). 5. Word budget: IJCNN 6 pages incl. refs; cut Sec. II to ~0.6 column, tables to 3. 6. Install LaTeX or use
Overleaf; IEEEtran conference template. 7. Repo/code release link.
