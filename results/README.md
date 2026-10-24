# Results index (Project Genesis)

Every experiment: question, script, run-name prefix in `results/runs/` (one JSON + one `.pt` model per seed),
write-up, and the code it ran on. Logs per run in `results/logs/`. Raw JSON holds full config (all hyperparameters),
per-task training curves, policy entropy per update, eval matrix after each stage, wall time.

## Common protocol
- Agent: PPO (ppo.py), small CNN (rl-starter-files), random init. CPU, 16 parallel envs, 128-step rollouts,
  4 epochs, 8 minibatches, lr 1e-3, gamma 0.99, GAE 0.95, clip 0.2. Entropy 0.01 until B1v2, 0.05 after.
- Seeds: final results use seeds 1-10 (some early ones 1-5). Hyperparameters were chosen ONLY on tuning
  seeds 101-109; final seeds were never used for tuning.
- Eval: separate seed range (>= 1e6) from training resets, stochastic policy, success = positive reward.
  50-200 episodes per test (stated per experiment).
- Stats: mean ± 95% CI with t-distribution over seeds (summarize.py `ci`). Fisher exact test where stated.
- Software: Python 3.14.4, torch 2.14.1+cu126, gymnasium 1.4.0, minigrid 3.2.0 (requirements.txt).
  Hardware: Ryzen 5 9600X (12 threads), 14 GB usable RAM, GTX 1050 Ti unused (CPU was faster for this model size).

## Experiments (chronological)
| # | Question | Script | Run prefix | Write-up |
|---|---|---|---|---|
| 0 | Setup works; random-agent floor | smoke_test.py | - | stage0.md |
| 1 | PPO from scratch on 8x8 tasks (B1) | run_b1.sh | B1_, B1long_ | stage1_b1.md |
| 1b | Curiosity (RND) tuning | (inline cmd) | tune_rnd* | stage1_b1.md |
| 1c | Stability diagnosis (lr, adv-norm, entropy) | (inline cmd) | diag_* | stage1_b1.md |
| 1d | Entropy 0.01 vs 0.05 on untouched seeds | run_b1v2.sh | B1_, B1v2_ | stage1_b1.md |
| 2 | Small tasks from scratch + sequential (B2) | run_small.sh | B1s_, B2_small | stage1_small.md |
| 3 | Plasticity loss diagnosis | run_plasticity.sh | P_none/adam/heads/shrink | stage1_small.md |
| 4 | Retention: distill vs EWC (tuning + final) | (inline cmd) | tuneR_*, R_distill, R_ewc | stage1_small.md |
| 5 | Generalisation to bigger unseen grids | gen_eval.py | (re-evaluates saved models) | generalisation.md, generalisation.json |
| 6 | Variety x working memory (LSTM) | run_gen.sh | G_cnn_5/mix, G_lstm_5/mix | generalisation.md |
| 7 | Continual + generalisation together | run_combo.sh | C_none/heads/distill/joint | combined.md |
| 8 | Blind continual learning (no task boundaries) | run_blind.sh | O_reset/replay/full | overnight_blind.md |
| 9 | Long sequence, 8 stages, 6 task kinds | run_long.sh (via safe_run.sh) | L_none/blind/distill/joint | long_sequence.md |
| 10 | Grounded words + held-out combinations (2a) | run_2a_pilot.sh, run_2a_nogoal.sh, run_2a_final.sh | p2a_* (pilot, tuning seeds), A_goal, A_goal_all, A_nogoal | grounded_words.md |
| 11 | Skill reuse for composite tasks (2b) | run_2b_pilot.sh, run_2b_final.sh | p2b_* (tuning 101-104), B_comp, B_finetuneUP, B_scratchUP | composition.md |
| 12 | Skills + composite inside blind continual learner (2c) | run_2c.sh | C2_life | composition.md (2c section) |
| 13 | Learned world model: causal probes, priors, imagination (3a) | worldmodel.py, wm_rollouts.py | results/wm/ (data.npz, model_*seed*.pt, metrics_*.json, rollout.json) | worldmodel.md |
| 14 | Skill-order feasibility for planning | skill_sequence.py | results/wm/skill_sequence.json | worldmodel.md |
| 15 | Planning over skills with world model (3b, v1-v4 + map memory, ablations) | planner.py, worldmodel.py collect_comp | results/wm/planner_v1-v4 json, planner_v4_map_random.json, model_comp_persist_seed*.pt | worldmodel.md (3b section) |
| 16 | Map memory (path integration, map, navigation) | mapmem.py (self-test in __main__) | - | worldmodel.md (v4 section) |
| 17 | Language L1: byte LM from scratch, grammar vs words seen | lm.py | results/lm/ | language.md |

## Code provenance (important for the paper)
- Runs of experiments 0-7 were produced before the repo existed. Commit d79f702 holds the code as it was right
  after experiment 7. Changes between runs of 0-7 only ADDED options (RND, retention, resets, LSTM, mixtures,
  entropy logging); default behaviour stayed the same except the entropy default (0.01 -> 0.05), and each JSON
  stores its own `config`, so every run's hyperparameters are exact.
- Experiments 8-9: commit 1453d15 / 6f9a04c.
- From commit ab20fd2 on, every result JSON stores `code.commit` and `code.ppo_py_modified`.
- Behaviour changes after d79f702 that affect re-runs: observation became flat [image + mission word ids]
  (image part identical; old models re-evaluated to the same scores), Dynamic-Obstacles action space widened to 7.

## Known caveats to report
- Eval layouts are drawn from finite layout spaces; true held-out tests are the unseen sizes / held-out pairs.
- Distill coef 0.3 and thresholds 80%/30% (Life) were set once and reused across later setups.
- Hours 01:00-07:30 on 2026-10-23 were lost to a watcher bug (overnight.md); no data was affected.
