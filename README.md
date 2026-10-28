# Boundary-blind continual reinforcement learning

Code, configurations, per-seed results and logs for the paper
**"Told Nothing, Forgetting Little: Boundary-Blind Continual Reinforcement Learning with Competence-Gated Memory
and Surprise Resets"** (submitted to IJCNN 2027).

A PPO agent faces a sequence of tasks and is never told when a task changes. It uses only its own success:
1. **Competence-gated memory**: while its success is >= 80%, it stores observations and its action probabilities
   in a fixed-size reservoir (8,192 items).
2. **Replay distillation**: every PPO minibatch adds 0.3 x KL(stored policy || current policy) on reservoir samples.
3. **Surprise reset**: when its success falls below 30% after mastery, it re-initialises the whole network and
   Adam; old skills come back only through the memory.

All of it is in `ppo.py` (class `Life`; baselines `Clear`, `Retention`, `on_task_switch`).

## Install
Python 3.14, CPU only (a GPU is not needed or used).
```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Reproduce the paper
Every run writes `results/runs/<name>_seed<k>.json` (full config, code commit, training curves, evaluation matrix)
and a `.pt` model; logs go to `results/logs/`. Run scripts skip finished runs, so they can be resumed.
`safe_run.sh` runs a script under a RAM cap (systemd scope); plain `bash <script>` also works.

| Paper item | Script | Run names |
|---|---|---|
| Ours, seeds 1-10 (MinAtar, MiniGrid L1, C1) | `run_fix_final.sh` | `MA_blind_all`, `L_blind_all`, `O_full_all` |
| MinAtar baselines + scratch normaliser | `run_minatar.sh`, `run_review.sh` | `MA_none`, `MA_clear`, `MA_joint`, `MA_rd`, `MAS_*` |
| MiniGrid baselines | `run_v2.sh`, `run_clear.sh`, `run_review.sh`, `run_rdres.sh` | `L_*`, `C_*` (`*_v2` = seeds 1-10 on final code) |
| Replication, seeds 11-20 | `run_seeds20.sh` | same names, seeds 11-20 |
| Ablations, order, sensitivity, gradual change | `run_robust.sh`, `run_fix_tune.sh`, `run_sens_ma.sh`, `run_drift.sh` | see `results/README.md` |

One run takes about 10-30 minutes (MiniGrid) or 1-3 hours (MinAtar) on one CPU thread; scripts run up to 12 in parallel.
Any single run can be reproduced exactly from its saved config (runs are bit-exact per seed on the final code):
```bash
.venv/bin/python ppo.py $(.venv/bin/python rerun_cmd.py results/runs/L_blind_all_seed1.json) --seed 1 --name check
```

## Tables, figures, paper
```bash
PYTHONPATH=. .venv/bin/python results/rep_analysis.py      # MinAtar tables (seeds 1-20, paired differences)
PYTHONPATH=. .venv/bin/python results/rep_analysis_mg.py   # MiniGrid tables
.venv/bin/python figures.py                                # paper/figs/*.pdf, *.png
node paper/make_docx.js                                    # paper/paper_ijcnn2027.docx (needs npm package docx)
```
`results/README.md` indexes every experiment (question, script, run names, write-up). Write-ups in `results/*.md`
report failures and negative results as well. Hyperparameters were tuned only on seeds 101+.

## Other material
Earlier exploratory work in the same repository (generalisation, grounded words, skill composition, world model and
planning, a small language model) is documented in `results/` and `PLAN.md`; it is not part of the paper.

## License
MIT, see `LICENSE`.
