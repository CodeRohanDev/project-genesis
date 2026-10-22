# C1: continual learning + generalisation together (2026-10-23)

Empty -> DoorKey -> Lava, one network, each task trained on mixed sizes (Empty 5/6/8, DoorKey 5/6/8, LavaGap S5/S6/S7),
600k steps per task. Joint = all 9 envs at once for 1.8M steps (non-sequential upper reference).
CNN, entropy 0.05, distill coef 0.3 (tuned earlier on 5x5, not re-tuned). Seeds 1-10, 50 eval episodes. Script: run_combo.sh.
Unseen sizes: Empty 10/12, DoorKey 10/12, Lava 8/10.

Final eval after all three tasks (mean ± 95% CI):
| Variant | Empty 5 | Empty 8 | Empty 10u | Empty 12u | DK 5 | DK 8 | DK 10u | DK 12u | Lava 5 | Lava 7 | Lava 8u | Lava 10u |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 51±33 | 33±23 | 33±25 | 35±27 | 43±23 | 25±14 | 23±12 | 20±11 | 59±23 | 45±26 | 35±22 | 23±16 |
| heads | 57±33 | 51±36 | 50±37 | 52±36 | 19±23 | 6±6 | 4±5 | 3±6 | 100±0 | 98±2 | 91±4 | 68±12 |
| **distill** | **100±0** | **100±0** | **100±0** | **100±0** | **100±0** | **99±1** | **94±4** | **83±10** | **96±8** | **93±12** | **90±11** | **76±16** |
| joint | 100±0 | 100±0 | 100±0 | 100±0 | 100±0 | 100±1 | 95±6 | 90±5 | 100±0 | 100±1 | 93±5 | 76±11 |

Worst single score per seed (all 12 tests): distill 74 74 54 18 82 78 92 74 92 82 | joint 96 78 68 72 82 46 88 76 60 86
| none / heads: 0-24.

Lava learned to 80% train success: none 3/10 (~358k), heads 10/10 (~159k), distill 9/10 (~75k).

## Findings
1. Sequential learner with replay-distillation matches the all-at-once learner within CI on all 12 tests,
   including unseen sizes. It learns new tasks, keeps old ones, and generalises, at the same time.
2. Combining variety + distill fixed the earlier problem that distill only "memorised": Empty at unseen 12x12 is
   100% here vs 1% when trained on 5x5 only (generalisation.md). Retention is only as general as what was learned.
3. Without protection: forgetting (DoorKey 100% -> 20-43%). Heads reset alone: learns Lava, loses DoorKey (3-19%).
4. Distill again made new learning faster: Lava ~75k steps vs ~159k with heads only.
5. Weak spots: one distill seed fell to 18% on one test; Lava at unseen 10 is 76% for both distill and joint.

## Limits (still true)
- Agent is told when the task changes; replay memory grows per task (2048 obs each).
- All tasks share one gridworld, one observation format, 7 actions. Transfer to a *different kind* of task is untested.
- Only 3 tasks. Distill coef tuned on a different setup.
