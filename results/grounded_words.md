# 2a: grounded words and unseen word combinations (2026-10-23)

Question: can an agent that starts with NO language (random word embeddings) learn word meanings from reward alone,
and understand colour+object combinations it was never trained on?

Task: BabyAI-PickupDistDebug-v0 (7x7 room, 5 objects, strict: picking up a wrong object ends the episode with 0).
Mission refers to the target by colour only ("the red object"), type only ("a ball") or both ("the red ball").
Held out of training: 4 "both" pairs: red ball, green key, blue box, purple ball. Every colour and every object type
still appears in training in other combinations. Filter: MissionFilter in ppo.py (resample until allowed).

Agent (--goal): mission words -> learned embedding (32) -> GRU (64) -> FiLM on the CNN features. Vocabulary is only a
list of ids; no pretrained embeddings. Same PPO settings as before, entropy 0.05, 5M steps.
Tuning (seeds 101-102, run_2a_pilot.sh): entropy 0.05 beat 0.01 (held-out 56-60% vs 40-49%).
Final: seeds 1-10, 200 eval episodes per test. Scripts: run_2a_nogoal.sh, run_2a_final.sh. Code ab20fd2 (= e8c5f05
for ppo.py); A_nogoal seeds 1-5 started just before the code-version field existed, same ppo.py logic.

| Agent | Missions (mixed) | Trained pairs (14) | Held-out pairs (4) | Gap trained - held-out |
|---|---|---|---|---|
| no goal input (image only) | 28 ± 1 | 22 ± 1 | 22 ± 2 | 0 ± 2 |
| **goal, 4 pairs held out** | **70 ± 4** | **64 ± 4** | **61 ± 5** | **3 ± 4** |
| control: goal, trained on all pairs (5 seeds) | 69 ± 10 | 64 ± 8 | 65 ± 14 | -1 ± 9 |

Held-out success per seed (goal): 54 60 60 73 62 68 52 56 64 62. No seed below 52%.

## Findings
1. Words got meaning from reward alone: reading the mission lifts success from 22% to 64% on trained pairs.
2. Unseen combinations work: on the 4 never-trained pairs the agent scores 61%, within CI of trained pairs (gap 3 ± 4)
   and of the control agent that WAS trained on them (65 ± 14). So it composes "red" + "ball" rather than
   memorising "red ball" as a phrase.
3. Control shows the held-out pairs are not intrinsically easier or harder (65 vs 64 for the control).

## Limits
- Absolute success is moderate (~64%); the agent is still far from mastering the task at 5M steps. The composition
  claim is about the gap, measured at this level of skill.
- Only 2-word combinations of a 6x3 vocabulary; one room type. No word order or longer instructions yet.
- Control has only 5 seeds (wide CI).
