# 2b: reusing learned skills for a composite task (2026-10-24)

Question: do separately learned skills combine into a task the agent was never trained on?
Components (goal agent, mission words, 5M steps, mixed 50/50):
- MiniGrid-Unlock-v0 "open the door": find key, pick it up, open the locked door (two rooms).
- BabyAI-PickupDistDebug-v0 "pick up the red box" etc.: pick the named object; wrong object = failure.
Composite tests (never trained in component phase):
- MiniGrid-UnlockPickup-v0 "pick up the <colour> box": key -> unlock -> enter -> pick up box (same 2-room layout).
- MiniGrid-KeyCorridorS3R1-v0 "pick up the <colour> ball": key in another room, locked room, ball.
Protocol: tuning seeds 101-104 (run_2b_pilot.sh), final seeds 1-10 (run_2b_final.sh), 200 eval episodes,
entropy 0.05, code 9e0bdeb.
- B_comp: components only. Composites evaluated zero-shot.
- B_finetuneUP: B_comp agent + 2M steps on UnlockPickup (no replay/retention).
- B_scratchUP: random init, 7M steps on UnlockPickup = same TOTAL steps as comp + fine-tune.

| Agent | Unlock | Pickup | UnlockPickup | KeyCorridor |
|---|---|---|---|---|
| components only (zero-shot on composites) | 100 ± 0 | 54 ± 2 | 0 ± 0 | 0 ± 0 |
| components + 2M fine-tune on UnlockPickup | 100 ± 0 | 29 ± 2 | **100 ± 0** | 8 ± 10 |
| scratch, 7M on UnlockPickup | 4 ± 4 | 13 ± 3 | **0 ± 0** | 5 ± 3 |

UnlockPickup training success >= 80%:
- fine-tune: 10/10 seeds, after 77k-579k steps (per seed k: 253 77 90 126 169 139 223 260 276 579; median ~155k)
- scratch: 0/10 seeds in 7M steps; best training success any seed ever reached: 2-7%.

## Findings
1. No zero-shot composition: told "pick up the box" with the box behind a locked door, the agent never first
   fetches the key. Inferring an unstated sub-goal (planning) is not something this agent does.
2. Strong transfer: with learned skills, UnlockPickup goes to 100% in ~0.1-0.6M steps on every seed; from scratch it is
   never learned within 7M steps (equal total budget). The skills turn an unlearnable task into an easy one.
3. Fine-tuning without replay forgets: Pickup 54% -> 29%. (Replay/Life fixed this kind of forgetting in L1/O1.)
4. KeyCorridor (ball, different layout) stays near 0-8% for everyone.

## Limits
- Pickup component is only ~54% mastered; composition was measured on top of partial skill.
- Both composite tasks share layouts/objects with components; no new object types or new verbs.

---

# 2c: same skills + composite task inside the blind continual learner (2026-10-24)

Script run_2c.sh, seeds 1-10, code 745937b. One run per seed: stage 1 Unlock|Pickup mixed (5M) -> stage 2
UnlockPickup (5M, longer than the 2M of B_finetuneUP, so a harsher retention test). Agent never told about the switch.
Life: reservoir 8192, surprise reset, distill 0.3, mastery threshold 0.7 (0.8 would never trigger: the mixed stage
plateaus at 63-81% because Pickup alone is ~55%).

| Agent | Unlock | Pickup | UnlockPickup | KeyCorridor |
|---|---|---|---|---|
| after stage 1 (= components) | 100 ± 0 | 53 ± 2 | 0 ± 0 | 5 ± 3 |
| after stage 2, Life (blind) | 100 ± 0 | **45 ± 3** | **100 ± 0** | 16 ± 8 |
| B_finetuneUP (told nothing, no retention, only 2M) | 100 ± 0 | 29 ± 2 | 100 ± 0 | 8 ± 10 |

- UnlockPickup learned 10/10 seeds, steps to 80%: 104k-565k (median ~122k), same speed as plain fine-tuning.
- Hidden switch detected on every seed ~20-25k steps after it (one surprise each, no false alarms). Reservoir full.
- Pickup forgetting cut from -25 points (B_finetuneUP, 2M) to -8 points (Life, 5M). Reduced, not eliminated: the
  reservoir only holds Pickup behaviour at ~55% skill (consolidation is gated on the mixed stage's 70%).

So the building blocks combine in one system: words (goal input) + skills + blind continual learning + transfer
to a composite task. Zero-shot composition is still absent (0% after stage 1).

## 2c rerun with the final method: full reset on surprise (2026-10-25)
Script run_2c_all.sh, code 2d08f11, seeds 1-10, otherwise identical to run_2c.sh (C2_life_all vs C2_life).
| Agent (after stage 2) | Unlock | Pickup | UnlockPickup | KeyCorridor |
|---|---|---|---|---|
| Life, heads reset (C2_life) | 100 ± 0 | 45 ± 3 | 100 ± 0 | 16 ± 8 |
| Life, full reset (C2_life_all) | 100 ± 0 | 43 ± 2 | 100 ± 0 | 14 ± 5 |
Paired full - heads: Pickup -2 ± 3, KeyCorridor -2 ± 10 (n.s.). One surprise per seed in both.
Steps to 80% on UnlockPickup: full reset 190k-301k (median ~258k) vs heads 104k-565k (median ~126k).
Reading: final skills are the same, but the full reset learns the composite task about 2x slower in the median
seed: it throws away the body features that Unlock/Pickup had built, i.e. it gives up forward transfer for
plasticity. This is the cost side of exp 21; report it in the paper's limitations.
