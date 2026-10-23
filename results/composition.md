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
