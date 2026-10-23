# L1: long sequence, 8 stages, 6 kinds of task (2026-10-23)

Empty -> DoorKey -> Lava -> DynamicObstacles -> SimpleCrossing -> Unlock -> DoorKey (revisit) -> Lava (revisit).
Each stage 500k steps, mixed sizes where available. Seeds 1-10, 30 eval episodes per test. Script: run_long.sh
(run via run_long_safe.sh: 6 parallel, hard 7 GB RAM cap, no swap). 40/40 runs completed, 65 min, no kills.
- L_none: blind, no protection
- L_blind: Life (reservoir 8192 + surprise reset), never told about switches
- L_distill: told switches, heads reset + distill
- L_joint: all 6 task families at once, 4M steps (same total budget)
"u" = unseen size. KeyCorridor = never trained by anyone (zero-shot).

Final eval after all 8 stages:
| Variant | E8 | E12u | D8 | D12u | L7 | L10u | Obst8 | Obst10u | Cross9 | Cross11u | Unlock | KeyCorridor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 47 | 41 | 17 | 18 | 93 | 65 | 14 | 13 | 14 | 8 | 26 | 0 |
| **blind** | **100** | **100** | **100** | **92±5** | **99** | **69±19** | **88±4** | **85±6** | **96±3** | **89±3** | **100** | 0 |
| distill (told) | 100 | 100 | 100 | 83±8 | 97 | 76±10 | 92±6 | 91±7 | 93±3 | 83±5 | 100 | 0 |
| joint | 90±23 | 90±23 | 100 | 76±12 | 99 | 72±12 | 89±22 | 88±22 | 51±20 | 34±20 | 100 | 0 |

Learning per stage (seeds reaching 80% train success / mean steps to get there):
| Stage | none | blind | distill |
|---|---|---|---|
| Empty | 10/10, 19k | 10/10, 19k | 10/10, 19k |
| DoorKey | 10/10, 77k | 10/10, 55k | 10/10, 46k |
| Lava | 4/10, 346k | 10/10, 94k | 10/10, 107k |
| Obstacles | 8/10, 31k | 10/10, 30k | 10/10, 45k |
| Crossing | 8/10, 304k | 10/10, 93k | 10/10, 127k |
| Unlock | 10/10, 93k | 10/10, 7k | 10/10, 16k |
| DoorKey again | 9/10, 25k | 10/10, 2k | 10/10, 11k |
| Lava again | 8/10, 185k | 10/10, 2k | 10/10, 18k |

Blind surprise events (stage switches every 500k): fired ~5-40k after the switches into DoorKey (some seeds), Lava,
Obstacles and Crossing; never on Unlock or the revisits, because the agent was already competent there.

## Findings
1. Scales to 8 stages / 6 task kinds: blind agent learns every stage 10/10 and keeps all earlier skills
   (final >= 85% on 10 of 11 trained tests). Matches told-boundary distill; beats all-at-once joint on Crossing
   (96 vs 51) and is more stable across seeds. Same total budget.
2. Revisits show retention directly: blind needs ~2k steps to be back at 80% on DoorKey/Lava (it never forgot);
   unprotected agent needs 25k / 185k (partial savings: relearns faster than first time, 77k / 346k).
3. Forward transfer: Unlock was ~97-100% before any Unlock training, right after DoorKey (key -> door skill reused).
4. Plasticity holds over many switches: no stage stalled for blind/distill, while unprotected agent failed Lava 6/10.
5. Weak spots: Lava at unseen 10 dipped (70 -> 34) during Obstacles training and only partly recovered (69 final).
6. Negative: KeyCorridor (never trained) 0% for everyone. Its goal (pick up a ball behind a locked door, new layout)
   needs a goal the agent never learned. Skill reuse works for close variants (Unlock), not for a new kind of goal.

## Limits
Still one gridworld family, same observation format, 7 actions. Reservoir/threshold settings not tuned for this
sequence. 30 eval episodes per test.
