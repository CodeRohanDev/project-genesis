# Overnight report (2026-10-23)

## What ran
- Phase 1 (O1, blind continual learning): DONE, 30 runs, 00:24-00:54. Details: overnight_blind.md.
- Phase 2 (L1, long 8-stage sequence): DID NOT RUN. My bug, see below.

## Phase 1 result
Agent is never told when the task changes. It keeps a fixed-size replay memory (8192 observations), fills it only
with situations it has mastered (own success >= 80%), replays them while learning, and treats a sudden collapse of
its own success as "the world changed" (surprise -> reset of output layers).

Final score after Empty -> DoorKey -> Lava, unseen sizes:
| | Empty 12x12 | DoorKey 12x12 | Lava 10x10 |
|---|---|---|---|
| no protection | 35% | 20% | 23% |
| told boundaries + distill (previous best) | 100% | 83% | 76% |
| all tasks at once (upper reference) | 100% | 90% | 76% |
| **blind: replay + surprise reset** | **100%** | **97%** | **74%** |
| blind: replay only | 100% | 95% | 75% |
| blind: surprise reset only | 43% | 5% | 62% |

- Blind agent matches or beats the boundary-given method and the all-at-once learner (within CI, 10 seeds).
- Real switches detected within 2-30k steps; no false alarms when replay was on.
- Memory stays fixed size.
- Replay alone also prevented plasticity loss; entropy logs show it keeps the policy less deterministic at the switch
  (0.55 vs 0.21). Correlation, not proven mechanism.

## Phase 2 failure (my mistake)
The watcher that was meant to auto-start Phase 2 waited with `pgrep -f run_blind.sh`. Its own command line contained
the text "run_blind.sh", so it always found itself and waited forever. At 00:55 I also checked with `pgrep -fc run_long.sh`,
which matched the watcher's own command line too, and I wrongly reported that Phase 2 was running.
Cost: ~6.5 hours of idle machine time. Found at 07:35, watcher stopped.
Lesson: wait on a PID (`wait`/`kill -0 PID`) or a result file, never on `pgrep -f` of a name that appears in the waiter;
verify "running" by log files growing, not by process-name match.

Phase 2 script (run_long.sh) is ready and smoke-tested. Estimated 2-2.5 hours.

## Update 09:05: Phase 2 run in the morning instead (07:56-09:01, 6 parallel, 7 GB RAM cap)
40/40 runs OK. Blind agent over 8 stages / 6 task kinds learned every stage and kept old skills; matched told-boundary
distill and beat all-at-once joint on Crossing. Details: long_sequence.md.
