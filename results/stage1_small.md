# Stage 1, small-task track: B1s + B2 (2026-10-22)

Tasks: Empty-5x5 -> DoorKey-5x5 -> LavaGapS5. PPO, entropy 0.05, 300k steps/task, seeds 1-10.
Eval: 100 episodes per task, held-out seed range. Mean ± 95% CI.

## B1s: each task from scratch
| Task | Eval success | Seeds solved | Steps to 80% |
|---|---|---|---|
| Empty-5x5 | 90% ± 23% | 9/10 (seed 3 collapsed to 0%) | ~15k |
| DoorKey-5x5 | 100% ± 0% | 10/10 | ~60k |
| LavaGapS5 | 100% ± 0% | 10/10 | ~63k |
All three are reliably learnable from scratch.

## B2: same network, trained sequentially, no protection
| after \ eval | Empty | DoorKey | Lava |
|---|---|---|---|
| Empty | 90% ± 23% | 0% | 18% ± 6% |
| DoorKey | 100% | 100% | 22% ± 1% |
| Lava | 66% ± 31% | 30% ± 20% | 21% ± 3% |

## Findings
1. Forward transfer (positive): DoorKey after Empty reached 80% in ~44k steps vs ~60k from scratch (~27% faster).
   DoorKey training also fixed Empty for seed 3 (backward transfer).
2. Forgetting: after Lava training, DoorKey 100% -> 30%, Empty 100% -> 66%.
3. Loss of plasticity (biggest, unexpected): Lava was learned 10/10 from scratch but 0/10 in sequence.
   Sequential Lava training curve stays flat at ~30% for all 300k steps; fresh network goes 7% -> 100% in ~60k.
   The previously trained network cannot learn the new task, AND it forgets the old ones: worst of both.

## Meaning for the AGI goal
A lifelong learner needs both stability (keep old skills) and plasticity (learn new ones).
Plain PPO fails on both after only 2 task switches. Plasticity loss was not in the original plan and is
now the first problem to diagnose: B3 (replay) and B4 (EWC) target forgetting only and would not fix it.

---

# Plasticity diagnosis (2026-10-22)

Same sequence and seeds as B2. Intervention applied when a new task starts (nested: adam < heads; shrink separate).
- none: nothing (= B2)
- adam: fresh Adam optimizer
- heads: fresh Adam + reinit actor & critic output layers
- shrink: fresh Adam + weights = 0.8 * old + 0.2 * fresh random init

| Variant | Lava learned (seeds >= 80%) | Lava final eval | Lava steps to 80% | Policy entropy at Lava start | DoorKey after Lava | Empty after Lava |
|---|---|---|---|---|---|---|
| none | 0/10 | 21% ± 3% | never | 0.01 | 30% ± 20% | 66% ± 31% |
| adam | 2/10 | 35% ± 24% | - | 0.03 | 8% ± 10% | 73% ± 27% |
| shrink | 7/10 | 78% ± 20% | ~189k | 0.19 | 18% ± 22% | 32% ± 26% |
| heads | 10/10 | 98% ± 4% | ~112k | 1.83 | 9% ± 8% | 44% ± 33% |
| (scratch, B1s) | 10/10 | 100% | ~63k | 1.94 | - | - |
(max entropy for 7 actions = 1.95)

Findings:
1. Plasticity tracks entropy at task start: 0.01 -> 0/10, 0.03 -> 2/10, 0.19 -> 7/10, 1.83 -> 10/10.
   After DoorKey the policy is almost deterministic, so it never tries the actions Lava needs.
   (Correlation over 4 variants; heads also resets the critic, so actor vs critic effect not separated.)
2. Optimizer state is a minor factor (adam: 0 -> 2/10).
3. Even with heads reset, Lava learning is slower than scratch (112k vs 63k): no positive transfer into Lava.
4. Stability-plasticity dilemma, measured: every intervention that restores learning keeps forgetting severe
   (DoorKey after Lava: 8-30% in all variants). Nothing yet gets both.

Next: keep heads reset (plasticity) and add a retention mechanism (replay-distillation / EWC).

---

# Retention: replay-distillation vs EWC, on top of heads reset (2026-10-22)

Tuning (seeds 101-103) picked distill coef 0.3 (tried 0.3/1/3) and EWC lambda 10 (tried 10/100/1000).
Higher distill coef -> Lava learned less (99% -> 70% -> 37%): stability-plasticity trade-off visible directly.

Final test, untouched seeds 1-10. Eval after all three tasks:
| Method | Empty | DoorKey | Lava | Lava steps to 80% |
|---|---|---|---|---|
| none (B2) | 66% ± 31% | 30% ± 20% | 21% ± 3% | never |
| heads reset | 44% ± 33% | 9% ± 8% | 98% ± 4% | ~112k |
| heads + EWC 10 | 54% ± 34% | 48% ± 26% | 79% ± 23% | - |
| **heads + distill 0.3** | **90% ± 23%** | **100% ± 0%** | **95% ± 9%** | **~67k** |

Per seed (Empty/DoorKey/Lava) distill: 100/100/100 100/100/100 0/100/100 100/100/100 100/100/100 100/100/97 100/100/100 100/100/61 100/100/99 100/100/97

Findings:
1. Distill keeps DoorKey at 100% in 10/10 seeds (heads-only: 0/10 above 30%). Lava still learned 9/10. First result
   where the agent learns a new task AND keeps old ones.
2. Distill also speeds up new learning: Lava ~67k steps vs ~112k with heads only (scratch: ~63k); DoorKey ~30k vs ~60k
   from scratch. Replaying old skills did not block new ones here; it seems to stabilise training.
3. EWC: partial retention, weaker learning. Worse than distill on every task in this setup.
4. Retention also preserves mistakes: seed 3 never learned Empty in stage 1 (0%). Plain B2 later fixed it by accident;
   distill faithfully kept the 0% policy. A memory that only copies the past cannot correct it.

Limits (must not be over-claimed):
- Task boundaries are given to the agent (it is told when the task switches). Not realistic.
- Memory grows linearly: 2048 stored observations per task.
- Only 3 small tasks, 300k steps each. Unknown whether this holds for longer sequences.
- Generalisation to unseen layouts/sizes not tested yet.
