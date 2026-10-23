#!/bin/bash
# 2c: skills + composite task inside the blind continual learner (Life), seeds 1-10.
# Stage 1: Unlock|Pickup mixed (5M). Stage 2: UnlockPickup (5M: --steps is per stage; longer than the 2M of
# B_finetuneUP, so a harder retention test). Agent is never told the switch.
# Compare with B_finetuneUP (same stages, no retention). Mastery threshold 0.7: the mixed stage plateaus near 75-80%
# (Pickup ~55%), so at 0.8 nothing would ever be consolidated; chosen from 2b numbers, not tuned on these seeds.
export OMP_NUM_THREADS=1
export EVAL="MiniGrid-Unlock-v0,BabyAI-PickupDistDebug-v0,MiniGrid-UnlockPickup-v0,MiniGrid-KeyCorridorS3R1-v0"
for s in 1 2 3 4 5 6 7 8 9 10; do echo $s; done | xargs -P 10 -I{} sh -c '
  [ -f results/runs/C2_life_seed{}.json ] && exit 0
  .venv/bin/python ppo.py --goal --blind --reservoir 8192 --surprise-reset --retain-coef 0.3 --master 0.7 \
    --tasks "MiniGrid-Unlock-v0|BabyAI-PickupDistDebug-v0,MiniGrid-UnlockPickup-v0" --eval-tasks "$EVAL" \
    --steps 5000000 --eval-episodes 200 --seed {} --name C2_life > results/logs/C2_life_seed{}.log 2>&1'
