#!/bin/bash
# Stronger skills: continue B_comp agents (Unlock + strict Pickup, 5M) for 10M more steps on the same 50/50 mix.
# Purpose: weak Pick-up (~54%) limited the planning test (3b diagnostic). Seeds 1-10. 9 parallel (GPU job uses 2 threads).
export OMP_NUM_THREADS=1
EVAL="MiniGrid-Unlock-v0,BabyAI-PickupDistDebug-v0,MiniGrid-UnlockPickup-v0,MiniGrid-KeyCorridorS3R1-v0"
for s in 1 2 3 4 5 6 7 8 9 10; do echo $s; done | xargs -P 9 -I{} sh -c '
  [ -f results/runs/S_comp15_seed{}.json ] && exit 0
  .venv/bin/python ppo.py --goal --init results/runs/B_comp_seed{}.pt \
    --tasks "MiniGrid-Unlock-v0|BabyAI-PickupDistDebug-v0" --eval-tasks "'"$EVAL"'" \
    --steps 10000000 --eval-episodes 200 --seed {} --name S_comp15 > results/logs/S_comp15_seed{}.log 2>&1'
