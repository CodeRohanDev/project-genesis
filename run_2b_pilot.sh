#!/bin/bash
# 2b pilot (tuning seeds 101-102). Components: Unlock ("open the door") + strict Pickup ("pick up the red box").
# Composites never trained: UnlockPickup, KeyCorridorS3R1. Step 1: component training (5M) + UnlockPickup from scratch (2M).
export OMP_NUM_THREADS=1
export COMP="MiniGrid-Unlock-v0|BabyAI-PickupDistDebug-v0"
export EVAL="MiniGrid-Unlock-v0,BabyAI-PickupDistDebug-v0,MiniGrid-UnlockPickup-v0,MiniGrid-KeyCorridorS3R1-v0"
{ for s in 101 102; do echo "p2b_comp $s COMP 5000000"; echo "p2b_scratchUP $s UP 2000000"; done; } | xargs -P 10 -L 1 sh -c '
  n=$0; s=$1; t=$COMP; [ "$2" = UP ] && t=MiniGrid-UnlockPickup-v0
  [ -f results/runs/${n}_seed$s.json ] && exit 0
  .venv/bin/python ppo.py --goal --tasks "$t" --eval-tasks "$EVAL" --steps $3 --eval-episodes 200 --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
# Step 2: fine-tune component agents on UnlockPickup for the same 2M steps as scratch.
for s in 101 102; do echo $s; done | xargs -P 10 -I{} sh -c '
  [ -f results/runs/p2b_finetuneUP_seed{}.json ] && exit 0
  .venv/bin/python ppo.py --goal --init results/runs/p2b_comp_seed{}.pt --tasks MiniGrid-UnlockPickup-v0 --eval-tasks "$EVAL" --steps 2000000 --eval-episodes 200 --seed {} --name p2b_finetuneUP > results/logs/p2b_finetuneUP_seed{}.log 2>&1'
