#!/bin/bash
# Plasticity diagnosis: same sequence as B2, different task-switch interventions. Seeds 1-10, 300k steps/task.
export OMP_NUM_THREADS=1
SEQ=MiniGrid-Empty-5x5-v0,MiniGrid-DoorKey-5x5-v0,MiniGrid-LavaGapS5-v0
for s in 1 2 3 4 5 6 7 8 9 10; do for m in none adam heads shrink; do echo "$m $s"; done; done |
  xargs -P 12 -n 2 sh -c '.venv/bin/python ppo.py --steps 300000 --tasks '$SEQ' --task-switch $0 --seed $1 --name P_$0 > results/logs/P_$0_seed$1.log 2>&1'
