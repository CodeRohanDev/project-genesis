#!/bin/bash
# Baseline B1: PPO from scratch on each task, 5 seeds. One CPU thread per run.
export OMP_NUM_THREADS=1
run() { .venv/bin/python ppo.py --tasks $1 --steps $2 --seed $3 --name B1_${1#MiniGrid-} > results/logs/B1_${1#MiniGrid-}_seed$3.log 2>&1; }
for s in 1 2 3 4 5; do
  run MiniGrid-Empty-8x8-v0 200000 $s &
  run MiniGrid-DoorKey-8x8-v0 1000000 $s &
  run MiniGrid-LavaCrossingS9N1-v0 1000000 $s &
done
wait
