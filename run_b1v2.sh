#!/bin/bash
# B1 vs B1v2 (entropy 0.01 vs 0.05) on DoorKey + Lava, seeds 1-10, 1M steps. Max 12 parallel.
export OMP_NUM_THREADS=1
jobs_list=()
for s in 1 2 3 4 5 6 7 8 9 10; do for t in DoorKey-8x8-v0 LavaCrossingS9N1-v0; do
  jobs_list+=("--ent-coef 0.05 --name B1v2_$t --tasks MiniGrid-$t --seed $s")
  [ $s -gt 5 ] && jobs_list+=("--ent-coef 0.01 --name B1_$t --tasks MiniGrid-$t --seed $s")
done; done
printf '%s\n' "${jobs_list[@]}" | xargs -P 12 -I{} sh -c '.venv/bin/python ppo.py --steps 1000000 {} > "results/logs/$(echo {} | tr " " "_" | tr -d "-").log" 2>&1'
