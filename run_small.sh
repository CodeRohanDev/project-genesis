#!/bin/bash
# Small-task track: B1 (each task from scratch) + B2 (sequential, no protection). Seeds 1-10, 300k steps/task.
export OMP_NUM_THREADS=1
SEQ=MiniGrid-Empty-5x5-v0,MiniGrid-DoorKey-5x5-v0,MiniGrid-LavaGapS5-v0
jobs_list=()
for s in 1 2 3 4 5 6 7 8 9 10; do
  jobs_list+=("--name B2_small --tasks $SEQ --seed $s")
  for t in ${SEQ//,/ }; do jobs_list+=("--name B1s_${t#MiniGrid-} --tasks $t --seed $s"); done
done
printf '%s\n' "${jobs_list[@]}" | xargs -P 12 -I{} sh -c '.venv/bin/python ppo.py --steps 300000 {} > "results/logs/$(echo {} | cut -d" " -f2)_$(echo {} | rev | cut -d" " -f1 | rev).log" 2>&1'
