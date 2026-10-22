#!/bin/bash
# C1: sequential tasks, each trained on mixed sizes. Does retention + generalisation hold together?
# Empty -> DoorKey -> Lava, 600k steps per task. Joint = all 9 envs at once (non-sequential reference), 1.8M.
export OMP_NUM_THREADS=1
M=MiniGrid
E="$M-Empty-5x5-v0|$M-Empty-6x6-v0|$M-Empty-8x8-v0"
D="$M-DoorKey-5x5-v0|$M-DoorKey-6x6-v0|$M-DoorKey-8x8-v0"
L="$M-LavaGapS5-v0|$M-LavaGapS6-v0|$M-LavaGapS7-v0"
EVAL="$M-Empty-5x5-v0,$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=10,$M-Empty-8x8-v0@size=12,\
$M-DoorKey-5x5-v0,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=10,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS5-v0,$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=8,$M-LavaGapS7-v0@size=10"
COMMON="--eval-tasks $EVAL --eval-episodes 50"
for s in 1 2 3 4 5 6 7 8 9 10; do
  echo "C_none $s --steps 600000 --tasks $E,$D,$L"
  echo "C_heads $s --steps 600000 --tasks $E,$D,$L --task-switch heads"
  echo "C_distill $s --steps 600000 --tasks $E,$D,$L --task-switch heads --retain distill --retain-coef 0.3"
  echo "C_joint $s --steps 1800000 --num-envs 18 --tasks $E|$D|$L"
done | xargs -P 12 -L 1 sh -c 'n=$0; s=$1; shift 1; .venv/bin/python ppo.py "$@" '"$COMMON"' --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
