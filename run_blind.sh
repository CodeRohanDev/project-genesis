#!/bin/bash
# O1: same as C1 but BLIND (agent never told about task switches). Compare with C_* results.
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
  echo "O_replay $s --steps 600000 --tasks $E,$D,$L --blind --reservoir 8192 --retain-coef 0.3"
  echo "O_reset $s --steps 600000 --tasks $E,$D,$L --blind --surprise-reset"
  echo "O_full $s --steps 600000 --tasks $E,$D,$L --blind --reservoir 8192 --surprise-reset --retain-coef 0.3"
done | xargs -P 12 -L 1 sh -c 'n=$0; s=$1; shift 1; .venv/bin/python ppo.py "$@" '"$COMMON"' --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
