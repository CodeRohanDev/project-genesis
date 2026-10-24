#!/bin/bash
# CLEAR baseline. MODE=tune: C1 setup (Empty->DoorKey->Lava, mixed sizes, 600k/task), coef 0.1/0.3/1.0, tuning seeds
# 101-103; pick the coef with the best mean over all 12 final tests (rule fixed before running).
# MODE=final COEF=x: C1 setup and L1 long sequence, seeds 1-10, compare with O_full/C_* and L_blind/L_*.
export OMP_NUM_THREADS=1
M=MiniGrid
E="$M-Empty-5x5-v0|$M-Empty-6x6-v0|$M-Empty-8x8-v0"
D="$M-DoorKey-5x5-v0|$M-DoorKey-6x6-v0|$M-DoorKey-8x8-v0"
L="$M-LavaGapS5-v0|$M-LavaGapS6-v0|$M-LavaGapS7-v0"
O="$M-Dynamic-Obstacles-5x5-v0|$M-Dynamic-Obstacles-6x6-v0|$M-Dynamic-Obstacles-8x8-v0"
X="$M-SimpleCrossingS9N1-v0|$M-SimpleCrossingS9N2-v0|$M-SimpleCrossingS9N3-v0"
U="$M-Unlock-v0"
EVAL_C="$M-Empty-5x5-v0,$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=10,$M-Empty-8x8-v0@size=12,\
$M-DoorKey-5x5-v0,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=10,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS5-v0,$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=8,$M-LavaGapS7-v0@size=10"
EVAL_L="$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=12,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=10,$M-Dynamic-Obstacles-8x8-v0,$M-Dynamic-Obstacles-8x8-v0@size=10,\
$M-SimpleCrossingS9N2-v0,$M-SimpleCrossingS11N5-v0,$M-Unlock-v0,$M-KeyCorridorS3R1-v0"
{
if [ "$MODE" = tune ]; then
  for c in 0.1 0.3 1.0; do for s in 101 102 103; do
    echo "tuneCLEAR_$c $s --steps 600000 --tasks $E,$D,$L --clear --reservoir 8192 --retain-coef $c --eval-tasks $EVAL_C --eval-episodes 50"
  done; done
else
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "C_clear $s --steps 600000 --tasks $E,$D,$L --clear --reservoir 8192 --retain-coef $COEF --eval-tasks $EVAL_C --eval-episodes 50"
    echo "L_clear $s --steps 500000 --tasks $E,$D,$L,$O,$X,$U,$D,$L --clear --reservoir 8192 --retain-coef $COEF --eval-tasks $EVAL_L --eval-episodes 30"
  done
fi
} | xargs -P 10 -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
