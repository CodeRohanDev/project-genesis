#!/bin/bash
# L1: longer sequence, 8 stages incl. new kinds of tasks and revisits. 500k steps per stage, seeds 1-10.
# Empty -> DoorKey -> Lava -> DynamicObstacles -> SimpleCrossing -> Unlock -> DoorKey (revisit) -> Lava (revisit)
# KeyCorridor is never trained: zero-shot test of key/door skills on a new layout type.
export OMP_NUM_THREADS=1
M=MiniGrid
E="$M-Empty-5x5-v0|$M-Empty-6x6-v0|$M-Empty-8x8-v0"
D="$M-DoorKey-5x5-v0|$M-DoorKey-6x6-v0|$M-DoorKey-8x8-v0"
L="$M-LavaGapS5-v0|$M-LavaGapS6-v0|$M-LavaGapS7-v0"
O="$M-Dynamic-Obstacles-5x5-v0|$M-Dynamic-Obstacles-6x6-v0|$M-Dynamic-Obstacles-8x8-v0"
X="$M-SimpleCrossingS9N1-v0|$M-SimpleCrossingS9N2-v0|$M-SimpleCrossingS9N3-v0"
U="$M-Unlock-v0"
SEQ="$E,$D,$L,$O,$X,$U,$D,$L"
EVAL="$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=12,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=10,$M-Dynamic-Obstacles-8x8-v0,$M-Dynamic-Obstacles-8x8-v0@size=10,\
$M-SimpleCrossingS9N2-v0,$M-SimpleCrossingS11N5-v0,$M-Unlock-v0,$M-KeyCorridorS3R1-v0"
COMMON="--eval-tasks $EVAL --eval-episodes 30"
for s in 1 2 3 4 5 6 7 8 9 10; do
  echo "L_none $s --steps 500000 --tasks $SEQ --blind"
  echo "L_blind $s --steps 500000 --tasks $SEQ --blind --reservoir 8192 --surprise-reset --retain-coef 0.3"
  echo "L_distill $s --steps 500000 --tasks $SEQ --task-switch heads --retain distill --retain-coef 0.3"
  echo "L_joint $s --steps 4000000 --tasks $E|$D|$L|$O|$X|$U"
done | xargs -P 12 -L 1 sh -c 'n=$0; s=$1; shift 1; .venv/bin/python ppo.py "$@" '"$COMMON"' --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
