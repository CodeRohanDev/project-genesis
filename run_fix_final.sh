#!/bin/bash
# Exp 21 final: fix A (--reset-scope all) on FINAL seeds 1-10, chosen on tuning seeds by the rule in
# run_fix_tune.sh (results/plasticity_fix.md). Same settings as MA_blind (run_minatar.sh), L_blind (run_long.sh)
# and O_full (run_blind.sh), plus --reset-scope all. Launch via ./safe_run.sh run_fix_final.sh. Finished runs skipped.
export OMP_NUM_THREADS=1
G="MinAtar-breakout MinAtar-space_invaders MinAtar-freeway MinAtar-asterix MinAtar-seaquest"
SEQ=$(echo $G | tr ' ' ',')
M=MiniGrid
E="$M-Empty-5x5-v0|$M-Empty-6x6-v0|$M-Empty-8x8-v0"
D="$M-DoorKey-5x5-v0|$M-DoorKey-6x6-v0|$M-DoorKey-8x8-v0"
L="$M-LavaGapS5-v0|$M-LavaGapS6-v0|$M-LavaGapS7-v0"
O="$M-Dynamic-Obstacles-5x5-v0|$M-Dynamic-Obstacles-6x6-v0|$M-Dynamic-Obstacles-8x8-v0"
X="$M-SimpleCrossingS9N1-v0|$M-SimpleCrossingS9N2-v0|$M-SimpleCrossingS9N3-v0"
U="$M-Unlock-v0"
EVAL_L="$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=12,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=10,$M-Dynamic-Obstacles-8x8-v0,$M-Dynamic-Obstacles-8x8-v0@size=10,\
$M-SimpleCrossingS9N2-v0,$M-SimpleCrossingS11N5-v0,$M-Unlock-v0,$M-KeyCorridorS3R1-v0"
EVAL_O="$M-Empty-5x5-v0,$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=10,$M-Empty-8x8-v0@size=12,\
$M-DoorKey-5x5-v0,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=10,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS5-v0,$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=8,$M-LavaGapS7-v0@size=10"
LIFE="--blind --reservoir 8192 --surprise-reset --retain-coef 0.3 --reset-scope all"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "MA_blind_all $s --tasks $SEQ --steps 2000000 $LIFE --eval-tasks $SEQ --eval-episodes 30"
  done
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "L_blind_all $s --steps 500000 --tasks $E,$D,$L,$O,$X,$U,$D,$L $LIFE --eval-tasks $EVAL_L --eval-episodes 30"
    echo "O_full_all $s --steps 600000 --tasks $E,$D,$L $LIFE --eval-tasks $EVAL_O --eval-episodes 50"
  done
} | xargs -P 8 -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
