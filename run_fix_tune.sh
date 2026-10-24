#!/bin/bash
# Plasticity fixes, TUNING seeds 101-105 only (final seeds 1-10 untouched). Question from results/minatar.md:
# later MinAtar games are learned far worse than from scratch. Candidates:
#   A  reset_scope all: a surprise resets the whole network (old skills return only via reservoir distill)
#   B  ReDo every 10 rollouts, tau 0.1 (Sokar et al. 2023), heads reset on surprise as before
#   none_redo: diagnostic, does ReDo alone restore learning without any protection?
# Also CLEAR coef 0.3 / 0.1 on MinAtar (coef 1.0 came from MiniGrid and blocked learning of new games).
# MiniGrid 8-stage check: A and B must not hurt the existing MiniGrid result.
# Selection rule (fixed before running): MinAtar mean score after all games / mean final-seed scratch score;
# a fix is adopted only if it beats TMA_blind there AND is not worse than TL_blind on MiniGrid by > 5 points.
export OMP_NUM_THREADS=1
G="MinAtar-breakout MinAtar-space_invaders MinAtar-freeway MinAtar-asterix MinAtar-seaquest"
SEQ=$(echo $G | tr ' ' ',')
EVAL="--eval-tasks $SEQ --eval-episodes 30"
B="--tasks $SEQ --steps 2000000 --blind --reservoir 8192 --surprise-reset --retain-coef 0.3 $EVAL"
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
LB="--steps 500000 --tasks $E,$D,$L,$O,$X,$U,$D,$L --blind --reservoir 8192 --surprise-reset --retain-coef 0.3 --eval-tasks $EVAL_L --eval-episodes 30"
{
  for s in 101 102 103 104 105; do
    echo "TL_blind $s $LB"
    echo "TL_blind_all $s $LB --reset-scope all"
    echo "TL_blind_redo $s $LB --redo-every 10"
  done
  for s in 101 102 103 104 105; do
    echo "TMA_blind $s $B"
    echo "TMA_blind_all $s $B --reset-scope all"
    echo "TMA_blind_redo $s $B --redo-every 10"
    echo "TMA_none_redo $s --tasks $SEQ --steps 2000000 --blind --redo-every 10 $EVAL"
    echo "TMA_clear_0.3 $s --tasks $SEQ --steps 2000000 --clear --reservoir 8192 --retain-coef 0.3 $EVAL"
    echo "TMA_clear_0.1 $s --tasks $SEQ --steps 2000000 --clear --reservoir 8192 --retain-coef 0.1 $EVAL"
  done
} | xargs -P 9 -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
