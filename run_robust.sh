#!/bin/bash
# Exp 23: robustness of the MinAtar result (final method = Life with --reset-scope all, exp 21). Seeds 1-10.
# No selection happens here: these are fixed ablations / checks, nothing is tuned.
#   MA_reset_all   full reset on surprise but NO reservoir/distill: how much does memory add?
#   MAR_*          reversed game order (Seaquest -> ... -> Breakout): is the ranking order-dependent?
#   MA_joint20M    joint with 2x the budget (20M mixed steps): is the joint baseline just under-trained?
#   O_reset_all    MiniGrid 3-task, full reset, no reservoir (same question as MA_reset_all, matches O_reset)
# Launch via ./safe_run.sh run_robust.sh. 7 parallel (LM runs on GPU at the same time). Finished runs skipped.
export OMP_NUM_THREADS=1
G="MinAtar-breakout MinAtar-space_invaders MinAtar-freeway MinAtar-asterix MinAtar-seaquest"
SEQ=$(echo $G | tr ' ' ',')
REV=$(echo $G | tr ' ' '\n' | tac | paste -sd,)
MIX=$(echo $G | tr ' ' '|')
EVAL="--eval-tasks $SEQ --eval-episodes 30"
M=MiniGrid
E="$M-Empty-5x5-v0|$M-Empty-6x6-v0|$M-Empty-8x8-v0"
D="$M-DoorKey-5x5-v0|$M-DoorKey-6x6-v0|$M-DoorKey-8x8-v0"
L="$M-LavaGapS5-v0|$M-LavaGapS6-v0|$M-LavaGapS7-v0"
EVAL_O="$M-Empty-5x5-v0,$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=10,$M-Empty-8x8-v0@size=12,\
$M-DoorKey-5x5-v0,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=10,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS5-v0,$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=8,$M-LavaGapS7-v0@size=10"
LIFE="--blind --reservoir 8192 --surprise-reset --retain-coef 0.3 --reset-scope all"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "MA_joint20M $s --tasks $MIX --steps 20000000 $EVAL"
    echo "MA_reset_all $s --tasks $SEQ --steps 2000000 --blind --surprise-reset --reset-scope all $EVAL"
    echo "MAR_blind_all $s --tasks $REV --steps 2000000 $LIFE $EVAL"
    echo "MAR_clear $s --tasks $REV --steps 2000000 --clear --reservoir 8192 --retain-coef 1.0 $EVAL"
    echo "MAR_none $s --tasks $REV --steps 2000000 --blind $EVAL"
  done
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "O_reset_all $s --steps 600000 --tasks $E,$D,$L --blind --surprise-reset --reset-scope all --eval-tasks $EVAL_O --eval-episodes 50"
  done
} | xargs -P 7 -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
