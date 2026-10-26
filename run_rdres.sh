#!/bin/bash
# Exp 26: why does told Reset & Distill collapse on 2/10 MiniGrid seeds (exp 24)? Control: OUR memory
# (competence-gated reservoir 8192, distill 0.3) but full reset at the GIVEN task boundaries (no surprise reset).
# vs L_rd/C_rd: only the memory differs. vs L_blind_all/O_full_all: only how the reset is triggered.
# Final seeds 1-10, fixed before running. Launch via ./safe_run.sh run_rdres.sh.
export OMP_NUM_THREADS=1
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
MEM="--blind --reservoir 8192 --retain-coef 0.3 --task-switch all"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "L_rdres $s --steps 500000 --tasks $E,$D,$L,$O,$X,$U,$D,$L $MEM --eval-tasks $EVAL_L --eval-episodes 30"
    echo "C_rdres $s --steps 600000 --tasks $E,$D,$L $MEM --eval-tasks $EVAL_O --eval-episodes 50"
  done
} | xargs -P ${P:-7} -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
