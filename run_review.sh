#!/bin/bash
# Exp 24: answers to likely reviewer questions. Launch via ./safe_run.sh run_review.sh. Finished runs skipped.
# Final seeds 1-10 (fixed comparisons, nothing selected):
#   MA_clear_0.1  CLEAR with its best MinAtar coef (0.1, tuning seeds, exp 21) - fairer CLEAR on MinAtar
#   MA_rd, L_rd, C_rd  Reset & Distill (Ahn et al. 2024) adapted: TOLD each task end, then full reset + fresh Adam,
#                      distill (KL, coef 0.3) toward stored last-rollout observations of every finished task
# Tuning seeds 101-105 (sensitivity of our method, MiniGrid 8-stage, one setting changed at a time):
#   TS_base (= final method), mastery 0.7/0.9, surprise 0.2/0.4, distill coef 0.1/1.0, reservoir 2048/32768
export OMP_NUM_THREADS=1
G="MinAtar-breakout MinAtar-space_invaders MinAtar-freeway MinAtar-asterix MinAtar-seaquest"
SEQ=$(echo $G | tr ' ' ',')
EVAL="--eval-tasks $SEQ --eval-episodes 30"
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
RD="--task-switch all --retain distill --retain-coef 0.3"
LSEQ="--steps 500000 --tasks $E,$D,$L,$O,$X,$U,$D,$L --eval-tasks $EVAL_L --eval-episodes 30"
LIFE="--blind --surprise-reset --reset-scope all"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do  # long MinAtar runs first
    echo "MA_clear_0.1 $s --tasks $SEQ --steps 2000000 --clear --reservoir 8192 --retain-coef 0.1 $EVAL"
    echo "MA_rd $s --tasks $SEQ --steps 2000000 $RD $EVAL"
  done
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "L_rd $s $LSEQ $RD"
    echo "C_rd $s --steps 600000 --tasks $E,$D,$L $RD --eval-tasks $EVAL_O --eval-episodes 50"
  done
  for s in 101 102 103 104 105; do
    echo "TS_base $s $LSEQ $LIFE --reservoir 8192 --retain-coef 0.3"
    echo "TS_master0.7 $s $LSEQ $LIFE --reservoir 8192 --retain-coef 0.3 --master 0.7"
    echo "TS_master0.9 $s $LSEQ $LIFE --reservoir 8192 --retain-coef 0.3 --master 0.9"
    echo "TS_surprise0.2 $s $LSEQ $LIFE --reservoir 8192 --retain-coef 0.3 --surprise 0.2"
    echo "TS_surprise0.4 $s $LSEQ $LIFE --reservoir 8192 --retain-coef 0.3 --surprise 0.4"
    echo "TS_coef0.1 $s $LSEQ $LIFE --reservoir 8192 --retain-coef 0.1"
    echo "TS_coef1.0 $s $LSEQ $LIFE --reservoir 8192 --retain-coef 1.0"
    echo "TS_res2048 $s $LSEQ $LIFE --reservoir 2048 --retain-coef 0.3"
    echo "TS_res32768 $s $LSEQ $LIFE --reservoir 32768 --retain-coef 0.3"
  done
} | xargs -P ${P:-7} -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
