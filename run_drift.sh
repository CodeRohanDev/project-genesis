#!/bin/bash
# Exp 27: gradual task change. MiniGrid 3-task (Empty -> DoorKey -> LavaGap, 600k steps each, as C_/O_ runs), but
# each new stage starts as a mix: episode is the new task with prob rising linearly 0 -> 1 over the first 200k steps
# (--drift 200000). Question: does the surprise still fire, and are old skills kept, when change is gradual?
# Abrupt counterparts already exist: O_full_all (ours), C_rdres (told reset + our memory), C_none.
# Final seeds 1-10, fixed before running. Launch via ./safe_run.sh run_drift.sh.
export OMP_NUM_THREADS=1
M=MiniGrid
E="$M-Empty-5x5-v0|$M-Empty-6x6-v0|$M-Empty-8x8-v0"
D="$M-DoorKey-5x5-v0|$M-DoorKey-6x6-v0|$M-DoorKey-8x8-v0"
L="$M-LavaGapS5-v0|$M-LavaGapS6-v0|$M-LavaGapS7-v0"
EVAL_O="$M-Empty-5x5-v0,$M-Empty-8x8-v0,$M-Empty-8x8-v0@size=10,$M-Empty-8x8-v0@size=12,\
$M-DoorKey-5x5-v0,$M-DoorKey-8x8-v0,$M-DoorKey-8x8-v0@size=10,$M-DoorKey-8x8-v0@size=12,\
$M-LavaGapS5-v0,$M-LavaGapS7-v0,$M-LavaGapS7-v0@size=8,$M-LavaGapS7-v0@size=10"
B="--steps 600000 --tasks $E,$D,$L --drift 200000 --eval-tasks $EVAL_O --eval-episodes 50"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "G_ours $s $B --blind --reservoir 8192 --retain-coef 0.3 --surprise-reset --reset-scope all"
    echo "G_told $s $B --blind --reservoir 8192 --retain-coef 0.3 --task-switch all"
    echo "G_none $s $B"
  done
} | xargs -P ${P:-7} -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
