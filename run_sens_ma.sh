#!/bin/bash
# Exp 25: sensitivity of our method on MinAtar (TUNING seeds 101-105), one setting changed at a time.
# Reference = TMA_blind_all (exp 21, same seeds, same flags). Same values as the MiniGrid check in exp 24
# (reservoir 32768 left out: bigger memory was not worse on MiniGrid). Launch via ./safe_run.sh run_sens_ma.sh.
export OMP_NUM_THREADS=1
G="MinAtar-breakout MinAtar-space_invaders MinAtar-freeway MinAtar-asterix MinAtar-seaquest"
SEQ=$(echo $G | tr ' ' ',')
B="--tasks $SEQ --steps 2000000 --blind --reservoir 8192 --surprise-reset --retain-coef 0.3 --reset-scope all --eval-tasks $SEQ --eval-episodes 30"
{
  for s in 101 102 103 104 105; do
    echo "TSM_master0.7 $s $B --master 0.7"
    echo "TSM_master0.9 $s $B --master 0.9"
    echo "TSM_surprise0.2 $s $B --surprise 0.2"
    echo "TSM_surprise0.4 $s $B --surprise 0.4"
    echo "TSM_coef0.1 $s $B --retain-coef 0.1"
    echo "TSM_coef1.0 $s $B --retain-coef 1.0"
    echo "TSM_res2048 $s $B --reservoir 2048"
  done
} | xargs -P ${P:-7} -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
