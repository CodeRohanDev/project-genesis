#!/bin/bash
# Exp 28b: six baseline groups (L_/C_ none, distill, joint) ran seeds 1-10 on pre-repo code (no code field) and do not
# reproduce on current code (REP_C_none seed 1 differs). Current code is deterministic (TS_base == TL_blind_all, all
# 5 seeds, matrices identical). So seeds 1-10 are rerun on current code as <name>_v2; with the seeds 11-20 from
# run_seeds20.sh (same current code) they give 20 seeds from one code version. Old runs are kept and reported.
export OMP_NUM_THREADS=1
cmd() { .venv/bin/python rerun_cmd.py results/runs/$1_seed1.json; }
{
  for n in L_distill L_none L_joint C_distill C_none C_joint; do
    c=$(cmd $n); for s in $(seq 1 10); do echo "${n}_v2 $s $c"; done
  done
} | xargs -P ${P:-7} -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
