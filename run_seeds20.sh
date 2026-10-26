#!/bin/bash
# Exp 28: replication of the main results on NEW seeds 11-20 (seeds 1-10 already done), plus gradual change on
# MinAtar (seeds 1-10). Every command is rebuilt from the seed-1 run's saved config (rerun_cmd.py), so settings are
# identical; only the seed differs. Rule fixed BEFORE running: main tables report seeds 1-20 pooled; a claim is kept
# only if seeds 11-20 alone point the same way, else it is dropped or weakened. MinAtar normaliser stays the
# seeds 1-10 scratch mean (MAS_*).
# REP_*: seed 1 rerun on current code, to check that old runs reproduce exactly (code-drift check).
# GM_ours: our final method on MinAtar with gradual change, new game mixed in over 1/3 of a stage (as exp 27).
# Launch via ./safe_run.sh run_seeds20.sh.
export OMP_NUM_THREADS=1
cmd() { .venv/bin/python rerun_cmd.py results/runs/$1_seed1.json; }
{
  echo "REP_C_none 1 $(cmd C_none)"
  echo "REP_L_distill 1 $(cmd L_distill)"
  for n in MA_blind_all MA_clear MA_rd MA_joint MA_none; do
    c=$(cmd $n); for s in $(seq 11 20); do echo "$n $s $c"; done
  done
  c=$(cmd MA_blind_all); for s in $(seq 1 10); do echo "GM_ours $s $c --drift 666667"; done
  for n in L_blind_all L_distill L_clear L_none L_joint L_rd L_rdres O_full_all C_distill C_clear C_none C_joint C_rd C_rdres; do
    c=$(cmd $n); for s in $(seq 11 20); do echo "$n $s $c"; done
  done
} | xargs -P ${P:-7} -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
