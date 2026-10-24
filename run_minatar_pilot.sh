#!/bin/bash
# MinAtar pilot (tuning seeds 101-102): PPO from scratch on each game, 3M steps. Which games are learnable, at what budget.
export OMP_NUM_THREADS=1
for g in breakout space_invaders freeway asterix seaquest; do for s in 101 102; do echo "$g $s"; done; done |
  xargs -P 10 -L 1 sh -c '[ -f results/runs/pilotMA_$0_seed$1.json ] && exit 0; .venv/bin/python ppo.py --tasks MinAtar-$0 --steps 3000000 --eval-episodes 30 --seed $1 --name pilotMA_$0 > results/logs/pilotMA_$0_seed$1.log 2>&1'
