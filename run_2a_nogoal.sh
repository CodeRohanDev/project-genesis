#!/bin/bash
# 2a final baseline: no-goal agent, seeds 1-5 (6-10 run later), 5M steps. Independent of the pilot outcome.
export OMP_NUM_THREADS=1
E=BabyAI-PickupDistDebug-v0
HOLD="red_ball+green_key+blue_box+purple_ball"
TRAINED=$(for c in red green blue purple yellow grey; do for o in ball box key; do echo ${c}_$o; done; done | grep -vxE "${HOLD//+/|}" | paste -sd+)
EVAL="$E@exclude=$HOLD,$E@only=$TRAINED,$E@only=$HOLD"
for s in ${SEEDS:-1 2 3 4 5}; do echo "A_nogoal $s"; done | xargs -P 10 -L 1 sh -c 'n=$0; s=$1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py --tasks "'"$E@exclude=$HOLD"'" --eval-tasks "'"$EVAL"'" --steps 5000000 --eval-episodes 200 --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
