#!/bin/bash
# 2a pilot (tuning seeds only): is 5M steps enough on strict PickupDist? entropy 0.05 vs 0.01.
export OMP_NUM_THREADS=1
E=BabyAI-PickupDistDebug-v0
HOLD="red_ball+green_key+blue_box+purple_ball"
TRAINED=$(for c in red green blue purple yellow grey; do for o in ball box key; do echo ${c}_$o; done; done | grep -vxE "${HOLD//+/|}" | paste -sd+)
EVAL="$E@exclude=$HOLD,$E@only=$TRAINED,$E@only=$HOLD"
{ for s in 101 102; do
  echo "p2a_goal $s --goal"; echo "p2a_goal_ent01 $s --goal --ent-coef 0.01"
done; echo "p2a_nogoal 101"; } | xargs -P 10 -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --tasks "'"$E@exclude=$HOLD"'" --eval-tasks "'"$EVAL"'" --steps 5000000 --eval-episodes 200 --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
