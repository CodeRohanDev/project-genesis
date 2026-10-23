#!/bin/bash
# 2a final (strict PickupDist, 5M steps, entropy 0.05):
#   A_goal      reads mission words, 4 colour+object pairs held out of training   seeds 1-10
#   A_goal_all  reads mission words, trained on ALL pairs (control: are held-out pairs just harder?)  seeds 1-5
#   A_nogoal    image only, same training as A_goal   seeds 6-10 (1-5 from run_2a_nogoal.sh)
export OMP_NUM_THREADS=1
E=BabyAI-PickupDistDebug-v0
HOLD="red_ball+green_key+blue_box+purple_ball"
TRAINED=$(for c in red green blue purple yellow grey; do for o in ball box key; do echo ${c}_$o; done; done | grep -vxE "${HOLD//+/|}" | paste -sd+)
export TRAIN_HELD="$E@exclude=$HOLD" TRAIN_ALL="$E" EVAL="$E@exclude=$HOLD,$E@only=$TRAINED,$E@only=$HOLD"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do echo "A_goal $s TRAIN_HELD --goal"; done
  for s in 1 2 3 4 5; do echo "A_goal_all $s TRAIN_ALL --goal"; done
  for s in 6 7 8 9 10; do echo "A_nogoal $s TRAIN_HELD"; done
} | xargs -P 10 -L 1 sh -c '
  n=$0; s=$1; eval t=\$$2; shift 2
  [ -f results/runs/${n}_seed$s.json ] && exit 0
  .venv/bin/python ppo.py --tasks "$t" --eval-tasks "$EVAL" --steps 5000000 --eval-episodes 200 --seed $s --name $n "$@" \
    > results/logs/${n}_seed$s.log 2>&1'
