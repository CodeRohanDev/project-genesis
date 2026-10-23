#!/bin/bash
# 2b final, seeds 1-10. Components: Unlock ("open the door") + strict Pickup ("pick up the red box"), 5M steps.
# Composites never trained during component phase: UnlockPickup, KeyCorridorS3R1.
#   B_comp        components 5M (zero-shot eval on composites)
#   B_finetuneUP  B_comp agent + 2M on UnlockPickup
#   B_scratchUP   UnlockPickup from random init, 7M (= same TOTAL steps as comp + finetune)
export OMP_NUM_THREADS=1
export COMP="MiniGrid-Unlock-v0|BabyAI-PickupDistDebug-v0"
export EVAL="MiniGrid-Unlock-v0,BabyAI-PickupDistDebug-v0,MiniGrid-UnlockPickup-v0,MiniGrid-KeyCorridorS3R1-v0"
run() {  # name seed tasks steps [extra args]
  n=$1; s=$2; t=$3; st=$4; shift 4
  [ -f results/runs/${n}_seed$s.json ] && return 0
  .venv/bin/python ppo.py --goal --tasks "$t" --eval-tasks "$EVAL" --steps $st --eval-episodes 200 --seed $s --name $n "$@" \
    > results/logs/${n}_seed$s.log 2>&1
}
export -f run
# each line: component run then its fine-tune (chained), or a scratch run
{ for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "run B_comp $s \"\$COMP\" 5000000 && run B_finetuneUP $s MiniGrid-UnlockPickup-v0 2000000 --init results/runs/B_comp_seed$s.pt"
    echo "run B_scratchUP $s MiniGrid-UnlockPickup-v0 7000000"
  done; } | xargs -P 10 -I{} bash -c '{}'
