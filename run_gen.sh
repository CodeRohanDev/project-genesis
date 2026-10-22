#!/bin/bash
# G1: does variety (mixed sizes) and/or working memory (LSTM) give generalisation to unseen sizes?
# DoorKey. Train 5x5 only vs mix of 5,6,8. Test 5,6,7,8,10,12 (7,10,12 never trained). Seeds 1-10, 1.5M steps.
export OMP_NUM_THREADS=1
D=MiniGrid-DoorKey
ONE="$D-5x5-v0"
MIX="$D-5x5-v0|$D-6x6-v0|$D-8x8-v0"
EVAL="$D-5x5-v0,$D-6x6-v0,$D-8x8-v0@size=7,$D-8x8-v0,$D-8x8-v0@size=10,$D-8x8-v0@size=12"
for s in 1 2 3 4 5 6 7 8 9 10; do
  echo "G_lstm_mix --memory $s $MIX"; echo "G_lstm_5 --memory $s $ONE"
  echo "G_cnn_mix --no-memory $s $MIX"; echo "G_cnn_5 --no-memory $s $ONE"
done | xargs -P 12 -L 1 sh -c 'm=$1; [ "$m" = "--no-memory" ] && m=""; .venv/bin/python ppo.py --steps 1500000 --tasks "$3" --eval-tasks "'"$EVAL"'" $m --seed $2 --name $0 > results/logs/$0_seed$2.log 2>&1'
