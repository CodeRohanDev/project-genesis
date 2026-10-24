#!/bin/bash
# Second environment: MinAtar. Breakout -> Space Invaders -> Freeway -> Asterix -> Seaquest, 2M steps per game,
# seeds 1-10. Same PPO / Life / CLEAR settings as MiniGrid (nothing tuned on MinAtar):
#   MA_none   sequential, no protection (blind)
#   MA_clear  CLEAR, reservoir 8192, coef 1.0 (chosen on MiniGrid)
#   MA_blind  ours: Life, reservoir 8192, surprise reset, distill 0.3, mastery 0.8 of own best score
#   MA_joint  all 5 games at once, 10M steps
#   MAS_<game> single game from scratch, 2M steps (normaliser: score / scratch score)
export OMP_NUM_THREADS=1
G="MinAtar-breakout MinAtar-space_invaders MinAtar-freeway MinAtar-asterix MinAtar-seaquest"
SEQ=$(echo $G | tr ' ' ',')
MIX=$(echo $G | tr ' ' '|')
EVAL="--eval-tasks $SEQ --eval-episodes 30"
{
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "MA_blind $s --tasks $SEQ --steps 2000000 --blind --reservoir 8192 --surprise-reset --retain-coef 0.3 $EVAL"
    echo "MA_clear $s --tasks $SEQ --steps 2000000 --clear --reservoir 8192 --retain-coef 1.0 $EVAL"
    echo "MA_none $s --tasks $SEQ --steps 2000000 --blind $EVAL"
    echo "MA_joint $s --tasks $MIX --steps 10000000 $EVAL"
  done
  for s in 1 2 3 4 5 6 7 8 9 10; do for g in $G; do
    echo "MAS_${g#MinAtar-} $s --tasks $g --steps 2000000 $EVAL"
  done; done
} | xargs -P 10 -L 1 sh -c 'n=$0; s=$1; shift 1; [ -f results/runs/${n}_seed$s.json ] && exit 0; .venv/bin/python ppo.py "$@" --seed $s --name $n > results/logs/${n}_seed$s.log 2>&1'
