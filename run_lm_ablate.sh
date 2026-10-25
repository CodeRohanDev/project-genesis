#!/bin/bash
# L3: split L2's +5.2 BLiMP gain into its three changes (L2 changed passes, tokenizer and size at once).
#   L1  bytes, d256, 4 passes   (done)
#   L3c bytes, d256, 10 passes  -> L3c - L1  = effect of more passes
#   L3b BPE 8k, d256, 10 passes -> L3b - L3c = effect of the tokenizer
#   L2  BPE 8k, d384, 10 passes -> L2 - L3b  = effect of model size  (done)
# One seed each (as L1/L2): differences of ~1 point are within noise.
D=data/babylm
run() { [ -f results/lm/$1.json ] && return; systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=2 .venv/bin/python lm.py --train $D/train_10M.txt --valid $D/valid.txt --epochs 10 --name "$@" \
  > results/lm/$1.log 2>&1; }
run L3b_babylm10m_bpe8k_d256 --bpe 8000 --dim 256
run L3c_babylm10m_bytes_d256_10ep --dim 256
# L4: bytes, d256, 20 passes - where does the still-rising L3c curve stop? (~12 h)
run L4_babylm10m_bytes_d256_20ep --dim 256 --epochs 20
