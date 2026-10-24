# Language track (L): child-like language learning, but faster (started 2026-10-24)

## Goal
Rohan's goal: the system should learn full English and Hindi the way a child does, but faster and more easily.
No system does this today. This track measures how much text a from-scratch learner needs for grammar, and tests
child-like ingredients one at a time to see which make learning more sample-efficient.

## Reference point: BabyLM
The BabyLM challenge trains models on child-scale input: ~10M words (strict-small, about what a child hears in the
first 2-5 years) or ~100M words, and evaluates grammar with BLiMP minimal pairs. Best 2024 strict-small BLiMP
(GPT-BERT) ~81% macro average; baselines ~70-74% ([findings](https://arxiv.org/pdf/2412.05149)).
Official English data: https://huggingface.co/datasets/BabyLM-community/babylm-eng (gated: needs Rohan's Hugging Face
login + access). Until then, development runs use 12.2M words of TinyStories (synthetic children's stories, open).
No Hindi child-directed corpus was found (results/lm/DATA_SOURCES.md); a gap to fill later.

## L1 baseline (running)
- Model: 6-layer transformer, d=256, 4.9M parameters, reads raw UTF-8 bytes (no tokenizer, nothing pretrained;
  Hindi works unchanged). Context 256 bytes. AdamW lr 1e-3 cosine, batch 32, 4 epochs. GPU (GTX 1050 Ti).
- Data: TinyStories, first 12.2M words. Validation: TinyStories valid split.
- Grammar: BLiMP, all 67 phenomena; fast subset (100 pairs each) at 12 checkpoints spaced by words seen, full
  67,000 pairs at the end. Ties count as wrong. Untrained model: 49.7% (chance).
- Script: lm.py. Output: results/lm/L1_ts12m_base.{json,log,pt}.

## Planned comparisons (one ingredient at a time, same words budget)
1. Curriculum: simple to complex input (child-directed speech first), vs shuffled.
2. Teacher feedback / interaction: the learner produces text and gets corrections, vs passive reading.
3. Grounding: words tied to objects/actions in the gridworld agent (from experiment 2a), vs text only.
Metric: words needed to reach a given BLiMP level, and BLiMP at a fixed budget.
Caveat: TinyStories is GPT-generated, not real child input; BabyLM data needed for claims about child-like input.
