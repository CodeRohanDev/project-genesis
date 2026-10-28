# Language track (L): child-like language learning, but faster (started 2026-10-24)

## Goal
The author's goal: the system should learn full English and Hindi the way a child does, but faster and more easily.
No system does this today. This track measures how much text a from-scratch learner needs for grammar, and tests
child-like ingredients one at a time to see which make learning more sample-efficient.

## Reference point: BabyLM
The BabyLM challenge trains models on child-scale input: ~10M words (strict-small, about what a child hears in the
first 2-5 years) or ~100M words, and evaluates grammar with BLiMP minimal pairs. Best 2024 strict-small BLiMP
(GPT-BERT) ~81% macro average; baselines ~70-74% ([findings](https://arxiv.org/pdf/2412.05149)).
Official English data: https://huggingface.co/datasets/BabyLM-community/babylm-eng (gated: needs the author's Hugging Face
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

## L1 results (2026-10-24)
Same model and settings for both corpora (4.9M params, byte-level, 4 epochs, GPU). Full BLiMP = all 67,000 pairs.
| Corpus | words | full BLiMP | grammar along training (fast BLiMP at words seen) |
|---|---|---|---|
| TinyStories (synthetic children's stories) | 12.2M | 56.3% | 49% @6M, 51% @10M, 53% @17M, 55% @49M (4 passes) |
| BabyLM English (real child-directed speech, children's books, simple wiki, subtitles), category-balanced 10M subset | 10.2M | 56.8% | 50% @5M, 52% @9M, 54% @14M, 56% @41M (4 passes) |
Chance = 50%. BabyLM 2024 strict-small (10M) baselines ~70-74%, best ~81% (GPT-BERT).

- TinyStories model writes fluent simple stories; the BabyLM model's text is less fluent (more varied input,
  same tiny model).
- Grammar emerges slowly and stays far below published baselines. Main suspects (to test, not assumed):
  model size (4.9M vs tens of millions of parameters in BabyLM baselines), byte-level input (longer sequences, sums
  over more positions), only 4 passes (baselines often 10-20), context 256 bytes.
- By phenomenon, BabyLM data helped most on NPI scope and determiner-noun agreement (+25-36 points) and hurt most on
  some question/ellipsis structures (-24-37 points).

## L2 results (2026-10-24): bigger model + subword tokenizer + more passes
Same BabyLM 10M data. Changes vs L1 (all at once, so the gain cannot be split between them):
BPE tokenizer (vocab 8k, ByteLevel, learned from the training file only), 13.8M params (vs 4.9M), 10 epochs (vs 4).
4.0 h on GTX 1050 Ti (sharing CPU with the MinAtar runs). Output: results/lm/L2_babylm10m_bpe8k_14M.{json,log,pt}.
| Model | full BLiMP | fast BLiMP along training (words seen) |
|---|---|---|
| L1 bytes, 4.9M, 4 epochs | 56.8% | 56% @41M |
| L2 BPE 8k, 13.8M, 10 epochs | **62.0%** | 53% @13M, 59% @36M, 61% @61M, 61% @102M |
- +5.2 points. Better on 39/67 phenomena, worse on 13 (>2 points). Largest gains: only_npi_licensor_present +57,
  irregular_past_participle_adjectives +47; largest losses: wh_questions_object_gap -20.
- Curve flattens after ~60M words (6 passes): more passes alone will not close the gap to baselines (~70-74%).
- Text samples: fluent dialogue-like English with grammatical errors and little coherence.
- Still far below BabyLM baselines. Next suspects: size (baselines 30-100M+ params; does not fit training time on
  the 4 GB GPU, a rented GPU would), context length, and the training objective.

## L3 results (2026-10-25): which of L2's three changes gave the +5.2?
L2 changed passes, tokenizer and size at once. L3 adds the missing steps (run_lm_ablate.sh, code 8cebd4e, one seed
each, GTX 1050 Ti):
| Model | tokens | params | epochs | full BLiMP | time |
|---|---|---|---|---|---|
| L1 | bytes | 4.9M | 4 | 56.8% | 2.5 h |
| L3c | bytes | 4.9M | 10 | **62.2%** | 6.1 h |
| L3b | BPE 8k | 6.9M | 10 | 60.2% | 2.4 h |
| L2 | BPE 8k | 13.8M | 10 | 62.0% | 4.0 h |
Decomposition (path L1 -> L3c -> L3b -> L2): more passes **+5.4**, BPE tokenizer -2.0, bigger model +1.8.
Phenomena better/worse (>0.5 point): passes 45/17, tokenizer 21/39, size 36/23.
- The whole L2 gain comes from more passes over the same 10M words. Tokenizer and size effects (~2 points, opposite
  signs) are within single-seed noise; we do not claim either.
- L3c's fast-BLiMP curve is still rising at 102M words (59% @61M, 61% @102M); L2's had flattened. Bytes need more
  passes but are not worse in the end.
- Cost: bytes are ~2.5x slower per pass on this GPU (longer sequences).
- Correction to L2 reading: "BPE + size helped" is NOT supported; repetition (seeing the same data many times,
  as a child hears the same words daily) is what helped.
- Next: more passes on bytes (20 epochs) is the cheapest test; seeds 2-3 for L3b/L3c if tokenizer/size claims matter.

## L4 results (2026-10-25): bytes, 4.9M params, 20 passes
run_lm_ablate.sh (code 1054ac7), one seed, 12.1 h on GTX 1050 Ti. Full BLiMP **65.5%** (L3c, 10 passes: 62.2%;
L1, 4 passes: 56.8%). Better on 42/67 phenomena, worse on 15 (>0.5 point); largest gain
distractor_agreement_relational_noun +20, largest loss complex_NP_island -8.
Fast-BLiMP along training: 54% @26M words, 57% @43M, 60% @72M, 64% @121M, 66% @204M; val loss still falling
(1.30 -> 1.26). Still rising at 20 passes: no plateau found yet.
- Same 10M words, 5x more passes than L1: +8.7 points. Repetition keeps paying off for this small byte model.
- Still below BabyLM baselines (~70-74%), and the words-seen count (204M) is now ~2x what a child hears by ~10 years
  (~100M): the gain comes from re-reading, which a child does not do in the same way. Do not call this child-like.
- Sample: grammatical-looking, meaningless English ("The personal ship was sounded in the forest."). No grounding.

Next (language track): scale-up within the 4 GB GPU (larger model, subword tokenizer learned from the 10M words only,
more passes), then the child-like ingredients (curriculum, feedback, grounding).
