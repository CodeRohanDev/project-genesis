"""Make child-scale text files from the BabyLM English 100M data (data/babylm/*.parquet, gated download).

train_10M.txt: random documents (seed 0) until ~10M whitespace words, same category mix as the full corpus
               (BabyLM strict-small size, ~ a child's input in the first years).
valid.txt:     ~0.5M words from documents not in train_10M.
Documents are separated by a blank line.

    python babylm_prep.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

D = Path("data/babylm")
t = pq.read_table(sorted(D.glob("train-*.parquet")), columns=["text", "category"]).to_pydict()
docs = list(zip(t["text"], t["category"]))
random.Random(0).shuffle(docs)
total = Counter()
for text, cat in docs:
    total[cat] += len(text.split())
grand = sum(total.values())
quota = {c: 10_000_000 * n / grand for c, n in total.items()}  # keep each category's share
train, valid, used, vwords = [], [], Counter(), 0
for text, cat in docs:
    w = len(text.split())
    if used[cat] + w <= quota[cat] * 1.02:
        train.append(text); used[cat] += w
    elif vwords < 500_000:
        valid.append(text); vwords += w
(D / "train_10M.txt").write_text("\n\n".join(train))
(D / "valid.txt").write_text("\n\n".join(valid))
stats = {"train_words": sum(used.values()), "valid_words": vwords, "train_docs": len(train),
         "train_words_by_category": dict(used), "full_corpus_words_by_category": dict(total)}
(D / "prep_stats.json").write_text(json.dumps(stats, indent=1))
print(json.dumps(stats, indent=1))
