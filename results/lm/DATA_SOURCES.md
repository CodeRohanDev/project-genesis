# Downloaded data (not in git; re-download with these URLs)
- BLiMP (67 x 1000 grammatical minimal pairs): https://github.com/alexwarstadt/blimp/tree/master/data -> data/blimp/*.jsonl
- TinyStories V2 (GPT-4 generated children's stories, CDLA-Sharing-1.0), dev corpus until BabyLM access:
  train: bytes 0-62,000,000 of https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
  (12.2M words) -> data/tinystories/train_head.txt ; valid: TinyStoriesV2-GPT4-valid.txt -> data/tinystories/valid.txt
- BabyLM English (gated, needs Hugging Face login + access): https://huggingface.co/datasets/BabyLM-community/babylm-eng
