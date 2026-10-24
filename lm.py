"""Language track L1: a small language model learned from scratch on child-scale text (bytes in, bytes out).

No pretrained weights, no pretrained tokenizer: the model reads raw UTF-8 bytes (so Hindi works unchanged).
Grammar is measured with BLiMP minimal pairs (67 phenomena): the model should give the grammatical sentence higher
probability. Ties count as wrong. Measured along training, so we get "grammar vs words seen".

    python lm.py --name ts10m --train data/tinystories/train_head.txt --valid data/tinystories/valid.txt
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BLIMP = Path("data/blimp")
OUT = Path("results/lm")
TOK = None  # optional subword tokenizer, learned ONLY from the training text (--bpe); None = raw bytes


def encode(text):
    return TOK.encode(text).ids if TOK else list(text.encode("utf-8"))


def decode(ids):
    return TOK.decode(ids) if TOK else bytes(ids).decode("utf-8", errors="replace")


def train_bpe(path, vocab, out):
    """Byte-level BPE trained from scratch on the training file; nothing pretrained."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train([str(path)], trainers.BpeTrainer(vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(str(out))
    return tok


class GPT(nn.Module):
    def __init__(self, layers=6, dim=256, heads=8, ctx=256, vocab=256):
        super().__init__()
        self.ctx = ctx
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(ctx, dim)
        block = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.1, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied input/output byte embeddings
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(ctx), persistent=False)

    def forward(self, x):
        t = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(t, device=x.device))
        h = self.blocks(h, mask=self.mask[:t, :t], is_causal=True)
        return self.head(self.norm(h))


def load_ids(path):
    if TOK is None:
        return np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    lines = Path(path).read_text().splitlines(keepends=True)
    ids = [i for enc in TOK.encode_batch(lines) for i in enc.ids]
    return np.array(ids, dtype=np.uint16)


def batch(data, n, ctx, device):
    i = np.random.randint(0, len(data) - ctx - 1, n)
    x = np.stack([data[j:j + ctx + 1] for j in i]).astype(np.int64)
    x = torch.as_tensor(x, device=device)
    return x[:, :-1], x[:, 1:]


@torch.no_grad()
def sentence_logprob(model, sentences, device, bs=128):
    """Total log-probability of each sentence's bytes, given a leading newline (start of text). Batched, padded."""
    out = []
    for i in range(0, len(sentences), bs):
        seqs = [encode("\n" + s)[: model.ctx + 1] for s in sentences[i:i + bs]]
        L = max(map(len, seqs))
        x = torch.zeros(len(seqs), L, dtype=torch.long, device=device)
        m = torch.zeros(len(seqs), L - 1, device=device)
        for k, q in enumerate(seqs):
            x[k, :len(q)] = torch.as_tensor(q)
            m[k, :len(q) - 1] = 1
        lp = F.log_softmax(model(x[:, :-1]), -1).gather(-1, x[:, 1:, None])[..., 0]
        out += (lp * m).sum(1).tolist()
    return out


@torch.no_grad()
def blimp(model, device, per_phenomenon=None):
    """Accuracy per phenomenon and macro average. per_phenomenon=None -> all 1000 pairs each."""
    model.eval()
    res = {}
    for f in sorted(BLIMP.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines()][:per_phenomenon]
        good = sentence_logprob(model, [r["sentence_good"] for r in rows], device)
        bad = sentence_logprob(model, [r["sentence_bad"] for r in rows], device)
        res[f.stem] = float(np.mean([g > b for g, b in zip(good, bad)]))
    model.train()
    return float(np.mean(list(res.values()))), res


@torch.no_grad()
def val_loss(model, data, ctx, device, n=64):
    model.eval()
    torch.manual_seed(0); np.random.seed(0)
    losses = []
    for _ in range(n // 16):
        x, y = batch(data, 16, ctx, device)
        losses.append(F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return float(np.mean(losses))


@torch.no_grad()
def sample(model, prompt, device, n=300, temp=0.8):
    model.eval()
    x = torch.as_tensor(encode(prompt), device=device)[None]
    for _ in range(n):
        p = F.softmax(model(x[:, -model.ctx:])[:, -1] / temp, -1)
        x = torch.cat([x, torch.multinomial(p, 1)], 1)
    model.train()
    return decode(x[0].tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--epochs", type=float, default=4.0, help="passes over the training text")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--ctx", type=int, default=256)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--evals", type=int, default=12, help="number of checkpoints with BLiMP (fast subset)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--bpe", type=int, default=0, help="subword vocab size learned from the training text; 0 = bytes")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    global TOK
    OUT.mkdir(parents=True, exist_ok=True)
    if args.bpe:
        TOK = train_bpe(args.train, args.bpe, OUT / f"{args.name}_tokenizer.json")
    train, valid = load_ids(args.train), load_ids(args.valid)
    words = len(Path(args.train).read_text(errors="ignore").split())
    model = GPT(args.layers, args.dim, ctx=args.ctx, vocab=args.bpe or 256).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    tokens_per_step = args.batch * args.ctx
    steps = int(args.epochs * len(train) / tokens_per_step)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    eval_at = set(np.unique(np.geomspace(max(1, steps // 300), steps, args.evals).astype(int)))
    log = {"config": vars(args), "device": device, "train_tokens": len(train), "train_words": words,
           "params": sum(p.numel() for p in model.parameters()), "curve": []}
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{log['params']/1e6:.1f}M params, {steps} steps, {words/1e6:.1f}M words x {args.epochs} epochs", flush=True)
    start = time.time()
    for step in range(1, steps + 1):
        x, y = batch(train, args.batch, args.ctx, device)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step in eval_at:
            acc, _ = blimp(model, device, per_phenomenon=100)
            point = {"step": step, "words_seen": step * tokens_per_step / len(train) * words,
                     "val_loss": val_loss(model, valid, args.ctx, device), "blimp_fast": acc,
                     "minutes": (time.time() - start) / 60}
            log["curve"].append(point)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in point.items()}), flush=True)
            (OUT / f"{args.name}.json").write_text(json.dumps(log, indent=1))
    acc, per = blimp(model, device)
    log.update(blimp_full=acc, blimp_per_phenomenon=per, minutes=(time.time() - start) / 60,
               samples=[sample(model, s, device) for s in ["Once upon a time", "The little girl", "Lily was sad because"]])
    (OUT / f"{args.name}.json").write_text(json.dumps(log, indent=1))
    torch.save(model.state_dict(), OUT / f"{args.name}.pt")
    print("BLiMP full:", round(acc, 3), "\n" + "\n---\n".join(log["samples"]))


if __name__ == "__main__":
    main()
