"""Multi-step imagination test for every trained world model variant -> results/wm/rollout.json.

    python wm_rollouts.py plain persist conv
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import worldmodel as W


class Copy:  # baseline: imagine that nothing ever changes
    def predict(self, c, act):
        return c


if __name__ == "__main__":
    torch.set_num_threads(2)
    d = np.load(W.OUT / "data.npz")
    data = {part: tuple(torch.as_tensor(d[f"{part}_{k}"]) for k in "xay") for part in ("test", "new")}
    path = W.OUT / "rollout.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["copy"] = {part: W.rollout_eval(Copy(), *xy) for part, xy in data.items()}
    for variant in sys.argv[1:]:
        tag = "" if variant == "plain" else f"{variant}_"
        for s in range(1, 6):
            f = W.OUT / f"model_{tag}seed{s}.pt"
            if not f.exists():
                continue
            m = W.ConvWorldModel() if variant == "conv" else W.WorldModel(persist=variant == "persist")
            m.load_state_dict(torch.load(f))
            res[f"{variant}_seed{s}"] = {part: W.rollout_eval(m, *xy) for part, xy in data.items()}
            print(variant, s, {p: round(r["k10_holding_acc"], 2) for p, r in res[f"{variant}_seed{s}"].items()}, flush=True)
            path.write_text(json.dumps(res, indent=1))
