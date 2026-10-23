"""Experiment 3a: a learned world model. Predicts the next 7x7 egocentric view from (view, action).

Data comes from our own trained agents (B_finetuneUP: Unlock + Pickup skills + UnlockPickup) acting with 30% random
actions, so failed actions (toggling a locked door without a key, walking into walls) are also seen.
Split: training data from agents 1-7 on training env seeds; test data from agents 8-10 on the separate eval seed range;
"new worlds" data from agents 8-10 on DoorKey-8x8 and KeyCorridorS3R1, which the model never saw.

    python worldmodel.py collect      # writes results/wm/data.npz
    python worldmodel.py train 1      # model seed 1 -> results/wm/model_seed1.pt + metrics JSON
    python worldmodel.py train 1 persist   # same with the object-permanence prior -> model_persist_seed1.pt
    python worldmodel.py train 1 conv      # locality prior (convolutions) + permanence -> model_conv_seed1.pt
"""
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from ppo import EVAL_SEED, IMG_DIM, Agent, make_env, to_tensor

OUT = Path("results/wm")
N_TYPE, N_COLOR, N_STATE = 11, 6, 3  # minigrid OBJECT/COLOR/STATE index ranges
TRAIN_ENVS = "MiniGrid-Unlock-v0|BabyAI-PickupDistDebug-v0|MiniGrid-UnlockPickup-v0"
NEW_ENVS = "MiniGrid-DoorKey-8x8-v0|MiniGrid-KeyCorridorS3R1-v0"
FRONT, AGENT = (3 * 7 + 5) * 3, (3 * 7 + 6) * 3  # flat offsets of the cell in front of the agent and the agent's own cell
TOGGLE, PICKUP, FORWARD = 5, 3, 2


@torch.no_grad()
def collect(agent_seeds, envs_spec, steps, env_seed, eps=0.3):
    """Roll out saved agents with eps-random actions; return (obs, action, next_obs) as uint8 arrays."""
    specs = envs_spec.split("|")
    xs, acts, ys = [], [], []
    for a_seed in agent_seeds:
        agent = Agent(goal=True)
        agent.load_state_dict(torch.load(f"results/runs/B_finetuneUP_seed{a_seed}.pt"))
        envs = gym.vector.SyncVectorEnv([make_env(specs[i % len(specs)]) for i in range(16)],
                                        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
        rng = np.random.default_rng(env_seed + a_seed)
        obs, _ = envs.reset(seed=env_seed + 1000 * a_seed)
        for _ in range(steps // 16 // len(agent_seeds)):
            dist, _, _ = agent(to_tensor(obs, "cpu"))
            act = dist.sample().numpy()
            rand = rng.random(16) < eps
            act[rand] = rng.integers(0, 6, rand.sum())  # 0-5: no 'done' action
            nxt, _, term, trunc, info = envs.step(act)
            done = term | trunc
            true_next = nxt.copy()
            if done.any():
                true_next[done] = np.stack(info["final_obs"][done])
            xs.append(obs[:, :IMG_DIM]); acts.append(act); ys.append(true_next[:, :IMG_DIM])
            obs = nxt
        envs.close()
    return np.concatenate(xs).astype(np.uint8), np.concatenate(acts).astype(np.uint8), np.concatenate(ys).astype(np.uint8)


class WorldModel(nn.Module):
    """MLP over the whole view (egocentric turns move every cell, so a global model is simpler than convs)."""

    def __init__(self, hidden=512, persist=False):
        super().__init__()
        # persist: "things stay as they are unless something changes them" built in (object permanence prior):
        # the current view is added to the output logits, so the network only has to learn changes.
        self.keep = nn.Parameter(torch.tensor(5.0)) if persist else None
        self.t, self.c, self.s = nn.Embedding(N_TYPE, 8), nn.Embedding(N_COLOR, 4), nn.Embedding(N_STATE, 4)
        self.a = nn.Embedding(7, 32)
        self.net = nn.Sequential(nn.Linear(49 * 16 + 32, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 49 * (N_TYPE + N_COLOR + N_STATE)))

    def forward(self, x, a):
        x = x.long().view(-1, 49, 3)
        h = torch.cat([self.t(x[..., 0]), self.c(x[..., 1]), self.s(x[..., 2])], -1).flatten(1)
        out = self.net(torch.cat([h, self.a(a.long())], -1)).view(-1, 49, N_TYPE + N_COLOR + N_STATE)
        if self.keep is not None:
            cur = torch.cat([nn.functional.one_hot(x[..., i], n) for i, n in enumerate((N_TYPE, N_COLOR, N_STATE))], -1)
            out = out + self.keep * cur
        return out.split([N_TYPE, N_COLOR, N_STATE], -1)

    def predict(self, x, a):
        t, c, s = self(x, a)
        return torch.stack([t.argmax(-1), c.argmax(-1), s.argmax(-1)], -1).flatten(1)  # same flat layout as x


class ConvWorldModel(WorldModel):
    """Same interface, locality prior: a stack of 3x3 convolutions over the 7x7 view (same rule everywhere in the
    view, cannot memorise where walls usually are). 6 layers give a 13x13 receptive field, enough for turns.
    Always uses the persistence prior."""

    def __init__(self, ch=64, layers=6):
        nn.Module.__init__(self)
        self.t, self.c, self.s = nn.Embedding(N_TYPE, 8), nn.Embedding(N_COLOR, 4), nn.Embedding(N_STATE, 4)
        self.a = nn.Embedding(7, 8)
        convs = [nn.Conv2d(16 + 8 + 2, ch, 3, padding=1), nn.ReLU()]
        for _ in range(layers - 2):
            convs += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU()]
        convs += [nn.Conv2d(ch, N_TYPE + N_COLOR + N_STATE, 3, padding=1)]
        self.net = nn.Sequential(*convs)
        self.keep = nn.Parameter(torch.tensor(5.0))
        # fixed (x, y) coordinate planes: the agent always sits at the same spot of its own view
        g = torch.linspace(-1, 1, 7)
        self.register_buffer("coords", torch.stack(torch.meshgrid(g, g, indexing="ij")))

    def forward(self, x, a):
        x = x.long().view(-1, 49, 3)
        h = torch.cat([self.t(x[..., 0]), self.c(x[..., 1]), self.s(x[..., 2])], -1).view(-1, 7, 7, 16).permute(0, 3, 1, 2)
        act = self.a(a.long())[:, :, None, None].expand(-1, -1, 7, 7)
        out = self.net(torch.cat([h, act, self.coords.expand(len(x), -1, -1, -1)], 1)).permute(0, 2, 3, 1).reshape(-1, 49, N_TYPE + N_COLOR + N_STATE)
        cur = torch.cat([nn.functional.one_hot(x[..., i], n) for i, n in enumerate((N_TYPE, N_COLOR, N_STATE))], -1)
        return (out + self.keep * cur).split([N_TYPE, N_COLOR, N_STATE], -1)


def loss_fn(model, x, a, y):
    y = y.long().view(-1, 49, 3)
    return sum(nn.functional.cross_entropy(o.flatten(0, 1), y[..., i].flatten()) for i, o in enumerate(model(x, a)))


def metrics(pred, x, a, y):
    """pred/x/y: (B, 147) int tensors. Compares the model with the 'nothing changes' baseline (pred = x)."""
    m = {}
    changed = y != x
    for name, p in [("model", pred), ("copy", x)]:
        m[f"{name}_cell_acc"] = (p == y).float().mean().item()
        m[f"{name}_exact_view"] = (p == y).all(1).float().mean().item()
        m[f"{name}_changed_cell_acc"] = ((p == y) & changed).sum().item() / max(1, changed.sum().item())
        # pickup/drop/toggle keep the view geometry fixed, so every cell of the next view is predictable in principle
        # (turns/moves reveal unseen cells, which no model can know)
        fixed = (a == PICKUP) | (a == 4) | (a == TOGGLE)
        m[f"{name}_exact_view_fixed_geometry"] = (p[fixed] == y[fixed]).all(1).float().mean().item()
    fr, ag = x[:, FRONT:FRONT + 3], x[:, AGENT:AGENT + 3]
    locked = (fr[:, 0] == 4) & (fr[:, 2] == 2) & (a == TOGGLE)
    has_key = (ag[:, 0] == 5) & (ag[:, 1] == fr[:, 1])
    probes = {
        "toggle_locked_door_with_key": locked & has_key,       # truth: door opens (state 0)
        "toggle_locked_door_without_key": locked & ~has_key,   # truth: stays locked (state 2)
        "pickup_facing_object_hands_empty": (a == PICKUP) & (fr[:, 0] >= 5) & (fr[:, 0] <= 7) & (ag[:, 0] == 1),
        "forward_into_wall": (a == FORWARD) & (fr[:, 0] == 2),
    }
    for k, mask in probes.items():
        n = int(mask.sum())
        m[f"probe_{k}_n"] = n
        if n:
            for name, p in [("model", pred), ("copy", x)]:
                # front cell (and own cell for pickup) predicted exactly right
                ok = (p[mask, FRONT:FRONT + 3] == y[mask, FRONT:FRONT + 3]).all(1)
                if k.startswith("pickup"):
                    ok &= (p[mask, AGENT:AGENT + 3] == y[mask, AGENT:AGENT + 3]).all(1)
                m[f"probe_{k}_{name}"] = ok.float().mean().item()
    return m


@torch.no_grad()
def evaluate(model, x, a, y, batch=8192):
    preds = torch.cat([model.predict(x[i:i + batch], a[i:i + batch]) for i in range(0, len(x), batch)])
    return metrics(preds, x.long(), a.long(), y.long())


@torch.no_grad()
def rollout_eval(model, x, a, y, ks=(1, 3, 5, 10, 20), n_env=16):
    """Imagination test: start from a real view, feed the real actions, never look at real views again.
    After k imagined steps, is the agent's own cell (what it is holding) right, and is the whole view right?
    Data rows are time-major over 16 envs (row t*16+e), so row i+16 continues row i unless an episode ended."""
    x, a, y = x.long(), a.long(), y.long()
    cont = (y[:-n_env] == x[n_env:]).all(1)  # row i is followed by row i+16 in the same episode
    out = {}
    for k in ks:
        starts = torch.arange(len(x) - k * n_env)
        ok = torch.ones(len(starts), dtype=torch.bool)
        for j in range(k - 1):
            ok &= cont[starts + j * n_env]
        starts = starts[ok][:20000]
        cur = x[starts]
        for j in range(k):
            cur = model.predict(cur, a[starts + j * n_env]).long()
        true = y[starts + (k - 1) * n_env]
        out[f"k{k}_n"] = len(starts)
        out[f"k{k}_holding_acc"] = (cur[:, AGENT:AGENT + 3] == true[:, AGENT:AGENT + 3]).all(1).float().mean().item()
        out[f"k{k}_cell_acc"] = (cur == true).float().mean().item()
    return out


def train(seed, persist=False, conv=False, epochs=8, batch=512):
    torch.manual_seed(seed)
    tag = "conv_" if conv else "persist_" if persist else ""
    d = np.load(OUT / "data.npz")
    T = {k: torch.as_tensor(d[k]) for k in d.files}
    model = ConvWorldModel() if conv else WorldModel(persist=persist)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    n = len(T["train_x"])
    start = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model, T["train_x"][idx], T["train_a"][idx], T["train_y"][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"seed {seed} epoch {ep + 1} loss {loss.item():.4f}", flush=True)
    res = {"seed": seed, "persist": persist or conv, "conv": conv, "epochs": epochs, "train_n": n, "wall_seconds": time.time() - start,
           "test": evaluate(model, T["test_x"], T["test_a"], T["test_y"]),
           "new_worlds": evaluate(model, T["new_x"], T["new_a"], T["new_y"])}
    torch.save(model.state_dict(), OUT / f"model_{tag}seed{seed}.pt")
    (OUT / f"metrics_{tag}seed{seed}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "collect":
        # agents rarely touch a locked door without the key, so part of the data is fully random play (eps=1),
        # which supplies failed actions (counterexamples) for the causal probes
        cat = lambda *ps: tuple(np.concatenate(z) for z in zip(*ps))
        parts = {"train": cat(collect(range(1, 8), TRAIN_ENVS, 700_000, 0),
                              collect(range(1, 8), TRAIN_ENVS, 150_000, 500_000, eps=1.0)),
                 "test": cat(collect(range(8, 11), TRAIN_ENVS, 150_000, EVAL_SEED),
                             collect(range(8, 11), TRAIN_ENVS, 30_000, EVAL_SEED + 500_000, eps=1.0)),
                 "new": cat(collect(range(8, 11), NEW_ENVS, 150_000, EVAL_SEED),
                            collect(range(8, 11), NEW_ENVS, 30_000, EVAL_SEED + 500_000, eps=1.0))}
        np.savez_compressed(OUT / "data.npz", **{f"{p}_{k}": v for p, (x, a, y) in parts.items() for k, v in zip("xay", (x, a, y))})
        print({p: len(v[0]) for p, v in parts.items()})
    else:
        train(int(sys.argv[2]), persist="persist" in sys.argv[3:], conv="conv" in sys.argv[3:])
