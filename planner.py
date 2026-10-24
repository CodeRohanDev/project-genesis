"""3b: planning over skills with the learned world model.

The agent (B_comp: trained only on "open the door" and "pick up <object>") faces UnlockPickup ("pick up the <c> box",
box behind a locked door), which neither the agent nor the world model (component-only data) has ever seen.

Skills = the same policy given different missions: "open the door", "pick up a key", "pick up a ball" (irrelevant),
and the real mission. Whether a skill is finished is read from the agent's own view (front cell is an open door /
the named object is in its hand).

At each decision point the planner IMAGINES each unfinished skill K times for up to H steps inside the world model,
starting from the current real view, and scores it:
  real mission:  fraction of imagined runs that end with the target in hand
  other skills:  fraction finished x critic's estimate of success for the REAL mission from the imagined end view
It runs the best skill in the real world until it finishes (or 40 steps), then decides again.

Compared in the same harness: real mission only (zero-shot), random skill choice, hand-written correct order.

    python planner.py            # all methods, agents 1-10 -> results/wm/planner.json
"""
import json
import sys

import numpy as np
import torch

from ppo import EVAL_SEED, IMG_DIM, VOCAB, WORD_ID, Agent, make_env, to_tensor
from worldmodel import AGENT, FRONT, WorldModel

SKILLS = {"open": "open the door", "key": "pick up a key", "ball": "pick up a ball"}
TYPE_ID = {"key": 5, "ball": 6, "box": 7}
COLOR_ID = {c: i for i, c in enumerate(["red", "green", "blue", "purple", "yellow", "grey"])}
K, H, SKILL_STEPS = 16, 40, 40


def words(text):
    return [WORD_ID[w] for w in text.split()]


def target_of(mission_ids):
    """'pick up the blue box' -> (type 7, colour 2); colour None if not given."""
    w = [VOCAB[i] for i in mission_ids]
    return TYPE_ID.get(w[-1]), COLOR_ID.get(w[-2])


def finished(view, skill, real_target):
    """view: (B, >=147) tensor. Read completion from the agent's own view."""
    front, hand = view[:, FRONT:FRONT + 3], view[:, AGENT:AGENT + 3]
    if skill == "open":
        return (front[:, 0] == 4) & (front[:, 2] == 0)
    t, c = (TYPE_ID[SKILLS[skill].split()[-1]], None) if skill in ("key", "ball") else real_target
    ok = hand[:, 0] == t
    return ok & (hand[:, 1] == c) if c is not None else ok


def with_words(img, ids):
    out = torch.zeros(len(img), IMG_DIM + 16)
    out[:, :IMG_DIM] = img[:, :IMG_DIM]
    out[:, IMG_DIM:IMG_DIM + len(ids)] = torch.as_tensor(ids, dtype=torch.float32)
    return out


@torch.no_grad()
def imagine(agent, wm, view, skill_ids, skill, real_ids, real_target):
    """K imagined runs of one skill from the current real view. Returns the plan score."""
    cur = view[None, :IMG_DIM].repeat(K, 1).long()
    done = torch.zeros(K, dtype=torch.bool)
    for _ in range(H):
        dist, _, _ = agent(with_words(cur.float(), skill_ids))
        nxt = wm.predict(cur, dist.sample()).long()
        cur = torch.where(done[:, None], cur, nxt)
        done |= finished(cur, skill, real_target)
        if done.all():
            break
    if skill == "real":
        return done.float().mean().item()
    _, value, _ = agent(with_words(cur.float(), real_ids))
    return (done.float() * value.clamp(min=0)).mean().item()


@torch.no_grad()
def episode(agent, wm, env, seed, method, rng):
    obs, _ = env.reset(seed=seed)
    real_ids = [int(t) for t in obs[IMG_DIM:] if t]
    real_target = target_of(real_ids)
    ids = {**{k: words(v) for k, v in SKILLS.items()}, "real": real_ids}
    order = ["open", "real"]  # hand-written correct order (only used by method 'correct')
    reward, done, chosen, completed = 0.0, False, [], set()  # completed: skills the agent remembers finishing
    while not done:
        view = torch.as_tensor(obs, dtype=torch.float32)
        open_ = [s for s in ids if s not in completed and not finished(view[None].long(), s, real_target)[0]] or ["real"]
        if method == "real_only":
            skill = "real"
        elif method == "random":
            skill = open_[rng.integers(len(open_))]
        elif method == "correct":
            skill = next((s for s in order if s in open_), "real")
        else:  # planner
            scores = {s: imagine(agent, wm, view, ids[s], s, real_ids, real_target) for s in open_}
            skill = max(scores, key=scores.get)
        chosen.append(skill)
        for _ in range(SKILL_STEPS):
            dist, _, _ = agent(with_words(torch.as_tensor(obs[None], dtype=torch.float32), ids[skill]))
            obs, reward, term, trunc, _ = env.step(dist.sample().item())
            done = term or trunc
            if done or finished(torch.as_tensor(obs[None]).long(), skill, real_target)[0]:
                completed.add(skill)
                break
    return float(reward > 0), chosen


def run_seed(s, methods=("real_only", "random", "correct", "planner"), episodes=100):
    torch.set_num_threads(1)
    if True:
        agent = Agent(goal=True)
        agent.load_state_dict(torch.load(f"results/runs/B_comp_seed{s}.pt"))
        wm = WorldModel(persist=True)
        wm.load_state_dict(torch.load(f"results/wm/model_comp_persist_seed{(s - 1) % 3 + 1}.pt"))
        env = make_env("MiniGrid-UnlockPickup-v0")()
        out = {}
        for m in methods:
            rng = np.random.default_rng(s)
            eps = [episode(agent, wm, env, EVAL_SEED + ep, m, rng) for ep in range(episodes)]
            out[m] = {"success": float(np.mean([e[0] for e in eps])),
                      "first_skills": {k: sum(e[1][0] == k for e in eps) for k in ["open", "key", "ball", "real"]},
                      "example_plans": [e[1][:6] for e in eps[:5]]}
        print(s, {m: out[m]["success"] for m in methods}, "planner first choices:",
              out.get("planner", {}).get("first_skills"), flush=True)
        return s, out


if __name__ == "__main__":
    from multiprocessing import Pool
    methods = sys.argv[1:] or ["real_only", "random", "correct", "planner"]
    with Pool(10) as pool:
        res = dict(pool.starmap(run_seed, [(s, methods) for s in range(1, 11)]))
    for m in methods:
        xs = [res[s][m]["success"] for s in res]
        print(f"{m:10s} {np.mean(xs):.0%}", [round(x * 100) for x in xs])
    json.dump(res, open("results/wm/planner.json", "w"), indent=1)
