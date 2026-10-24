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
VERSION = int(__import__("os").environ.get("PLANNER_VERSION", 4))
ENV_SPEC = __import__("os").environ.get("PLANNER_ENV", "MiniGrid-UnlockPickup-v0")
# 5 = same as 4 under tighter step budgets (PLANNER_ENV=...@max_steps=N): wrong choices become costly
# 4 = map memory (mapmem.py): imagine skills from views rendered at remembered objects; see episode_v4
# 1 = first run (planner_v1.json), 2 = fairer score for the real mission (planner_v2.json),
# 3 = curiosity rule: goal in sight -> go for it; otherwise do the most doable unfinished skill (imagined), to change
#     the world and reveal more. A skill is only considered if its object (key / ball / door) has actually been seen
#     in this episode: the world model sometimes imagines objects that are not there. Threshold 0.2.
EPISODE_OFFSET = 10_000  # v2 is evaluated on new episode seeds, not the ones used to diagnose v1


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


def target_visible(view, real_target):
    t, c = real_target
    cells = view[:IMG_DIM].long().view(49, 3)
    return bool(((cells[:, 0] == t) & ((cells[:, 1] == c) if c is not None else True)).any())


@torch.no_grad()
def imagine(agent, wm, view, skill_ids, skill, real_ids, real_target, doable_only=False, horizon=H):
    """K imagined runs of one skill from the current real view. Returns the plan score."""
    cur = view[None, :IMG_DIM].repeat(K, 1).long()
    done = torch.zeros(K, dtype=torch.bool)
    for _ in range(horizon):
        dist, _, _ = agent(with_words(cur.float(), skill_ids))
        nxt = wm.predict(cur, dist.sample()).long()
        cur = torch.where(done[:, None], cur, nxt)
        done |= finished(cur, skill, real_target)
        if done.all():
            break
    if doable_only:
        return done.float().mean().item()
    _, value, _ = agent(with_words(cur.float(), real_ids))
    value = value.clamp(min=0)
    if skill == "real":
        if VERSION == 1:  # v1: only imagined success counted (unfair when the target is out of view)
            return done.float().mean().item()
        return torch.where(done, torch.ones_like(value), value).mean().item()  # v2: same yardstick as other skills
    return (done.float() * value).mean().item()


@torch.no_grad()
def episode(agent, wm, env, seed, method, rng):
    obs, _ = env.reset(seed=seed)
    real_ids = [int(t) for t in obs[IMG_DIM:] if t]
    real_target = target_of(real_ids)
    ids = {**{k: words(v) for k, v in SKILLS.items()}, "real": real_ids}
    order = ["open", "real"]  # hand-written correct order (only used by method 'correct')
    reward, done, chosen, completed, seen = 0.0, False, [], set(), False
    seen_types = set()  # object types the agent has really seen this episode  # completed: skills the agent remembers finishing
    while not done:
        view = torch.as_tensor(obs, dtype=torch.float32)
        open_ = [s for s in ids if s not in completed and not finished(view[None].long(), s, real_target)[0]] or ["real"]
        if method == "real_only":
            skill = "real"
        elif method == "random":
            skill = open_[rng.integers(len(open_))]
        elif method == "correct":
            skill = next((s for s in order if s in open_), "real")
        elif VERSION == 3:  # planner, curiosity rule
            seen = seen or target_visible(view, real_target)
            seen_types |= set(view[:IMG_DIM].long().view(49, 3)[:, 0].tolist())
            needs = {"open": 4, "key": 5, "ball": 6}
            others = [s for s in open_ if s != "real" and needs[s] in seen_types]
            doable = {s: imagine(agent, wm, view, ids[s], s, real_ids, real_target, doable_only=True) for s in others}
            best = max(doable, key=doable.get) if doable else None
            skill = "real" if seen or best is None or doable[best] < 0.2 else best
        else:  # planner v1/v2
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


DROP = 4


@torch.no_grad()
def episode_v4(agent, wm, env, seed, method, rng=None):
    """Planner with map memory. method: 'planner' (decides by imagination) or 'correct_map' (hand-written order,
    same map/navigation machinery: key -> open -> explore until box seen -> drop if holding -> real),
    or 'map_random' (ablation: same machinery and candidates as the planner, but choices made at random instead of
    by imagination)."""
    from mapmem import Map
    obs, _ = env.reset(seed=seed)
    real_ids = [int(t) for t in obs[IMG_DIM:] if t]
    real_target = target_of(real_ids)
    ids = {**{k: words(v) for k, v in SKILLS.items()}, "real": real_ids}
    m, chosen = Map(), []
    m.observe(obs)
    state = {"obs": obs, "done": False, "reward": 0.0, "steps": 0}

    def act(a):
        before = state["obs"]
        state["obs"], r, term, trunc, _ = env.step(a)
        m.step(a, before)
        m.observe(state["obs"])
        state["done"], state["reward"], state["steps"] = term or trunc, r, state["steps"] + 1

    def hand():
        return tuple(int(v) for v in state["obs"][AGENT:AGENT + 3])

    def run_skill(skill, n):
        for _ in range(n):
            if state["done"]:
                return
            dist, _, _ = agent(with_words(torch.as_tensor(state["obs"][None], dtype=torch.float32), ids[skill]))
            act(dist.sample().item())
            if finished(torch.as_tensor(state["obs"][None]).long(), skill, real_target)[0]:
                return

    def go_face(cell):
        for _ in range(60):
            if state["done"]:
                return False
            path = m.path_to_face(cell)
            if path is None:
                return False
            if not path:
                return True
            act(path[0])
        return False

    def drop():
        """Put the held object down; needs an empty cell in front, so turn until there is one (max 4 turns)."""
        for _ in range(4):
            if state["done"]:
                return
            if int(state["obs"][FRONT]) == 1:
                act(DROP)
                return
            act(0)

    def facing_view(cell, h):
        """Rendered view from the end of the path to `cell`, with hand h; None if unreachable."""
        path = m.path_to_face(cell)
        if path is None:
            return None
        pos, d = m.pos.copy(), m.dir
        for a in path:
            if a == 0: d = (d - 1) % 4
            elif a == 1: d = (d + 1) % 4
            else: pos = pos + np.array(__import__("mapmem").DIRS[d])
        return torch.as_tensor(m.render(pos, d, h), dtype=torch.float32)

    def doable(view, skill):
        return imagine(agent, wm, view, ids[skill], skill, real_ids, real_target, doable_only=True, horizon=10)

    while not state["done"]:
        t, c = real_target
        boxes = [b for b in m.find(t, c) if tuple(b) != tuple(m.pos)]
        h = hand()
        if method == "correct_map":
            keys, doors = m.find(5), [d for d in m.find(4) if m.cells[d[0], d[1], 2] != 0]
            if boxes:
                if h[0] != 1:
                    chosen.append("drop"); drop(); continue
                chosen.append("real"); go_face(boxes[0]); run_skill("real", 10)
            elif h[0] != 5 and keys:
                chosen.append("key"); go_face(keys[0]); run_skill("key", 20)
            elif h[0] == 5 and doors:
                chosen.append("open"); go_face(doors[0]); run_skill("open", 20)
            else:
                chosen.append("explore"); run_skill("real", 40)
            continue
        # planner: decide by imagination from views rendered at remembered objects
        if boxes:
            v = facing_view(boxes[0], h)
            empty_hand = tuple(int(x) for x in wm.predict(torch.as_tensor(state["obs"][None, :IMG_DIM]).long(),
                                                        torch.tensor([DROP]))[0, AGENT:AGENT + 3])
            vd = facing_view(boxes[0], empty_hand) if h[0] != 1 else None
            if method == "map_random":
                s_real, s_drop = rng.random(), (rng.random() if h[0] != 1 else -1.0)
            else:
                s_real = doable(v, "real") if v is not None else 0.0
                s_drop = doable(vd, "real") if vd is not None else -1.0
            if s_drop > s_real:
                chosen.append("drop"); drop()
            else:
                chosen.append("real"); go_face(boxes[0]); run_skill("real", 10)
            continue
        cands = {}
        for skill, ty in (("key", 5), ("open", 4), ("ball", 6)):
            for cell in m.find(ty):
                if skill == "open" and m.cells[cell[0], cell[1], 2] == 0:
                    continue  # already open
                if skill == "key" and h[0] == 5:
                    continue  # already holding a key
                v = facing_view(cell, h)
                if v is not None:
                    cands[(skill, tuple(cell))] = rng.random() if method == "map_random" else doable(v, skill)
        best = max(cands, key=cands.get) if cands else None
        if best is not None and cands[best] >= 0.2:
            chosen.append(best[0]); go_face(best[1]); run_skill(best[0], 20)
        else:
            chosen.append("explore"); run_skill("real", 40)
    return float(state["reward"] > 0), chosen


def run_seed(s, methods=("real_only", "random", "correct", "correct_map", "planner"), episodes=100):
    torch.set_num_threads(1)
    if True:
        agent = Agent(goal=True)
        agent.load_state_dict(torch.load(f"results/runs/B_comp_seed{s}.pt"))
        wm = WorldModel(persist=True)
        wm.load_state_dict(torch.load(f"results/wm/model_comp_persist_seed{(s - 1) % 3 + 1}.pt"))
        env = make_env(ENV_SPEC)()
        out = {}
        for m in methods:
            rng = np.random.default_rng(s)
            ep_fn = (lambda sd: episode_v4(agent, wm, env, sd, m, rng)) if m in ("planner", "correct_map", "map_random") and VERSION >= 4 \
                else (lambda sd: episode(agent, wm, env, sd, m, rng))
            eps = [ep_fn(EVAL_SEED + EPISODE_OFFSET * (VERSION - 1) + ep) for ep in range(episodes)]
            out[m] = {"success": float(np.mean([e[0] for e in eps])),
                      "first_skills": {k: sum(e[1][0] == k for e in eps) for k in ["open", "key", "ball", "real", "drop", "explore"]},
                      "example_plans": [e[1][:6] for e in eps[:5]]}
        print(s, {m: out[m]["success"] for m in methods}, "planner first choices:",
              out.get("planner", {}).get("first_skills"), flush=True)
        return s, out


if __name__ == "__main__":
    from multiprocessing import Pool
    methods = sys.argv[1:] or ["real_only", "random", "correct", "correct_map", "planner"]
    with Pool(10) as pool:
        res = dict(pool.starmap(run_seed, [(s, methods) for s in range(1, 11)]))
    for m in methods:
        xs = [res[s][m]["success"] for s in res]
        print(f"{m:10s} {np.mean(xs):.0%}", [round(x * 100) for x in xs])
    tag = ("_" + "_".join(methods) if sys.argv[1:] else "") + ("_" + ENV_SPEC.split("@")[1] if "@" in ENV_SPEC else "")
    json.dump(res, open(f"results/wm/planner_v{VERSION}{tag}.json", "w"), indent=1)
