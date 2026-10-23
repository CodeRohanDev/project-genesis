"""3b feasibility: can learned skills solve UnlockPickup if someone ELSE picks the order of missions?
No planning, no world model. Upper bound / control for the skill planner.

The agent (B_comp: trained only on Unlock + Pickup) gets a fake mission in its observation; the environment keeps its
real goal ("pick up the <c> box") and reward. Orders tested:
  pickup_only: real mission all the time (= zero-shot, expected ~0%)
  correct:     "open the door" until the door is open, then the real mission
  wrong:       real mission for half the budget, then "open the door"

    python skill_sequence.py
"""
import json

import numpy as np
import torch

from ppo import EVAL_SEED, IMG_DIM, MAX_WORDS, WORD_ID, Agent, make_env, to_tensor

OPEN_DOOR = [WORD_ID[w] for w in "open the door".split()]


def with_mission(obs, words):
    obs = obs.copy()
    obs[IMG_DIM:] = 0
    obs[IMG_DIM:IMG_DIM + len(words)] = words
    return obs


@torch.no_grad()
def run(agent, order, episodes=100):
    env = make_env("MiniGrid-UnlockPickup-v0")()
    wins = 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + ep)
        real = list(obs[IMG_DIM:][obs[IMG_DIM:] > 0])
        door = next(o for o in env.unwrapped.grid.grid if o is not None and o.type == "door")
        t, done = 0, False
        while not done:
            door_open = door.is_open  # used only to switch the fake mission; the agent never sees this flag
            if order == "correct":
                words = real if door_open else OPEN_DOOR
            elif order == "wrong":
                words = real if t < 144 else OPEN_DOOR
            else:
                words = real
            dist, _, _ = agent(to_tensor(with_mission(obs, words)[None], "cpu"))
            obs, reward, term, trunc, _ = env.step(dist.sample().item())
            done, t = term or trunc, t + 1
        wins += reward > 0
    return wins / episodes


if __name__ == "__main__":
    torch.set_num_threads(1)
    res = {}
    for s in range(1, 11):
        agent = Agent(goal=True)
        agent.load_state_dict(torch.load(f"results/runs/B_comp_seed{s}.pt"))
        res[s] = {order: run(agent, order) for order in ("pickup_only", "correct", "wrong")}
        print(s, res[s], flush=True)
    for order in ("pickup_only", "correct", "wrong"):
        xs = [r[order] for r in res.values()]
        print(order, f"{np.mean(xs):.0%}", [round(x * 100) for x in xs])
    json.dump(res, open("results/wm/skill_sequence.json", "w"), indent=1)
