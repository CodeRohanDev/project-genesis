"""Generalisation test: evaluate saved agents, without any further training, on bigger unseen grids.
Same 7x7 egocentric view, so the network applies unchanged; only the world is larger.

    python gen_eval.py
"""
import json
import statistics
from multiprocessing import Pool
from pathlib import Path

import torch

from ppo import Agent, evaluate
from summarize import ci

FAMILIES = {
    "Empty": ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-6x6-v0", "MiniGrid-Empty-8x8-v0"],
    "DoorKey": ["MiniGrid-DoorKey-5x5-v0", "MiniGrid-DoorKey-6x6-v0", "MiniGrid-DoorKey-8x8-v0"],
    "Lava": ["MiniGrid-LavaGapS5-v0", "MiniGrid-LavaGapS6-v0", "MiniGrid-LavaGapS7-v0"],
}
# group -> families it was trained on (scratch agents only tested on their own family)
GROUPS = {
    "untrained": list(FAMILIES),
    "B1s_Empty-5x5-v0": ["Empty"],
    "B1s_DoorKey-5x5-v0": ["DoorKey"],
    "B1s_LavaGapS5-v0": ["Lava"],
    "P_none": list(FAMILIES),
    "P_heads": list(FAMILIES),
    "R_ewc": list(FAMILIES),
    "R_distill": list(FAMILIES),
}
SEEDS = range(1, 11)
EPISODES = 50


def job(args):
    group, seed, env_id = args
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    agent = Agent()
    if group != "untrained":
        agent.load_state_dict(torch.load(f"results/runs/{group}_seed{seed}.pt"))
    return group, seed, env_id, evaluate(agent, env_id, EPISODES, "cpu")


if __name__ == "__main__":
    jobs = [(g, s, e) for g, fams in GROUPS.items() for s in SEEDS for f in fams for e in FAMILIES[f]]
    with Pool(12) as pool:
        rows = pool.map(job, jobs)
    Path("results/generalisation.json").write_text(json.dumps(rows))

    envs = [e for f in FAMILIES.values() for e in f]
    short = lambda e: e.removeprefix("MiniGrid-").removesuffix("-v0")
    print("| group | " + " | ".join(short(e) for e in envs) + " |")
    print("|---" * (len(envs) + 1) + "|")
    for g in GROUPS:
        cells = []
        for e in envs:
            xs = [r for gg, s, ee, r in rows if gg == g and ee == e]
            if not xs:
                cells.append("")
                continue
            m, h = ci(xs)
            cells.append(f"{m:.0%} ± {h:.0%}")
        print(f"| {g} | " + " | ".join(cells) + " |")
