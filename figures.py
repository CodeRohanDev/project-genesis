"""Paper figures from results/runs/*.json -> paper/figs/*.pdf (+ .png previews). Mean ± 95% CI over seeds.

    python figures.py
"""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize import ci

OUT = Path("paper/figs")
# validated categorical palette (dataviz skill, light mode); slots 3-4 are low-contrast on white, so every series
# also has its own marker / line style and a legend (secondary encoding for print and colour-blind readers)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e4e4e0"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 5, "pdf.fonttype": 42})


def runs(prefix):
    return [json.load(open(f)) for f in sorted(glob.glob(f"results/runs/{prefix}_seed*.json"))]


def mci(xs):
    m, h = ci(xs)
    return 100 * m, 100 * h


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_combined():
    """Experiments 7-8: final success after Empty -> DoorKey -> Lava, training sizes vs unseen sizes."""
    methods = [("C_none", "No protection", MUTED, ""), ("C_heads", "Head reset", YELLOW, "//"),
               ("C_distill", "Told + distill", AQUA, ".."), ("C_clear", "CLEAR (replay all)", MAGENTA, "xx"),
               ("O_full", "Blind (ours)", BLUE, ""), ("C_joint", "Joint (all at once)", ORANGE, "\\\\")]
    rs0 = runs("C_none")[0]
    evals = rs0["eval_tasks"]
    unseen = [i for i, e in enumerate(evals) if "size=" in e]
    seen = [i for i in range(len(evals)) if i not in unseen]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(2)
    w = 0.135
    for k, (p, label, color, hatch) in enumerate(methods):
        rs = runs(p)
        vals = [mci([np.mean([r["matrix"][-1][i] for i in idx]) for r in rs]) for idx in (seen, unseen)]
        ax.bar(x + (k - 2.5) * w, [v[0] for v in vals], w * 0.9, yerr=[v[1] for v in vals], color=color, label=label,
               hatch=hatch, edgecolor="white", linewidth=0, error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 1.5})
    ax.set_xticks(x, ["Training sizes", "Unseen sizes"])
    ax.set_ylabel("Success after all tasks (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.3))
    save(fig, "fig_combined")


def fig_long():
    """Experiment 9: mean success over all task families learned so far, after each of 8 stages."""
    fams = {"E": [0, 1], "D": [2, 3], "L": [4, 5], "O": [6, 7], "X": [8, 9], "U": [10]}
    order = ["E", "D", "L", "O", "X", "U", "D", "L"]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    for p, label, color, marker, ls in [("L_none", "No protection", MUTED, "o", ":"),
                                        ("L_clear", "CLEAR (replay all)", MAGENTA, "P", (0, (3, 1, 1, 1))),
                                        ("L_distill", "Told + distill", AQUA, "s", "--"),
                                        ("L_blind", "Blind (ours)", BLUE, "D", "-")]:
        rs = runs(p)
        ys, hs = [], []
        for st in range(8):
            learned = sorted({i for f in order[:st + 1] for i in fams[f]})
            m, h = mci([np.mean([r["matrix"][st][i] for i in learned]) for r in rs])
            ys.append(m); hs.append(h)
        xs = np.arange(1, 9)
        ax.plot(xs, ys, color=color, marker=marker, linestyle=ls, label=label)
        ax.fill_between(xs, np.array(ys) - hs, np.array(ys) + hs, color=color, alpha=0.15, linewidth=0)
    joint = runs("L_joint")
    jm, jh = mci([np.mean(r["matrix"][-1][:11]) for r in joint])
    ax.axhline(jm, color=ORANGE, linewidth=1.2, linestyle="-.", label=f"Joint, end ({jm:.0f}%)")
    ax.set_xticks(range(1, 9), ["Empty", "DoorKey", "Lava", "Obst.", "Cross.", "Unlock", "DoorKey", "Lava"],
                  rotation=40, ha="right")
    ax.set_ylabel("Mean success, tasks\nlearned so far (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.34))
    save(fig, "fig_long_sequence")


def fig_plasticity():
    """Experiment 3: policy entropy when a new task starts vs whether it can be learned (Lava, 3rd task)."""
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for p, label, color, marker in [("P_none", "No reset", MUTED, "o"), ("P_adam", "Optimizer reset", YELLOW, "v"),
                                    ("P_shrink", "Shrink & perturb", AQUA, "s"), ("P_heads", "Head reset", BLUE, "D")]:
        rs = runs(p)
        ent = [r["train"][2]["entropy"][0] for r in rs]
        lava = [100 * r["matrix"][2][2] for r in rs]
        ax.scatter(ent, lava, color=color, marker=marker, s=22, label=label, edgecolors="white", linewidths=0.6)
    ax.set_xlabel("Policy entropy at start of new task (nats; max 1.95)")
    ax.set_ylabel("New task success (%)")
    ax.set_ylim(-3, 105)
    ax.legend(fontsize=6.5, loc="center right")
    save(fig, "fig_plasticity")


def fig_generalisation():
    """Experiment 6: success vs grid size; shaded = sizes never seen by any agent."""
    sizes = [5, 6, 7, 8, 10, 12]
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    for x0 in (7, 10, 12):
        ax.axvspan(x0 - 0.4, x0 + 0.4, color=GRID, alpha=0.7, linewidth=0)
    for p, label, color, marker, ls in [("G_cnn_5", "Train 5x5", MUTED, "o", ":"),
                                        ("G_lstm_5", "Train 5x5 + memory", YELLOW, "v", ":"),
                                        ("G_lstm_mix", "Mixed sizes + memory", AQUA, "s", "--"),
                                        ("G_cnn_mix", "Mixed sizes", BLUE, "D", "-")]:
        rs = runs(p)
        ys = [mci([r["matrix"][0][j] for r in rs]) for j in range(len(sizes))]
        ax.errorbar(sizes, [y[0] for y in ys], yerr=[y[1] for y in ys], color=color, marker=marker, linestyle=ls,
                    label=label, capsize=1.5, elinewidth=0.8)
    ax.set_xlabel("DoorKey grid size (shaded: never trained on)")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 105)
    ax.set_xticks(sizes)
    ax.legend(ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.34))
    save(fig, "fig_generalisation")


def fig_minatar():
    """Experiment 20: MinAtar, score per game after all 5 games, divided by the 2M-step scratch score."""
    games = ["breakout", "space_invaders", "freeway", "asterix", "seaquest"]
    scratch = np.array([np.mean([r["matrix"][-1][i] for r in runs(f"MAS_{g}")]) for i, g in enumerate(games)])
    methods = [("MA_none", "No protection", MUTED, ""), ("MA_clear", "CLEAR (replay all)", MAGENTA, "xx"),
               ("MA_blind", "Blind, heads reset", BLUE, ""), ("MA_joint", "Joint (all at once)", ORANGE, "\\\\"),
               ("MA_blind_all", "Blind, full reset (ours)", AQUA, "//")]
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    x = np.arange(6)
    w = 0.16
    for k, (p, label, color, hatch) in enumerate(methods):
        fin = np.array([r["matrix"][-1] for r in runs(p)]) / scratch
        vals = [mci(fin[:, i]) for i in range(5)] + [mci(fin.mean(1))]
        ax.bar(x + (k - 2) * w, [v[0] for v in vals], w * 0.9, yerr=[v[1] for v in vals], color=color, label=label,
               hatch=hatch, edgecolor="white", linewidth=0, error_kw={"elinewidth": 0.8, "ecolor": INK, "capsize": 1.5})
    ax.set_xticks(x, ["Breakout", "Space Inv.", "Freeway", "Asterix", "Seaquest", "Mean"], rotation=30, ha="right")
    ax.set_ylabel("Score after all games\n(% of scratch)")
    ax.legend(ncol=2, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.34))
    save(fig, "fig_minatar")


if __name__ == "__main__":
    for f in (fig_combined, fig_long, fig_plasticity, fig_generalisation, fig_minatar):
        f()
        print("ok", f.__name__)
