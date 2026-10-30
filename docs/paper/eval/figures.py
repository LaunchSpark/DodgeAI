"""Render the report's figures from runs/stats.json and the raw JSONL."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((RUNS / "stats.json").read_text())
SUM, TEL = S["summary"], S["telemetry"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
Q1C, MEANC, Q4C = "#D55E00", "#E69F00", "#CC79A7"
K1C, K4C, ARGC = "#009E73", "#444444", "#0072B2"
CAUSE_COLORS = {"normal": "#0072B2", "kamikaze": "#CC79A7",
                "static_rect": "#56B4E9", "moving_rect": "#E69F00"}
CAUSE_LABELS = {"normal": "normal enemy", "kamikaze": "kamikaze (body + blast)",
                "static_rect": "static pattern rectangle",
                "moving_rect": "moving pattern rectangle"}
ONE_COL = 3.5


def frames(name):
    return np.array([json.loads(line)["frames"]
                     for line in (RUNS / f"{name}.jsonl").open()], float)


def bar_groups(conds, labels, path, ylabel):
    fig, ax = plt.subplots(figsize=(ONE_COL, 2.2))
    x = np.arange(len(conds))
    w = 0.26
    for j, (key, color, lab) in enumerate(
            [("q1", Q1C, "Q1"), ("mean", MEANC, "mean"), ("q4", Q4C, "Q4")]):
        vals = [SUM[c][key] for c in conds]
        bars = ax.bar(x + (j - 1) * w, vals, w, color=color, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=5.5, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(SUM[c]["q4"] for c in conds) * 1.28)
    ax.legend(ncol=3, loc="upper left", frameon=False)
    fig.savefig(OUT / path)
    plt.close(fig)


# Fig: decision interval x action mode for the latest checkpoint
grid = ["latest_k4_sampled", "latest_k1_sampled", "latest_k4_argmax", "latest_k1_argmax"]
if all(c in SUM for c in grid):
    bar_groups(grid, ["4-frame\nsampled", "1-frame\nsampled", "4-frame\nargmax",
                      "1-frame\nargmax"], "fig_k_grid.pdf",
               "alive frames (512 seeds)")

# Fig: checkpoints compared at the training configuration
ckpts = ["c100_k1_argmax", "v2_k1_argmax", "latest_k1_argmax"]
if all(c in SUM for c in ckpts):
    bar_groups(ckpts, ["can-reach-100\n(5,691 upd.)", "v2\n(8,855 upd.)",
                       "v2-latest\n(21,051 upd.)"], "fig_ckpts.pdf",
               "alive frames, argmax, $K$=1")

# Fig: survival curves
curves = [("latest_k1_argmax", "v2-latest, 1-frame, argmax", K1C, "-", "D"),
          ("latest_k1_sampled", "v2-latest, 1-frame, sampled", MEANC, "-", "s"),
          ("latest_k4_argmax", "v2-latest, 4-frame, argmax", ARGC, "-", "^"),
          ("latest_k4_sampled", "v2-latest, 4-frame, sampled", K4C, "-", "o"),
          ("gradient_k1", "gradient-field heuristic", "#999999", "--", "x"),
          ("random_k1", "uniform random", "#BBBBBB", ":", "")]
fig, ax = plt.subplots(figsize=(ONE_COL, 2.3))
top = 0
for name, label, color, ls, mk in curves:
    if name not in SUM:
        continue
    f = frames(name)
    m = np.arange(0, f.max() + 1000, 1000)
    alive = [(f >= t).mean() * 100 for t in m]
    top = max(top, m[-1])
    ax.plot(m / 1000, alive, ls, color=color, marker=mk, ms=3, lw=1.1,
            label=f"{label} (mean {f.mean():,.0f})")
ax.axhline(50, color="#999999", lw=0.5, ls=":")
ax.set_xlabel("gameplay frames survived (thousands)")
ax.set_ylabel("games still alive (%)")
ax.set_ylim(0, 102)
ax.set_xlim(0, top / 1000)
ax.legend(frameon=False, fontsize=5.8)
fig.savefig(OUT / "fig_survival.pdf")
plt.close(fig)

# Fig: killer class by window, K=4 vs K=1 (argmax)
pairs = [("latest_k4_argmax", "4f"), ("latest_k1_argmax", "1f")]
if all(p in TEL for p, _ in pairs):
    fig, ax = plt.subplots(figsize=(7.16, 2.2))
    wnames = [w["window"] for w in TEL[pairs[0][0]]["cause_by_window"]]
    xs, ticks = [], []
    for i, wn in enumerate(wnames):
        for j, (name, tag) in enumerate(pairs):
            x = i * 3 + j
            win = TEL[name]["cause_by_window"][i]
            bottom = 0
            for cause in CAUSE_COLORS:
                h = win["causes"][cause] * 100
                ax.bar(x, h, 0.8, bottom=bottom, color=CAUSE_COLORS[cause],
                       label=CAUSE_LABELS[cause] if i == 0 and j == 0 else None)
                if h >= 8:
                    ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center",
                            va="center", fontsize=5.5, color="white")
                bottom += h
            ax.text(x, 103, f"{win['share_of_games']:.0%}", ha="center", fontsize=5.5)
            xs.append(x)
            ticks.append(tag)
    ax.set_xticks(xs, ticks, fontsize=6)
    for i, wn in enumerate(wnames):
        ax.text(i * 3 + 0.5, -21, wn, ha="center", fontsize=7)
    ax.set_ylim(0, 110)
    ax.set_ylabel("deaths in window (%)")
    ax.text(len(wnames) * 1.5 - 0.5, -32, "gameplay frame window in which the game ended",
            ha="center", fontsize=7)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.36), frameon=False)
    fig.savefig(OUT / "fig_causes.pdf")
    plt.close(fig)

# Fig: per-pattern lethality (K=1 argmax, patterns met >= 30 times)
name = "latest_k1_argmax"
if name in TEL:
    pats = TEL[name]["patterns"]
    pats4 = TEL.get("latest_k4_argmax", {}).get("patterns", {})
    rows = [(int(p), v) for p, v in pats.items() if v["encounters"] >= 30]
    rows.sort(key=lambda r: -r[1]["lethality"])
    fig, ax = plt.subplots(figsize=(7.16, 2.0))
    for i, (p, v) in enumerate(rows):
        n, k = v["encounters"], v["deaths"]
        ph = k / n
        # Wilson interval
        z = 1.96
        c = (ph + z * z / (2 * n)) / (1 + z * z / n)
        hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        ax.errorbar(i, ph * 100, yerr=[[max(0, ph - (c - hw)) * 100],
                                       [max(0, (c + hw) - ph) * 100]],
                    fmt="o", ms=3, color="#0072B2", lw=0.8)
        q = pats4.get(str(p))
        if q:
            ax.plot(i, q["lethality"] * 100, "D", ms=3, mfc="none", color="#444444")
    ax.set_xticks(range(len(rows)), [str(p) for p, _ in rows], fontsize=6)
    ax.set_xlabel("pattern id (met at least 30 times at 1-frame), sorted by 1-frame lethality")
    ax.set_ylabel("encounters ending\nin death (%)")
    ax.plot([], [], "o", color="#0072B2", ms=3, label="1-frame (95% Wilson CI)")
    ax.plot([], [], "D", mfc="none", color="#444444", ms=3, label="4-frame")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 100)
    fig.savefig(OUT / "fig_patterns.pdf")
    plt.close(fig)

# Fig: deaths per 10k frames, pattern on vs off, per window
fig, ax = plt.subplots(figsize=(ONE_COL, 2.1))
WINDOWS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 10**9)]
for name, color, tag in [("latest_k1_argmax", K1C, "1f"), ("latest_k4_argmax", K4C, "4f")]:
    if name not in SUM:
        continue
    games = [json.loads(line) for line in (RUNS / f"{name}.jsonl").open()]
    on_rates, off_rates = [], []
    for lo, hi in WINDOWS:
        on_f = off_f = on_d = off_d = 0
        for g in games:
            # pattern exposure is not tracked per window; apportion a game's
            # pattern share uniformly across the frames it lived in the window
            lived = max(0, min(g["frames"], hi) - lo)
            if lived <= 0:
                continue
            share = g["pattern_frames"] / max(1, g["frames"])
            on_f += lived * share
            off_f += lived * (1 - share)
            if g["died"] and lo <= g["frames"] < hi:
                if g["pattern"] is not None:
                    on_d += 1
                else:
                    off_d += 1
        on_rates.append(on_d / max(on_f, 1) * 1e4)
        off_rates.append(off_d / max(off_f, 1) * 1e4)
    x = np.arange(len(WINDOWS))
    ax.semilogy(x, on_rates, "-s", color=color, ms=3, label=f"{tag}, pattern active")
    ax.semilogy(x, [max(r, 1e-2) for r in off_rates], "--o", color=color, ms=3,
                mfc="white", label=f"{tag}, pattern inactive")
ax.set_xticks(range(5), ["0-1k", "1k-2k", "2k-3k", "3k-5k", "5k+"])
ax.set_xlabel("gameplay frame window")
ax.set_ylabel("deaths per 10,000 frames")
ax.legend(frameon=False, fontsize=5.8)
fig.savefig(OUT / "fig_hazard.pdf")
plt.close(fig)
print("figures written to", OUT)
