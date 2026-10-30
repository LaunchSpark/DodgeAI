"""Fig. 9 counterpart: last 60 frames before death inside the deadliest static patterns."""
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PATHS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
PANELS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
MAX_PATHS = 45
RECT_C, ENEMY_C = "#0072B2", "#E69F00"

plt.rcParams.update({"font.family": "serif", "font.size": 7,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

games = [json.loads(line) for line in PATHS.open()]
mismatched = sum(not g["replay_match"] for g in games)
by_pattern = Counter(g["pattern"] for g in games)
top = [p for p, _ in by_pattern.most_common(PANELS)]


def layout(g):
    """The lethal rectangles at the death frame, rounded so layouts compare."""
    return tuple(sorted((round(r["x"]), round(r["y"]), round(r["w"]), round(r["h"]))
                        for r in g["rects"] if r["sh"] >= 2))


fig, axes = plt.subplots(1, len(top), figsize=(7.16, 7.16 / len(top) + 0.25))
for ax, pat in zip(axes, top):
    members = [g for g in games if g["pattern"] == pat]
    # Static rectangles move within a pattern, so no one layout is
    # representative: shade each pixel by the share of these deaths at which
    # it lay inside a lethal rectangle.
    cover = np.zeros((128, 128))
    for g in members:
        mask = np.zeros((128, 128), bool)
        for x, y, w, h in layout(g):
            mask[max(0, y):max(0, y + h), max(0, x):max(0, x + w)] = True
        cover += mask
    cover /= max(1, len(members))
    ax.imshow(cover, cmap="Blues", vmin=0, vmax=1.6, extent=(0, 128, 128, 0),
              interpolation="nearest", alpha=0.9)
    # The first MAX_PATHS by seed, an unselected sample; rect deaths are drawn
    # last so they are not buried under enemy deaths.
    shown = sorted(members, key=lambda g: g["seed"])[:MAX_PATHS]
    for g in sorted(shown, key=lambda g: g["cause"] == "static_rect"):
        color = RECT_C if g["cause"] == "static_rect" else ENEMY_C
        xs = [p[0] for p in g["path"]]
        ys = [p[1] for p in g["path"]]
        ax.plot(xs, ys, color=color, lw=0.6, alpha=0.75)
        ax.plot(xs[-1], ys[-1], "o", color=color, ms=1.6)
    rect_n = sum(g["cause"] == "static_rect" for g in members)
    ax.set_title(f"pattern {pat}\n{len(members)} deaths, {rect_n} to a rectangle",
                 fontsize=6.5)
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_linewidth(0.8)
fig.savefig(OUT)
print(f"wrote {OUT}; patterns {top}; deaths per panel "
      f"{[by_pattern[p] for p in top]}; replay mismatches {mismatched}")
