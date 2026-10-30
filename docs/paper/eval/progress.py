"""Progress bars and running stats for the evaluation batch."""
import json
import sys
from pathlib import Path

import numpy as np

RUNS = Path(sys.argv[1])
PLAN = [  # (file, label, target games)
    ("latest_k1_argmax", "v2-latest  1f argmax ", 512),
    ("latest_k1_sampled", "v2-latest  1f sampled", 512),
    ("latest_k4_argmax", "v2-latest  4f argmax ", 512),
    ("latest_k4_sampled", "v2-latest  4f sampled", 512),
    ("c100_k1_argmax", "can-reach  1f argmax ", 256),
    ("random_k1", "random     1f        ", 256),
    ("gradient_k1", "gradient   1f        ", 128),
]
WIDTH = 24


def q(x, top):
    x = np.sort(x)
    k = max(1, len(x) // 4)
    return (x[-k:] if top else x[:k]).mean()


done_all = target_all = 0
print(f"{'run':22s} {'progress':{WIDTH + 2}s} {'games':>9s}   "
      f"{'Q1':>5s} {'median':>6s} {'mean':>5s} {'Q4':>5s} {'best':>5s} {'>=1k':>5s}")
for name, label, target in PLAN:
    p = RUNS / f"{name}.jsonl"
    games = [json.loads(line) for line in p.open()] if p.exists() else []
    n = len(games)
    done_all += n
    target_all += target
    filled = round(WIDTH * n / target)
    bar = "#" * filled + "." * (WIDTH - filled)
    line = f"{label:22s} [{bar}] {n:4d}/{target:<4d}"
    if n:
        f = np.array([g["frames"] for g in games], float)
        tag = "final" if n >= target else "partial, biased low"
        line += (f"   {q(f, False):5.0f} {np.median(f):6.0f} {f.mean():5.0f} "
                 f"{q(f, True):5.0f} {f.max():5.0f} {(f >= 1000).mean():5.0%}  ({tag})")
    print(line)
filled = round(WIDTH * done_all / target_all)
print(f"\n{'all runs':22s} [{'#' * filled}{'.' * (WIDTH - filled)}] "
      f"{done_all}/{target_all} games ({done_all / target_all:.0%})")
