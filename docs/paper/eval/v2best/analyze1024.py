"""Statistics for the paper refresh, from the eval1024* runs (dashboard records).

Same estimators as docs/paper/eval/analyze.py: Q1/Q4 are quarter means, single-
condition intervals bootstrap 4,000x (mean) and 1,000x (Q1), paired differences
bootstrap 2,000x by seed. Games are uncapped, as on the dashboard; capped-at-18k
figures are given alongside for comparison with the earlier draft and Turner.

    python analyze1024.py  ->  paper1024/stats.json
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent / "runs"
OUT = HERE / "paper1024"
OUT.mkdir(exist_ok=True)
CAP = 18_000
RUNS = {"k1_argmax": "eval1024", "k1_sampled": "eval1024_k1_sampled", "k4_argmax": "eval1024_k4_argmax",
        "k4_sampled": "eval1024_k4_sampled", "c100": "eval1024_can-reach-100",
        "random": "eval1024_random", "gradient": "eval1024_gradient"}
rng = np.random.default_rng(0)


def q1(x):
    x = np.sort(x)
    return x[: max(1, len(x) // 4)].mean()


def q4(x):
    x = np.sort(x)
    return x[-max(1, len(x) // 4):].mean()


def summary(f):
    n = len(f)
    idx = rng.integers(0, n, (4000, n))
    bm = f[idx].mean(1)
    bq = np.array([q1(f[i]) for i in idx[:1000]])
    c = np.minimum(f, CAP)
    return {"n": n, "worst": f.min(), "q1": q1(f), "median": float(np.median(f)), "mean": f.mean(),
            "q4": q4(f), "best": f.max(), "mean_ci": np.percentile(bm, [2.5, 97.5]).tolist(),
            "q1_ci": np.percentile(bq, [2.5, 97.5]).tolist(), "capped_mean": c.mean(), "capped_q1": q1(c),
            "over_cap": int((f >= CAP).sum()),
            "past": {m: float((f > m).mean()) for m in (1000, 2000, 4000, 8000, 18000)}}


def paired(a, b, stat):
    n = len(a)
    idx = rng.integers(0, n, (2000, n))
    boots = np.array([stat(a[i]) - stat(b[i]) for i in idx])
    return {"diff": stat(a) - stat(b), "ci": np.percentile(boots, [2.5, 97.5]).tolist(),
            "better": int((a > b).sum()), "worse": int((a < b).sum())}


def telemetry(games, figures):
    died = [g for g in games if g["died"]]
    cause = lambda g: "kamikaze" if g["cause"] in ("kamikaze", "kamikaze_blast") else g["cause"]
    causes = {c: sum(cause(g) == c for g in died) for c in ("normal", "kamikaze", "static_rect", "moving_rect")}
    in_pattern = [g for g in died if g["pattern"] not in (-1, None)]
    enemy = [g for g in died if cause(g) in ("normal", "kamikaze")]
    enemy_in_pattern = [g for g in enemy if g["pattern"] not in (-1, None)]
    frames = sum(g["frames"] for g in games)
    pframes = sum(g["pattern_frames"] for g in games)
    return {"deaths": len(died), "causes": causes,
            "cause_shares": {c: v / max(1, len(died)) for c, v in causes.items()},
            "active_frame_share": pframes / frames, "deaths_in_pattern": len(in_pattern),
            "death_share_in_pattern": len(in_pattern) / max(1, len(died)),
            "enemy_deaths": len(enemy), "enemy_in_pattern": len(enemy_in_pattern),
            "rect_deaths_in_pattern": sum(cause(g) in ("static_rect", "moving_rect") for g in in_pattern),
            "hazard": figures["hazard"]["overall"], "anti_camping": figures["anti_camping"],
            "fyou_games": sum(g["fyou_triggers"] > 0 for g in games),
            "by_window": figures["causes"], "hazard_windows": figures["hazard"]["windows"],
            "patterns": figures["patterns"], "survival": figures["survival"]}


data, stats = {}, {"summary": {}, "telemetry": {}, "paired": {}}
for key, folder in RUNS.items():
    path = HERE / folder / "games.json"
    if not path.exists():
        print("missing", key)
        continue
    games = json.loads(path.read_text())
    figures = json.loads((HERE / folder / "figures.json").read_text())
    data[key] = (games, np.array([g["frames"] for g in games], float))
    stats["summary"][key] = summary(data[key][1])
    stats["telemetry"][key] = telemetry(games, figures)
    s = stats["summary"][key]
    print(f"{key:>11}: n={s['n']} mean {s['mean']:,.0f} [{s['mean_ci'][0]:,.0f}-{s['mean_ci'][1]:,.0f}] "
          f"Q1 {s['q1']:,.0f} median {s['median']:,.0f} Q4 {s['q4']:,.0f} best {s['best']:,.0f} "
          f"capped mean {s['capped_mean']:,.0f} over cap {s['over_cap']}")

pairs = {"k4_minus_k1_argmax": ("k4_argmax", "k1_argmax"), "k4_minus_k1_sampled": ("k4_sampled", "k1_sampled"),
         "argmax_minus_sampled_k1": ("k1_argmax", "k1_sampled"), "argmax_minus_sampled_k4": ("k4_argmax", "k4_sampled"),
         "best_minus_c100": ("k1_argmax", "c100")}
for name, (a, b) in pairs.items():
    if a in data and b in data:
        fa, fb = data[a][1], data[b][1]
        assert [g["seed"] for g in data[a][0]] == [g["seed"] for g in data[b][0]]
        stats["paired"][name] = {"mean": paired(fa, fb, np.mean), "q1": paired(fa, fb, q1),
                                 "capped_mean": paired(np.minimum(fa, CAP), np.minimum(fb, CAP), np.mean)}
        m, q = stats["paired"][name]["mean"], stats["paired"][name]["q1"]
        print(f"{name}: mean {m['diff']:+,.0f} [{m['ci'][0]:+,.0f}, {m['ci'][1]:+,.0f}]  "
              f"Q1 {q['diff']:+,.0f} [{q['ci'][0]:+,.0f}, {q['ci'][1]:+,.0f}]")
(OUT / "stats.json").write_text(json.dumps(stats, default=float, indent=1))
print("wrote", OUT / "stats.json")
