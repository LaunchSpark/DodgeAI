"""Summarise evaluation runs into the statistics the report needs (JSON + text)."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(sys.argv[1])
RNG = np.random.default_rng(0)
B = 4000
WINDOWS = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, 10**9)]
WNAMES = ["0-1k", "1k-2k", "2k-3k", "3k-5k", "5k+"]
CAUSES = ["normal", "kamikaze", "static_rect", "moving_rect"]


def load(name):
    p = RUNS / f"{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in p.open()]


def q1(x):
    x = np.sort(x)
    return x[: max(1, len(x) // 4)].mean()


def q4(x):
    x = np.sort(x)
    return x[-max(1, len(x) // 4):].mean()


def summary(games):
    f = np.array([g["frames"] for g in games], float)
    s = np.array([g["score"] for g in games], float)
    idx = RNG.integers(0, len(f), size=(B, len(f)))
    boot_mean = f[idx].mean(1)
    boot_q1 = np.array([q1(f[i]) for i in idx[:1000]])
    return {
        "n": len(f), "worst": f.min(), "q1": q1(f), "median": np.median(f),
        "mean": f.mean(), "q4": q4(f), "best": f.max(),
        "mean_ci": list(np.percentile(boot_mean, [2.5, 97.5])),
        "q1_ci": list(np.percentile(boot_q1, [2.5, 97.5])),
        "score_mean": s.mean(), "score_median": float(np.median(s)),
        "capped": int(sum(g["capped"] for g in games)),
        "survive": {m: float((f >= m).mean()) for m in range(0, 18001, 1000)},
    }


def paired(a, b, stat):
    """Bootstrap stat(a) - stat(b) resampling shared seeds."""
    da = {g["seed"]: g["frames"] for g in a}
    db = {g["seed"]: g["frames"] for g in b}
    seeds = sorted(set(da) & set(db))
    xa = np.array([da[s] for s in seeds], float)
    xb = np.array([db[s] for s in seeds], float)
    point = stat(xa) - stat(xb)
    idx = RNG.integers(0, len(seeds), size=(2000, len(seeds)))
    boots = np.array([stat(xa[i]) - stat(xb[i]) for i in idx])
    return {"n": len(seeds), "diff": point,
            "ci": list(np.percentile(boots, [2.5, 97.5])),
            "p_le0": float((boots <= 0).mean())}


def cause_class(c):
    return "kamikaze" if c and c.startswith("kamikaze") else c


def telemetry(games):
    out = {}
    died = [g for g in games if g["died"]]
    # cause by window
    bywin = []
    for (lo, hi), wn in zip(WINDOWS, WNAMES):
        w = [g for g in died if lo <= g["frames"] < hi]
        c = Counter(cause_class(g["cause"]) for g in w)
        bywin.append({"window": wn, "share_of_games": len(w) / len(games),
                      "n": len(w),
                      "causes": {k: c.get(k, 0) / max(1, len(w)) for k in CAUSES}})
    out["cause_by_window"] = bywin
    c = Counter(cause_class(g["cause"]) for g in died)
    out["cause_overall"] = {k: c.get(k, 0) / max(1, len(died)) for k in CAUSES}
    # pattern hazard
    total_frames = sum(g["frames"] for g in games)
    pat_frames = sum(g["pattern_frames"] for g in games)
    pat_deaths = sum(1 for g in died if g["pattern"] is not None)
    out["pattern_frame_share"] = pat_frames / total_frames
    out["pattern_death_share"] = pat_deaths / max(1, len(died))
    rate_on = pat_deaths / max(pat_frames, 1) * 1e4
    rate_off = (len(died) - pat_deaths) / max(total_frames - pat_frames, 1) * 1e4
    out["deaths_per_10k_pattern_on"] = rate_on
    out["deaths_per_10k_pattern_off"] = rate_off
    out["hazard_ratio"] = rate_on / max(rate_off, 1e-9)
    enemy = [g for g in died if cause_class(g["cause"]) in ("normal", "kamikaze")]
    out["enemy_deaths_during_pattern_share"] = (
        sum(1 for g in enemy if g["pattern"] is not None) / max(1, len(enemy)))
    # per-pattern lethality
    enc = Counter()
    kills = Counter()
    rect_kills = Counter()
    for g in games:
        for p in g["encounters"]:
            enc[p] += 1
        if g["died"] and g["pattern"] is not None:
            kills[g["pattern"]] += 1
            if g["cause"] in ("static_rect", "moving_rect"):
                rect_kills[g["pattern"]] += 1
    out["patterns"] = {int(p): {"encounters": enc[p], "deaths": kills[p],
                                "rect_deaths": rect_kills[p],
                                "lethality": kills[p] / enc[p]}
                       for p in enc}
    early = [g for g in died if g["frames"] < 1000 and g["pattern"] is not None]
    out["early_killers"] = Counter(g["pattern"] for g in early).most_common(5)
    out["early_deaths"] = sum(1 for g in died if g["frames"] < 1000)
    # anti-camping guard
    out["fyou_games_share"] = sum(1 for g in games if g["fyou_frames"]) / len(games)
    out["fyou_triggers"] = sum(len(g["fyou_frames"]) for g in games)
    out["fyou_killer"] = sum(1 for g in enemy if g["killer_fyou"])
    out["enemy_kills"] = len(enemy)
    out["death_within_300_of_fyou"] = sum(
        1 for g in died if any(0 <= g["frames"] - t <= 300 for t in g["fyou_frames"]))
    return out


def main():
    names = sorted(p.stem for p in RUNS.glob("*.jsonl"))
    result = {"summary": {}, "telemetry": {}, "contrasts": {}}
    for n in names:
        games = load(n)
        if not games:
            continue
        result["summary"][n] = summary(games)
        result["telemetry"][n] = telemetry(games)
    pairs = [
        ("latest_k1_sampled", "latest_k4_sampled"),
        ("latest_k1_argmax", "latest_k4_argmax"),
        ("latest_k1_argmax", "latest_k1_sampled"),
        ("latest_k4_argmax", "latest_k4_sampled"),
        ("latest_k1_argmax", "v2_k1_argmax"),
        ("latest_k1_argmax", "c100_k1_argmax"),
        ("v2_k1_argmax", "c100_k1_argmax"),
        ("latest_k1_sampled", "v2_k1_sampled"),
        ("latest_k1_sampled", "c100_k1_sampled"),
        ("latest_k1_argmax", "gradient_k1"),
    ]
    for a, b in pairs:
        ga, gb = load(a), load(b)
        if ga and gb:
            result["contrasts"][f"{a} - {b}"] = {
                "mean": paired(ga, gb, np.mean), "q1": paired(ga, gb, q1)}
    out = RUNS / "stats.json"
    out.write_text(json.dumps(result, indent=1, default=float))

    print(f"{'condition':22s} {'n':>4s} {'worst':>6s} {'Q1':>6s} {'median':>7s} "
          f"{'mean':>6s} {'Q4':>6s} {'best':>6s} {'>=1k':>5s} {'score':>6s}  mean 95% CI")
    for n, s in result["summary"].items():
        print(f"{n:22s} {s['n']:4d} {s['worst']:6.0f} {s['q1']:6.0f} {s['median']:7.0f} "
              f"{s['mean']:6.0f} {s['q4']:6.0f} {s['best']:6.0f} {s['survive'][1000]:5.1%} "
              f"{s['score_mean']:6.1f}  [{s['mean_ci'][0]:.0f}, {s['mean_ci'][1]:.0f}]")
    print()
    for k, v in result["contrasts"].items():
        print(f"{k:42s} mean {v['mean']['diff']:+7.0f} [{v['mean']['ci'][0]:+.0f}, "
              f"{v['mean']['ci'][1]:+.0f}]   Q1 {v['q1']['diff']:+6.0f} "
              f"[{v['q1']['ci'][0]:+.0f}, {v['q1']['ci'][1]:+.0f}]")
    print()
    for n, t in result["telemetry"].items():
        print(f"{n}: pattern frames {t['pattern_frame_share']:.0%}, deaths in pattern "
              f"{t['pattern_death_share']:.0%}, hazard x{t['hazard_ratio']:.1f}, "
              f"causes {', '.join(f'{k} {v:.0%}' for k, v in t['cause_overall'].items())}, "
              f"fyou games {t['fyou_games_share']:.1%}, fyou killers "
              f"{t['fyou_killer']}/{t['enemy_kills']}")


if __name__ == "__main__":
    main()
