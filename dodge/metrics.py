"""The evaluation figures from the paper, computed from finished games.

Every function takes a list of game records, the dicts DodgeVecEnv reports as
info["run"] and play_games returns: frames, died, cause, pattern,
pattern_frames, encounters, fyou_triggers. All of them are pure, so the
dashboard can recompute them on whatever window of games it holds.
"""

import math

import numpy as np

# Where in a game it ended, the paper's windows.
WINDOWS = ((0, 1000), (1000, 2000), (2000, 3000), (3000, 5000), (5000, None))
CAUSES = ("normal", "kamikaze", "static_rect", "moving_rect")
# Explosions are kamikaze deaths; the paper reports them together.
CAUSE_GROUP = {"kamikaze_blast": "kamikaze"}


def window_label(low, high):
    def k(value):
        return f"{value // 1000}k" if value else "0"
    return f"{k(low)}-{k(high)}" if high is not None else f"{k(low)}+"


def quarter_means(frames):
    """Mean of the worst and best quarter of games (not the 25th percentile)."""
    ordered = np.sort(np.asarray(frames, dtype=float))
    k = max(1, len(ordered) // 4)
    return float(ordered[:k].mean()), float(ordered[-k:].mean())


def summary(games):
    """Worst, Q1, median, mean, Q4, best and the share past 1,000 frames."""
    frames = np.array([game["frames"] for game in games], dtype=float)
    if not len(frames):
        return None
    q1, q4 = quarter_means(frames)
    return {
        "games": len(frames),
        "worst": float(frames.min()),
        "q1": q1,
        "median": float(np.median(frames)),
        "mean": float(frames.mean()),
        "q4": q4,
        "best": float(frames.max()),
        "past_1000": float((frames >= 1000).mean()),
    }


SURVIVAL_STEPS = (25, 50, 100, 250, 500, 1000)


def survival_curve(games, step=None):
    """Share of games still alive at each multiple of `step` frames.

    The paper's step is 1,000 frames; by default the step shrinks for short
    games so the curve keeps at least ten points.
    """
    frames = np.array([game["frames"] for game in games], dtype=float)
    if not len(frames):
        return []
    if step is None:
        step = next((s for s in reversed(SURVIVAL_STEPS) if frames.max() / s >= 10),
                    SURVIVAL_STEPS[0])
    top = int(frames.max()) + step
    return [{"frames": t, "alive": float((frames >= t).mean())}
            for t in range(0, top, step)]


def _in_window(frames, low, high):
    return frames >= low and (high is None or frames < high)


def causes_by_window(games):
    """For each window: its share of all deaths, and what killed those games."""
    deaths = [game for game in games if game.get("died")]
    rows = []
    for low, high in WINDOWS:
        ended = [game for game in deaths if _in_window(game["frames"], low, high)]
        causes = {cause: 0 for cause in CAUSES}
        for game in ended:
            cause = CAUSE_GROUP.get(game.get("cause"), game.get("cause"))
            if cause in causes:
                causes[cause] += 1
        rows.append({
            "window": window_label(low, high),
            "games": len(ended),
            "share_of_games": len(ended) / len(deaths) if deaths else 0.0,
            "causes": {cause: count / len(ended) if ended else 0.0
                       for cause, count in causes.items()},
        })
    return rows


def pattern_hazard(games):
    """Deaths per 10,000 frames lived with a pattern active, and without.

    Overall, and per window. A game's pattern exposure is only known in total,
    so within a window it is apportioned uniformly over the frames the game
    lived there.
    """
    def rates(rows):
        on_frames = sum(row[0] for row in rows)
        off_frames = sum(row[1] for row in rows)
        on_deaths = sum(row[2] for row in rows)
        off_deaths = sum(row[3] for row in rows)
        return {
            "active": on_deaths / on_frames * 1e4 if on_frames else None,
            "inactive": off_deaths / off_frames * 1e4 if off_frames else None,
            "active_frame_share": on_frames / (on_frames + off_frames)
            if on_frames + off_frames else 0.0,
        }

    overall = []
    windows = {window_label(low, high): [] for low, high in WINDOWS}
    for game in games:
        frames = max(game["frames"], 1)
        share = game.get("pattern_frames", 0) / frames
        died = bool(game.get("died"))
        during = died and game.get("pattern", -1) >= 0
        overall.append((frames * share, frames * (1 - share),
                        int(during), int(died and not during)))
        for low, high in WINDOWS:
            lived = max(0, min(game["frames"], high or game["frames"]) - low)
            if lived <= 0:
                continue
            ended = died and _in_window(game["frames"], low, high)
            windows[window_label(low, high)].append(
                (lived * share, lived * (1 - share),
                 int(ended and during), int(ended and not during)))
    return {
        "overall": rates(overall),
        "windows": [{"window": label, **rates(rows)} for label, rows in windows.items()],
    }


def wilson(successes, trials, z=1.96):
    """95% Wilson interval for a proportion."""
    if not trials:
        return (0.0, 0.0)
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = (z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
            / (1 + z * z / trials))
    return (max(0.0, centre - half), min(1.0, centre + half))


def pattern_lethality(games, min_encounters=10):
    """Share of encounters with each pattern that ended in death, deadliest first."""
    encounters, deaths = {}, {}
    for game in games:
        for pattern, count in game.get("encounters", {}).items():
            pattern = int(pattern)
            encounters[pattern] = encounters.get(pattern, 0) + count
        if game.get("died") and game.get("pattern", -1) >= 0:
            pattern = int(game["pattern"])
            deaths[pattern] = deaths.get(pattern, 0) + 1
    rows = []
    for pattern, met in encounters.items():
        if met < min_encounters:
            continue
        killed = min(deaths.get(pattern, 0), met)
        low, high = wilson(killed, met)
        rows.append({"pattern": pattern, "encounters": met, "deaths": killed,
                     "lethality": killed / met, "low": low, "high": high})
    rows.sort(key=lambda row: -row["lethality"])
    return rows


def anti_camping(games):
    """How often the anti-camping wave fires, and in how many games."""
    if not games:
        return None
    triggered = [game for game in games if game.get("fyou_triggers", 0) > 0]
    return {
        "games_triggered": len(triggered) / len(games),
        "triggers_per_game": float(np.mean([game.get("fyou_triggers", 0) for game in games])),
    }


def evaluation(games):
    """Every figure at once, as the dashboard sends it."""
    return {
        "summary": summary(games),
        "survival": survival_curve(games),
        "causes": causes_by_window(games),
        "hazard": pattern_hazard(games),
        "patterns": pattern_lethality(games),
        "anti_camping": anti_camping(games),
    }
