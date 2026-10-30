"""Record how games end inside each pattern, for the death-path viewer.

A frozen policy plays a block of fixed seeds. Every lane keeps its last
TAIL frames in a ring; when a game dies while a pattern is active, that tail
becomes a run: the player's path, the pattern's state and rectangles, how many
enemies were on screen, the gap to the nearest one, and the killer's path.

The engine removes the enemy that kills the player, so the killer is found in
the frame before death as the enemy closest to the player, and traced back by
matching each frame's enemies to the next (enemies move under 4px a frame).
"""

import math
from collections import deque

import numpy as np

from .env import DEATH_CAUSES, make_engine

TAIL = 120
# Encounters before a pattern's death rate is ranked.
MIN_RANKED = 5
# How far an enemy may move between frames and still be the same enemy.
MATCH_PIXELS = 6.0
KILLER_KINDS = {"normal": 0, "kamikaze": 1, "kamikaze_blast": -1}


def _pattern_state(pattern):
    """0 no pattern, 1 charging (not lethal yet), 2 lethal."""
    if pattern is None:
        return 0
    lethal = pattern["tpe"] == 1 or any(rect["sh"] == 2 for rect in pattern["rects"])
    return 2 if lethal else 1


def _gap(state):
    """Pixels between the player's box and the nearest dangerous enemy's."""
    size = state["size"]
    left, top = state["px"] - size, state["py"] - size
    right, bottom = state["px"] + size, state["py"] + size
    best = None
    for enemy in state["enemies"]:
        if enemy["p"] >= 2:
            continue
        dx = max(enemy["x"] - right, left - (enemy["x"] + enemy["s"]), 0.0)
        dy = max(enemy["y"] - bottom, top - (enemy["y"] + enemy["s"]), 0.0)
        gap = math.hypot(dx, dy)
        best = gap if best is None else min(best, gap)
    return None if best is None else round(best, 1)


def _frame(state):
    pattern = state["cp"]
    return {
        "x": state["px"], "y": state["py"], "size": state["size"],
        "st": _pattern_state(pattern),
        "pid": pattern["id"] if pattern is not None else -1,
        "tpe": pattern["tpe"] if pattern is not None else None,
        "rects": ([[r["x"], r["y"], r["w"], r["h"],
                    int(pattern["tpe"] == 1 or r["sh"] == 2)] for r in pattern["rects"]]
                  if pattern is not None else []),
        "enemies": [(e["x"], e["y"], e["s"], e["p"]) for e in state["enemies"]],
        "gap": _gap(state),
    }


def _killer_path(frames, kind):
    """Trace the killing enemy back from the frame before death."""
    personality = KILLER_KINDS.get(kind)
    if personality is None or len(frames) < 2:
        return None, [None] * len(frames)
    last = frames[-2]
    candidates = [e for e in last["enemies"] if e[3] == personality] or last["enemies"]
    if not candidates:
        return None, [None] * len(frames)

    def distance(enemy, frame):
        return math.hypot(enemy[0] + enemy[2] / 2 - frame["x"],
                          enemy[1] + enemy[2] / 2 - frame["y"])

    killer = min(candidates, key=lambda e: distance(e, last))
    path = [None] * len(frames)
    path[-2] = killer
    current = killer
    for index in range(len(frames) - 3, -1, -1):
        options = [e for e in frames[index]["enemies"] if e[3] == current[3]]
        if not options:
            break
        nearest = min(options, key=lambda e: math.hypot(e[0] - current[0], e[1] - current[1]))
        if math.hypot(nearest[0] - current[0], nearest[1] - current[1]) > MATCH_PIXELS:
            break
        path[index] = nearest
        current = nearest
    # The death frame has no killer left on screen; hold its last position.
    path[-1] = killer
    age = sum(1 for point in path if point is not None)
    return {"p": killer[3], "size": round(killer[2], 1), "age": age}, [
        None if point is None else [round(point[0], 2), round(point[1], 2)] for point in path]


def _run(seed, frames, result):
    tail = list(frames)
    pid = result["pattern"]
    kind = result["cause"]
    since = 0
    for frame in reversed(tail):
        if frame["pid"] != pid:
            break
        since += 1
    killer, killer_path = _killer_path(tail, kind)
    death = tail[-1]
    return {
        "pid": pid,
        "seed": seed,
        "cause": "rect" if kind in ("static_rect", "moving_rect") else "enemy",
        "kind": kind or "unknown",
        "frames": result["frames"],
        "since": since,
        "score": round(result["score"], 1),
        "psize": death["size"],
        "x": [round(frame["x"], 2) for frame in tail],
        "y": [round(frame["y"], 2) for frame in tail],
        "st": [frame["st"] if frame["pid"] == pid else 0 for frame in tail],
        "rects": [frame["rects"] if frame["pid"] == pid else None for frame in tail],
        "n": [sum(1 for e in frame["enemies"] if e[3] < 2) for frame in tail],
        "gap": [frame["gap"] for frame in tail],
        "kp": killer_path,
        "killer": killer,
        "alive": [sum(1 for e in death["enemies"] if e[3] == p) for p in (0, 1, -1)],
        "armed": any(st == 2 for st in (frame["st"] for frame in tail if frame["pid"] == pid)),
    }


def record(policy, seeds, *, frame_cap=18_000, progress=None, **game_settings):
    """Play one game per seed and return the viewer's data."""
    seeds = np.asarray(seeds, dtype=np.uint32)
    engine = make_engine(**game_settings)
    engine.reset_ml_batch(seeds)
    count = len(seeds)
    rings = [deque(maxlen=TAIL) for _ in range(count)]
    for lane in range(count):
        rings[lane].append(_frame(engine.lane_state(lane)))
    done = [False] * count
    frames = np.zeros(count, dtype=np.int64)
    encounters, deaths, types = {}, {}, {}
    previous = [-1] * count
    runs = []
    finished = 0
    filler = 2**31
    while finished < count:
        observations = engine.observe_velocity()["observation"]
        actions = np.asarray(policy(observations, engine.lane_state), dtype=np.uint8)
        stepped = engine.step_ml_batch(actions.reshape(count))
        frames += stepped["frames_advanced"]
        state = engine.observe_velocity()
        # Finished lanes run throwaway games; a death there only needs a reset.
        restart = [lane for lane in range(count) if done[lane] and state["dead"][lane]]
        for lane in range(count):
            if done[lane]:
                continue
            frame = _frame(engine.lane_state(lane))
            rings[lane].append(frame)
            if frame["pid"] >= 0 and frame["pid"] != previous[lane]:
                encounters[frame["pid"]] = encounters.get(frame["pid"], 0) + 1
                types[frame["pid"]] = "moving" if frame["tpe"] == 1 else "static"
            previous[lane] = frame["pid"]
            died = bool(state["dead"][lane])
            if not died and frames[lane] < frame_cap:
                continue
            done[lane] = True
            finished += 1
            restart.append(lane)
            if progress is not None:
                progress(finished, count)
            pattern = int(state["death_pattern"][lane])
            if died and pattern >= 0:
                deaths[pattern] = deaths.get(pattern, 0) + 1
                runs.append(_run(int(seeds[lane]), rings[lane], {
                    "pattern": pattern, "frames": int(frames[lane]),
                    "score": float(state["score"][lane]),
                    "cause": DEATH_CAUSES.get(int(state["death_cause"][lane])),
                }))
        if restart:
            lanes = np.array(restart, dtype=np.uint32)
            engine.reset_ml_lanes(lanes, np.arange(filler, filler + len(lanes), dtype=np.uint32))
            filler += len(lanes)

    stats = {}
    for pattern, met in encounters.items():
        killed = min(deaths.get(pattern, 0), met)
        stats[pattern] = {"type": types.get(pattern), "encounters": met, "deaths": killed,
                          "rate": round(100 * killed / met, 1) if met else 0.0}
    # Deadliest first, but a pattern met a handful of times has a meaningless
    # rate, so those follow the rest.
    order = sorted(stats, key=lambda p: (stats[p]["encounters"] < MIN_RANKED,
                                         -stats[p]["rate"], -stats[p]["encounters"], p))
    return {"pids": order, "stats": stats, "runs": runs, "games": count}
