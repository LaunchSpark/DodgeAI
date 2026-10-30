"""Record every pattern appearance in a block of games, for the death-path viewer.

A frozen policy plays a block of fixed seeds with the engine's gameplay trace
on (Taylor Turner's recorder, `dodge_native` `lane_trace`). When a game ends,
by death or by the frame cap, its trace gives every pattern appearance in it:
the pattern, when it started and ended, the player's path through it, its
rectangles, and how it ended. A death outside any pattern becomes an
appearance of "no pattern" (pid -1) holding the last frames before it.

Every decision each game took is kept, so any appearance can be replayed
(`dodge.pattern_replay`).
"""

import uuid
from dataclasses import dataclass, field

import numpy as np

from .env import make_engine

# Encounters before a pattern's death rate is ranked.
MIN_RANKED = 5
# Frames of path kept for a death outside any pattern.
LOOSE_TAIL = 120
NO_PATTERN = -1

# The trace's cause names, as the env names them (dodge.env.DEATH_CAUSES).
KINDS = {
    "normal_enemy": "normal",
    "kamikaze_enemy": "kamikaze",
    "kamikaze_explosion": "kamikaze_blast",
    "static_pattern_rect": "static_rect",
    "moving_pattern_rect": "moving_rect",
    "unknown": "unknown",
}
RECT_KINDS = ("static_rect", "moving_rect")


def _points(values):
    return [round(value, 2) for value in values]


def _rects(rects):
    return [[round(x, 2), round(y, 2), round(w, 2), round(h, 2), lethal]
            for x, y, w, h, lethal in rects]


def _death(death, score):
    killer = death["killer"]
    active = death["active_pattern"]
    return {
        "frame": death["frame"],
        "score": round(score, 1),
        "since": active["since"] if active is not None else None,
        "alive": death["enemies_alive"],
        "killer": None if killer is None else {
            "personality": killer["personality"],
            "size": round(killer["size"], 1),
            "age": killer["age"],
            "fyou_spawned": killer["fyou_spawned"],
        },
    }


def _appearances(seed, trace, score):
    """Every appearance in one finished game's trace, in order."""
    death = trace["death"]
    active = death["active_pattern"] if death is not None else None
    kind = KINDS.get(death["cause"], "unknown") if death is not None else None
    cause = None if kind is None else ("rect" if kind in RECT_KINDS else "enemy")
    found = []
    for span in trace["patterns"]:
        base = {"seed": seed, "pid": span["id"],
                "type": "moving" if span["type"] == 1 else "static",
                "start": span["start"], "cause": None, "kind": None, "death": None,
                "x": _points(span["x"]), "y": _points(span["y"]),
                "rects": _rects(span["rects"])}
        if span["end"] is not None:
            found.append({**base, "end": span["end"], "outcome": "survived"})
        elif active is not None and active["start"] == span["start"]:
            if not base["rects"]:
                # Killed before any rectangle turned lethal: draw them as they were.
                base["rects"] = _rects(death["rects"])
            found.append({**base, "end": death["frame"], "outcome": "died", "cause": cause,
                          "kind": kind, "death": _death(death, score)})
        else:
            found.append({**base, "end": trace["frame"], "outcome": "unfinished"})
    if death is not None and active is None:
        recent = death["recent"]
        # The last recorded frame is the one before death; the player has not
        # moved since, so the death frame repeats its position.
        x = (recent["x"][-(LOOSE_TAIL - 1):] + [death["player"]["x"]])
        y = (recent["y"][-(LOOSE_TAIL - 1):] + [death["player"]["y"]])
        found.append({"seed": seed, "pid": NO_PATTERN, "type": None,
                      "start": death["frame"] - len(x) + 1, "end": death["frame"],
                      "outcome": "died", "cause": cause, "kind": kind,
                      "death": _death(death, score), "x": _points(x), "y": _points(y),
                      "rects": []})
    return found


@dataclass
class Recording:
    """One sampling run. An appearance's id is its index in `appearances`."""

    run_id: str
    settings: dict
    games: int
    actions: dict = field(repr=False)
    appearances: list = field(repr=False)

    def summary(self):
        stats = {}
        for appearance in self.appearances:
            entry = stats.setdefault(appearance["pid"], {
                "pid": appearance["pid"], "type": appearance["type"],
                "encounters": 0, "deaths": 0})
            entry["encounters"] += 1
            entry["deaths"] += appearance["outcome"] == "died"
        for entry in stats.values():
            entry["rate"] = (round(100 * entry["deaths"] / entry["encounters"], 1)
                             if entry["pid"] != NO_PATTERN else None)
        loose = stats.pop(NO_PATTERN, None)
        # Deadliest first, but a pattern met a handful of times has a
        # meaningless rate, so those follow the rest; no pattern comes last.
        order = sorted(stats.values(), key=lambda p: (p["encounters"] < MIN_RANKED,
                                                      -p["rate"], -p["encounters"], p["pid"]))
        return {"run_id": self.run_id, "games": self.games,
                "patterns": order + ([loose] if loose is not None else [])}

    def pattern(self, pid):
        appearances = [a for a in self.appearances if a["pid"] == pid]
        layouts = {tuple(map(tuple, a["rects"])) for a in appearances if a["rects"]}
        layout = [list(rect) for rect in next(iter(layouts))] if len(layouts) == 1 else None
        return {"run_id": self.run_id, "pid": pid, "layout": layout,
                "appearances": appearances}


def record(policy, seeds, *, frame_cap=18_000, progress=None, **game_settings):
    """Play one game per seed with the trace on and return the Recording."""
    seeds = np.asarray(seeds, dtype=np.uint32)
    count = len(seeds)
    engine = make_engine(**game_settings)
    engine.set_trace_enabled(True)
    engine.reset_ml_batch(seeds)
    decisions = [bytearray() for _ in range(count)]
    frames = np.zeros(count, dtype=np.int64)
    done = np.zeros(count, dtype=bool)
    appearances = []
    finished = 0
    filler = 2**31
    while finished < count:
        observations = engine.observe_velocity()["observation"]
        actions = np.asarray(policy(observations, engine.lane_state),
                             dtype=np.uint8).reshape(count)
        for lane in np.flatnonzero(~done):
            decisions[lane].append(int(actions[lane]))
        stepped = engine.step_ml_batch(actions)
        frames += stepped["frames_advanced"]
        over = np.asarray(stepped["done"], dtype=bool)
        # Finished lanes run throwaway games; a death there only needs a reset.
        restart = np.flatnonzero(done & over).tolist()
        ending = np.flatnonzero(~done & (over | (frames >= frame_cap)))
        if len(ending):
            scores = engine.observe_velocity()["score"]
            for lane in ending:
                trace = engine.lane_trace(int(lane))
                appearances.extend(_appearances(int(seeds[lane]), trace, float(scores[lane])))
                done[lane] = True
                finished += 1
                restart.append(int(lane))
                if progress is not None:
                    progress(finished, count)
        if restart:
            lanes = np.array(restart, dtype=np.uint32)
            engine.reset_ml_lanes(lanes, np.arange(filler, filler + len(lanes), dtype=np.uint32))
            filler += len(lanes)
    for index, appearance in enumerate(appearances):
        appearance["id"] = index
    return Recording(
        run_id=uuid.uuid4().hex,
        settings={"step_frames": 1, "frame_cap": frame_cap, **game_settings},
        games=count,
        actions={int(seed): np.frombuffer(bytes(decisions[lane]), dtype=np.uint8)
                 for lane, seed in enumerate(seeds)},
        appearances=appearances)
