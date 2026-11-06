"""Re-record the K=1 argmax games that died inside a static pattern, with the trace on.

The dashboard's death-path recorder (dodge.death_paths.record) on exactly the seeds
whose eval1024 game ended during a static pattern; each recorded death is checked
against the original game (same frame, same cause). Writes paper1024/paths.json:
the died appearances, plus a pid -> static/moving map from every appearance seen.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "D:/projects/DodgeAI/.claude/worktrees/train-plan")
from stable_baselines3 import PPO

from dodge.dashboard import _game_settings
from dodge.death_paths import record
from dodge.evaluate import argmax_policy

HERE = Path(__file__).parent / "runs"
games = {g["seed"]: g for g in json.loads((HERE / "eval1024" / "games.json").read_text())}
settings = {**json.loads(Path("D:/projects/DodgeAI/game-config.json").read_text()), "action_repeat": 1}
act = argmax_policy(PPO.load("D:/projects/DodgeAI/artifacts/models/velocity-flow-v2-best.zip", device="cuda"))

# Pattern types are only in the trace: record every died-in-pattern game, then keep the static ones.
seeds = sorted(s for s, g in games.items() if g["died"] and g["pattern"] not in (-1, None))
print(f"recording {len(seeds)} games that died during a pattern", flush=True)
rec = record(lambda obs, _: act(obs), seeds, **_game_settings(settings))
types = {}
for a in rec.appearances:
    if a["pid"] != -1:
        types[a["pid"]] = a["type"]
died = [a for a in rec.appearances if a["outcome"] == "died"]
CAUSE = {"normal": "normal", "kamikaze": "kamikaze", "kamikaze_blast": "kamikaze",
         "static_rect": "static_rect", "moving_rect": "moving_rect"}
match = 0
for a in died:
    g = games[a["seed"]]
    a["replay_match"] = (a["end"] == g["frames"] and CAUSE.get(a["kind"]) == CAUSE.get(g["cause"]))
    match += a["replay_match"]
print(f"{len(died)} deaths recorded, {match} match the original frame and cause", flush=True)
out = [{**a, "x": [round(float(v), 2) for v in a["x"][-60:]], "y": [round(float(v), 2) for v in a["y"][-60:]]}
       for a in died]
(HERE / "paper1024" / "paths.json").write_text(json.dumps({"types": types, "deaths": out}))
print("wrote paths.json", flush=True)
