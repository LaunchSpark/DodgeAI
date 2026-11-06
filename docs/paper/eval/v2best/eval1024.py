"""The dashboard's Evaluation-page run, outside the dashboard, on the best v2 checkpoint.

Same seeds (EVALUATION_FIRST_SEED onward), same play_games, same game settings,
argmax, uncapped; then the same metrics.evaluation and chart builders the page
uses. Loading the checkpoint into the dashboard would resume PPO training on it.

    python eval1024.py [games]
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "D:/projects/DodgeAI/.claude/worktrees/train-plan")
from stable_baselines3 import PPO

from dodge import charts, metrics
from dodge.dashboard import EVALUATION_FIRST_SEED, _game_settings
from dodge.env import play_games
from dodge.evaluate import argmax_policy

GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
CAP = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None   # smoke tests only; the page is uncapped
K = int(sys.argv[3]) if len(sys.argv) > 3 else 1                # frames per decision
MODE = sys.argv[4] if len(sys.argv) > 4 else "argmax"           # argmax | sampled
POLICY = sys.argv[5] if len(sys.argv) > 5 else "velocity-flow-v2-best"   # a model name, 'random' or 'gradient'
CHECKPOINT = f"D:/projects/DodgeAI/artifacts/models/{POLICY}.zip"
NAME = "eval1024" if (K, MODE) == (1, "argmax") else f"eval1024_k{K}_{MODE}"
if POLICY != "velocity-flow-v2-best":
    NAME = f"eval1024_{POLICY}"
OUT = Path(__file__).parent / "runs" / (NAME if CAP is None else "eval-smoke")
OUT.mkdir(exist_ok=True)

settings = {**json.loads(Path("D:/projects/DodgeAI/game-config.json").read_text()), "action_repeat": K}
if POLICY == "random":
    import numpy as np
    rng = np.random.default_rng(0)
    act = lambda obs: rng.integers(0, 9, len(obs))
elif POLICY == "gradient":
    from dodge.gradient import GradientBot
    bot = GradientBot(action_repeat=K)
elif MODE == "argmax":
    model = PPO.load(CHECKPOINT, device="cuda")
    act = argmax_policy(model)      # predict's argmax, through the policy's CUDA graphs
else:
    import torch
    torch.manual_seed(0)
    model = PPO.load(CHECKPOINT, device="cuda")
    act = lambda obs: model.predict(obs, deterministic=False)[0]   # the page's sampled mode
started = time.time()
last = [0.0]


def progress(finished, total):
    if time.time() - last[0] > 60 or finished == total:
        last[0] = time.time()
        print(f"{finished}/{total} games done, {time.time() - started:.0f}s", flush=True)


games = play_games(bot.policy if POLICY == "gradient" else (lambda obs, _: act(obs)), range(EVALUATION_FIRST_SEED, EVALUATION_FIRST_SEED + GAMES),
                   progress=progress, frame_cap=CAP, **_game_settings(settings))
print(f"played {len(games)} games in {time.time() - started:.0f}s", flush=True)
(OUT / "games.json").write_text(json.dumps(games, default=float))

figures = metrics.evaluation(games)
label = f"Evaluation, {len(games)} fixed seeds"
specs = {"survival": charts.survival_curves({label: figures["survival"]}),
         "causes": charts.causes_by_window(figures["causes"]),
         "hazard": charts.pattern_hazard(figures["hazard"])}
if figures["patterns"]:
    specs["patterns"] = charts.pattern_lethality(figures["patterns"])
(OUT / "figures.json").write_text(json.dumps(figures, default=float))
(OUT / "specs.json").write_text(json.dumps(specs, default=float))
print("wrote", OUT, flush=True)
